"""Continuous AlphaZero loop: self-play workers + trainer in one process (plan §3).

    python continuous.py --run-dir runs/T5-4x48 --engine-bin target/release/taflzero.exe \
        --channels 48 --blocks 4 --workers 8 --train-threads 2 --batch 512

Three threads share a `ReplayBuffer`:

  ingest     tails data/worker<N>.bin every `ingest_interval`, adds whole games
             to the buffer, keeps self-play statistics, toggles the PAUSE file
             when training lags too far behind generation;
  trainer    paces itself to `steps * batch <= sample_reuse * generated`,
             exports latest.onnx/.qnxx every `sync_every` steps (atomically,
             so workers with --net-watch pick it up), checkpoints every
             `checkpoint_every`;
  supervisor spawns the engine workers, restarts crashed ones, and waits for
             them on STOP.

The STOP file is the only stop mechanism (SIGINT just creates it); shutdown
follows plan §3.8: trainer finishes its step and checkpoints, workers finish
their games and exit, ingest reads the tails, state.json is rewritten, exit 0.
`--resume <run_dir>` continues from state.json + latest.qnxx + optimizer.pt and
re-reads the replay window from the per-file `window_start` offsets.

Any option can also come from a JSON file (`--config run.json`, keys = CLI long
names without the leading dashes, dashes or underscores inside); CLI flags
override the file.  The effective config is written to <run_dir>/config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from az_micro_net import TaflAlphaZeroNet
from dataset import geometry
from export_onnx import export_model_to_onnx
from ingest import TERMINAL_COLUMNS, Ingester, SelfplayStats, chunk_outcome
from qnxx_io import load_qnxx, save_qnxx
from replay import ReplayBuffer, replay_window
from train import compute_batch_loss

METRICS_COLUMNS = [
    "time", "step", "generated", "games", "window", "buffer", "lr", "loss", "policy",
    "value", "aux_p", "corner", "moves_left", "grad_norm", "unknown_frac", "steps_per_s",
    "samples_per_s", "paused", "gpu_util", "gpu_mem_gb",
]
SELFPLAY_COLUMNS = [
    "time", "games", "generated", "atk_wins", "def_wins", "undecided", "atk_pct", "avg_len",
] + TERMINAL_COLUMNS
SELFPLAY_ROW_EVERY_GAMES = 500
SELFPLAY_ROW_EVERY_SECONDS = 120.0
SELFPLAY_ROW_MIN_GAMES = 50

WORKER_RESTART_DELAY_S = 5.0
WORKER_CRASH_LIMIT = 10
WORKER_CRASH_WINDOW_S = 600.0
WORKER_STOP_GRACE_S = 120.0
BAD_LOSS_LIMIT = 20

SMOKE_PRESET = {
    "workers": 2, "channels": 48, "blocks": 4, "full_nodes": 20, "cheap_nodes": 8,
    "min_samples": 2000, "sync_every": 10, "checkpoint_every": 50, "max_steps": 120,
}


# ─── Config ─────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Continuous self-play + training loop (plan §3)")
    p.add_argument("--config", type=Path, default=None, help="JSON file with option values (keys = long names)")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--engine-bin", type=Path, default=None, help="taflzero executable (required if --workers > 0)")
    p.add_argument("--resume", type=Path, default=None, help="Continue a run from this run_dir")
    p.add_argument("--start-checkpoint", type=Path, default=None, help="Initial weights (.qnxx); steps start at 0")
    p.add_argument("--smoke", action="store_true", help="Small preset for the integration test (plan §3.10)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--amp", choices=["none", "bf16"], default="none",
                   help="mixed precision for the trainer's forward pass (CUDA); weights stay fp32")
    p.add_argument("--seed", type=int, default=1)

    # network architecture (same flags as train.py)
    p.add_argument("--board-size", type=int, default=11)
    p.add_argument("--channels", type=int, default=48)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--value-channels", type=int, default=1)
    p.add_argument("--se", action="store_true")
    p.add_argument("--gpool-channels", type=int, default=0)
    p.add_argument("--aux-heads", action="store_true")
    p.add_argument("--moves-left", action="store_true")

    # loop
    p.add_argument("--workers", type=int, default=None, help="engine processes (default cpu_count - train_threads - 1, max 8); 0 = trainer only")
    p.add_argument("--train-threads", type=int, default=2)
    p.add_argument("--decode-threads", type=int, default=1)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=None, help="default: sgd 0.02*batch/1024, adamw 5e-4")
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--sample-reuse", type=float, default=4.0)
    p.add_argument("--min-samples", type=int, default=100_000)
    p.add_argument("--window-base", type=int, default=250_000)
    p.add_argument("--window-cap", type=int, default=2_000_000)
    p.add_argument("--window-alpha", type=float, default=0.75)
    p.add_argument("--window-beta", type=float, default=0.4)
    p.add_argument("--buffer-max-bytes", type=int, default=0, help="0 = unlimited; else shrink the window to fit")
    p.add_argument("--max-training-lag", type=int, default=200_000)
    p.add_argument("--sync-every", type=int, default=40)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ingest-interval", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = run until STOP")
    p.add_argument("--export-fp16", action="store_true", help="also write latest.fp16.onnx and point workers at it")

    # loss
    p.add_argument("--value-lambda", type=float, default=0.5)
    p.add_argument("--value-gamma", type=float, default=1.0)
    p.add_argument("--aux-policy-weight", type=float, default=0.15)
    p.add_argument("--corner-weight", type=float, default=0.1)
    p.add_argument("--decided-policy-k", type=float, default=0.0)
    p.add_argument("--moves-left-weight", type=float, default=0.1)

    # engine
    p.add_argument("--full-nodes", type=int, default=160)
    p.add_argument("--cheap-nodes", type=int, default=40)
    p.add_argument("--full-prob", type=float, default=0.25)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--max-no-capture", type=int, default=None)
    p.add_argument("--variant", type=str, default="copenhagen11x11")
    p.add_argument("--gumbel", type=int, default=None, help="Gumbel root search with this many considered moves")
    p.add_argument("--worker-threads", type=int, default=1,
                   help="game threads per worker process, served by one batching net (engine --threads)")
    p.add_argument("--mlh-alpha", type=float, default=0.15)
    return p


_META_KEYS = {"config", "resume"}


def _normalize_keys(d: dict) -> dict:
    return {k.lstrip("-").replace("-", "_"): v for k, v in d.items()}


def _coerce_for_parser(parser: argparse.ArgumentParser, d: dict) -> dict:
    """Make JSON values match the parser's types (Path fields, ints from floats)."""
    actions = {a.dest: a for a in parser._actions}
    out = {}
    for k, v in d.items():
        a = actions.get(k)
        if a is None:
            raise SystemExit(f"unknown config key: {k!r}")
        if v is not None and a.type is Path:
            v = Path(v)
        elif v is not None and a.type is int and isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


def parse_config(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI > --config file > <resume dir>/config.json > --smoke preset > defaults."""
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    defaults: dict = {}
    if pre.smoke:
        defaults.update(SMOKE_PRESET)
    if pre.resume is not None:
        cfg_path = Path(pre.resume) / "config.json"
        if not cfg_path.exists():
            raise SystemExit(f"--resume: {cfg_path} not found")
        defaults.update(_normalize_keys(json.loads(cfg_path.read_text(encoding="utf-8"))))
        defaults["run_dir"] = str(pre.resume)
    if pre.config is not None:
        defaults.update(_normalize_keys(json.loads(Path(pre.config).read_text(encoding="utf-8"))))
    defaults = {k: v for k, v in defaults.items() if k not in _META_KEYS}
    parser.set_defaults(**_coerce_for_parser(parser, defaults))
    args = parser.parse_args(argv)
    if args.run_dir is None:
        parser.error("--run-dir is required")
    if args.workers is None:
        args.workers = max(1, min(8, (os.cpu_count() or 4) - args.train_threads - 1))
    if args.workers > 0 and args.engine_bin is None:
        parser.error("--engine-bin is required when --workers > 0")
    if args.lr is None:
        args.lr = 0.02 * args.batch / 1024 if args.optimizer == "sgd" else 5e-4
    return args


def config_to_json(args: argparse.Namespace) -> dict:
    out = {}
    for k, v in sorted(vars(args).items()):
        if k in _META_KEYS:
            continue
        out[k] = str(v) if isinstance(v, Path) else v
    return out


# ─── Small helpers ──────────────────────────────────────────────────────────


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def replace_with_retry(src: Path, dst: Path, attempts: int = 10, delay: float = 0.2) -> None:
    """os.replace that tolerates a reader holding `dst` open for a moment
    (ORT reads the model file while building a session)."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


class Log:
    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self._lock:
            print(line, flush=True)
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# Console children of a windowless parent would each open a console window on Windows.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def gpu_stats() -> tuple[str, str]:
    """(utilisation %, memory used GB) of GPU 0 from nvidia-smi; blanks if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW,
        ).stdout.strip().splitlines()[0]
        util, mem = (x.strip() for x in out.split(","))
        return util, f"{float(mem) / 1024:.2f}"
    except Exception:
        return "", ""


class CsvLog:
    def __init__(self, path: Path, columns: list[str]) -> None:
        self.columns = columns
        if path.exists() and path.stat().st_size > 0:
            # A resumed run whose file predates a column change: rewrite it under
            # the new header so rows stay aligned (missing columns are blank).
            with open(path, newline="", encoding="utf-8") as fh:
                old = list(csv.DictReader(fh))
            with open(path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])
            if header != columns:
                tmp = path.with_name(path.name + ".tmp")
                with open(tmp, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(columns)
                    for r in old:
                        w.writerow([r.get(c, "") for c in columns])
                os.replace(tmp, path)
        new = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(columns)
            self._fh.flush()

    def row(self, values: dict) -> None:
        self._w.writerow([values.get(c, "") for c in self.columns])
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def pacing_allowed_steps(generated: int, batch: int, sample_reuse: float) -> int:
    """Max steps allowed so far: steps * batch <= sample_reuse * generated."""
    return int(math.floor(sample_reuse * generated / batch))


def training_lag(generated: int, step: int, batch: int, sample_reuse: float) -> float:
    """How far generation is ahead of training, in reuse-weighted samples."""
    return sample_reuse * generated - step * batch


def build_model(args: argparse.Namespace) -> TaflAlphaZeroNet:
    return TaflAlphaZeroNet(
        board_size=args.board_size, trunk_channels=args.channels, num_blocks=args.blocks,
        value_channels=args.value_channels, use_se=args.se, gpool_channels=args.gpool_channels,
        aux_heads=args.aux_heads, moves_left=args.moves_left,
    )


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                               weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def lr_at(step: int, base_lr: float, warmup: int) -> float:
    """Linear warmup from 0 over `warmup` steps, then constant (step is 1-based)."""
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    return base_lr


def worker_command(args: argparse.Namespace, run_dir: Path, worker_id: int, net_path: Path) -> list[str]:
    """Engine CLI for one self-play worker (plan §3.7)."""
    data = run_dir / "data"
    cmd = [
        str(args.engine_bin),
        "--net", str(net_path),
        "--net-watch",
        "--datagen", str(data / f"worker{worker_id}.bin"),
        "--datagen-count", "0",
        "--gamelog", str(data / f"worker{worker_id}.gamelog"),
        "--stop-file", str(run_dir / "STOP"),
        "--pause-file", str(run_dir / "PAUSE"),
        "--worker-id", str(worker_id),
        "--full-nodes", str(args.full_nodes),
        "--cheap-nodes", str(args.cheap_nodes),
        "--full-prob", str(args.full_prob),
        "--max-plies", str(args.max_plies),
        "--variant", str(args.variant),
    ]
    if args.max_no_capture is not None:
        cmd += ["--max-no-capture", str(args.max_no_capture)]
    if args.gumbel is not None:
        cmd += ["--gumbel", str(args.gumbel)]
    if args.worker_threads > 1:
        cmd += ["--threads", str(args.worker_threads)]
    return cmd


# ─── Run state ──────────────────────────────────────────────────────────────


@dataclass
class StepStats:
    loss: float
    policy: float
    value: float
    aux_p: float
    corner: float
    moves_left: float
    grad_norm: float
    unknown_frac: float
    lr: float


class ContinuousRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        self.data_dir = self.run_dir / "data"
        self.ckpt_dir = self.run_dir / "ckpt"
        self.stop_file = self.run_dir / "STOP"
        self.pause_file = self.run_dir / "PAUSE"
        self.latest_onnx = self.run_dir / "latest.onnx"
        self.latest_fp16 = self.run_dir / "latest.fp16.onnx"
        self.latest_qnxx = self.run_dir / "latest.qnxx"
        self.state_path = self.run_dir / "state.json"
        self.optimizer_path = self.run_dir / "optimizer.pt"
        self.worker_net = self.latest_fp16 if args.export_fp16 else self.latest_onnx

        for d in (self.run_dir, self.data_dir, self.ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.run_dir / "log.txt")

        self.geom = geometry(args.board_size)
        self.device = torch.device(args.device)
        self.amp_dtype = torch.bfloat16 if args.amp == "bf16" and self.device.type == "cuda" else None
        torch.set_num_threads(args.train_threads)

        self.lock = threading.Lock()
        self.buffer = ReplayBuffer(
            self.geom, z_lambda=args.value_lambda, augment=True, value_gamma=args.value_gamma,
            max_bytes=args.buffer_max_bytes, decode_threads=args.decode_threads,
        )
        self.stats = SelfplayStats()
        self.step = 0
        self.export_count = 0
        self._last_ckpt_step = -1
        self.paused = False
        self.error: str | None = None
        self._stop_event = threading.Event()      # internal: threads wind down
        self._bytes_warned = False
        self.rng = np.random.default_rng(args.seed)
        self.ingester = Ingester(self.data_dir, self.geom)

        self.metrics = CsvLog(self.run_dir / "metrics.csv", METRICS_COLUMNS)
        self.selfplay_csv = CsvLog(self.run_dir / "selfplay.csv", SELFPLAY_COLUMNS)
        self._selfplay_row_stats = SelfplayStats()   # snapshot at the last selfplay.csv row
        self._selfplay_row_time = time.time()

        self.model: TaflAlphaZeroNet
        self.optimizer: torch.optim.Optimizer
        if args.resume is not None:
            self._init_resume()
        else:
            self._init_fresh()
        self.use_aux = "aux_policy_head" in self.model._modules and (
            args.aux_policy_weight > 0.0 or args.corner_weight > 0.0
        )

    # ─── init ───────────────────────────────────────────────────────────────

    def _init_fresh(self) -> None:
        args = self.args
        if self.state_path.exists():
            raise SystemExit(f"{self.run_dir} already holds a run; use --resume {self.run_dir}")
        torch.manual_seed(args.seed)
        if args.start_checkpoint is not None:
            self.model = load_qnxx(Path(args.start_checkpoint))
            if args.moves_left and not self.model.has_moves_left:
                self.model.enable_moves_left()
            self.log(f"[init] weights from {args.start_checkpoint}: {self.model.model_kwargs}")
        else:
            self.model = build_model(args)
            self.log(f"[init] new model {self.model.model_kwargs}")
        self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, args)
        atomic_write_text(self.run_dir / "config.json", json.dumps(config_to_json(args), indent=2))
        self.export_latest()
        self.save_state()

    def _init_resume(self) -> None:
        args = self.args
        if not self.state_path.exists() or not self.latest_qnxx.exists():
            raise SystemExit(f"--resume: {self.run_dir} has no state.json / latest.qnxx")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.model = load_qnxx(self.latest_qnxx).to(self.device)
        self.optimizer = build_optimizer(self.model, args)
        if self.optimizer_path.exists():
            opt = torch.load(self.optimizer_path, map_location="cpu", weights_only=False)
            try:
                self.optimizer.load_state_dict(opt["optimizer"])
            except (ValueError, KeyError) as e:  # optimizer type changed on resume
                self.log(f"[resume] optimizer state not loaded ({e}); starting it fresh")
            if "torch_rng" in opt:
                torch.set_rng_state(opt["torch_rng"])
        self.step = int(state.get("step", 0))
        self.export_count = int(state.get("export_count", 0))
        self.stats = SelfplayStats.from_dict(state.get("selfplay", {}))
        self._selfplay_row_stats = self.stats.copy()
        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]
        offsets = {k: int(v) for k, v in state.get("offsets", {}).items()}
        self.ingester.offsets = dict(offsets)
        self.ingester.gamelog_offsets = {k: int(v) for k, v in state.get("gamelog_offsets", {}).items()}

        # Re-read the replay window: each file from the first game still in the
        # window up to the ingest offset (both are game boundaries).
        window_start = state.get("window_start", {})
        window = self.current_window()
        reloaded = 0
        for path in self.ingester.bin_files():
            start = int(window_start.get(path.name, offsets.get(path.name, 0)))
            end = offsets.get(path.name, 0)
            if end <= start:
                continue
            chunks, _ = self.ingester.read_range(path, start, end)
            for c in chunks:
                self.buffer.add(c, window)
                reloaded += len(c)
        # Rewrite config.json so CLI overrides on resume are recorded.
        atomic_write_text(self.run_dir / "config.json", json.dumps(config_to_json(args), indent=2))
        self.log(f"[resume] step={self.step} generated={self.stats.generated} games={self.stats.games} "
                 f"reloaded {reloaded} samples into the buffer")
        if not self.worker_net.exists():
            self.export_latest()

    # ─── shared state ───────────────────────────────────────────────────────

    def current_window(self) -> int:
        a = self.args
        return replay_window(self.stats.generated, a.window_base, a.window_alpha, a.window_beta, a.window_cap)

    def stop_requested(self) -> bool:
        return self._stop_event.is_set() or self.stop_file.exists()

    def request_stop(self, reason: str) -> None:
        if not self.stop_file.exists():
            try:
                self.stop_file.touch()
            except OSError:
                pass
        self.log(f"[stop] {reason}")

    def fail(self, where: str, exc: BaseException) -> None:
        self.error = f"{where}: {exc!r}"
        self.log(f"[error] {where}: {traceback.format_exc()}")
        self.request_stop(f"fatal error in {where}")

    # ─── export / checkpoint ────────────────────────────────────────────────

    def _export_onnx_to(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        was_training = self.model.training
        self.model.eval()
        # Module.cpu() moves in place and returns self, so remember whether to move back.
        moved = self.device.type != "cpu"
        model_cpu = self.model.cpu() if moved else self.model
        try:
            export_model_to_onnx(model_cpu, tmp, fp16=False)
            replace_with_retry(tmp, path)
            if self.args.export_fp16:
                tmp16 = self.latest_fp16.with_name(self.latest_fp16.name + ".tmp")
                import onnx
                from onnxconverter_common import float16
                m = onnx.load(str(path))
                onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), str(tmp16))
                replace_with_retry(tmp16, self.latest_fp16)
        finally:
            if moved:
                self.model.to(self.device)
            if was_training:
                self.model.train()

    def export_latest(self) -> None:
        """Atomically refresh latest.onnx (+ latest.fp16.onnx) and latest.qnxx."""
        self._export_onnx_to(self.latest_onnx)
        tmp = self.latest_qnxx.with_name(self.latest_qnxx.name + ".tmp")
        save_qnxx(self.model, tmp)
        replace_with_retry(tmp, self.latest_qnxx)
        self.export_count += 1

    def state_dict(self) -> dict:
        with self.lock:
            window_start = self.buffer.window_starts()
            selfplay = self.stats.to_dict()
            buffer_len = len(self.buffer)
        return {
            "step": self.step,
            "generated": selfplay["generated"],
            "games": selfplay["games"],
            "export_count": self.export_count,
            "offsets": dict(self.ingester.offsets),
            "gamelog_offsets": dict(self.ingester.gamelog_offsets),
            "window_start": window_start,
            "window": self.current_window(),
            "buffer": buffer_len,
            "selfplay": selfplay,
            "rng": self.rng.bit_generator.state,
            "time": time.time(),
        }

    def save_state(self) -> None:
        atomic_write_text(self.state_path, json.dumps(self.state_dict(), indent=2))

    def save_checkpoint(self) -> None:
        if self._last_ckpt_step == self.step:
            return  # already on disk (e.g. STOP right after a periodic checkpoint)
        self._last_ckpt_step = self.step
        tag = f"step{self.step:06d}"
        was_training = self.model.training
        self.model.eval()
        try:
            self._export_onnx_to(self.ckpt_dir / f"{tag}.onnx")
            save_qnxx(self.model, self.ckpt_dir / f"{tag}.qnxx")
            # Where the run was when this net was saved: the evaluator plots by positions.
            with self.lock:
                meta = {"step": self.step, "generated": self.stats.generated, "games": self.stats.games,
                        "time": time.time()}
            atomic_write_text(self.ckpt_dir / f"{tag}.json", json.dumps(meta))
        finally:
            if was_training:
                self.model.train()
        tmp = self.optimizer_path.with_name(self.optimizer_path.name + ".tmp")
        torch.save({"optimizer": self.optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "step": self.step}, tmp)
        replace_with_retry(tmp, self.optimizer_path)
        self.save_state()
        self.log(f"[ckpt] {tag} saved")

    # ─── trainer ────────────────────────────────────────────────────────────

    def train_step(self, batch) -> StepStats | None:
        a = self.args
        lr = lr_at(self.step + 1, a.lr, a.warmup_steps)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.model.train()
        total, p_loss, v_loss, ap_loss, c_loss, ml_loss = compute_batch_loss(
            self.model, batch, self.device, 1.0, a.aux_policy_weight, a.corner_weight,
            self.use_aux, decided_policy_k=a.decided_policy_k, moves_left_weight=a.moves_left_weight,
            amp_dtype=self.amp_dtype,
        )
        loss = float(total.item())
        if not math.isfinite(loss):
            return None
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), a.grad_clip))
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return None
        self.optimizer.step()
        unknown_frac = 1.0 - float(batch[11].mean().item())
        return StepStats(loss, float(p_loss.item()), float(v_loss.item()), float(ap_loss.item()),
                         float(c_loss.item()), float(ml_loss.item()), grad_norm, unknown_frac, lr)

    def sample_batch(self, window: int):
        with self.lock:
            idx = self.buffer.draw_indices(self.args.batch, window, self.rng)
            located = [self.buffer.locate(int(k)) for k in idx]
            syms = self.rng.integers(0, 8, size=self.args.batch)
        return self.buffer.collate(located, syms)

    def trainer_loop(self) -> None:
        a = self.args
        acc: list[StepStats] = []
        bad = 0
        t_last = time.time()
        step_at_last = self.step
        while not self.stop_requested():
            with self.lock:
                n = len(self.buffer)
                generated = self.stats.generated
            if n == 0 or n < a.min_samples:
                time.sleep(0.05)
                continue
            if self.step >= pacing_allowed_steps(generated, a.batch, a.sample_reuse):
                time.sleep(0.05)
                continue
            window = self.current_window()
            batch = self.sample_batch(window)
            st = self.train_step(batch)
            if st is None:
                bad += 1
                self.log(f"[train] non-finite loss at step {self.step + 1} ({bad}/{BAD_LOSS_LIMIT}), skipped")
                if bad >= BAD_LOSS_LIMIT:
                    raise RuntimeError(f"{BAD_LOSS_LIMIT} consecutive non-finite losses")
                continue
            bad = 0
            self.step += 1
            acc.append(st)
            if self.step % a.sync_every == 0:
                self.export_latest()
            if self.step % a.log_every == 0:
                now = time.time()
                dt = max(now - t_last, 1e-9)
                steps_per_s = (self.step - step_at_last) / dt
                self._write_metrics(acc, window, steps_per_s)
                acc = []
                t_last, step_at_last = now, self.step
            if a.checkpoint_every > 0 and self.step % a.checkpoint_every == 0:
                self.save_checkpoint()
            if a.max_steps > 0 and self.step >= a.max_steps:
                self.request_stop(f"max_steps {a.max_steps} reached")
                break
        # §3.8: finish the step, then checkpoint before anything else winds down.
        self.save_checkpoint()

    def _write_metrics(self, acc: list[StepStats], window: int, steps_per_s: float) -> None:
        n = max(1, len(acc))
        mean = lambda f: sum(f(s) for s in acc) / n  # noqa: E731
        with self.lock:
            generated, games, buf = self.stats.generated, self.stats.games, len(self.buffer)
            atk_pct, avg_len = self.stats.atk_pct, self.stats.avg_len
        row = {
            "time": f"{time.time():.1f}", "step": self.step, "generated": generated, "games": games,
            "window": window, "buffer": buf, "lr": f"{acc[-1].lr:.6g}" if acc else "",
            "loss": f"{mean(lambda s: s.loss):.5f}", "policy": f"{mean(lambda s: s.policy):.5f}",
            "value": f"{mean(lambda s: s.value):.5f}", "aux_p": f"{mean(lambda s: s.aux_p):.5f}",
            "corner": f"{mean(lambda s: s.corner):.5f}", "moves_left": f"{mean(lambda s: s.moves_left):.5f}",
            "grad_norm": f"{mean(lambda s: s.grad_norm):.4f}",
            "unknown_frac": f"{mean(lambda s: s.unknown_frac):.4f}",
            "steps_per_s": f"{steps_per_s:.3f}", "samples_per_s": f"{steps_per_s * self.args.batch:.1f}",
            "paused": int(self.paused),
        }
        if self.device.type == "cuda":
            row["gpu_util"], row["gpu_mem_gb"] = gpu_stats()
        self.metrics.row(row)
        self.log(f"step {self.step} gen {generated} games {games} atk {atk_pct:.1f}% len {avg_len:.0f} "
                 f"loss {row['loss']} (p {row['policy']} v {row['value']}) {row['samples_per_s']} samples/s"
                 + (" PAUSED" if self.paused else ""))

    # ─── ingest ─────────────────────────────────────────────────────────────

    def ingest_once(self) -> int:
        """One poll of the worker files; returns the number of new samples."""
        res = self.ingester.poll()
        with self.lock:
            for c in res.chunks:
                self.stats.add_game(chunk_outcome(c), len(c))
                self.buffer.add(c, self.current_window())
            for name, cnt in res.terminals.items():
                self.stats.by_terminal[name] += cnt
            capped = self.buffer.bytes_capped
            games = self.stats.games
        if capped and not self._bytes_warned:
            self._bytes_warned = True
            self.log(f"[buffer] exceeds --buffer-max-bytes {self.args.buffer_max_bytes}; window shrunk")
        self._update_pause()
        # A row is one logging window; a window of a handful of games makes the per-window
        # shares jump to 0/100, so never close one before SELFPLAY_ROW_MIN_GAMES.
        since = games - self._selfplay_row_stats.games
        due_by_games = since >= SELFPLAY_ROW_EVERY_GAMES
        due_by_time = since >= SELFPLAY_ROW_MIN_GAMES and time.time() - self._selfplay_row_time >= SELFPLAY_ROW_EVERY_SECONDS
        if due_by_games or due_by_time:
            self._write_selfplay_row()
        return res.samples

    def _update_pause(self) -> None:
        a = self.args
        with self.lock:
            lag = training_lag(self.stats.generated, self.step, a.batch, a.sample_reuse)
        threshold = a.sample_reuse * a.max_training_lag
        if lag > threshold and not self.paused:
            self.paused = True
            self.pause_file.touch()
            self.log(f"[pace] training lags {lag / a.sample_reuse:.0f} samples; PAUSE created")
        elif lag < threshold / 2 and self.paused:
            self.paused = False
            self.pause_file.unlink(missing_ok=True)
            self.log("[pace] caught up; PAUSE removed")

    def _write_selfplay_row(self) -> None:
        with self.lock:
            cur = self.stats.copy()
        recent = cur.diff(self._selfplay_row_stats)
        row = {
            "time": f"{time.time():.1f}", "games": cur.games, "generated": cur.generated,
            "atk_wins": cur.atk_wins, "def_wins": cur.def_wins, "undecided": cur.undecided,
            "atk_pct": f"{recent.atk_pct:.2f}", "avg_len": f"{recent.avg_len:.1f}",
        }
        for col in TERMINAL_COLUMNS:
            row[col] = cur.by_terminal.get(col, 0)
        self.selfplay_csv.row(row)
        self._selfplay_row_stats = cur
        self._selfplay_row_time = time.time()

    def ingest_loop(self) -> None:
        while not self._stop_event.is_set():
            self.ingest_once()
            self._stop_event.wait(self.args.ingest_interval)

    # ─── supervisor ─────────────────────────────────────────────────────────

    def supervisor_loop(self) -> None:
        a = self.args
        if a.workers <= 0:
            return
        procs: dict[int, subprocess.Popen | None] = {}
        restart_at: dict[int, float] = {}
        logs: dict[int, object] = {}
        crashes: deque[float] = deque()
        env = dict(os.environ, TAFLZERO_ORT_THREADS="1", TAFLZERO_MLH_ALPHA=str(a.mlh_alpha))
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | NO_WINDOW if os.name == "nt" else 0

        def spawn(i: int) -> None:
            if i not in logs:
                logs[i] = open(self.data_dir / f"worker{i}.log", "a", encoding="utf-8")
            cmd = worker_command(a, self.run_dir, i, self.worker_net)
            procs[i] = subprocess.Popen(
                cmd, cwd=str(self.run_dir), stdin=subprocess.DEVNULL, stdout=logs[i],
                stderr=subprocess.STDOUT, env=env, creationflags=creationflags,
            )
            self.log(f"[W{i}] started pid {procs[i].pid}")

        try:
            self.log(f"[supervisor] launching {a.workers} workers: {' '.join(worker_command(a, self.run_dir, 0, self.worker_net))}")
            for i in range(a.workers):
                spawn(i)
            while not self.stop_requested():
                now = time.time()
                for i in range(a.workers):
                    p = procs.get(i)
                    if p is None:
                        if now >= restart_at.get(i, 0.0):
                            spawn(i)
                        continue
                    rc = p.poll()
                    if rc is None:
                        continue
                    if self.stop_file.exists():
                        continue
                    crashes.append(now)
                    while crashes and now - crashes[0] > WORKER_CRASH_WINDOW_S:
                        crashes.popleft()
                    self.log(f"[W{i}] exited with code {rc} ({len(crashes)} crashes in the last 10 min); restart in {WORKER_RESTART_DELAY_S:.0f}s")
                    if len(crashes) >= WORKER_CRASH_LIMIT:
                        self.error = f"{WORKER_CRASH_LIMIT} worker crashes within {WORKER_CRASH_WINDOW_S:.0f}s"
                        self.request_stop(self.error)
                        break
                    procs[i] = None
                    restart_at[i] = now + WORKER_RESTART_DELAY_S
                time.sleep(1.0)
            # STOP: workers finish their current game and exit on their own.
            deadline = time.time() + WORKER_STOP_GRACE_S
            for i, p in procs.items():
                if p is None:
                    continue
                try:
                    p.wait(timeout=max(0.0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    self.log(f"[W{i}] did not exit within {WORKER_STOP_GRACE_S:.0f}s; terminating")
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
            self.log("[supervisor] all workers exited")
        finally:
            for fh in logs.values():
                fh.close()

    # ─── main ───────────────────────────────────────────────────────────────

    def _thread(self, name: str, fn) -> threading.Thread:
        def runner():
            try:
                fn()
            except BaseException as e:  # noqa: BLE001
                self.fail(name, e)
        t = threading.Thread(target=runner, name=name, daemon=True)
        t.start()
        return t

    def run(self) -> int:
        a = self.args
        self.pause_file.unlink(missing_ok=True)
        if self.stop_file.exists():
            self.log("[init] stale STOP file removed")
            self.stop_file.unlink()
        self.log(f"[init] run_dir={self.run_dir} workers={a.workers} batch={a.batch} lr={a.lr} "
                 f"optimizer={a.optimizer} train_threads={a.train_threads} device={self.device} "
                 f"amp={'bf16' if self.amp_dtype is not None else 'none'}")

        def on_sigint(signum, frame):
            self.request_stop("SIGINT")

        try:
            signal.signal(signal.SIGINT, on_sigint)
        except ValueError:
            pass  # not the main thread (tests)

        t_ingest = self._thread("ingest", self.ingest_loop)
        t_train = self._thread("trainer", self.trainer_loop)
        t_super = self._thread("supervisor", self.supervisor_loop)

        while not self.stop_requested():
            time.sleep(0.5)
        # §3.8 order: trainer (checkpoint) -> workers -> ingest tails -> state.
        t_train.join()
        t_super.join()
        self._stop_event.set()
        t_ingest.join()
        self.ingest_once()
        self.save_state()
        self.pause_file.unlink(missing_ok=True)
        self.buffer.close()
        rc = 1 if self.error else 0
        self.log(f"[exit] step={self.step} generated={self.stats.generated} games={self.stats.games} rc={rc}")
        self.metrics.close()
        self.selfplay_csv.close()
        self.log.close()
        return rc


def main(argv: list[str] | None = None) -> int:
    args = parse_config(argv)
    run = ContinuousRun(args)
    return run.run()


if __name__ == "__main__":
    sys.exit(main())
