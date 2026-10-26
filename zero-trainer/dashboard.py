"""Training dashboard: start / stop / resume continuous runs and watch their progress.

    python zero-trainer/dashboard.py [--port 8787] [--runs-dir zero-trainer/runs]
                                     [--engine-bin target/release/taflzero.exe]

Opens http://127.0.0.1:8787. One page: a list of runs under --runs-dir, a "new run"
form built from zero-trainer/presets/*.json, and for the selected run the live
numbers (step, positions, attacker share, speed) with charts of Elo against the
anchors, losses, self-play outcome shares and the blind-spot metrics, all read
straight from the run directory (metrics.csv, selfplay.csv, elo.jsonl, eval.jsonl,
state.json). Nothing but the standard library on the server side; Chart.js on the page.

Starting a run spawns `continuous.py` and, when the preset has anchors,
`eval_anchor.py` as detached processes writing to <run>/dashboard-*.log; stopping one
creates the STOP file, exactly like Ctrl-C would. Closing the dashboard does not stop
runs: on restart it finds them again through <run>/dashboard.pids.json.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PRESETS = HERE / "presets"
PAGE = HERE / "dashboard" / "index.html"

LOCK = threading.Lock()


# ─── process helpers ─────────────────────────────────────────────────────────

def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # OpenProcess + GetExitCodeProcess instead of `tasklist`: spawning tasklist cost
        # ~200 ms a pid, and every request checks every run's pids (3 s per page load).
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def process_cmdline(pid: int) -> str:
    """Command line of a live process ('' if unknown or gone)."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"],
                capture_output=True, text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
            return out.stdout.strip()
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""


SIDE_CARS = {"eval": "eval_anchor.py", "ladder": "eval_ladder.py"}


def stop_stale_side_cars(run_dir: Path) -> list[int]:
    """Stop the eval/ladder side-cars a previous start of this run left running.

    A side-car exits on its own once the run's STOP file has stayed for its grace period,
    but a resume deletes STOP first, so the old side-cars carried on next to the new ones:
    every checkpoint was evaluated twice and the results appended twice. Only a live pid
    whose command line is that side-car for this run directory is stopped (with its
    match/engine children), so a reused pid is never touched."""
    pids = read_json(run_dir / "dashboard.pids.json") or {}
    stopped = []
    for role, script in SIDE_CARS.items():
        pid = int(pids.get(role) or 0)
        if not pid or not pid_alive(pid):
            continue
        cmd = process_cmdline(pid)
        if script not in cmd or str(run_dir) not in cmd:
            continue
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.kill(pid, 15)
        stopped.append(pid)
    return stopped


def spawn_detached(cmd: list[str], log_path: Path, cwd: Path) -> int:
    log = open(log_path, "a", encoding="utf-8")
    kw: dict = {}
    if os.name == "nt":
        # A hidden console rather than none (DETACHED_PROCESS): every console program the
        # trainer or a side-car starts (engine, nvidia-smi, match.py) inherits it instead of
        # opening a window of its own.
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=str(cwd), **kw)
    return p.pid


# ─── run directory readers ───────────────────────────────────────────────────

_FILE_CACHE: dict[tuple, tuple[float, int, object]] = {}
_FILE_CACHE_LOCK = threading.Lock()


def _cached(fn):
    """Re-parse a file only when its size or mtime changed: a run's CSVs are thousands of
    rows, and every page refresh asked for all of them (1.3 s for a stopped run whose
    files never change). Callers must not mutate the returned rows."""
    def wrapper(path: Path, *args):
        try:
            st = path.stat()
        except OSError:
            return fn(path, *args)
        key = (fn.__name__, str(path), args)
        with _FILE_CACHE_LOCK:
            hit = _FILE_CACHE.get(key)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return hit[2]
        value = fn(path, *args)
        with _FILE_CACHE_LOCK:
            _FILE_CACHE[key] = (st.st_mtime, st.st_size, value)
        return value
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def latest_timeline(rows: list[dict], key: str) -> list[dict]:
    """Rows as one timeline in `key` (step, generated): where the key goes back (a resume
    from an earlier checkpoint after a crash logs those steps again), the rows written
    before the restart from that point on are dropped and the newer ones kept."""
    out: list[dict] = []
    for r in rows:
        try:
            k = float(r.get(key) or "nan")
        except ValueError:
            out.append(r)
            continue
        while out and float(out[-1].get(key) or "-inf") >= k:
            out.pop()
        out.append(r)
    return out


@_cached
def read_csv(path: Path, max_rows: int = 2000, monotonic: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if monotonic:
        rows = latest_timeline(rows, monotonic)
    if len(rows) > max_rows:
        step = len(rows) / max_rows
        rows = [rows[int(i * step)] for i in range(max_rows)] + [rows[-1]]
    out = []
    for r in rows:
        o = {}
        for k, v in r.items():
            try:
                o[k] = float(v)
            except (TypeError, ValueError):
                o[k] = v
        out.append(o)
    return out


@_cached
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def tail(path: Path, n: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-n:]


def run_status(run_dir: Path) -> dict:
    pids = read_json(run_dir / "dashboard.pids.json") or {}
    trainer_alive = pid_alive(int(pids.get("trainer", 0)))
    eval_alive = pid_alive(int(pids.get("eval", 0)))
    state = read_json(run_dir / "state.json") or {}
    stop = (run_dir / "STOP").exists()
    if trainer_alive:
        status = "stopping" if stop else "running"
    elif stop or state:
        status = "stopped"
    else:
        status = "new"
    return {
        "name": run_dir.name, "status": status, "trainer_alive": trainer_alive, "eval_alive": eval_alive,
        "step": state.get("step", 0), "generated": state.get("generated", 0), "games": state.get("games", 0),
        "mtime": run_dir.stat().st_mtime,
    }


def match_sides(path: Path) -> dict:
    """Per match (key 'ckpt|anchor'): wins, losses and draws by the side the checkpoint
    played, from tools/match.py's games log: {key: {"atk": [w, l, d], "def": [w, l, d]}}."""
    out = {}
    for m in read_jsonl(path):
        sides = {"atk": [0, 0, 0], "def": [0, 0, 0]}
        for g in m.get("games", []):
            side = sides.get(g.get("a_side"))
            i = {"w": 0, "l": 1, "d": 2}.get(g.get("score"))
            if side is not None and i is not None:
                side[i] += 1
        out[m["key"]] = sides
    return out


def ladder_detail(run_dir: Path) -> dict:
    """The checkpoint ladder as Ordo prints it: every rated player (checkpoints, anchors,
    external anchors) with points and games, and the head-to-head score of every pairing
    played at the fit's node count, both orders of a pairing folded into one."""
    lad = read_json(run_dir / "ladder.ratings.json") or {}
    nodes = lad.get("nodes")
    pairs: dict[tuple[str, str], list[int]] = {}
    for m in read_jsonl(run_dir / "ladder.jsonl"):
        if m.get("nodes") != nodes:
            continue
        a, b, w, l, d = m["a"], m["b"], int(m["w"]), int(m["l"]), int(m["d"])
        if a > b:
            a, b, w, l = b, a, l, w
        s = pairs.setdefault((a, b), [0, 0, 0])
        s[0] += w
        s[1] += l
        s[2] += d
    points: dict[str, list[float]] = {}
    for (a, b), (w, l, d) in pairs.items():
        pa, pb = points.setdefault(a, [0.0, 0]), points.setdefault(b, [0.0, 0])
        pa[0] += w + 0.5 * d
        pb[0] += l + 0.5 * d
        pa[1] += w + l + d
        pb[1] += w + l + d
    players = []
    for p in lad.get("players", []):
        pts, games = points.get(p["name"], [0.0, 0])
        players.append(dict(p, points=pts, played=games))
    return {"nodes": nodes, "reference": lad.get("reference"), "updated": lad.get("updated"), "players": players,
            "pairs": [[a, b, w, l, d] for (a, b), (w, l, d) in pairs.items()]}


def run_detail(run_dir: Path) -> dict:
    d = run_status(run_dir)
    d["config"] = read_json(run_dir / "config.json")
    d["state"] = {k: v for k, v in (read_json(run_dir / "state.json") or {}).items() if k != "rng"}
    d["metrics"] = read_csv(run_dir / "metrics.csv", 2000, "step")
    d["selfplay"] = read_csv(run_dir / "selfplay.csv", 2000, "generated")
    d["elo"] = read_jsonl(run_dir / "elo.jsonl")
    d["sides"] = match_sides(run_dir / "eval.games.jsonl")
    d["eval"] = read_jsonl(run_dir / "eval.jsonl")
    d["fort"] = read_jsonl(run_dir / "fort.jsonl")  # tools/eval_fort.py backfill and anchors
    d["eval_progress"] = read_json(run_dir / "eval.progress.json")
    d["ladder"] = read_json(run_dir / "ladder.ratings.json")
    d["rr"] = rr_state(run_dir)
    d["nets"] = {"ckpts": [p.stem for p in sorted((run_dir / "ckpt").glob("step*.onnx"), reverse=True)],
                 "anchors": list(anchor_nets(run_dir))}
    d["anchor_labels"] = {name: anchor_label(run_dir, name) for name in anchor_nets(run_dir)}
    d["retired_anchors"] = (read_json(run_dir / "dashboard.eval.json") or {}).get("retired_anchors", [])
    d["log"] = tail(run_dir / "log.txt")
    d["eval_log"] = tail(run_dir / "eval.log", 15)
    d["paused"] = (run_dir / "PAUSE").exists()
    return d


# ─── starting runs ───────────────────────────────────────────────────────────

# ─── round robin between chosen nets (round_robin.py) ───────────────────────

def anchor_nets(run_dir: Path) -> dict[str, Path]:
    """Nets a round robin can use besides the run's checkpoints: the best 11x11 net
    and the run's anchors."""
    out = {"copenhagen11x11": REPO / "copenhagen11x11.onnx"}
    for a in (read_json(run_dir / "dashboard.eval.json") or {}).get("anchors", []):
        p = (REPO / a).resolve()
        if p.exists():
            out.setdefault(p.stem, p)
    return out


def net_origin(run_dir: Path, name: str) -> str:
    """Where a net in this run's games comes from, for the game viewer's player cards: the
    run a checkpoint or anchor checkpoint belongs to (its folder under runs/), or what
    the fixed anchor is."""
    if name == "copenhagen11x11":
        return "T2 gen0225 (best 11x11 net)"
    path = anchor_nets(run_dir).get(name)
    if path is None:
        return run_dir.name
    parts = path.parts
    if "runs" in parts and parts.index("runs") + 1 < len(parts):
        return parts[parts.index("runs") + 1]
    return path.stem


def anchor_label(run_dir: Path, name: str) -> str:
    """How the dashboard names an anchor: its stem plus where it comes from, e.g.
    'T6 step486399' for runs/T6-10x128/ckpt/step486399.onnx, 'copenhagen11x11 (T2 gen0225)'.
    Display only: elo.jsonl and every file keep the bare stem."""
    if name == "copenhagen11x11":
        return "copenhagen11x11 (T2 gen0225)"
    origin = net_origin(run_dir, name)
    if origin in (run_dir.name, name):
        return name
    tag = origin.split("-", 1)[0]
    return f"{tag} {name}"


def rr_state(run_dir: Path) -> dict | None:
    st = read_json(run_dir / "rr" / "state.json")
    if st and st.get("status") == "running" and not pid_alive(int(st.get("pid", 0))):
        st["status"] = "stopped"
    return st


def load_presets() -> dict[str, dict]:
    out = {}
    if PRESETS.exists():
        for p in sorted(PRESETS.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                out[p.stem] = {"description": f"broken preset: {e}", "trainer": {}, "eval": {}}
    return out


class Dashboard:
    def __init__(self, runs_dir: Path, engine_bin: Path):
        self.runs_dir = runs_dir
        self.engine_bin = engine_bin
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dirs(self) -> list[Path]:
        return sorted([p for p in self.runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)

    def start(self, name: str, preset_name: str, overrides: dict, resume: bool = False) -> dict:
        if not name or any(c in name for c in '\\/:*?"<>| '):
            return {"error": "run name must be a plain folder name"}
        run_dir = self.runs_dir / name
        status = run_status(run_dir) if run_dir.exists() else {"status": "new"}
        if status.get("trainer_alive"):
            return {"error": f"{name} is already running"}
        presets = load_presets()
        preset = presets.get(preset_name)
        if preset is None and not resume:
            return {"error": f"unknown preset {preset_name!r}"}
        engine = Path(overrides.pop("engine_bin", "") or self.engine_bin).resolve()
        if not engine.exists():
            return {"error": f"engine binary not found: {engine}"}
        run_dir.mkdir(parents=True, exist_ok=True)
        stop_stale_side_cars(run_dir)
        (run_dir / "STOP").unlink(missing_ok=True)

        if resume:
            # A resumed run keeps the engine it was running with (config.json's engine_bin),
            # not whatever the dashboard was started with: two runs may need two builds.
            own = (read_json(run_dir / "config.json") or {}).get("engine_bin")
            if own and Path(own).exists():
                engine = Path(own)
            cmd = [sys.executable, str(HERE / "continuous.py"), "--resume", str(run_dir), "--engine-bin", str(engine)]
            eval_cfg = (read_json(run_dir / "dashboard.eval.json") or {})
        else:
            trainer_cfg = dict(preset.get("trainer", {}))
            trainer_cfg.update({k: v for k, v in overrides.items() if v not in (None, "")})
            # `smoke` is a CLI preset switch, not a config value: pass it as the flag.
            smoke = bool(trainer_cfg.pop("smoke", False))
            cfg_path = run_dir / "dashboard.config.json"
            cfg_path.write_text(json.dumps(trainer_cfg, indent=2), encoding="utf-8")
            cmd = [sys.executable, str(HERE / "continuous.py"), "--config", str(cfg_path), "--run-dir", str(run_dir),
                   "--engine-bin", str(engine)] + (["--smoke"] if smoke else [])
            eval_cfg = dict(preset.get("eval", {}))
            (run_dir / "dashboard.eval.json").write_text(json.dumps(eval_cfg, indent=2), encoding="utf-8")

        pids = {"trainer": spawn_detached(cmd, run_dir / "dashboard-trainer.log", REPO), "started": time.time()}
        anchors = [str((REPO / a).resolve()) for a in eval_cfg.get("anchors", [])]
        anchors = [a for a in anchors if Path(a).exists()]
        # "retired_anchors": stems the anchor eval no longer plays (the ladder still does).
        retired = set(eval_cfg.get("retired_anchors", []))
        eval_anchors = [a for a in anchors if Path(a).stem not in retired]
        if eval_anchors:
            ecmd = [sys.executable, str(HERE / "eval_anchor.py"), "--run-dir", str(run_dir), "--engine-bin", str(engine),
                    "--anchors", ",".join(eval_anchors), "--pairs", str(eval_cfg.get("pairs", 30)),
                    "--nodes", str(eval_cfg.get("nodes", 400)), "--workers", str(eval_cfg.get("workers", 2)),
                    "--playout-every", str(eval_cfg.get("playout_every", 5)),
                    "--max-plies", str(eval_cfg.get("max_plies", 400)),
                    "--min-gap", str(eval_cfg.get("min_gap", 0))]
            if eval_cfg.get("blind", False):
                ecmd.append("--blind")
            pids["eval"] = spawn_detached(ecmd, run_dir / "dashboard-eval.log", REPO)
        # Optional cheap all-checkpoint ladder: "ladder": {"nodes": 20, "pairs": 4, "opponents": 8}
        ladder = eval_cfg.get("ladder")
        if ladder:
            lcmd = [sys.executable, str(HERE / "eval_ladder.py"), "--run-dir", str(run_dir), "--engine-bin", str(engine),
                    "--nodes", str(ladder.get("nodes", 20)), "--pairs", str(ladder.get("pairs", 4)),
                    "--opponents", str(ladder.get("opponents", 8)), "--workers", str(ladder.get("workers", 1))]
            if anchors:
                lcmd += ["--anchors", ",".join(anchors)]
            pids["ladder"] = spawn_detached(lcmd, run_dir / "dashboard-ladder.log", REPO)
        (run_dir / "dashboard.pids.json").write_text(json.dumps(pids), encoding="utf-8")
        return {"ok": True, "name": name, "pids": pids}

    def stop(self, name: str) -> dict:
        run_dir = self.runs_dir / name
        if not run_dir.exists():
            return {"error": "no such run"}
        (run_dir / "STOP").write_text("", encoding="utf-8")
        return {"ok": True}

    def rr_start(self, name: str, nets: list[str], nodes: int, pairs: int, workers: int) -> dict:
        run_dir = self.runs_dir / name
        if not run_dir.is_dir():
            return {"error": "no such run"}
        st = rr_state(run_dir)
        if st and st.get("status") == "running":
            return {"error": "a round robin is already running"}
        known = anchor_nets(run_dir)
        paths = []
        for n in dict.fromkeys(nets):
            p = known.get(n) or run_dir / "ckpt" / f"{n}.onnx"
            if not p.exists():
                return {"error": f"no such net: {n}"}
            paths.append(str(p))
        if len(paths) < 2:
            return {"error": "pick at least two nets"}
        (run_dir / "rr").mkdir(exist_ok=True)
        cmd = [sys.executable, str(HERE / "round_robin.py"), "--run-dir", str(run_dir), "--engine-bin", str(self.engine_bin),
               "--nets", ",".join(paths), "--nodes", str(max(1, nodes)), "--pairs", str(max(1, pairs)),
               "--workers", str(max(1, workers))]
        return {"ok": True, "pid": spawn_detached(cmd, run_dir / "rr" / "log.txt", REPO)}

    def rr_stop(self, name: str) -> dict:
        """Stops at once: the STOP file for a clean exit, then the process tree."""
        run_dir = self.runs_dir / name
        st = rr_state(run_dir)
        if not st:
            return {"error": "no round robin"}
        (run_dir / "rr").mkdir(exist_ok=True)
        (run_dir / "rr" / "STOP").write_text("", encoding="utf-8")
        pid = int(st.get("pid", 0))
        if pid_alive(pid):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.kill(pid, 9)
        st["status"] = "stopped"
        st["current"] = []
        (run_dir / "rr" / "state.json").write_text(json.dumps(st), encoding="utf-8")
        return {"ok": True}

    def unpause(self, name: str) -> dict:
        (self.runs_dir / name / "PAUSE").unlink(missing_ok=True)
        return {"ok": True}


# ─── game viewer (docs/plan-game-viewer.md) ─────────────────────────────────
#
# Self-play games come from data/worker<w>.bin: every ply of a game is one sample and
# the last carries FLAG_LAST_OF_GAME, so a game is a run of samples. A 50 GB file
# cannot be scanned per request, so a per-file index of game boundaries
# (data/games<w>.idx.json) is built once in a background thread and then extended
# from where it stopped. Anchor-match games come from eval.games.jsonl (moves and evals
# recorded by tools/match.py); their positions are replayed by `taflzero --replay`,
# since the trainer has no rules engine of its own.
#
# The sample layout is read here directly (not through dataset.py) on purpose:
# dataset.py imports torch, which costs the dashboard seconds of start-up and memory
# for a few byte offsets. The constants mirror dataset.py's docstring.

VIEWER_BIT_POSITION_BYTES = 50   # 3 x 16 plane bytes + stm + rep
VIEWER_TAIL_BYTES = 4            # value, root_q, flags, king_corner
VIEWER_FLAG_POLICY_VALID = 1 << 0
VIEWER_FLAG_LAST_OF_GAME = 1 << 1
VIEWER_FLAG_RESULT_UNKNOWN = 1 << 2
GAME_INDEX_MAX_LIST = 1000       # most games one list request returns
GAME_INDEX_VERSION = 2           # 1 kept only the newest 5000 games; 2 keeps every game
GAME_INDEX_SAVE_SECONDS = 30     # a long scan saves its progress this often, not per chunk
GAME_INDEX_SYNC_BYTES = 64 << 20  # a smaller backlog is indexed inside the request
GAME_INDEX_CHUNK = 64 << 20       # bytes read per step of the scan

# The per-sample feature record of `taflzero --features` (docs/probe-input-planes.md):
# a 16-byte header (magic, u32 record size, u32 reserved), then one record per sample;
# byte 77 of the record of a game's last sample says how the game ended (codes from
# src/search/features.rs on the probe branch; 0 = written before the byte existed,
# 255 = the replay could not tell).
FEAT_MAGIC = b"TZFEAT01"
FEAT_HEADER_BYTES = 16
FEAT_TERMINAL_OFFSET = 77
FEAT_TERMINAL_CODES = {1: "atk_capture", 2: "atk_surrounded", 3: "atk_no_moves", 4: "atk_threefold",
                       5: "def_corner", 6: "def_edge", 7: "def_fort", 8: "def_no_moves", 9: "max_plies"}


def legal_mask_bytes(n: int) -> int:
    return (n * n * 4 * (n - 1) + 7) // 8


def sample_size(buf, pos: int, n: int) -> int | None:
    """Byte size of the sample at `pos` in `buf`, None when `buf` ends inside it."""
    at = pos + VIEWER_BIT_POSITION_BYTES + legal_mask_bytes(n)
    if at + 2 > len(buf):
        return None
    size = at + 2 - pos + int.from_bytes(buf[at:at + 2], "little") * 4 + VIEWER_TAIL_BYTES
    return size if pos + size <= len(buf) else None


def scan_games(buf, n: int, base_offset: int, base_sample: int) -> tuple[list[list], int, int]:
    """Whole games in `buf` (which starts on a game boundary): ([[offset, samples,
    first_sample, bytes, result], …], bytes consumed, samples consumed). The consumed
    counts stop at the end of the last complete game, so a game still being written
    (or a sample cut by the read) is picked up by the next scan from its start. The
    result is read here from the game's last sample, so listing games later reads
    nothing but the index."""
    hdr = VIEWER_BIT_POSITION_BYTES + legal_mask_bytes(n)
    games: list[list] = []
    pos = start = 0
    count = samples = 0
    end = len(buf)
    while True:
        at = pos + hdr
        if at + 2 > end:
            break
        size = hdr + 2 + (buf[at] | buf[at + 1] << 8) * 4 + VIEWER_TAIL_BYTES
        if pos + size > end:
            break
        count += 1
        flags = buf[pos + size - 2]
        if flags & VIEWER_FLAG_LAST_OF_GAME:
            value = buf[pos + size - 4]
            result = game_result(buf[pos + 48], value - 256 if value > 127 else value, flags)
            games.append([base_offset + start, count, base_sample + samples, pos + size - start, result])
            samples += count
            start, count = pos + size, 0
        pos += size
    return games, start, samples


class GameIndex:
    """Game boundaries of one self-play file, persisted next to it.

    `{"bin_size": <bytes indexed>, "samples": <samples before bin_size>, "games":
    [[offset, samples, first_sample, bytes, result], …]}`: bin_size always ends on a
    game boundary, games holds every game in file order (a 60M-sample file has ~360k games,
    a few MB of JSON), so a list can be the newest games or a spread over the whole run. Extending is a forward scan
    from bin_size, in the request for a small backlog, else in a background thread whose
    progress the API reports instead of blocking (the first index of a 50 GB file takes
    minutes). `lock` guards the fields, `scan_lock` lets one scan run at a time."""

    def __init__(self, bin_path: Path, idx_path: Path, board_size: int):
        self.bin_path, self.idx_path, self.n = bin_path, idx_path, board_size
        self.lock = threading.Lock()
        self.scan_lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.progress: float | None = None
        self.error: str | None = None
        self.bin_size, self.samples, self.games = 0, 0, []
        st = read_json(idx_path)
        if (st and st.get("version") == GAME_INDEX_VERSION and isinstance(st.get("games"), list)
                and all(len(g) == 5 for g in st["games"])):
            self.bin_size, self.samples, self.games = int(st["bin_size"]), int(st.get("samples", 0)), st["games"]

    def _save(self) -> None:
        tmp = self.idx_path.with_suffix(".tmp")
        with self.lock:
            state = {"version": GAME_INDEX_VERSION, "bin_size": self.bin_size, "samples": self.samples,
                     "games": list(self.games)}
        tmp.write_text(json.dumps(state, separators=(",", ":")),
                       encoding="utf-8")
        os.replace(tmp, self.idx_path)

    def _extend(self, upto: int, report: bool) -> None:
        """Scans [bin_size, upto) chunk by chunk, saving after each chunk."""
        start = self.bin_size
        want = GAME_INDEX_CHUNK
        last_save = time.time()
        with open(self.bin_path, "rb") as f:
            while self.bin_size < upto:
                f.seek(self.bin_size)
                buf = f.read(min(want, upto - self.bin_size))
                games, used, samples = scan_games(buf, self.n, self.bin_size, self.samples)
                if used == 0:
                    if len(buf) == upto - self.bin_size:
                        break  # the tail is an unfinished game
                    if want >= 1 << 30:
                        # No game is a gigabyte long: the file is not what we think it is.
                        raise ValueError(f"no complete game in {len(buf)} bytes at {self.bin_size}")
                    want *= 2  # a game longer than one read: read more at once
                    continue
                want = GAME_INDEX_CHUNK
                with self.lock:
                    self.games.extend(games)
                    self.bin_size += used
                    self.samples += samples
                    if report:
                        self.progress = (self.bin_size - start) / max(1, upto - start)
                # Saving rewrites the whole list: every 30 s during a long scan, not per chunk.
                if time.time() - last_save > GAME_INDEX_SAVE_SECONDS:
                    self._save()
                    last_save = time.time()
        if self.bin_size > start:
            self._save()

    def _run(self, upto: int) -> None:
        try:
            self._extend(upto, report=True)
        except Exception as e:  # reported with the game list; the next request retries
            self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.progress = None
                self.thread = None
            self.scan_lock.release()

    def refresh(self) -> float | None:
        """Brings the index up to the file's current size. Returns None when it is
        current, else the fraction done of a background build still running."""
        if not self.scan_lock.acquire(blocking=False):
            with self.lock:  # a scan is running: report it rather than wait for it
                return self.progress if self.thread is not None else None
        try:
            size = self.bin_path.stat().st_size
            if size < self.bin_size:  # the file was replaced: start over
                with self.lock:
                    self.bin_size, self.samples, self.games = 0, 0, []
            backlog = size - self.bin_size
            if backlog <= GAME_INDEX_SYNC_BYTES:
                if backlog > 0:
                    self._extend(size, report=False)
                self.scan_lock.release()
                return None
            with self.lock:
                self.error, self.progress = None, 0.0
                self.thread = threading.Thread(target=self._run, args=(size,), daemon=True)
            self.thread.start()  # owns scan_lock from here on and releases it when done
            return 0.0
        except BaseException:
            if self.thread is None:
                self.scan_lock.release()
            raise


GAME_INDEXES: dict[Path, GameIndex] = {}
GAME_INDEXES_LOCK = threading.Lock()
MATCH_REPLAYS: dict[str, list[bytes]] = {}  # match game id -> positions (BitPosition bytes)


def run_geometry(run_dir: Path) -> tuple[int, str]:
    cfg = read_json(run_dir / "config.json") or {}
    return int(cfg.get("board_size") or 11), str(cfg.get("variant") or "copenhagen11x11")


def game_index(run_dir: Path, worker: int) -> GameIndex:
    bin_path = run_dir / "data" / f"worker{worker}.bin"
    with GAME_INDEXES_LOCK:
        idx = GAME_INDEXES.get(bin_path)
        if idx is None:
            idx = GameIndex(bin_path, bin_path.with_name(f"games{worker}.idx.json"), run_geometry(run_dir)[0])
            GAME_INDEXES[bin_path] = idx
        return idx


def feat_terminal(feat_path: Path, sample: int) -> str:
    """How the game ended, from the feature record of its last sample; 'unknown' when
    there is no side file, it does not reach that sample, or the byte is unset."""
    try:
        with open(feat_path, "rb") as f:
            head = f.read(FEAT_HEADER_BYTES)
            if len(head) < FEAT_HEADER_BYTES or head[:8] != FEAT_MAGIC:
                return "unknown"
            rec = int.from_bytes(head[8:12], "little")
            f.seek(FEAT_HEADER_BYTES + sample * rec + FEAT_TERMINAL_OFFSET)
            b = f.read(1)
    except OSError:
        return "unknown"
    return FEAT_TERMINAL_CODES.get(b[0], "unknown") if b else "unknown"


def sample_fields(raw: bytes, n: int) -> dict:
    """The scalar fields of one sample and its sparse policy target."""
    at = VIEWER_BIT_POSITION_BYTES + legal_mask_bytes(n)
    plen = int.from_bytes(raw[at:at + 2], "little")
    pol = [(int.from_bytes(raw[at + 2 + 4 * i:at + 4 + 4 * i], "little"),
            int.from_bytes(raw[at + 4 + 4 * i:at + 6 + 4 * i], "little")) for i in range(plen)]
    tail = at + 2 + 4 * plen
    value, root_q = [int.from_bytes(raw[tail + i:tail + i + 1], "little", signed=True) for i in (0, 1)]
    return {"stm": raw[48], "policy": pol, "value": value, "root_q": root_q, "flags": raw[tail + 2]}


def game_result(stm: int, value: int, flags: int) -> str:
    """Winner from the last sample: `value` is z from the side to move (0 attackers)."""
    if flags & VIEWER_FLAG_RESULT_UNKNOWN:
        return "draw"
    if value == 0:
        return "unknown"
    return "atk" if (value > 0) == (stm == 0) else "def"


def board_pieces(planes: bytes, n: int) -> list[dict]:
    """Pieces of a BitPosition: square bit i of plane p is planes[16p + i/8] bit i%8.
    Squares are the engine's flat index row * n + col with row 0 = rank 1, which is
    also taflground's numbering (row 0 drawn at the bottom)."""
    out = []
    for p, kind in enumerate(("attacker", "defender", "king")):
        for sq in range(n * n):
            if planes[16 * p + sq // 8] >> (sq % 8) & 1:
                out.append({"sq": sq, "type": kind})
    return out


def recover_move(prev: bytes, cur: bytes, n: int) -> dict | None:
    """The move between two consecutive positions (as feature_dump.rs recovers it): the
    one square of the mover (prev's side to move; the king counts as a defender) that
    empties, and the one empty square that a mover piece fills. Captures only remove
    opponent pieces, so the rule holds for every pair in a game; None if it does not."""
    bit = lambda bp, p, sq: bp[16 * p + sq // 8] >> (sq % 8) & 1  # noqa: E731
    occ = lambda bp, sq: bit(bp, 0, sq) or bit(bp, 1, sq) or bit(bp, 2, sq)  # noqa: E731
    mine = (lambda bp, sq: bit(bp, 0, sq)) if prev[48] == 0 else (lambda bp, sq: bit(bp, 1, sq) or bit(bp, 2, sq))
    froms = [sq for sq in range(n * n) if mine(prev, sq) and not occ(cur, sq)]
    tos = [sq for sq in range(n * n) if not occ(prev, sq) and mine(cur, sq)]
    return {"from": froms[0], "to": tos[0]} if len(froms) == 1 and len(tos) == 1 else None


def policy_move(index: int, n: int) -> dict:
    """Policy index -> move: from * 4(n-1) + direction * (n-1) + distance - 1, with
    directions Up (row - d), Down (row + d), Left (col - d), Right (col + d) as in
    move_to_policy_index (src/search/mcts/utils.rs)."""
    md = n - 1
    frm, d, dist = index // (4 * md), (index // md) % 4, index % md + 1
    r, c = divmod(frm, n)
    r, c = {0: (r - dist, c), 1: (r + dist, c), 2: (r, c - dist), 3: (r, c + dist)}[d]
    return {"from": frm, "to": r * n + c}


def selfplay_games(run_dir: Path, worker: int, count: int, pick: str = "newest") -> dict:
    """`count` indexed games of one worker, newest first: the newest ones, or with
    pick="spread" evenly spaced over the whole file (first and last game included)."""
    bin_path = run_dir / "data" / f"worker{worker}.bin"
    workers = sorted(int(p.stem[6:]) for p in (run_dir / "data").glob("worker*.bin") if p.stem[6:].isdigit())
    if not bin_path.exists():
        return {"error": f"no {bin_path.name}", "workers": workers}
    n = run_geometry(run_dir)[0]
    idx = game_index(run_dir, worker)
    indexing = idx.refresh()
    with idx.lock:
        total = len(idx.games)
        if pick == "spread" and total > count:
            step = (total - 1) / max(1, count - 1)
            games = [idx.games[round(i * step)] for i in range(count)]
        else:
            games = list(idx.games[-count:])
        error = idx.error
    feat = bin_path.with_suffix(".feat")
    out = [{"id": f"{worker}:{first}", "start_sample": first, "plies": samples, "len_plies": samples,
            "result": result, "reason": game_reason(feat, first + samples - 1, result)}
           for _offset, samples, first, _nbytes, result in reversed(games)]
    res = {"games": out, "workers": workers, "feat": feat.exists(), "indexed_samples": idx.samples,
           "indexed_games": total, "pick": pick}
    if indexing is not None:
        res["indexing"] = round(indexing, 4)
    if error:
        res["index_error"] = error
    return res


def game_reason(feat: Path, last_sample: int, result: str) -> str:
    """How a self-play game ended: the side file's terminal byte when there is one;
    otherwise only an undecided game has a known reason (the ply / no-capture limit is
    the only way a game ends undecided)."""
    reason = feat_terminal(feat, last_sample) if feat.exists() else "unknown"
    if reason == "unknown" and result == "draw":
        reason = "max_plies"
    return reason


def selfplay_game(run_dir: Path, worker: int, first: int) -> dict:
    n = run_geometry(run_dir)[0]
    idx = game_index(run_dir, worker)
    with idx.lock:
        # games are in file order, so first_sample is ascending: binary search.
        i = bisect.bisect_left(idx.games, first, key=lambda g: g[2])
        hit = idx.games[i] if i < len(idx.games) and idx.games[i][2] == first else None
    if hit is None:
        return {"error": "game not in the index"}
    offset, samples, _, nbytes, _ = hit
    bin_path = run_dir / "data" / f"worker{worker}.bin"
    with open(bin_path, "rb") as f:
        f.seek(offset)
        buf = f.read(nbytes)
    raws, pos = [], 0
    for _ in range(samples):
        size = sample_size(buf, pos, n)
        if size is None:
            return {"error": "game cut short in the file"}
        raws.append(buf[pos:pos + size])
        pos += size
    plies = []
    for i, raw in enumerate(raws):
        s = sample_fields(raw, n)
        total = sum(v for _, v in s["policy"]) or 1
        top = sorted(s["policy"], key=lambda e: -e[1])[:5]
        plies.append({
            "pieces": board_pieces(raw[:48], n), "stm": s["stm"],
            "move": recover_move(raws[i - 1], raw, n) if i else None,
            "root_q": round(s["root_q"] / 127, 4), "z": s["value"],
            "policy_valid": bool(s["flags"] & VIEWER_FLAG_POLICY_VALID), "cheap": not s["flags"] & VIEWER_FLAG_POLICY_VALID,
            "top": [dict(policy_move(mi, n), visits=v, share=round(v / total, 4)) for mi, v in top],
        })
    last = sample_fields(raws[-1], n)
    result = game_result(last["stm"], last["value"], last["flags"])
    reason = game_reason(bin_path.with_suffix(".feat"), first + samples - 1, result)
    return {"source": "selfplay", "id": f"{worker}:{first}", "board_size": n, "plies": plies,
            "result": result, "reason": reason}


def match_games(run_dir: Path, count: int) -> dict:
    """Games of eval.games.jsonl that carry their moves, newest first."""
    out = []
    for line, m in enumerate(read_jsonl(run_dir / "eval.games.jsonl")):
        for i, g in enumerate(m.get("games", [])):
            if g.get("moves"):
                ck, _, anchor = str(m.get("key", "")).partition("|")
                out.append({"id": f"m:{line}:{i}", "key": m.get("key"), "ckpt": ck, "anchor": anchor,
                            "time": m.get("time"), "nodes": m.get("nodes"), "a_side": g.get("a_side"),
                            "score": g.get("score"), "reason": g.get("reason"), "plies": g.get("plies")})
    return {"games": out[::-1][:count]}


def replay_moves(engine: Path, variant: str, moves: str) -> list[bytes]:
    """Positions of a move list (start position first) from `taflzero --replay`."""
    proc = subprocess.run([str(engine), "--replay", moves, "--variant", variant], capture_output=True, text=True,
                          timeout=60, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if proc.returncode != 0:
        raise ValueError((proc.stderr or proc.stdout).strip()[-300:] or f"exit {proc.returncode}")
    out = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            out.append(bytes.fromhex(parts[1]))
    return out


def match_game(run_dir: Path, engine: Path, gid: str) -> dict:
    try:
        _, line, index = gid.split(":")
        m = read_jsonl(run_dir / "eval.games.jsonl")[int(line)]
        g = m["games"][int(index)]
    except (ValueError, IndexError, KeyError):
        return {"error": "no such match game"}
    moves = str(g.get("moves") or "").split()
    if not moves:
        return {"error": "this game has no recorded moves"}
    n, variant = run_geometry(run_dir)
    key = f"{run_dir.name}:{gid}"
    positions = MATCH_REPLAYS.get(key)
    if positions is None:
        try:
            positions = replay_moves(engine, variant, " ".join(moves))
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            return {"error": f"replay failed: {e}"}
        if len(MATCH_REPLAYS) > 200:
            MATCH_REPLAYS.clear()
        MATCH_REPLAYS[key] = positions
    if len(positions) != len(moves) + 1:
        return {"error": f"replay gave {len(positions)} positions for {len(moves)} moves"}
    evals = list(g.get("evals") or [])
    opening = len(moves) - len(evals)
    ck, _, anchor = str(m.get("key", "")).partition("|")
    a_attacks = g.get("a_side") == "atk"
    plies = []
    for i, bp in enumerate(positions):
        stm = bp[48]
        # evals[j] is the search at position opening + j, from its mover's view.
        ev = evals[i - opening] if opening <= i < len(moves) else None
        mover_is_a = (stm == 0) == a_attacks
        mv = moves[i - 1] if i else None
        plies.append({"pieces": board_pieces(bp[:48], n), "stm": stm,
                      "move": algebraic_move(mv, n) if mv else None,
                      "root_q": ev, "engine": (ck if mover_is_a else anchor) if ev is not None else None,
                      # By role, not name: a net can play its own anchor.
                      "role": ("ckpt" if mover_is_a else "anchor") if ev is not None else None,
                      "opening": i < opening, "top": []})
    score = g.get("score")
    result = "draw" if score == "d" else ("atk" if (score == "w") == a_attacks else "def")
    return {"source": "match", "id": gid, "board_size": n, "plies": plies, "result": result,
            "reason": g.get("reason") or "unknown", "ckpt": ck, "anchor": anchor, "a_side": g.get("a_side"),
            "opening": opening}


# ─── encirclements (atk_surrounded endings) ─────────────────────────────────
# What the attackers had locked in when a game ended by encirclement, from its final
# position: every defender and the king are inside (that is the rule), so the count of
# encircled pieces is what was left of the defence; around them a ring of attackers,
# sometimes with attackers inside it too.

ENCIRCLE_CACHE: dict[str, dict] = {}  # "<run>:<game id>" -> encircle_info + game fields
ENCIRCLE_SCAN_PER_WORKER = 20000      # newest games of one worker looked at for one request


def encircle_info(planes: bytes, n: int) -> dict:
    """Shape of an encirclement in a final position (BitPosition planes): `defenders`
    (king included) inside, the `region` they can move in (squares reachable from them
    without crossing an attacker, their own squares included), the `ring` (attackers
    next to that region) and the attackers `inside` (cut off from every edge square by
    the region: reached from the edge only through it)."""
    bit = lambda p, sq: planes[16 * p + sq // 8] >> (sq % 8) & 1  # noqa: E731
    atk = {sq for sq in range(n * n) if bit(0, sq)}
    dfn = [sq for sq in range(n * n) if bit(1, sq) or bit(2, sq)]
    king = any(bit(2, sq) for sq in range(n * n))

    def nbrs(sq):
        r, c = divmod(sq, n)
        if r > 0: yield sq - n
        if r < n - 1: yield sq + n
        if c > 0: yield sq - 1
        if c < n - 1: yield sq + 1

    region, stack = set(dfn), list(dfn)
    while stack:
        for t in nbrs(stack.pop()):
            if t not in region and t not in atk:
                region.add(t)
                stack.append(t)
    edge = [sq for sq in range(n * n) if sq // n in (0, n - 1) or sq % n in (0, n - 1)]
    outside = {sq for sq in edge if sq not in region}
    stack = list(outside)
    while stack:
        for t in nbrs(stack.pop()):
            if t not in outside and t not in region:
                outside.add(t)
                stack.append(t)
    ring = {a for a in atk if any(t in region for t in nbrs(a))}
    return {"defenders": len(dfn), "king": king, "attackers": len(atk), "region": len(region),
            "ring": len(ring), "inside": len(atk - outside),
            "region_closed": not any(sq in region for sq in edge)}


def feat_terminals(feat_path: Path, samples: list[int]) -> list[str]:
    """feat_terminal for many samples with one open."""
    out = []
    try:
        with open(feat_path, "rb") as f:
            head = f.read(FEAT_HEADER_BYTES)
            if len(head) < FEAT_HEADER_BYTES or head[:8] != FEAT_MAGIC:
                return ["unknown"] * len(samples)
            rec = int.from_bytes(head[8:12], "little")
            for s in samples:
                f.seek(FEAT_HEADER_BYTES + s * rec + FEAT_TERMINAL_OFFSET)
                b = f.read(1)
                out.append(FEAT_TERMINAL_CODES.get(b[0], "unknown") if b else "unknown")
    except OSError:
        return ["unknown"] * len(samples)
    return out


def last_sample(f, offset: int, nbytes: int, n: int) -> bytes | None:
    """The raw final sample of a game, read from its tail only (`f` an open .bin): a
    sample is header + u16 policy_len + 4 bytes per target + tail, so the last one
    starts where the u16 read at that distance from the end equals its own target count."""
    hdr = VIEWER_BIT_POSITION_BYTES + legal_mask_bytes(n)
    take = min(nbytes, hdr + 2 + VIEWER_TAIL_BYTES + 4 * 1024)
    f.seek(offset + nbytes - take)
    buf = f.read(take)
    moves = n * n * 4 * (n - 1)
    for p in range(0, 1025):
        size = hdr + 2 + 4 * p + VIEWER_TAIL_BYTES
        pos = len(buf) - size
        if pos < 0:
            break
        if int.from_bytes(buf[pos + hdr:pos + hdr + 2], "little") != p or buf[pos + 48] not in (0, 1):
            continue
        # A u16 inside the targets can equal its distance by chance: the planes must be a
        # position (no square twice, nothing past the last square, at most one king) and
        # every target a move of the sample's own legal mask.
        planes = [int.from_bytes(buf[pos + 16 * i:pos + 16 * i + 16], "little") for i in range(3)]
        if (any(pl >> (n * n) for pl in planes) or planes[0] & planes[1] or planes[0] & planes[2]
                or planes[1] & planes[2] or planes[2].bit_count() > 1):
            continue
        mask = int.from_bytes(buf[pos + VIEWER_BIT_POSITION_BYTES:pos + hdr], "little")
        mine = planes[0] if buf[pos + 48] == 0 else planes[1] | planes[2]
        froms = 0
        m = mask
        while m:
            low = m & -m
            froms |= 1 << ((low.bit_length() - 1) // (4 * (n - 1)))
            m ^= low
        if not mask or mask >> moves or froms & ~mine:
            continue
        ok = True
        for i in range(p):
            mi = int.from_bytes(buf[pos + hdr + 2 + 4 * i:pos + hdr + 4 + 4 * i], "little")
            if mi >= moves or not buf[pos + VIEWER_BIT_POSITION_BYTES + mi // 8] >> (mi % 8) & 1:
                ok = False
                break
        if ok:
            return buf[pos:]
    return None


def encircling_move(raw: bytes, n: int) -> bytes | None:
    """The position after the attackers' last move of a game won by encirclement.

    A self-play file stores the position before each move, so its last sample is one
    attacker move short of the ring. That move is not stored (it was drawn from the
    search's visits), so this tries the searched moves, most visited first, then every
    other attacker move, and takes the first that leaves no defender a path to the edge.
    Captures of defenders next to the landing square are applied (custodial, against an
    attacker, a corner or the empty throne; shield walls are not), since they change the
    count of pieces encircled but never the region."""
    planes = bytearray(raw[:48])
    if raw[48] != 0:
        return None
    bit = lambda p, sq: planes[16 * p + sq // 8] >> (sq % 8) & 1  # noqa: E731
    occupied = lambda sq: bit(0, sq) or bit(1, sq) or bit(2, sq)  # noqa: E731
    corners = {0, n - 1, n * (n - 1), n * n - 1}
    throne = n * n // 2
    seen, order = set(), []
    for mi, _v in sorted(sample_fields(raw, n)["policy"], key=lambda e: -e[1]):
        m = policy_move(mi, n)
        order.append((m["from"], m["to"]))
    for frm in range(n * n):
        if not bit(0, frm):
            continue
        r, c = divmod(frm, n)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            while 0 <= rr < n and 0 <= cc < n and not occupied(rr * n + cc):
                to = rr * n + cc
                if to not in corners and to != throne:
                    order.append((frm, to))
                rr, cc = rr + dr, cc + dc

    def put(bp, p, sq, on):
        if on:
            bp[16 * p + sq // 8] |= 1 << (sq % 8)
        else:
            bp[16 * p + sq // 8] &= ~(1 << (sq % 8)) & 0xFF

    for frm, to in order:
        if (frm, to) in seen or not bit(0, frm) or occupied(to):
            continue
        seen.add((frm, to))
        bp = bytearray(planes)
        put(bp, 0, frm, False)
        put(bp, 0, to, True)
        r, c = divmod(to, n)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r1, c1, r2, c2 = r + dr, c + dc, r + 2 * dr, c + 2 * dc
            if not (0 <= r2 < n and 0 <= c2 < n):
                continue
            mid, far = r1 * n + c1, r2 * n + c2
            hostile = (bp[far // 8] >> (far % 8) & 1) or far in corners or (
                far == throne and not (bp[32 + far // 8] >> (far % 8) & 1))
            if bp[16 + mid // 8] >> (mid % 8) & 1 and hostile:
                put(bp, 1, mid, False)
        info = encircle_info(bytes(bp), n)
        if info["region_closed"]:
            return bytes(bp)
    return None


def encircle_selfplay(run_dir: Path, count: int) -> list[dict]:
    """The newest `count` self-play games that ended by encirclement, over all workers."""
    n = run_geometry(run_dir)[0]
    data = run_dir / "data"
    workers = sorted(int(p.stem[6:]) for p in data.glob("worker*.bin") if p.stem[6:].isdigit())
    per = max(1, -(-count // max(1, len(workers))))
    out = []
    for w in workers:
        idx = game_index(run_dir, w)
        idx.refresh()
        with idx.lock:
            games = idx.games[-ENCIRCLE_SCAN_PER_WORKER:]
        atk = [g for g in reversed(games) if g[4] == "atk"]
        bin_path = data / f"worker{w}.bin"
        feat = bin_path.with_suffix(".feat")
        found = 0
        with open(bin_path, "rb") as f:
            for i in range(0, len(atk), 500):
                chunk = atk[i:i + 500]
                for g, reason in zip(chunk, feat_terminals(feat, [g[2] + g[1] - 1 for g in chunk])):
                    if reason != "atk_surrounded":
                        continue
                    offset, samples, first, nbytes, _ = g
                    key = f"{run_dir.name}:{w}:{first}"
                    info = ENCIRCLE_CACHE.get(key)
                    if info is None:
                        raw = last_sample(f, offset, nbytes, n)
                        planes = encircling_move(raw, n) if raw else None
                        if planes is None:
                            continue
                        info = dict(encircle_info(planes, n), id=f"{w}:{first}", plies=samples,
                                    pieces=board_pieces(planes, n), sample=first)
                        ENCIRCLE_CACHE[key] = info
                    out.append(info)
                    found += 1
                    if found >= per:
                        break
                if found >= per:
                    break
    return sorted(out, key=lambda e: -e["sample"])


# The whole run: every encirclement of a worker file, found by a background scan of its
# game index and kept in data/encircle<w>.json (one row per encirclement, the final
# position as hex), extended from where it stopped like the game index itself.
ENCIRCLE_SCAN_VERSION = 1
ENCIRCLE_BUCKET = 500_000  # positions per point of the whole-run chart
ENCIRCLE_COLS = ["sample", "plies", "defenders", "king", "region", "ring", "inside", "attackers", "planes"]


class EncircleScan:
    def __init__(self, run_dir: Path, worker: int):
        self.run_dir, self.worker = run_dir, worker
        self.n = run_geometry(run_dir)[0]
        self.path = run_dir / "data" / f"encircle{worker}.json"
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.progress: float | None = None
        self.error: str | None = None
        self.backlog = 0
        self.done, self.rows = 0, []
        st = read_json(self.path)
        if st and st.get("version") == ENCIRCLE_SCAN_VERSION:
            self.done, self.rows = int(st["games_done"]), st["rows"]

    def _save(self) -> None:
        with self.lock:
            state = {"version": ENCIRCLE_SCAN_VERSION, "columns": ENCIRCLE_COLS, "games_done": self.done,
                     "rows": list(self.rows)}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.path)

    def _run(self, upto: int) -> None:
        try:
            idx = game_index(self.run_dir, self.worker)
            bin_path = self.run_dir / "data" / f"worker{self.worker}.bin"
            feat = bin_path.with_suffix(".feat")
            start, last_save = self.done, time.time()
            with open(bin_path, "rb") as f:
                while self.done < upto:
                    with idx.lock:
                        chunk = idx.games[self.done:min(upto, self.done + 5000)]
                    atk = [g for g in chunk if g[4] == "atk"]
                    rows = []
                    for g, reason in zip(atk, feat_terminals(feat, [g[2] + g[1] - 1 for g in atk])):
                        if reason != "atk_surrounded":
                            continue
                        raw = last_sample(f, g[0], g[3], self.n)
                        planes = encircling_move(raw, self.n) if raw else None
                        if planes is None:
                            continue
                        e = encircle_info(planes, self.n)
                        rows.append([g[2], g[1], e["defenders"], int(e["king"]), e["region"], e["ring"],
                                     e["inside"], e["attackers"], planes.hex()])
                    with self.lock:
                        self.rows.extend(rows)
                        self.done += len(chunk)
                        self.progress = (self.done - start) / max(1, upto - start)
                    if time.time() - last_save > GAME_INDEX_SAVE_SECONDS:
                        self._save()
                        last_save = time.time()
            self._save()
        except Exception as e:  # reported with the result; the next request retries
            self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.progress, self.thread = None, None

    def refresh(self) -> float | None:
        """Starts a scan of the games indexed since the last one; returns the fraction
        done while a scan runs, None when the rows are current."""
        with self.lock:
            if self.thread is not None:
                return self.progress if self.backlog > 2000 else None
        idx = game_index(self.run_dir, self.worker)
        if idx.refresh() is not None:
            return 0.0  # the game index itself is still being built
        with idx.lock:
            total = len(idx.games)
        backlog = total - self.done
        if backlog < 1:
            return None
        with self.lock:
            self.error, self.progress, self.backlog = None, 0.0, backlog
            self.thread = threading.Thread(target=self._run, args=(total,), daemon=True)
            self.thread.start()
        # The games of the last few minutes are caught up in a second or two: only a real
        # backlog is reported (and polled for) as a scan in progress.
        return 0.0 if backlog > 2000 else None


ENCIRCLE_SCANS: dict[Path, EncircleScan] = {}


def encircle_whole_run(run_dir: Path, k: str) -> dict:
    """Every encirclement of the run: per-bucket counts by pieces encircled (and all games
    per bucket, for shares), per-count averages, and up to 60 example boards: the largest
    encirclements, or with `k` set those of that many pieces spread over the run."""
    data = run_dir / "data"
    workers = sorted(int(p.stem[6:]) for p in data.glob("worker*.bin") if p.stem[6:].isdigit())
    rows, games_x, progress, errors, done, total = [], [], [], [], 0, 0
    for w in workers:
        with GAME_INDEXES_LOCK:
            scan = ENCIRCLE_SCANS.get(data / f"worker{w}.bin")
            if scan is None:
                scan = ENCIRCLE_SCANS[data / f"worker{w}.bin"] = EncircleScan(run_dir, w)
        p = scan.refresh()
        if p is not None:
            progress.append(p)
        if scan.error:
            errors.append(scan.error)
        idx = game_index(run_dir, w)
        with scan.lock:
            rows += [(w, r) for r in scan.rows]
            d = scan.done
        with idx.lock:
            games_x += [g[2] for g in idx.games[:d]]
            total += len(idx.games)
        done += d
    nb = (max(games_x) // ENCIRCLE_BUCKET + 1) if games_x else 0
    games = [0] * nb
    for x in games_x:
        games[x // ENCIRCLE_BUCKET] += 1
    by_k: dict[int, list[int]] = {}
    stats: dict[int, dict] = {}
    for _w, r in rows:
        kk = r[2]
        by_k.setdefault(kk, [0] * nb)[r[0] // ENCIRCLE_BUCKET] += 1
        s = stats.setdefault(kk, {"games": 0, "region": 0, "ring": 0, "inside": 0, "attackers": 0, "plies": 0})
        s["games"] += 1
        for c in ("region", "ring", "inside", "attackers", "plies"):
            s[c] += r[ENCIRCLE_COLS.index(c)]
    for s in stats.values():
        for c in ("region", "ring", "inside", "attackers", "plies"):
            s[c] = round(s[c] / s["games"], 2)
    if k:
        pick = sorted((x for x in rows if x[1][2] == int(k)), key=lambda x: x[1][0])
        if len(pick) > 60:
            step = (len(pick) - 1) / 59
            pick = [pick[round(i * step)] for i in range(60)]
    else:
        pick = sorted(rows, key=lambda x: (-x[1][2], -x[1][0]))[:60]
    examples = []
    for w, r in pick:
        e = dict(zip(ENCIRCLE_COLS, r))
        planes = bytes.fromhex(e.pop("planes"))
        e.update(id=f"{w}:{e['sample']}", king=bool(e["king"]), pieces=board_pieces(planes, run_geometry(run_dir)[0]))
        examples.append(e)
    res = {"source": "selfplay", "scope": "all", "bucket": ENCIRCLE_BUCKET, "games_per_bucket": games,
           "by_k": {str(kk): v for kk, v in sorted(by_k.items())}, "stats": {str(kk): v for kk, v in sorted(stats.items())},
           "max_defenders": max(by_k) if by_k else None, "encirclements": len(rows), "games_done": done,
           "games_total": total, "examples": examples, "k": k or None}
    if progress:
        res["scanning"] = round(min(progress), 4)
    if errors:
        res["scan_error"] = errors[0]
    return res


def encircle_matches(run_dir: Path, engine: Path, count: int) -> list[dict]:
    """The newest `count` anchor-match games that ended by encirclement, either side
    encircling (`by`: "ckpt" when the checkpoint attacked, "anchor" otherwise)."""
    n, variant = run_geometry(run_dir)
    found = []
    for line, m in enumerate(read_jsonl(run_dir / "eval.games.jsonl")):
        for i, g in enumerate(m.get("games", [])):
            if g.get("reason") == "atk_surrounded" and g.get("moves"):
                found.append((line, i, m, g))
    out = []
    for line, i, m, g in reversed(found[-count:]):
        gid = f"m:{line}:{i}"
        key = f"{run_dir.name}:{gid}"
        info = ENCIRCLE_CACHE.get(key)
        if info is None:
            try:
                positions = replay_moves(engine, variant, g["moves"])
            except (OSError, ValueError, subprocess.TimeoutExpired):
                continue
            if not positions:
                continue
            ck, _, anchor = str(m.get("key", "")).partition("|")
            info = dict(encircle_info(positions[-1][:48], n), id=gid, plies=g.get("plies"),
                        pieces=board_pieces(positions[-1][:48], n), ckpt=ck, anchor=anchor,
                        by="ckpt" if g.get("a_side") == "atk" else "anchor", time=m.get("time"))
            ENCIRCLE_CACHE[key] = info
        out.append(info)
    return out


def encirclements(run_dir: Path, engine: Path, source: str, count: int) -> dict:
    games = encircle_matches(run_dir, engine, count) if source == "match" else encircle_selfplay(run_dir, count)
    if len(ENCIRCLE_CACHE) > 20000:
        ENCIRCLE_CACHE.clear()
    return {"source": source, "games": games}


def algebraic_move(mv: str, n: int) -> dict | None:
    """'d1d3' -> {from, to} squares (file letter + rank number, rank 1 = row 0)."""
    m = re.fullmatch(r"([a-z])(\d+)([a-z])(\d+)", mv)
    if not m:
        return None
    sq = lambda f, r: (int(r) - 1) * n + (ord(f) - 97)  # noqa: E731
    return {"from": sq(m[1], m[2]), "to": sq(m[3], m[4])}


VENDOR = HERE / "dashboard" / "vendor"
VENDOR_TYPES = {".js": "text/javascript", ".cjs": "text/javascript", ".css": "text/css", ".ts": "text/plain",
                ".map": "application/json"}


def vendor_file(url_path: str) -> tuple[bytes, str] | None:
    """A file under dashboard/vendor/ for /vendor/<path>, never outside it."""
    p = (VENDOR / url_path[len("/vendor/"):]).resolve()
    if VENDOR.resolve() not in p.parents or not p.is_file():
        return None
    return p.read_bytes(), VENDOR_TYPES.get(p.suffix, "application/octet-stream")


# ─── HTTP ────────────────────────────────────────────────────────────────────

def make_handler(dash: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/presets":
                self._json({"presets": load_presets(), "engine_bin": str(dash.engine_bin),
                            "engine_exists": dash.engine_bin.exists(), "cpu_count": os.cpu_count()})
            elif url.path == "/api/runs":
                with LOCK:
                    self._json({"runs": [run_status(p) for p in dash.run_dirs()]})
            elif url.path == "/api/rr":
                run_dir = dash.runs_dir / q.get("name", [""])[0]
                self._json({"rr": rr_state(run_dir) if run_dir.is_dir() else None})
            elif url.path == "/api/run":
                name = q.get("name", [""])[0]
                run_dir = dash.runs_dir / name
                if not name or not run_dir.is_dir():
                    self._json({"error": "no such run"}, 404)
                    return
                with LOCK:
                    self._json(run_detail(run_dir))
            elif url.path == "/api/ladder":
                run_dir = dash.runs_dir / q.get("name", [""])[0]
                if not q.get("name", [""])[0] or not run_dir.is_dir():
                    self._json({"error": "no such run"}, 404)
                    return
                self._json(ladder_detail(run_dir))
            elif url.path == "/api/encircle":
                # Not under LOCK, like the game viewer: it reads worker files and replays games.
                name = q.get("name", [""])[0]
                run_dir = dash.runs_dir / name
                if not name or not run_dir.is_dir():
                    self._json({"error": "no such run"}, 404)
                    return
                try:
                    if q.get("scope", [""])[0] == "all":
                        res = encircle_whole_run(run_dir, q.get("k", [""])[0])
                    else:
                        n = max(1, min(2000, int(q.get("n", ["300"])[0])))
                        res = encirclements(run_dir, dash.engine_bin, q.get("source", ["selfplay"])[0], n)
                except (ValueError, OSError) as e:
                    res = {"error": f"{type(e).__name__}: {e}"}
                self._json(res)
            elif url.path in ("/api/games", "/api/game"):
                # Game viewer. Not under LOCK: the first index of a big file runs in its
                # own thread and these requests must not hold up the page's refresh.
                name = q.get("name", [""])[0]
                run_dir = dash.runs_dir / name
                if not name or not run_dir.is_dir():
                    self._json({"error": "no such run"}, 404)
                    return
                try:
                    if url.path == "/api/games":
                        n = max(1, min(GAME_INDEX_MAX_LIST, int(q.get("n", ["50"])[0])))
                        if q.get("source", ["selfplay"])[0] == "match":
                            res = match_games(run_dir, n)
                        else:
                            res = selfplay_games(run_dir, int(q.get("worker", ["0"])[0]), n,
                                                 q.get("pick", ["newest"])[0])
                    else:
                        gid = q.get("id", [""])[0]
                        if gid.startswith("m:"):
                            res = match_game(run_dir, dash.engine_bin, gid)
                        else:
                            w, _, first = gid.partition(":")
                            res = selfplay_game(run_dir, int(w), int(first))
                        if "error" not in res:
                            res["run"] = run_dir.name
                            if res.get("source") == "match":
                                res["ckpt_origin"] = net_origin(run_dir, res["ckpt"])
                                res["anchor_origin"] = net_origin(run_dir, res["anchor"])
                except (ValueError, OSError) as e:
                    res = {"error": f"{type(e).__name__}: {e}"}
                self._json(res, 400 if "error" in res and "games" not in res else 200)
            elif url.path.startswith("/vendor/"):
                hit = vendor_file(url.path)
                if hit is None:
                    self._json({"error": "not found"}, 404)
                    return
                body, ctype = hit
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            with LOCK:
                if url.path == "/api/start":
                    res = dash.start(payload.get("name", ""), payload.get("preset", ""), dict(payload.get("overrides", {})))
                elif url.path == "/api/resume":
                    res = dash.start(payload.get("name", ""), "", {}, resume=True)
                elif url.path == "/api/stop":
                    res = dash.stop(payload.get("name", ""))
                elif url.path == "/api/rr/start":
                    res = dash.rr_start(payload.get("name", ""), list(payload.get("nets", [])), int(payload.get("nodes", 20)),
                                        int(payload.get("pairs", 2)), int(payload.get("workers", 1)))
                elif url.path == "/api/rr/stop":
                    res = dash.rr_stop(payload.get("name", ""))
                elif url.path == "/api/unpause":
                    res = dash.unpause(payload.get("name", ""))
                else:
                    res = {"error": "not found"}
            self._json(res, 400 if "error" in res else 200)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--engine-bin", default=str(REPO / "target" / "release" / ("taflzero.exe" if os.name == "nt" else "taflzero")))
    args = ap.parse_args()
    dash = Dashboard(Path(args.runs_dir).resolve(), Path(args.engine_bin).resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dash))
    print(f"dashboard: http://{args.host}:{args.port}  runs: {dash.runs_dir}  engine: {dash.engine_bin}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
