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
import csv
import json
import os
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
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return False
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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

def read_csv(path: Path, max_rows: int = 2000) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
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
        return json.loads(path.read_text(encoding="utf-8"))
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


def run_detail(run_dir: Path) -> dict:
    d = run_status(run_dir)
    d["config"] = read_json(run_dir / "config.json")
    d["state"] = {k: v for k, v in (read_json(run_dir / "state.json") or {}).items() if k != "rng"}
    d["metrics"] = read_csv(run_dir / "metrics.csv")
    d["selfplay"] = read_csv(run_dir / "selfplay.csv")
    d["elo"] = read_jsonl(run_dir / "elo.jsonl")
    d["eval"] = read_jsonl(run_dir / "eval.jsonl")
    d["eval_progress"] = read_json(run_dir / "eval.progress.json")
    d["ladder"] = read_json(run_dir / "ladder.ratings.json")
    d["log"] = tail(run_dir / "log.txt")
    d["eval_log"] = tail(run_dir / "eval.log", 15)
    d["paused"] = (run_dir / "PAUSE").exists()
    return d


# ─── starting runs ───────────────────────────────────────────────────────────

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
        (run_dir / "STOP").unlink(missing_ok=True)

        if resume:
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
        if anchors:
            ecmd = [sys.executable, str(HERE / "eval_anchor.py"), "--run-dir", str(run_dir), "--engine-bin", str(engine),
                    "--anchors", ",".join(anchors), "--pairs", str(eval_cfg.get("pairs", 30)),
                    "--nodes", str(eval_cfg.get("nodes", 400)), "--workers", str(eval_cfg.get("workers", 2)),
                    "--playout-every", str(eval_cfg.get("playout_every", 5)),
                    "--max-plies", str(eval_cfg.get("max_plies", 400))]
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

    def unpause(self, name: str) -> dict:
        (self.runs_dir / name / "PAUSE").unlink(missing_ok=True)
        return {"ok": True}


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
            elif url.path == "/api/run":
                name = q.get("name", [""])[0]
                run_dir = dash.runs_dir / name
                if not name or not run_dir.is_dir():
                    self._json({"error": "no such run"}, 404)
                    return
                with LOCK:
                    self._json(run_detail(run_dir))
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
