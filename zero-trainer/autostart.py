"""Bring the training back after a power loss or crash (run at logon by Task Scheduler).

    python autostart.py [--port 8787] [--engine-bin engines/6e2eb57/taflzero.exe] [--max-age-hours 72]

1. Starts the dashboard if nothing answers on the port.
2. Resumes every run that was cut off: a run with a state.json, no STOP file (a run
   stopped from the dashboard, by max_steps or by a fatal error keeps its STOP), a
   trainer pid that is no longer alive, and a log written in the last --max-age-hours.
   The resume itself repairs what the crash left (continuous.py: torn worker tails cut,
   weights from the saved step's checkpoint, log rows after the saved state dropped).

Everything it does goes to runs/autostart.log. Register it once with:

    schtasks /Create /TN "TaflZero autostart" /SC ONLOGON /DELAY 0001:00 /RL LIMITED ^
        /TR "\"<python.exe>\" \"<repo>\\zero-trainer\\autostart.py\""
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
LOG = RUNS / "autostart.log"


def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def api(port: int, path: str, payload: dict | None = None, timeout: float = 30.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def dashboard_up(port: int) -> bool:
    try:
        api(port, "/api/runs", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


def start_dashboard(port: int, engine_bin: Path) -> None:
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    out = open(HERE / f"dashboard-{port}.log", "a", encoding="utf-8")
    err = open(HERE / f"dashboard-{port}.err", "a", encoding="utf-8")
    cmd = [sys.executable, "dashboard.py", "--engine-bin", str(engine_bin)]
    if port != 8787:
        cmd += ["--port", str(port)]
    subprocess.Popen(cmd, cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags)
    say(f"dashboard started: {' '.join(cmd)}")


def cut_off_runs(max_age_hours: float) -> list[str]:
    sys.path.insert(0, str(HERE))
    from dashboard import pid_alive  # noqa: E402

    out = []
    now = time.time()
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        if not (run_dir / "state.json").exists() or (run_dir / "STOP").exists():
            continue
        log = run_dir / "log.txt"
        if not log.exists() or now - log.stat().st_mtime > max_age_hours * 3600:
            continue
        try:
            pids = json.loads((run_dir / "dashboard.pids.json").read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            pids = {}
        if pids.get("trainer") and pid_alive(int(pids["trainer"])):
            continue
        out.append(run_dir.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--engine-bin", type=Path, default=HERE / "engines" / "6e2eb57" / "taflzero.exe",
                    help="the dashboard's default engine; a resumed run keeps its own")
    ap.add_argument("--max-age-hours", type=float, default=72.0,
                    help="only runs whose log was written this recently are resumed")
    ap.add_argument("--dry-run", action="store_true", help="say what would be done, do nothing")
    a = ap.parse_args()

    runs = cut_off_runs(a.max_age_hours)
    say(f"autostart: runs cut off: {runs or 'none'}" + (" (dry run)" if a.dry_run else ""))
    if a.dry_run:
        return 0
    if not dashboard_up(a.port):
        start_dashboard(a.port, a.engine_bin)
        for _ in range(60):
            time.sleep(2)
            if dashboard_up(a.port):
                break
        else:
            say("dashboard did not come up in 120 s; giving up")
            return 1
    for name in runs:
        try:
            res = api(a.port, "/api/resume", {"name": name}, timeout=120)
            say(f"resume {name}: {res}")
        except Exception as e:  # noqa: BLE001
            say(f"resume {name} failed: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
