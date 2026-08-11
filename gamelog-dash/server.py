#!/usr/bin/env python3
"""Realtime self-play gamelog dashboard — backend.

Incrementally tails a selfplay*.gamelog (append-only `terminal,length` lines)
and serves per-chunk aggregates as JSON.  The frontend (index.html) polls
/api/stats every few seconds and draws live charts + raw stats.

    python gamelog-dash/server.py --path zero-trainer/selfplay-t2.bin.gamelog --port 8420
    open http://localhost:8420

Stdlib only — no dependencies.
"""
import argparse
import json
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
STATE = {"path": None, "offset": 0, "games": []}  # games: list[(terminal, length)]


def read_new():
    """Read only bytes appended since last call (handles truncation/rotation)."""
    p = STATE["path"]
    if not p or not p.exists():
        return
    size = p.stat().st_size
    if size < STATE["offset"]:            # file shrank -> reset and re-read
        STATE["offset"] = 0
        STATE["games"] = []
    with open(p, "rb") as f:
        f.seek(STATE["offset"])
        data = f.read()
    nl = data.rfind(b"\n")                # only consume up to the last complete line
    if nl < 0:
        return
    STATE["offset"] += nl + 1
    for line in data[:nl + 1].decode("utf-8", "ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) == 2:
            try:
                STATE["games"].append((parts[0], int(parts[1])))
            except ValueError:
                pass


def compute(step):
    read_new()
    games = STATE["games"]
    chunks = []
    for i in range(0, len(games), step):
        seg = games[i:i + step]
        counts = defaultdict(int)
        tl = 0
        for t, l in seg:
            counts[t] += 1
            tl += l
        n = len(seg)
        chunks.append({
            "start": i,
            "n": n,
            "avg_len": round(tl / n, 1) if n else 0,
            "counts": dict(counts),
        })
    return {"total": len(games), "step": step, "chunks": chunks}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            try:
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, b"index.html missing", "text/plain")
        elif u.path == "/api/stats":
            q = parse_qs(u.query)
            try:
                step = max(1, int(q.get("step", ["5000"])[0]))
            except ValueError:
                step = 5000
            self._send(200, json.dumps(compute(step)).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="zero-trainer/selfplay-t2.bin.gamelog")
    ap.add_argument("--port", type=int, default=8420)
    args = ap.parse_args()
    STATE["path"] = Path(args.path).resolve()
    print(f"gamelog: {STATE['path']}")
    print(f"serving  http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
