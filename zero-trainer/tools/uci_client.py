"""Minimal UCI driver for taflzero.exe: one engine process, synchronous searches.

Used by the evaluators (match.py, eval_blind.py). Never closes stdin while a search
runs — the engine panics on that — so every `go` waits for its `bestmove`.
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading


class Engine:
    def __init__(self, exe: str, net: str, variant: str = "copenhagen11x11", env: dict | None = None):
        import os
        self.proc = subprocess.Popen(
            [os.path.abspath(exe), "--net", os.path.abspath(net)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._send("uci")
        self._until("uciok")
        self._send(f"setoption name Variant value {variant}")
        self._send("isready")
        self._until("readyok")

    def _pump(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")  # type: ignore[union-attr]
        self.proc.stdin.flush()

    def _until(self, prefix: str, timeout: float = 600.0) -> tuple[str, list[str]]:
        seen: list[str] = []
        while True:
            line = self.lines.get(timeout=timeout)
            if line is None:
                raise RuntimeError(f"engine exited; last output: {seen[-5:]}")
            if line.startswith(prefix):
                return line, seen
            seen.append(line)

    def position(self, moves: list[str], fen: str | None = None) -> None:
        base = f"position fen {fen}" if fen else "position startpos"
        self._send(base + (" moves " + " ".join(moves) if moves else ""))

    def search(self, moves: list[str], nodes: int, fen: str | None = None) -> tuple[str | None, float | None]:
        """Best move (None on a terminal position) and the root winrate in percent for
        the side to move, as the engine reports it."""
        self.position(moves, fen)
        self._send(f"go nodes {nodes}")
        best, seen = self._until("bestmove")
        info = next((s for s in reversed(seen) if s.startswith("info ")), "")
        wr = re.search(r"winrate ([\d.]+)%", info)
        mv = best.split()[1]
        return (None if mv == "(none)" else mv), (float(wr.group(1)) if wr else None)

    def result(self, moves: list[str], fen: str | None = None) -> str | None:
        """'attackers' / 'defenders' when the position is won, else None."""
        self.position(moves, fen)
        self._send("result")
        line, _ = self._until("result")
        who = line.split()[1]
        return None if who == "none" else who

    def random_move(self, moves: list[str], fen: str | None = None) -> str | None:
        self.position(moves, fen)
        self._send("randommove")
        line, _ = self._until("randommove")
        mv = line.split()[1]
        return None if mv == "(none)" else mv

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
