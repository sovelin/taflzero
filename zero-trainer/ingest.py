"""Tail the self-play workers' append-only files into game chunks (plan §3.2).

Each worker writes `data/worker<N>.bin` (v2 samples, one whole game per
write+flush) and `data/worker<N>.gamelog` (`terminal,len` lines).  `Ingester`
keeps a byte offset per file, reads whatever was appended since the last poll,
splits it into whole samples with `dataset.parse_samples`, groups samples into
games by FLAG_LAST_OF_GAME and defers both a truncated trailing sample and an
unfinished trailing game to the next poll (the offset stays at their start).

Per game it derives `plies_left[i] = len - 1 - i`, the outcome (attackers /
defenders / undecided) and the length, and returns one `Chunk` per game.

Next to every `worker<N>.bin` a current engine writes `worker<N>.feat`, the feature
side file (one 144-byte record per sample, record index = sample index in the .bin).
When it exists, a game is ingested only once its records are there too (the worker
writes the .bin first, so the .feat may lag by a game) and the chunk carries them
(`Chunk.feat`). Files without one (T6's data) give chunks with `feat=None`, which can
only train the base planes.
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataset import (
    BIT_POSITION_BYTES,
    FEAT_HEADER_BYTES,
    FEAT_RECORD_BYTES,
    BoardGeometry,
    FLAG_LAST_OF_GAME,
    RawSample,
    feature_count,
    parse_samples,
    TAIL_BYTES_V2,
)
from replay import Chunk

WORKER_BIN_RE = re.compile(r"^worker(\d+)\.bin$")

# selfplay.csv by_terminal columns.  The gamelog names are `<winner>_<how>`
# (atk_capture, def_corner, ...); the two limit terminals carry no winner.
TERMINAL_COLUMNS = [
    "king_captured", "def_corner", "fort", "threefold", "no_moves_atk", "no_moves_def",
    "surrounded", "edge", "max_plies", "no_capture", "other",
]
_TERMINAL_MAP = {
    "atk_capture": "king_captured",
    "def_corner": "def_corner",
    "def_fort": "fort",
    "atk_threefold": "threefold",
    "atk_no_moves": "no_moves_atk",   # attackers win: defenders had no legal move
    "def_no_moves": "no_moves_def",   # defenders win: attackers had no legal move
    "atk_surrounded": "surrounded",
    "def_edge": "edge",
}


def terminal_column(name: str) -> str:
    """Map a gamelog terminal name onto a TERMINAL_COLUMNS entry."""
    name = name.strip()
    if name in _TERMINAL_MAP:
        return _TERMINAL_MAP[name]
    if "nocapture" in name or "no_capture" in name:
        return "no_capture"
    if "limit" in name or "plies" in name:
        return "max_plies"
    return "other"


def game_outcome(first: RawSample) -> str:
    """'atk' / 'def' / 'undecided' from the first sample of a game.

    `value` is z from the side-to-move perspective; stm 0 = attackers."""
    if not first.result_known or first.value == 0:
        return "undecided"
    stm_won = first.value > 0
    attackers_to_move = first.stm == 0
    return "atk" if stm_won == attackers_to_move else "def"


@dataclass
class SelfplayStats:
    """Cumulative self-play counters (persisted in state.json)."""

    games: int = 0
    generated: int = 0
    atk_wins: int = 0
    def_wins: int = 0
    undecided: int = 0
    total_len: int = 0
    by_terminal: Counter = field(default_factory=Counter)

    def add_game(self, outcome: str, length: int) -> None:
        self.games += 1
        self.generated += length
        self.total_len += length
        if outcome == "atk":
            self.atk_wins += 1
        elif outcome == "def":
            self.def_wins += 1
        else:
            self.undecided += 1

    def add_terminal(self, name: str) -> None:
        self.by_terminal[terminal_column(name)] += 1

    @property
    def avg_len(self) -> float:
        return self.total_len / self.games if self.games else 0.0

    @property
    def atk_pct(self) -> float:
        return 100.0 * self.atk_wins / self.games if self.games else 0.0

    def to_dict(self) -> dict:
        return {
            "games": self.games, "generated": self.generated, "atk_wins": self.atk_wins,
            "def_wins": self.def_wins, "undecided": self.undecided, "total_len": self.total_len,
            "by_terminal": dict(self.by_terminal),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfplayStats":
        s = cls()
        for k in ("games", "generated", "atk_wins", "def_wins", "undecided", "total_len"):
            setattr(s, k, int(d.get(k, 0)))
        s.by_terminal = Counter(d.get("by_terminal", {}))
        return s

    def diff(self, older: "SelfplayStats") -> "SelfplayStats":
        """Counters accumulated since `older` (for per-window CSV rows)."""
        d = SelfplayStats(
            games=self.games - older.games, generated=self.generated - older.generated,
            atk_wins=self.atk_wins - older.atk_wins, def_wins=self.def_wins - older.def_wins,
            undecided=self.undecided - older.undecided, total_len=self.total_len - older.total_len,
        )
        d.by_terminal = Counter({k: self.by_terminal[k] - older.by_terminal.get(k, 0)
                                 for k in self.by_terminal})
        return d

    def copy(self) -> "SelfplayStats":
        return SelfplayStats.from_dict(self.to_dict())


@dataclass
class IngestResult:
    chunks: list[Chunk] = field(default_factory=list)
    samples: int = 0
    games: int = 0
    outcomes: Counter = field(default_factory=Counter)   # atk / def / undecided
    terminals: Counter = field(default_factory=Counter)  # TERMINAL_COLUMNS keys

    @property
    def total_len(self) -> int:
        return self.samples


def group_games(samples: list[RawSample], base_offset: int = 0, source: str = "") -> tuple[list[Chunk], int]:
    """Split consecutive samples into whole games.

    Returns (chunks, consumed_bytes): `consumed_bytes` covers only complete
    games; an unfinished trailing game is left for the next call so that
    `plies_left` is computed from a known game end.  Byte positions are
    relative to the start of `samples[0]`, shifted by `base_offset` for the
    chunk's `offset` field.
    """
    chunks: list[Chunk] = []
    pos = 0
    game_start_pos = 0
    game: list[RawSample] = []
    for s in samples:
        game.append(s)
        pos += len(s.raw)
        if s.flags & FLAG_LAST_OF_GAME:
            chunks.append(Chunk.from_samples(game, source=source, offset=base_offset + game_start_pos))
            game = []
            game_start_pos = pos
    return chunks, game_start_pos


def feature_file(bin_path: Path) -> Path:
    """`worker<N>.bin` -> `worker<N>.feat`."""
    return bin_path.with_suffix(".feat")


def samples_before(path: Path, geom: BoardGeometry, byte_offset: int, legacy: bool = False) -> int:
    """Number of samples in `path` before `byte_offset` (a sample boundary).  Only
    needed when a resumed run's state predates sample counting."""
    tail = 2 if legacy else 4
    pos = 0
    n = 0
    with open(path, "rb") as f:
        while pos < byte_offset:
            f.seek(pos + BIT_POSITION_BYTES + geom.legal_mask_bytes)
            raw = f.read(2)
            if len(raw) < 2:
                break
            pos += geom.fixed_header + struct.unpack("<H", raw)[0] * 4 + tail
            n += 1
    if pos != byte_offset:
        raise ValueError(f"{path}: byte {byte_offset} is not a sample boundary")
    return n


def trim_torn_tail(path: Path, geom: BoardGeometry, start_offset: int, start_sample: int) -> dict | None:
    """Cut a worker's files back to the last complete game after an unclean stop.

    A power loss can leave `workerN.bin` ending in part of a game (the worker writes a
    game in one write, but the OS need not flush all of it), and the `.feat` side file
    with more or fewer records than that. A restarted worker appends after whatever is
    there, so a torn tail would misalign every later sample. From `start_offset` (a game
    boundary already ingested, sample `start_sample`) this finds the last game that is
    whole in the `.bin` and has all its records in the `.feat`, and truncates `.bin`,
    `.feat` and the `.gamelog`'s unfinished last line there. The cut-off bytes are kept
    next to each file as `<name>.torn-tail`. Returns what was cut, or None if the files
    already end cleanly."""
    size = path.stat().st_size
    feat = path.with_suffix(".feat")
    have_feat = feat.exists()
    feat_records = ((feat.stat().st_size - FEAT_HEADER_BYTES) // FEAT_RECORD_BYTES) if have_feat else None
    pos, n = start_offset, start_sample
    cut, cut_samples = start_offset, start_sample
    with open(path, "rb") as f:
        while True:
            f.seek(pos + BIT_POSITION_BYTES + geom.legal_mask_bytes)
            raw = f.read(2)
            if len(raw) < 2:
                break
            end = pos + geom.fixed_header + struct.unpack("<H", raw)[0] * 4 + TAIL_BYTES_V2
            if end > size:
                break
            f.seek(end - 2)
            flags = f.read(1)[0]
            pos, n = end, n + 1
            if flags & FLAG_LAST_OF_GAME and (feat_records is None or n <= feat_records):
                cut, cut_samples = pos, n
    feat_cut = FEAT_HEADER_BYTES + cut_samples * FEAT_RECORD_BYTES
    gamelog = path.with_suffix(".gamelog")
    targets = [(path, cut)]
    if have_feat:
        targets.append((feat, feat_cut))
    if gamelog.exists():
        data = gamelog.read_bytes()
        targets.append((gamelog, data.rfind(b"\n") + 1))
    cuts = [(p, at, p.stat().st_size) for p, at in targets if p.stat().st_size > at]
    if not cuts:
        return None
    for p, at, sz in cuts:
        with open(p, "rb") as f:
            f.seek(at)
            tail = f.read()
        with open(p.with_name(p.name + ".torn-tail"), "ab") as t:
            t.write(tail)
        with open(p, "r+b") as f:
            f.truncate(at)
    return {"file": path.name, "cut_at": cut, "samples": cut_samples,
            "trimmed": {p.name: sz - at for p, at, sz in cuts}}


def read_feature_records(path: Path, first: int, count: int) -> np.ndarray:
    """Records `first .. first + count` of a side file, (count, 144) uint8."""
    with open(path, "rb") as f:
        f.seek(FEAT_HEADER_BYTES + first * FEAT_RECORD_BYTES)
        raw = f.read(count * FEAT_RECORD_BYTES)
    if len(raw) != count * FEAT_RECORD_BYTES:
        raise ValueError(f"{path}: records {first}..{first + count} are not all there")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, FEAT_RECORD_BYTES).copy()


class Ingester:
    def __init__(self, data_dir: Path, geom: BoardGeometry, legacy: bool = False,
                 offsets: dict[str, int] | None = None,
                 gamelog_offsets: dict[str, int] | None = None,
                 sample_offsets: dict[str, int] | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.geom = geom
        self.legacy = legacy
        self.offsets: dict[str, int] = dict(offsets or {})
        self.gamelog_offsets: dict[str, int] = dict(gamelog_offsets or {})
        # Samples consumed per file (the side-file record index of the next game).
        self.sample_offsets: dict[str, int] = dict(sample_offsets or {})

    def sample_index_at(self, path: Path, byte_offset: int) -> int:
        """Sample index of `byte_offset` in `path`: from the running count when the
        offset is the ingest position, else by counting."""
        if byte_offset == 0:
            return 0
        if self.offsets.get(path.name) == byte_offset and path.name in self.sample_offsets:
            return self.sample_offsets[path.name]
        return samples_before(path, self.geom, byte_offset, self.legacy)

    # ─── discovery ──────────────────────────────────────────────────────────

    def bin_files(self) -> list[Path]:
        if not self.data_dir.is_dir():
            return []
        files = [p for p in self.data_dir.iterdir() if WORKER_BIN_RE.match(p.name)]
        return sorted(files, key=lambda p: int(WORKER_BIN_RE.match(p.name).group(1)))

    def gamelog_files(self) -> list[Path]:
        if not self.data_dir.is_dir():
            return []
        return sorted(p for p in self.data_dir.iterdir()
                      if p.name.startswith("worker") and p.suffix == ".gamelog")

    # ─── samples ────────────────────────────────────────────────────────────

    @staticmethod
    def _read_from(path: Path, start: int, end: int | None = None) -> bytes:
        with open(path, "rb") as f:
            f.seek(start)
            return f.read() if end is None else f.read(max(0, end - start))

    def read_range(self, path: Path, start: int, end: int | None = None,
                   first_sample: int | None = None) -> tuple[list[Chunk], int]:
        """Parse [start, end) of one file into complete games.

        Returns (chunks, new_offset): `new_offset` is the absolute offset just
        past the last complete game (start of a deferred game / truncated
        sample).  With `first_sample` (the sample index at `start`) and a side file
        next to `path`, games whose records are not written yet are deferred too, and
        the others carry their records."""
        buf = self._read_from(path, start, end)
        if not buf:
            return [], start
        samples, _consumed = parse_samples(buf, self.geom, self.legacy)
        chunks, used = group_games(samples, base_offset=start, source=path.name)
        feat = feature_file(path)
        if first_sample is None or not feat.exists():
            return chunks, start + used
        available = feature_count(feat)
        kept: list[Chunk] = []
        index = first_sample
        for c in chunks:
            if index + len(c) > available:
                break
            c.first_sample = index
            index += len(c)
            kept.append(c)
        if kept:
            records = read_feature_records(feat, first_sample, index - first_sample)
            at = 0
            for c in kept:
                c.attach_feat(records[at:at + len(c)])
                at += len(c)
        new_off = kept[-1].offset + kept[-1].nbytes_raw if kept else start
        return kept, new_off

    def poll(self) -> IngestResult:
        """Read everything appended since the last poll, as complete games."""
        res = IngestResult()
        for path in self.bin_files():
            name = path.name
            off = self.offsets.get(name, 0)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= off:
                continue
            # Sample counting only matters for files with a side file (and costs a scan
            # when a resumed state predates it).
            first = None
            if feature_file(path).exists():
                if name not in self.sample_offsets:
                    self.sample_offsets[name] = self.sample_index_at(path, off)
                first = self.sample_offsets[name]
            chunks, new_off = self.read_range(path, off, first_sample=first)
            self.offsets[name] = new_off
            if first is not None:
                self.sample_offsets[name] += sum(len(c) for c in chunks)
            for c in chunks:
                res.chunks.append(c)
                res.samples += len(c)
                res.games += 1
                res.outcomes[chunk_outcome(c)] += 1
        res.terminals = self.poll_gamelogs()
        return res

    # ─── gamelog ────────────────────────────────────────────────────────────

    def poll_gamelogs(self) -> Counter:
        """New complete `terminal,len` lines since the last poll, bucketed by
        TERMINAL_COLUMNS.  A partial trailing line is left for next time."""
        counts: Counter = Counter()
        for path in self.gamelog_files():
            name = path.name
            off = self.gamelog_offsets.get(name, 0)
            try:
                if path.stat().st_size <= off:
                    continue
            except OSError:
                continue
            buf = self._read_from(path, off)
            last_nl = buf.rfind(b"\n")
            if last_nl < 0:
                continue
            complete = buf[:last_nl + 1]
            self.gamelog_offsets[name] = off + len(complete)
            for line in complete.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                terminal = line.split(",", 1)[0]
                counts[terminal_column(terminal)] += 1
        return counts


def chunk_outcome(chunk: Chunk) -> str:
    """Outcome of a chunk from its first sample's value/stm/known bit."""
    if len(chunk) == 0:
        return "undecided"
    if not bool(chunk.result_known[0]) or int(chunk.value[0]) == 0:
        return "undecided"
    stm_won = int(chunk.value[0]) > 0
    attackers_to_move = int(chunk.stm[0]) == 0
    return "atk" if stm_won == attackers_to_move else "def"
