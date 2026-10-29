"""Tests for the continuous training loop (plan §3.11 + §1.5 pytest part).

Run from the repo root:  python -m pytest zero-trainer -q
No engine binary is needed: samples are synthesised in the v2 layout.
"""

from __future__ import annotations

import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dataset import (
    BIT_POSITION_BYTES,
    CORNER_CLASS_NONE,
    FLAG_LAST_OF_GAME,
    FLAG_POLICY_VALID,
    FLAG_RESULT_UNKNOWN,
    KING_CORNER_NONE_RAW,
    MOVES_LEFT_SCALE,
    TAIL_BYTES_V2,
    _read_sample_at,
    decode_sample,
    geometry,
    parse_samples,
)
from ingest import Ingester, SelfplayStats, group_games, chunk_outcome, terminal_column
from replay import Chunk, ReplayBuffer, replay_window
import continuous
from continuous import (
    ContinuousRun,
    lr_at,
    parse_lr_schedule,
    pacing_allowed_steps,
    parse_config,
    training_lag,
    worker_command,
)

GEOM = geometry(11)
REPO_ROOT = Path(__file__).resolve().parents[1]
PA_BIN = Path(__file__).resolve().parent / "selfplay-pA.bin"


# ─── synthetic v2 samples ───────────────────────────────────────────────────


def _pack_squares(squares: list[int]) -> bytes:
    bits = np.zeros(128, dtype=np.uint8)
    for sq in squares:
        bits[sq] = 1
    return np.packbits(bits, bitorder="little").tobytes()


def make_sample(
    *,
    stm: int = 0,
    value: int = 1,
    root_q: int = 50,
    flags: int = FLAG_POLICY_VALID,
    policy: tuple[tuple[int, int], ...] = ((5, 10), (7, 30)),
    corner: int = KING_CORNER_NONE_RAW,
    rep: int = 0,
    attackers: tuple[int, ...] = (3, 4, 5, 60, 70),
    defenders: tuple[int, ...] = (49, 71),
    king: int = 60,
) -> bytes:
    """One v2 sample (plan §1.2 layout).  Legal mask = the policy's moves."""
    planes = _pack_squares(list(attackers)) + _pack_squares(list(defenders)) + _pack_squares([king])
    legal_bits = np.zeros(GEOM.legal_mask_bytes * 8, dtype=np.uint8)
    for mi, _ in policy:
        legal_bits[mi] = 1
    legal = np.packbits(legal_bits, bitorder="little").tobytes()[: GEOM.legal_mask_bytes]
    out = bytearray(planes + bytes([stm, rep]) + legal)
    out += struct.pack("<H", len(policy))
    for mi, visits in policy:
        out += struct.pack("<HH", mi, visits)
    out += struct.pack("<bbBB", value, root_q, flags, corner)
    assert len(out) == GEOM.fixed_header + len(policy) * 4 + TAIL_BYTES_V2
    return bytes(out)


def make_game(length: int, winner: str = "atk", first_stm: int = 0, root_q: int = 40,
              policy_valid: bool = True) -> list[bytes]:
    """A game of `length` samples with consistent z/stm and the last-of-game flag.

    winner: 'atk' / 'def' / 'undecided' (sets FLAG_RESULT_UNKNOWN, value 0)."""
    samples = []
    for i in range(length):
        stm = (first_stm + i) % 2
        flags = FLAG_POLICY_VALID if policy_valid else 0
        if i == length - 1:
            flags |= FLAG_LAST_OF_GAME
        if winner == "undecided":
            flags |= FLAG_RESULT_UNKNOWN
            value = 0
        else:
            stm_is_atk = stm == 0
            value = 1 if (winner == "atk") == stm_is_atk else -1
        # vary the policy a little so neighbours differ
        pol = ((5 + i % 3, 10 + i), (7, 30))
        samples.append(make_sample(stm=stm, value=value, root_q=root_q, flags=flags, policy=pol))
    return samples


def write_games(path: Path, games: list[list[bytes]], mode: str = "ab") -> None:
    with open(path, mode) as f:
        for g in games:
            f.write(b"".join(g))


# ─── replay_window ──────────────────────────────────────────────────────────


def test_replay_window_control_values() -> None:
    assert replay_window(250_000) == 250_000
    assert replay_window(100_000) == 100_000
    # Formula values with base 250k, alpha .75, beta .4 (plan §3.3):
    assert abs(replay_window(1_000_000) - 493_700) < 2_000
    assert abs(replay_window(4_000_000) - 1_183_000) < 2_000
    assert abs(replay_window(10_000_000) - 2_237_000) < 3_000
    assert replay_window(10_000_000, cap=2_000_000) == 2_000_000


# ─── parse_samples / grouping / plies_left ─────────────────────────────────


def test_parse_samples_stops_at_truncated_tail() -> None:
    s1, s2, s3 = make_sample(policy=((1, 5),)), make_sample(policy=((2, 5), (3, 6))), make_sample()
    buf = s1 + s2 + s3[:-7]  # third sample cut mid-policy
    samples, consumed = parse_samples(buf, GEOM)
    assert len(samples) == 2
    assert consumed == len(s1) + len(s2)
    assert samples[0].raw == s1 and samples[1].raw == s2
    assert samples[1].policy_len == 2
    # cut inside the fixed header (before policy_len can be read)
    samples, consumed = parse_samples(s1 + s2[:20], GEOM)
    assert len(samples) == 1 and consumed == len(s1)
    # exact boundary
    samples, consumed = parse_samples(s1 + s2, GEOM)
    assert len(samples) == 2 and consumed == len(s1) + len(s2)
    assert parse_samples(b"", GEOM) == ([], 0)


def test_raw_sample_fields() -> None:
    raw = make_sample(stm=1, value=-1, root_q=-20, flags=FLAG_POLICY_VALID | FLAG_RESULT_UNKNOWN)
    (s,), _ = parse_samples(raw, GEOM)
    assert (s.stm, s.value, s.root_q, s.flags) == (1, -1, -20, FLAG_POLICY_VALID | FLAG_RESULT_UNKNOWN)
    assert s.policy_valid and not s.last_of_game and not s.result_known


def test_group_games_two_and_a_half() -> None:
    g1, g2, g3 = make_game(3), make_game(4, winner="def"), make_game(5)
    half = g3[:2]
    samples, consumed = parse_samples(b"".join(g1 + g2 + half), GEOM)
    assert len(samples) == 3 + 4 + 2
    chunks, used = group_games(samples, base_offset=1000, source="worker0.bin")
    assert [len(c) for c in chunks] == [3, 4]
    assert used == sum(len(s) for s in g1 + g2)  # tail (2 samples) deferred
    assert chunks[0].offset == 1000 and chunks[1].offset == 1000 + sum(len(s) for s in g1)
    assert chunks[0].source == "worker0.bin"


def test_plies_left_for_game_of_five() -> None:
    samples, _ = parse_samples(b"".join(make_game(5)), GEOM)
    (chunk,), _ = group_games(samples)
    assert chunk.plies_left.tolist() == [4, 3, 2, 1, 0]
    assert chunk.result_known.all()


def test_chunk_outcome_from_first_sample() -> None:
    def chunk_for(winner, first_stm):
        samples, _ = parse_samples(b"".join(make_game(3, winner=winner, first_stm=first_stm)), GEOM)
        return group_games(samples)[0][0]

    assert chunk_outcome(chunk_for("atk", 0)) == "atk"
    assert chunk_outcome(chunk_for("atk", 1)) == "atk"
    assert chunk_outcome(chunk_for("def", 0)) == "def"
    assert chunk_outcome(chunk_for("def", 1)) == "def"
    assert chunk_outcome(chunk_for("undecided", 0)) == "undecided"
    assert not chunk_for("undecided", 0).result_known.any()


# ─── pacing ─────────────────────────────────────────────────────────────────


def test_lr_schedule_is_constant_without_entries_and_steps_down_with_them() -> None:
    assert parse_lr_schedule(None) == []
    assert parse_lr_schedule(" ") == []
    assert parse_lr_schedule("60000:0.002, 20000:0.007") == [(20000, 0.007), (60000, 0.002)]
    with pytest.raises(ValueError):
        parse_lr_schedule("20000")
    with pytest.raises(ValueError):
        parse_lr_schedule("20000:0.007,20000:0.001")
    sched = parse_lr_schedule("20000:0.007,60000:0.002")
    # No schedule: the old behaviour, warmup then constant.
    assert lr_at(250, 0.02, 500) == pytest.approx(0.01)
    assert lr_at(10_000, 0.02, 500) == 0.02
    # With one: base lr until the first step, then each entry's lr from its step on.
    assert lr_at(19_999, 0.02, 500, sched) == 0.02
    assert lr_at(20_000, 0.02, 500, sched) == 0.007
    assert lr_at(59_999, 0.02, 500, sched) == 0.007
    assert lr_at(60_000, 0.02, 500, sched) == 0.002
    # Warmup scales whatever the schedule gives for those steps.
    assert lr_at(250, 0.02, 500, [(1, 0.01)]) == pytest.approx(0.005)


def test_pacing_allows_exactly_forty_steps() -> None:
    assert pacing_allowed_steps(1000, 100, 4) == 40
    # the trainer loop trains while step < allowed
    assert 39 < pacing_allowed_steps(1000, 100, 4) <= 40
    assert pacing_allowed_steps(1000 + 24, 100, 4) == 40  # 4*1024/100 = 40.96 -> 40
    assert pacing_allowed_steps(1025, 100, 4) == 41
    assert training_lag(1000, 40, 100, 4) == 0
    assert training_lag(2000, 40, 100, 4) == 4000


# ─── value target (plan §1.4 / §1.5) ────────────────────────────────────────


def test_value_target_known_vs_unknown() -> None:
    known = make_sample(value=1, root_q=63, flags=FLAG_POLICY_VALID)
    unknown = make_sample(value=0, root_q=63, flags=FLAG_POLICY_VALID | FLAG_RESULT_UNKNOWN)
    cheap_known = make_sample(value=-1, root_q=63, flags=0)
    cheap_unknown = make_sample(value=0, root_q=-63, flags=FLAG_RESULT_UNKNOWN)
    q = 63 / 127.0
    d = decode_sample(known, GEOM, z_lambda=0.5)
    assert d.result_known and abs(d.value_target - (0.5 * 1 + 0.5 * q)) < 1e-6
    d = decode_sample(unknown, GEOM, z_lambda=0.5)
    assert not d.result_known and abs(d.value_target - q) < 1e-6
    d = decode_sample(cheap_known, GEOM, z_lambda=0.5)
    assert abs(d.value_target - (-1.0)) < 1e-6  # cheap search: pure z
    d = decode_sample(cheap_unknown, GEOM, z_lambda=0.5)
    assert abs(d.value_target - (-q)) < 1e-6  # no z: root_q even for cheap search
    # z_lambda = 1 with unknown result still gives root_q, never 0
    d = decode_sample(unknown, GEOM, z_lambda=1.0)
    assert abs(d.value_target - q) < 1e-6


def test_batch_value_target_and_moves_left_mask() -> None:
    buf = ReplayBuffer(GEOM, z_lambda=0.5, augment=False)
    known_s, _ = parse_samples(b"".join(make_game(4, winner="atk", root_q=63)), GEOM)
    unknown_s, _ = parse_samples(b"".join(make_game(3, winner="undecided", root_q=63)), GEOM)
    (ck,), _ = group_games(known_s)
    (cu,), _ = group_games(unknown_s)
    buf.add(ck, 10_000)
    buf.add(cu, 10_000)
    q = 63 / 127.0
    located = [(ck, 0), (ck, 3), (cu, 0), (cu, 2)]
    batch = buf.collate(located, [0, 0, 0, 0])
    assert len(batch) == 12
    value, ml_target, ml_valid = batch[3], batch[10], batch[11]
    # known: 0.5 * z + 0.5 * q, z = +1 for attackers-to-move in an attacker win
    assert torch.allclose(value[0], torch.tensor(0.5 + 0.5 * q))
    assert torch.allclose(value[1], torch.tensor(0.5 * -1 + 0.5 * q))  # stm=1 lost
    assert torch.allclose(value[2], torch.tensor(q))
    assert torch.allclose(value[3], torch.tensor(q))
    assert ml_valid.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert torch.allclose(ml_target, torch.tensor([3, 0, 2, 0], dtype=torch.float32) / MOVES_LEFT_SCALE)


def test_value_loss_masks_unknown_outcome() -> None:
    """Undecided samples (result unknown) must not train the value head."""
    from az_micro_net import TaflAlphaZeroNet
    from train import compute_batch_loss

    buf = ReplayBuffer(GEOM, z_lambda=0.5, augment=False)
    known_s, _ = parse_samples(b"".join(make_game(4, winner="atk", root_q=63)), GEOM)
    unknown_s, _ = parse_samples(b"".join(make_game(3, winner="undecided", root_q=63)), GEOM)
    (ck,), _ = group_games(known_s)
    (cu,), _ = group_games(unknown_s)
    buf.add(ck, 10_000)
    buf.add(cu, 10_000)
    torch.manual_seed(0)
    model = TaflAlphaZeroNet(trunk_channels=8, num_blocks=1)
    model.eval()
    dev = torch.device("cpu")

    def v_loss(located):
        batch = buf.collate(located, [0] * len(located))
        with torch.no_grad():
            return compute_batch_loss(model, batch, dev, 1.0, 0.0, 0.0, use_aux=False)[2]

    mixed = v_loss([(ck, 0), (ck, 3), (cu, 0), (cu, 2)])
    known_only = v_loss([(ck, 0), (ck, 3)])
    assert torch.allclose(mixed, known_only), (mixed, known_only)
    unknown_only = v_loss([(cu, 0), (cu, 2)])
    assert torch.isfinite(unknown_only) and unknown_only.item() == 0.0


def test_bf16_autocast_matches_fp32_loss() -> None:
    """amp_dtype only lowers the forward precision: every loss term stays close to
    fp32, comes back as float32, and the weights keep their fp32 dtype."""
    from az_micro_net import TaflAlphaZeroNet
    from train import compute_batch_loss

    buf = ReplayBuffer(GEOM, z_lambda=0.5, augment=False)
    samples, _ = parse_samples(b"".join(make_game(6, winner="atk", root_q=40)), GEOM)
    (c,), _ = group_games(samples)
    buf.add(c, 10_000)
    batch = buf.collate([(c, i) for i in range(6)], [0] * 6)
    torch.manual_seed(0)
    model = TaflAlphaZeroNet(trunk_channels=16, num_blocks=2, aux_heads=True, moves_left=True)
    model.eval()
    dev = torch.device("cpu")
    with torch.no_grad():
        ref = compute_batch_loss(model, batch, dev, 1.0, 0.15, 0.1, True, moves_left_weight=0.1)
        low = compute_batch_loss(model, batch, dev, 1.0, 0.15, 0.1, True, moves_left_weight=0.1,
                                 amp_dtype=torch.bfloat16)
    for a, b in zip(ref, low):
        assert b.dtype == torch.float32
        assert abs(a.item() - b.item()) <= 0.03 * max(abs(a.item()), 1e-3), (a.item(), b.item())
    assert all(p.dtype == torch.float32 for p in model.parameters())


# ─── replay buffer ──────────────────────────────────────────────────────────


def _chunk(length: int, source="worker0.bin", offset=0, winner="atk") -> Chunk:
    samples, _ = parse_samples(b"".join(make_game(length, winner=winner)), GEOM)
    (c,), _ = group_games(samples, base_offset=offset, source=source)
    return c


def test_replay_buffer_eviction_and_window_starts() -> None:
    buf = ReplayBuffer(GEOM)
    off = 0
    for n in (10, 10, 10, 10):
        c = _chunk(n, offset=off)
        buf.add(c, window=25)
        off += c.nbytes
    # keeps >= window samples: 40 -> evict the oldest while 30 >= 25 -> 30 left
    assert len(buf) == 30 and buf.games == 3
    assert buf.window_starts() == {"worker0.bin": _chunk(10).nbytes}
    assert buf.bytes_used == sum(c.nbytes for c in list(buf._chunks))
    # never evicts the chunk just added even if it alone exceeds the window
    buf.add(_chunk(50), window=5)
    assert len(buf) == 50 and buf.games == 1


def test_replay_buffer_max_bytes() -> None:
    one = _chunk(10)
    buf = ReplayBuffer(GEOM, max_bytes=one.nbytes * 2 + 10)
    for _ in range(4):
        buf.add(_chunk(10), window=1_000_000)
    assert buf.games == 2 and buf.bytes_capped


def test_replay_buffer_sample_layout_and_aux() -> None:
    buf = ReplayBuffer(GEOM, augment=True)
    buf.add(_chunk(6), window=1000)
    buf.add(_chunk(5, winner="def"), window=1000)
    rng = np.random.default_rng(0)
    batch = buf.sample(8, 1000, rng)
    planes, legal, pi, value, pvalid, aux_p, aux_legal, aux_valid, corner, root_q, ml_t, ml_v = batch
    assert planes.shape == (8, 11, 11, 11) and planes.dtype == torch.float32
    assert legal.shape == (8, GEOM.policy_size) and legal.dtype == torch.bool
    assert pi.shape == (8, GEOM.policy_size) and torch.allclose(pi.sum(1), torch.ones(8))
    assert aux_legal.dtype == torch.bool and corner.dtype == torch.int64
    assert value.shape == (8,) and root_q.shape == (8,) and ml_t.shape == (8,) and ml_v.shape == (8,)
    assert (legal.int() * pi).sum(1).allclose(torch.ones(8))  # policy mass only on legal moves
    # aux policy = next sample of the same game (sym 0, direct check)
    c = buf._chunks[0]
    row = buf.collate([(c, 2)], [0])
    nxt = decode_sample(c.raw[3], GEOM)
    assert torch.allclose(row[5][0], torch.from_numpy(nxt.policy)) and row[7].item() == 1.0
    last = buf.collate([(c, 5)], [0])
    assert last[7].item() == 0.0 and last[5].sum() == 0  # last sample has no reply


def test_replay_buffer_draws_only_from_window() -> None:
    buf = ReplayBuffer(GEOM, augment=False)
    for _ in range(5):
        buf.add(_chunk(10), window=1_000_000)
    rng = np.random.default_rng(1)
    idx = buf.draw_indices(200, window=10, rng=rng)
    assert idx.min() >= 40 and idx.max() < 50


def test_decode_matches_read_sample_at_on_engine_data() -> None:
    if not PA_BIN.exists():
        pytest.skip("selfplay-pA.bin not present")
    with open(PA_BIN, "rb") as f:
        head = f.read(200_000)
        samples, _ = parse_samples(head, GEOM)
        assert len(samples) > 1
        for i in (0, 1, len(samples) - 1):
            off = sum(len(s.raw) for s in samples[:i])
            ref = _read_sample_at(f, off, GEOM, z_lambda=0.5)
            d = decode_sample(samples[i].raw, GEOM, z_lambda=0.5)
            assert np.array_equal(ref[0].numpy(), d.planes)
            assert np.array_equal(ref[1].numpy(), d.legal_mask)
            assert np.array_equal(ref[2].numpy(), d.policy)
            assert float(ref[3]) == np.float32(d.value_target)
            assert (ref[4], ref[5], ref[6]) == (d.policy_valid, d.last_of_game, d.corner_class)
            assert ref[7] == off + d.size and ref[7] == off + len(samples[i].raw)
            assert ref[9] is True  # old files: result known


def test_decode_sample_legacy_v1() -> None:
    v2 = make_sample(value=1, root_q=0, flags=FLAG_POLICY_VALID)
    v1 = v2[:-2]  # drop flags + king_corner
    d = decode_sample(v1, GEOM, z_lambda=0.5, legacy=True)
    assert d.size == len(v1) and d.policy_valid and d.last_of_game and d.result_known
    assert d.corner_class == CORNER_CLASS_NONE and abs(d.value_target - 0.5) < 1e-6
    samples, consumed = parse_samples(v1 + v1, GEOM, legacy=True)
    assert len(samples) == 2 and consumed == 2 * len(v1)


# ─── ingest ─────────────────────────────────────────────────────────────────


def test_ingester_defers_truncated_sample_and_incomplete_game(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    g1, g2, g3 = make_game(3), make_game(4, winner="def"), make_game(5, winner="undecided")
    f0 = data / "worker0.bin"
    write_games(f0, [g1, g2])
    with open(f0, "ab") as f:
        f.write(g3[0] + g3[1][:30])  # incomplete game + truncated sample
    (data / "worker0.gamelog").write_bytes(b"atk_capture,3\ndef_corner,4\ndraw_li")

    ing = Ingester(data, GEOM)
    res = ing.poll()
    assert res.games == 2 and res.samples == 7
    assert res.outcomes == {"atk": 1, "def": 1}
    assert ing.offsets["worker0.bin"] == sum(len(s) for s in g1 + g2)
    assert res.terminals == {"king_captured": 1, "def_corner": 1}
    assert ing.gamelog_offsets["worker0.gamelog"] == len("atk_capture,3\ndef_corner,4\n")

    # nothing new -> nothing read, offsets untouched
    res = ing.poll()
    assert res.games == 0 and ing.offsets["worker0.bin"] == sum(len(s) for s in g1 + g2)

    # finish the game and the gamelog line
    with open(f0, "ab") as f:
        f.write(g3[1][30:] + b"".join(g3[2:]))
    with open(data / "worker0.gamelog", "ab") as f:
        f.write(b"mit,5\n")
    res = ing.poll()
    assert res.games == 1 and res.samples == 5 and res.outcomes == {"undecided": 1}
    assert res.chunks[0].plies_left.tolist() == [4, 3, 2, 1, 0]
    assert res.chunks[0].offset == sum(len(s) for s in g1 + g2)
    assert ing.offsets["worker0.bin"] == f0.stat().st_size
    assert res.terminals == {"max_plies": 1}

    # offsets survive a round trip through JSON (state.json)
    ing2 = Ingester(data, GEOM, offsets=json.loads(json.dumps(ing.offsets)),
                    gamelog_offsets=json.loads(json.dumps(ing.gamelog_offsets)))
    assert ing2.poll().games == 0


def test_ingester_read_range_and_file_order(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    games = [make_game(2), make_game(3), make_game(2)]
    write_games(data / "worker10.bin", games)
    write_games(data / "worker2.bin", games[:1])
    ing = Ingester(data, GEOM)
    assert [p.name for p in ing.bin_files()] == ["worker2.bin", "worker10.bin"]
    start = sum(len(s) for s in games[0])
    end = start + sum(len(s) for s in games[1])
    chunks, new_off = ing.read_range(data / "worker10.bin", start, end)
    assert len(chunks) == 1 and len(chunks[0]) == 3 and chunks[0].offset == start and new_off == end


def test_terminal_column_mapping() -> None:
    assert terminal_column("atk_capture") == "king_captured"
    assert terminal_column("atk_threefold") == "threefold"
    assert terminal_column("def_fort") == "fort"
    assert terminal_column("draw_nocapture") == "no_capture"
    assert terminal_column("undecided_nocapture") == "no_capture"
    assert terminal_column("draw_limit") == "max_plies"
    assert terminal_column("max_plies") == "max_plies"
    assert terminal_column("weird") == "other"


def test_selfplay_stats_roundtrip_and_diff() -> None:
    s = SelfplayStats()
    s.add_game("atk", 10)
    s.add_game("def", 20)
    s.add_game("undecided", 30)
    s.add_terminal("atk_capture")
    snap = s.copy()
    s.add_game("atk", 40)
    d = s.diff(snap)
    assert (d.games, d.atk_wins, d.generated, d.avg_len) == (1, 1, 40, 40.0)
    assert SelfplayStats.from_dict(json.loads(json.dumps(s.to_dict()))).to_dict() == s.to_dict()
    assert abs(s.atk_pct - 50.0) < 1e-9


# ─── config / CLI ───────────────────────────────────────────────────────────


def test_parse_config_json_then_cli_override(tmp_path: Path) -> None:
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"run-dir": str(tmp_path / "r"), "batch": 64, "workers": 0,
                               "sync_every": 7, "se": True, "optimizer": "adamw"}), encoding="utf-8")
    args = parse_config(["--config", str(cfg)])
    assert args.batch == 64 and args.sync_every == 7 and args.se and args.workers == 0
    assert args.lr == 5e-4  # adamw default
    args = parse_config(["--config", str(cfg), "--batch", "128", "--optimizer", "sgd"])
    assert args.batch == 128 and args.sync_every == 7
    assert abs(args.lr - 0.02 * 128 / 1024) < 1e-12
    with pytest.raises(SystemExit):
        parse_config(["--run-dir", str(tmp_path / "r")])  # workers default > 0 but no engine
    with pytest.raises(SystemExit):
        cfg.write_text(json.dumps({"bogus": 1}), encoding="utf-8")
        parse_config(["--config", str(cfg)])


def test_smoke_preset_and_worker_command(tmp_path: Path) -> None:
    args = parse_config(["--run-dir", str(tmp_path), "--engine-bin", "taflzero.exe", "--smoke",
                         "--gumbel", "16", "--max-no-capture", "300", "--full-nodes", "33"])
    assert args.workers == 2 and args.channels == 48 and args.blocks == 4
    assert args.cheap_nodes == 8 and args.full_nodes == 33  # CLI overrides the preset
    assert (args.min_samples, args.sync_every, args.checkpoint_every, args.max_steps) == (2000, 10, 50, 120)
    cmd = worker_command(args, tmp_path, 3, tmp_path / "latest.onnx")
    s = " ".join(cmd)
    assert cmd[0] == "taflzero.exe"
    for flag in ("--net-watch", "--datagen-count 0", "--stop-file", "--pause-file", "--worker-id 3",
                 "--full-nodes 33", "--cheap-nodes 8", "--full-prob 0.25", "--max-plies 400",
                 "--variant copenhagen", "--gumbel 16", "--max-no-capture 300"):
        assert flag in s, flag
    assert str(tmp_path / "data" / "worker3.bin") in cmd
    assert str(tmp_path / "data" / "worker3.gamelog") in cmd
    assert str(tmp_path / "STOP") in cmd and str(tmp_path / "PAUSE") in cmd
    assert "--search-batch" not in s  # the engine default needs no flag
    args = parse_config(["--run-dir", str(tmp_path), "--engine-bin", "taflzero.exe", "--smoke",
                         "--worker-threads", "2048", "--worker-search-batch", "1"])
    s = " ".join(worker_command(args, tmp_path, 0, tmp_path / "latest.onnx"))
    assert "--threads 2048" in s and "--search-batch 1" in s


# ─── trainer loop end to end (no engine) ────────────────────────────────────


def _tiny_args(run_dir: Path, **over) -> list[str]:
    base = {
        "run-dir": str(run_dir), "workers": 0, "channels": 8, "blocks": 1, "batch": 16,
        "min-samples": 40, "sync-every": 2, "checkpoint-every": 3, "log-every": 2,
        "max-steps": 6, "warmup-steps": 2, "ingest-interval": 0.05, "train-threads": 1,
        "window-base": 1000, "sample-reuse": 100,
    }
    base.update(over)
    argv = []
    for k, v in base.items():
        if v is True:
            argv.append(f"--{k}")
        elif v is not False and v is not None:
            argv += [f"--{k}", str(v)]
    return argv


def test_trainer_loop_runs_checkpoints_and_resumes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    data = run_dir / "data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(0)
    games = []
    for i in range(12):
        winner = ["atk", "def", "undecided"][i % 3]
        games.append(make_game(int(rng.integers(4, 9)), winner=winner, first_stm=i % 2,
                               policy_valid=(i % 4 != 3)))
    write_games(data / "worker0.bin", games[:8])
    write_games(data / "worker1.bin", games[8:])
    total = sum(len(g) for g in games)

    args = parse_config(_tiny_args(run_dir, **{"aux-heads": True, "moves-left": True}))
    run = ContinuousRun(args)
    assert (run_dir / "latest.onnx").exists() and (run_dir / "latest.qnxx").exists()
    assert run.export_count == 1
    assert json.loads((run_dir / "config.json").read_text())["batch"] == 16
    rc = run.run()
    assert rc == 0
    assert (run_dir / "STOP").exists()  # created by max_steps; left for the operator to see

    state = json.loads((run_dir / "state.json").read_text())
    assert state["step"] == 6
    assert state["generated"] == total and state["games"] == 12
    assert state["export_count"] >= 1 + 3  # initial + steps 2, 4, 6
    assert set(state["offsets"]) == {"worker0.bin", "worker1.bin"}
    assert state["offsets"]["worker0.bin"] == (data / "worker0.bin").stat().st_size
    assert set(state["window_start"]) == {"worker0.bin", "worker1.bin"}
    assert state["selfplay"]["undecided"] == 4
    assert (run_dir / "ckpt" / "step000003.onnx").exists()
    assert (run_dir / "ckpt" / "step000006.qnxx").exists()
    assert (run_dir / "optimizer.pt").exists()
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [2, 4, 6]
    assert all(np.isfinite(float(r["loss"])) for r in rows)
    assert 0.0 <= float(rows[-1]["unknown_frac"]) <= 1.0
    assert rows[0]["paused"] == "0"
    assert (run_dir / "selfplay.csv").read_text().splitlines()[0].startswith("time,games,generated,atk_wins")

    # resume: STOP must be cleared by the caller (it stopped the run); continue to step 9
    (run_dir / "STOP").unlink()
    args2 = parse_config(["--resume", str(run_dir), "--max-steps", "9"])
    assert args2.batch == 16 and args2.aux_heads and args2.moves_left and args2.workers == 0
    run2 = ContinuousRun(args2)
    assert run2.step == 6 and len(run2.buffer) == total and run2.stats.games == 12
    assert run2.run() == 0
    state2 = json.loads((run_dir / "state.json").read_text())
    assert state2["step"] == 9 and state2["generated"] == total
    assert (run_dir / "ckpt" / "step000009.onnx").exists()
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [2, 4, 6, 8]


def test_seed_run_opens_the_window_with_its_newest_games(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    (seed_dir / "data").mkdir(parents=True)
    lengths = [5, 7, 4, 6, 8, 5, 6, 7]
    games = [make_game(n, winner="atk" if i % 2 else "def", first_stm=i % 2) for i, n in enumerate(lengths)]
    write_games(seed_dir / "data" / "worker0.bin", games)
    size = (seed_dir / "data" / "worker0.bin").stat().st_size
    (seed_dir / "state.json").write_text(json.dumps(
        {"offsets": {"worker0.bin": size}, "sample_offsets": {"worker0.bin": sum(lengths)}}))
    # The newest 15 positions start inside game 6 (6 + 7 = 13 < 15 <= 18): the games
    # from the first one starting at or after that point, 6 and 7, are taken.
    run_dir = tmp_path / "run"
    args = parse_config(_tiny_args(run_dir, **{"seed-run": str(seed_dir), "seed-samples": 15, "min-samples": 10}))
    run = ContinuousRun(args)
    assert len(run.buffer) == 6 + 7
    assert run.current_window() == replay_window(15, 1000, args.window_alpha, args.window_beta, args.window_cap)
    assert all(c.source == "seed:worker0.bin" for c in run.buffer._chunks)

    # Own games come after the seed ones; the window bookkeeping of the run's own file
    # is untouched by them, and a resume loads the seed again ahead of the own games.
    own = [make_game(5, winner="atk"), make_game(6, winner="def")]
    write_games(run_dir / "data" / "worker0.bin", own)
    assert run.run() == 0
    state = json.loads((run_dir / "state.json").read_text())
    assert state["window_start"]["worker0.bin"] == 0
    (run_dir / "STOP").unlink()
    run2 = ContinuousRun(parse_config(["--resume", str(run_dir), "--max-steps", "7"]))
    assert len(run2.buffer) == 6 + 7 + 5 + 6
    assert [c.source for c in run2.buffer._chunks][-2:] == ["worker0.bin", "worker0.bin"]


def test_fresh_run_refuses_existing_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{}")
    args = parse_config(_tiny_args(run_dir))
    with pytest.raises(SystemExit):
        ContinuousRun(args)


def test_ladder_fit_recovers_a_known_gap():
    from eval_ladder import fit_ratings
    # 75% for a over b is +191 Elo; b against the reference is even.
    matches = [{"a": "a", "b": "b", "w": 300, "l": 100, "d": 0}, {"a": "b", "b": "ref", "w": 200, "l": 200, "d": 0}]
    fit = fit_ratings(matches, "ref")
    assert abs(fit["ref"][0]) < 1e-6
    assert abs(fit["b"][0]) < 5
    assert abs(fit["a"][0] - 191) < 5
    assert 0 < fit["a"][1] < 60


def test_fort_bench_takes_attacker_to_move_prefixes():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    from build_fort_bench import fort_rows
    moves = " ".join(f"m{i}" for i in range(61))  # defenders made the last (61st) move
    rows = fort_rows({"id": 0, "attacker": "x", "opening": 16, "moves": moves})
    assert [r["k"] for r in rows] == [4, 8, 16, 24, 32]  # k=48 would cut into the opening
    for r in rows:
        n = len(r["moves"].split())
        assert n % 2 == 0 and 61 - r["k"] - 1 <= n <= 61 - r["k"]


# ─── eval side-car spacing ───────────────────────────────────────────────────


def test_eval_min_gap_spaces_checkpoints_and_finishes_started_ones() -> None:
    import eval_anchor as e
    done = {"step174000|copenhagen11x11", "step184000|step017005"}
    ck = lambda n: Path(f"step{n:06d}.onnx")  # noqa: E731
    assert e.spaced_out(ck(190000), done, 0)
    assert e.spaced_out(ck(184000), done, 20000)  # the newest rated one, one anchor left: finish it
    assert not e.spaced_out(ck(174000), done, 20000)  # an older half-rated one is left alone
    assert not e.spaced_out(ck(190000), done, 20000)  # only 6000 past 184000
    assert e.spaced_out(ck(204000), done, 20000)
    assert e.spaced_out(ck(2000), set(), 20000)  # nothing rated yet


def test_resume_drops_log_rows_written_after_the_saved_state(tmp_path: Path) -> None:
    from continuous import trim_csv_after
    p = tmp_path / "metrics.csv"
    p.write_text("time,step,loss\n1,100,5\n2,200,4\n3,300,3\n", encoding="utf-8")
    assert trim_csv_after(p, "step", 200) == 1
    assert p.read_text(encoding="utf-8").splitlines() == ["time,step,loss", "1,100,5", "2,200,4"]
    assert trim_csv_after(p, "step", 200) == 0
    assert trim_csv_after(tmp_path / "missing.csv", "step", 0) == 0


def test_dashboard_keeps_the_latest_timeline() -> None:
    from dashboard import latest_timeline
    rows = [{"step": s, "v": v} for s, v in [(100, "a"), (200, "a"), (300, "a"), (200, "b"), (300, "b"), (400, "b")]]
    assert [(r["step"], r["v"]) for r in latest_timeline(rows, "step")] == [(100, "a"), (200, "b"), (300, "b"), (400, "b")]


def test_trim_torn_tail_cuts_back_to_the_last_complete_game(tmp_path: Path) -> None:
    from ingest import trim_torn_tail
    data = tmp_path / "data"
    data.mkdir()
    g1, g2, g3 = make_game(5), make_game(4, winner="def"), make_game(6)
    whole = b"".join(g1) + b"".join(g2)
    torn = b"".join(g3[:3]) + g3[3][:100]           # half a game, then half a sample
    (data / "worker0.bin").write_bytes(whole + torn)
    feat = np.zeros((5 + 4 + 3, 144), dtype=np.uint8)  # records ran ahead of the samples
    (data / "worker0.feat").write_bytes(b"TZFEAT01" + bytes(8) + feat.tobytes())
    (data / "worker0.gamelog").write_bytes(b"def_corner,5\natk_capture,4\nkin")
    cut = trim_torn_tail(data / "worker0.bin", GEOM, 0, 0)
    assert cut["cut_at"] == len(whole) and cut["samples"] == 9
    assert (data / "worker0.bin").stat().st_size == len(whole)
    assert (data / "worker0.feat").stat().st_size == 16 + 9 * 144
    assert (data / "worker0.gamelog").read_bytes() == b"def_corner,5\natk_capture,4\n"
    assert (data / "worker0.bin.torn-tail").read_bytes() == torn
    assert trim_torn_tail(data / "worker0.bin", GEOM, 0, 0) is None    # already clean
    # records behind the samples: the cut goes back to the last game they all cover
    (data / "worker0.feat").write_bytes(b"TZFEAT01" + bytes(8) + np.zeros((7, 144), np.uint8).tobytes())
    cut = trim_torn_tail(data / "worker0.bin", GEOM, 0, 0)
    assert cut["cut_at"] == len(b"".join(g1)) and cut["samples"] == 5


def test_resume_after_a_crash_uses_the_saved_step_weights_and_cuts_the_tail(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    data = run_dir / "data"
    data.mkdir(parents=True)
    games = [make_game(6, winner=["atk", "def"][i % 2], first_stm=i % 2) for i in range(12)]
    write_games(data / "worker0.bin", games)
    run = ContinuousRun(parse_config(_tiny_args(run_dir, **{"max-steps": 3, "checkpoint-every": 3})))
    assert run.run() == 0
    (run_dir / "STOP").unlink()
    # the crash: newer weights in latest.qnxx than the state, a torn game at the end
    from qnxx_io import load_qnxx, save_qnxx
    m = load_qnxx(run_dir / "latest.qnxx")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    save_qnxx(m, run_dir / "latest.qnxx")
    size = (data / "worker0.bin").stat().st_size
    with open(data / "worker0.bin", "ab") as f:
        f.write(b"".join(make_game(5)[:2]) + b"\x01" * 50)
    run2 = ContinuousRun(parse_config(["--resume", str(run_dir), "--max-steps", "3"]))
    ck = load_qnxx(run_dir / "ckpt" / "step000003.qnxx")
    for a, b in zip(run2.model.state_dict().values(), ck.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())
    assert (data / "worker0.bin").stat().st_size == size
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    assert "torn tail cut" in log and "weights from step000003.qnxx" in log
