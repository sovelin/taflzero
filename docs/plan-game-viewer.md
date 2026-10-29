# Dashboard game viewer

Status: specification, 2026-10-22. Branch off `feat/continuous`. Nothing here touches
a running trainer, the sample format, or the engine's search; the engine changes are
confined to what the match mode prints.

## Goal

From the dashboard, open a modal, pick a game and step through it on a real board with
the engine's evaluation per ply, the result and how it ended. Two sources:

* **Self-play** games of the selected run, read from `data/workerN.bin` (every ply is a
  sample: position, `root_q`, final result `z`, the visit-count policy target, flags) and
  `data/workerN.feat` (the per-sample feature record written next to the samples; see
  `docs/probe-input-planes.md`, record layout, and byte 77 `terminal` added by
  `docs/plan-T7-planes.md`).
* **Anchor matches** of the eval side-car (`eval.games.jsonl`, one line per match, a
  `games` list), once the match mode records moves and evaluations (part 1).

The board is `taflground` (Vue 3 component, controlled: it renders `piecesList` and
emits nothing we need). A built copy is in `D:\projects\hnef\node_modules\taflground`
(`dist/index.umd.cjs`, `dist/index.js`, `dist/style.css`, `dist/index.d.ts` for the
`Piece`/`Move` types and the square convention; read the .d.ts before using it). It is a
private package: copy `dist/` into `zero-trainer/dashboard/vendor/taflground/` (keep its
LICENSE if there is one) and serve it from `dashboard.py` with a static route for
`/vendor/...`. Vue 3 comes from cdnjs (`vue/3.5.x/vue.global.prod.min.js`, exact
version). The dashboard page stays a single vanilla-JS page; the board is a small Vue
app mounted inside the modal. Verify in the .d.ts how squares are numbered and map our
flat `Square` (0..120, see `src/board/types.rs` and `get_square_from_algebraic`) onto
it; add a JS unit check in the page's self-test (a `?selftest` query already exists? if
not, a tiny `console.assert` block run once on load) that `a1`, `k11`, the throne map
to the right squares.

## Part 1 — match mode records moves and evaluations

`src/search/match_play.rs`: `GameOutcome` gains `moves: Vec<Move>` (the opening moves
first, then the searched ones) and `evals: Vec<f32>` (one per searched move: the root
value of that search from the **mover's** perspective, as `mcts_search` leaves it on the
tree root; openings have no eval). `run_match` in `src/main.rs` prints, per game, after
the existing `game …` line, one line `moves <pair> <a_attacks> <m1> <m2> …` in algebraic
(`Board::move_to_algebraic` or whatever the UCI output uses, same notation as UCI
`bestmove`) and one line `evals <pair> <a_attacks> <v1> <v2> …` (4 decimals). The `done`
line stays last. `zero-trainer/tools/match.py` parses both lines and stores
`"moves": "<space-separated>"` and `"evals": [..]` on each game record it writes to
`eval.games.jsonl` (older records without them keep working everywhere). The existing
`match_reports_every_game_once` test is extended: moves count equals `plies`, evals
count equals `plies - opening`.

## Part 2 — dashboard API

`dashboard.py`:

* `GET /api/games?name=<run>&source=selfplay&n=50&worker=0` → the last `n` games of
  `data/worker<w>.bin`: `[{id, start_sample, plies, result: "atk"|"def"|"draw"|"unknown",
  reason: <REASON key or "unknown">, len_plies, time?}]`, newest first. `id` is
  `"<worker>:<start_sample_index>"`. Implementation: a per-run game index
  `data/games<w>.idx.json` (`{"bin_size": <bytes indexed>, "games": [[offset, samples], …]}`)
  built incrementally by scanning the sample stream from the indexed size forward using
  `dataset.parse_samples`/`sample_size_at` and the `FLAG_LAST_OF_GAME` flag (truncated
  trailing game ignored until complete). First build of a 40 GB file is a background
  thread with a progress field in the response (`{"indexing": <fraction>}`); the index
  keeps only the last 5,000 games. Reason: from the `.feat` record of the game's last
  sample (byte 77) when the `.feat` exists and covers that sample, else `"unknown"`.
  Result: from the last sample's `value` sign and `stm` (`result_known` false → "draw").
* `GET /api/game?name=<run>&id=<worker>:<start>` → `{plies: [{fen?, pieces:
  [{sq, type}], stm, move: {from, to} | null, root_q, z, policy_valid, top: [{from, to,
  visits}] (top 5 of the policy target, decoded with dataset's policy index mapping),
  cheap: bool}], result, reason}`. `move` of ply i is the move that led **to** position
  i (null for the first), recovered as the one mover square that empties and the one that
  fills between consecutive positions (same rule as `recover_move` in
  `src/search/feature_dump.rs`; it holds for every sample pair in a game). Positions are
  decoded from `BitPosition` (`dataset.decode_sample` already unpacks the three piece
  planes).
* `GET /api/games?name=<run>&source=match` → the matches of `eval.games.jsonl` that have
  `moves`, newest first: `[{id: "<line>:<index>", key, time, nodes, a_side, score,
  reason, plies}]`; `GET /api/game?...&id=m:<line>:<index>` replays the moves with a
  Python board: there is none in `zero-trainer` and we do not want one, so the engine
  provides it: `taflzero --replay "<moves>"` prints one line per ply with the
  `BitPosition` as hex (50 bytes) after each move (add this tiny mode to `main.rs`;
  it uses `Board::make_move_simple` and the variant's start position). The dashboard
  runs it once per request (sub-second) and caches by id. Evaluations come from the
  record's `evals` (checkpoint's and anchor's moves alternate; show both, labelled by
  who moved).

## Part 3 — the modal

A `Games` button next to `Details` on the Elo card opens a modal:

* Left: source select (`self-play`, `anchor matches`), worker select for self-play, a
  table of games (newest first; result, reason, plies, for matches the checkpoint,
  anchor and side), click selects.
* Right: the `TaflBoard` with the position of the current ply, `highlightMove` = the
  move that led to it; below it a slider plus ◀ ▶ buttons and keyboard arrows, `ply i / n`,
  side to move, result and reason in words (the `REASON` table already in the page).
* Under the board a small Chart.js line of `root_q` per ply (self-play; sign-normalised
  to the attackers' perspective: `root_q` is from the mover's view, flip it on defender
  plies) with the current ply marked, and the final `z` as a dashed line; for matches
  the two engines' `evals` as two series. Hovering the chart moves the board.
* Cheap-search plies (`policy_valid = 0`) shown in grey on the chart; the top-5 policy
  moves of the current ply listed under the board with their visit shares.
* The modal keeps the chosen game across dashboard refreshes (no re-fetch while open
  unless the game list is refreshed by the user).

## Acceptance

* `cargo fmt --all --check`, `cargo test --no-default-features`, `python -m pytest
  zero-trainer -q`.
* Tests: the game index on a small synthetic `.bin` (3 games, one truncated) returns
  two games with the right sample counts and ignores the truncated one, then picks it
  up once completed; `/api/game` on that file reproduces the moves of a hand-made game;
  `match.py` parsing of `moves`/`evals` lines; `--replay` of a 3-move sequence gives the
  expected positions.
* Manual: open `http://127.0.0.1:8787/`, open the Games modal for `T6-10x128`, step
  through a recent self-play game, screenshot it (Playwright is available via the
  `mcp__playwright__*` tools if you have them; otherwise describe what you verified).

## Out of scope

* Any change to the sample format, the search, `fill_input`, `continuous.py`,
  `eval_anchor.py`, or the running processes. The dashboard process may be restarted
  (plain `Stop-Process` on `dashboard.py`'s pid, **never** `taskkill /T`), it holds no
  children that matter.
* Rules or legality in JavaScript: the viewer only replays recorded games.
