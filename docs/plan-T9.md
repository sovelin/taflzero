# T9: value target, extra value heads, position weighting, soft resignation

Status: implemented on branch feat/t9 (worktree D:\projects\zevratafl-t9), not started as a run.
T8 keeps running unchanged meanwhile.

T9 = T8 (10x128, SE, gpool, planes `base,last2,rep_dst,king12,ply,cap,mob`, batch 1024,
lr 0.02, Gumbel 16, 160/40 nodes, full_prob 0.25, KataGo window) plus every change
below at once. If T9 beats T8 at equal positions we keep all of it; if not, we bisect
(order in §9).

Hard rules for the implementation:

- Every new option defaults to off / the old behaviour. An existing run resumed with a
  new trainer or engine must train exactly as before (same as `lr_schedule`).
- The new engine is built into its own `zero-trainer/engines/<sha>/` directory. T8's
  pinned exe (`engines/6e2eb57`) is never rebuilt or overwritten.
- New nets must stay playable by old engines, the WASM build and the dashboard: ONNX
  output 0 = policy, output 1 = scalar value in [-1, 1], output 2 = moves-left (when the
  net has it). Nothing new is exported.
- No change to the `.bin` sample format. The two free bytes of the `.feat` record
  (offsets 78, 79) carry the new per-sample data; old `.feat` files have zeros there,
  which must mean "no information, weight 1".

## 1. Evidence from T8 (latest 4000 games, 735k positions, 2026-10-24 03:00)

| measurement | value |
|---|---|
| positions after the game is decided (\|root_q\| > 0.9 for 6 plies) | 23.2 % of known-result positions |
| sign errors in those decided positions (root_q vs result) | 0.3 % (def to move), 0.5 % (atk to move) |
| games cut by the ply limit (result unknown, dropped from the value loss) | 549 of 4000 = 13.7 % |
| cheap-search positions | 75 % of all positions, all in the value loss at full weight |

Calibration of root_q against the real result, full-search positions, known results:

| root_q bucket | mean result, atk to move | mean result, def to move |
|---|---|---|
| -0.8 | -0.57 | -0.91 |
| -0.6 | -0.01 | -0.65 |
| -0.4 | 0.00 | -0.38 |
| 0.0 | +0.22 | +0.07 |
| +0.4 | +0.69 | +0.63 |

The defenders' search is calibrated, the attackers' is pessimistic by 0.2-0.6 in the
losing half. With `value_lambda 0.5` half of every value target is this root_q, so the
bias is trained back into the net. Draw games are not in this table; counting them as
0 moves the attacker means toward 0 but cannot produce the asymmetry.

Also measured and rejected: averaging the T8 checkpoints 40000/42000/44000 (KataGo's
exported nets are weight averages) played the raw step044000 43-44-13, -3.5 Elo
(CI +-67). Not part of T9.

## 2. A. Value target = game result (config only)

`value_lambda: 1.0` (existing key). Full-search samples then train on z, like cheap ones
already do. root_q stays in the data and feeds the short-term heads (B) and the
weighting (E).

Expected: removes the attacker pessimism above. It is not a fix for forts: T6 learned
forts (its fort wave at 19-27M positions) on the same `value_lambda 0.5`, so the mixed
target does not block them; at most a calibrated attacker value may make the attack
learn faster. Risk: noisier targets early on; B is the counterweight.

## 3. B. Short-term value heads (trainer only)

KataGo trains its main value head on the result and three extra heads on exponentially
weighted future root values (python/katago/train/metrics_pytorch.py, td value terms;
cpp/dataio/trainingwrite.cpp `fillValueTDTargets`). Same here.

Targets, per game, computed once when the game enters the replay buffer
(`replay.Chunk`, new array `td: float32 (len, H)`), from the stored root_q of every
sample (full and cheap) and the result. Values are from the side-to-move view and the
side alternates every ply (tafl has no passes), so for horizon h with lambda = 1/h:

```
v[last] = lambda * q[last] + (1 - lambda) * z[last]
v[i]    = lambda * q[i]    + (1 - lambda) * (-v[i+1])
```

z of a game with unknown result: excluded (mask) unless C is on, then 0 (draw).

- Config: `td_horizons: "40,12,4"` (plies; empty = off), `td_weight: 0.5` (per head).
  KataGo's 6/16/50-turn horizons on 19x19 are about a quarter of a Go game at the long
  end; our self-play games are ~160 plies, hence 40.
- Net: one extra head `td_head` (same shape as the value head, `H` tanh outputs), only in
  `forward_with_aux`, never exported.
- Loss: MSE per horizon, masked like the value loss, weight `td_weight` each.
- Metrics: `td` (mean of the H losses) in metrics.csv and the dashboard loss chart.

## 4. C. Win / draw / loss value head

Both reference engines use a 3-way value head (KataGo: win/loss/no-result; lc0: WDL).
Ours is one tanh output with MSE, and the 13.7 % ply-limit games are masked out.

- Config: `value_wdl: true` (default false).
- Net: the value MLP ends in 3 logits (win, draw, loss from the side to move) instead
  of 1 + tanh. A new `value_wdl` flag in the qnxx payload; loading an old checkpoint
  without it builds the old head.
- Targets: decided game -> one-hot win or loss; ply-limit game -> draw. These games
  were excluded from the value loss; with C on they are included. Threefold stays an
  attacker win (rules, unchanged).
- Loss: cross-entropy, weight `value_weight` (new key, default 1.0, applies to MSE too).
- Export: the ONNX graph computes `p_win - p_loss` and returns it as output 1. The
  engine, WASM build, matches and anchors see a normal scalar value. Draw probability is
  not exported in T9.
- Moves-left stays masked for ply-limit games (their length is the cap, not a result).
- Metrics: `value` loss column becomes the CE when on (log the key so the dashboard
  labels it), plus `draw_pred` (mean predicted draw probability) and `draw_frac`
  (share of draw targets in the batch).

## 5. D. Soft resignation (engine)

**Revised 2026-10-24 (gradual, as KataGo).** The first version below cut all at once at
the threshold (cheap search, weight 0.1, with an exit margin). KataGo cuts in proportion:
reduction = ((v - 0.9) / 0.1)^2 over the last 3 values, so 0.95 gives 0.25 and only a dead
game gets the full cut. With the step version T9's attack trailed T8's (self-play 19.9 % vs
26.2 % at 4.0-4.5M positions) and lost the direct match at 4.1M 38-58-4, all of it on the
attacking side. Now: full searches shrink from full_nodes toward cheap_nodes, the weight
from 1 toward reduced_weight, both by the reduction; no state, no exit margin. The text
below is the original version.

KataGo never resigns in self-play; when the winrate passes 0.9 for 3 turns it lowers
the visits and the row weight (`reduceVisits`, `reducedVisitsWeight 0.1`,
cpp/configs/training/selfplay8b.cfg). Same here, with our two search sizes.

- Engine flags: `--soft-resign <threshold>` (off when absent), `--soft-resign-plies N`
  (default 3).
- State per game: the last N root values in the attackers' view. When all N are beyond
  +threshold or all beyond -threshold, the game is "reduced": every further move uses
  the cheap search (no policy target, no noise), whatever full_prob draws. The game
  leaves the reduced state when the latest root value falls below threshold - 0.1 on the
  winning side (hysteresis; entering again needs N fresh plies). Simulated on T8's root
  values: without the margin games left the state 1.55 times each (13 % of positions
  reduced), with it 0.86 times (16 %); 0.22 % of reduced positions belonged to the side
  that went on to lose.
- Every sample searched in the reduced state gets bit 0 of `.feat` byte 79 set
  (`FEAT2_REDUCED`).
- The game is always played to the end; results, reasons and the gamelog are unchanged.
- Trainer: `reduced_weight: 0.1` multiplies the sampling weight of reduced samples (E).
- Continuous/dashboard: new config keys `soft_resign` (0 = off) and `soft_resign_plies`,
  passed to the workers; `metrics.csv` gets `reduced` (share of reduced samples among
  the games added since the last log line).

Expected: ~16 % of positions move to the cheap search (simulation above), so roughly
10-15 % more games per hour, and fewer near-uniform policy targets from won positions. The old
`decided_policy_k` stays available but is not needed with D on.

## 6. E. Position weighting by surprise (engine + trainer)

KataGo writes positions where the search disagreed with the net more often
(`policySurpriseDataWeight 0.5`, `valueSurpriseDataWeight 0.1`, play.cpp ~2110-2170),
keeping each game's total weight. We keep games whole in the buffer (aux policy and
B need neighbours), so the weight is applied when drawing samples, not when writing.

Engine: for every full-search sample, `.feat` byte 78 = policy surprise
KL(policy target || raw net prior at the root), in nats, capped at 4, quantized as
`round(kl * 63.75)` (0-255). Cheap samples write 0.

Trainer, when a game enters the buffer:

```
ps[i] = byte78[i] / 63.75                       (policy surprise; full searches only)
vs[i] = min(1, (z - q[i])**2 / 4)               (value surprise, from data we have; known results only)
w[i]  = (1 - a - b) + a * ps[i] / mean_full(ps) + b * vs[i] / mean(vs)
        (policy term: mean over the game's full searches, other samples get a flat a, so
        weight is reordered among full searches but not moved from cheap ones; a term
        whose mean is 0 contributes its share uniformly)
w[i] *= reduced_weight if reduced[i]
w     = w * len(game) / sum(w)                  (a game keeps its total weight)
```

KataGo computes value surprise from the raw net value; we have no raw value in the
data, root_q is the stand-in. Documented deviation.

- Config: `surprise_policy: 0.5`, `surprise_value: 0.1` (both 0 = off, uniform sampling
  as now). `reduced_weight` from D also goes through this path.
- Sampling: `ReplayBuffer.draw_indices` draws over the window proportionally to w
  (cumulative weights per chunk + searchsorted), keeping the window boundary exact.
  With all weights 1 it must return the same distribution as today (test).
- Pacing (`sample_reuse`) is unchanged: steps x batch <= reuse x generated.
- Metrics: `surprise_mean`, `w_max` per log line; dashboard shows them in the details.

## 7. F. Soft policy (trainer only)

KataGo adds a cross-entropy on the policy target raised to the power 0.25 and
renormalized over legal moves, weight 8 (`-soft-policy-weight-scale 8.0`,
metrics_pytorch.py ~600). It trains the ranking of the moves after the best one.

- Config: `soft_policy_weight: 8.0` (0 = off).
- Net: a second policy head `soft_policy_head` (same as the policy head), only in
  `forward_with_aux`, never exported.
- Loss on full-search samples only (policy_valid), legal-masked, weight as configured.
- Metric: `soft_p`.

## 8. G. Cheap positions in the value loss (config only)

Existing key `cheap_value_weight`. KataGo trains no row from a cheap search
(`cheapSearchTargetWeight 0`); lc0 keeps 1 position in 32 to avoid correlated rows.

Checked on T8 with `tools/value_overfit.py` (value MSE against the result z, known
results only, ~5.5k full and ~16k cheap positions per range):

| checkpoint | range | z, full | z, cheap |
|---|---|---|---|
| step044000 (11.28M) | fresh, not trained yet | 0.481 | 0.460 |
| | seen, last 200k | 0.475 | 0.496 |
| | out of window, 4M back | 0.558 | 0.555 |
| step048000 (12.30M) | fresh, not trained yet | 0.477 | 0.483 |
| | seen, last 200k | 0.465 | 0.450 |
| | out of window, 4M back | 0.556 | 0.527 |

Fresh and seen positions score the same within +-0.035, and the sign of the gap flips
between the two checkpoints: no memorisation. Verdict: `cheap_value_weight` stays 1.0.
Re-run the check on T9 at ~10M positions, since pure-z targets (A) are more prone to
memorising whole games.

## 9. T9 preset and evaluation

`presets/t9-10x128-cuda.json` = T8's preset plus:

```
value_lambda 1.0
td_horizons "40,12,4", td_weight 0.5
value_wdl true, value_weight 1.0
soft_resign 0.9, soft_resign_plies 3, reduced_weight 0.1
surprise_policy 0.5, surprise_value 0.1
soft_policy_weight 8.0
cheap_value_weight 1.0 (§8: no overfit on T8)
lr 0.02, lr_schedule "" (same lr story as T8)
```

Anchors: copenhagen11x11 and T6 step486399 as T8, plus T8's best checkpoint at the time
T9 starts. Compare with T8 at equal positions on both fixed anchors (pooled per 3M
positions), the ladder, and direct matches T9 vs T8 at the same position count every
~3M.

Note for comparisons: with D on, positions per hour rise, so compare by positions and
by wall-clock both. "Positions" counts every written sample, reduced ones included.

All changes go in together, the risky ones included: dropping them would mean never
learning whether they mattered, and the worst case is one lost run.

Early tripwires (watch from the first checkpoints, act before 10M positions):

| change | signal of harm | when it shows |
|---|---|---|
| C (WDL) | `draw_pred` drifts far from the real draw share (~14 % of self-play games on T8), or the value/policy loss ratio departs sharply from T8's at equal positions | 1-2M positions |
| A+B (pure z) | self-play attacker win share and the copenhagen11x11 match score clearly below T8 at equal positions | 3-5M positions |
| E (surprise) | `w_max` explodes or a few positions dominate the draws (`surprise_mean` jumping) | from the first log lines |
| F (soft policy) | `grad_norm` well above T8's at equal positions (soft policy is ~85 % of the total loss at weight 8, KataGo's ratio but new for our lr) | first 1M positions |
| D (soft resign) | `reduced` far above ~20 %, or sign errors in reduced positions above ~1 % (re-run the decided-position stats) | 1M positions |

If a tripwire fires, turn that change off and restart. If T9 is worse than T8 at 5-10M
positions with no tripwire pointing at a culprit, bisect in this order, one short run
each (~10M positions): C off (the riskiest, changes the head), then E off, then A+B back
to `value_lambda 0.5`.

If T9 is better, we will not know which change did it. For the goal that does not
matter; per-change ablations can wait for a free GPU.

## 10. Implementation order

1. G check script and its verdict (hours).
2. A, B, F and the trainer side of C, E (trainer, replay, net, export, tests). The
   `.feat` bytes 78/79 are read as zeros until the engine writes them, so this part
   can be smoke-tested on T8's data.
3. Engine: surprise byte, soft resignation, flags plumbing; Rust tests; new
   `engines/<sha>/` build.
4. continuous.py / dashboard config keys, metrics columns, docs/continuous.md.
5. Smoke run (`--smoke`), then the T9 preset.

Tests (python -m pytest test_continuous.py, cargo test --no-default-features):

- td targets on a hand-made 4-ply game, both result signs, unknown result with C off/on.
- WDL target mapping, CE loss, export: the ONNX value output equals p_win - p_loss, an
  old engine plays the net, output count unchanged.
- An old qnxx (no new flags) loads and trains exactly as before; an old `.feat`
  (bytes 78/79 zero) gives weight 1 everywhere.
- Weighted draw: uniform when all weights are 1; proportional otherwise; never outside
  the window.
- Engine: soft resignation enters and leaves the reduced state, writes the flag, never
  ends a game early; surprise byte equals the quantized KL for a known prior/target.

## 11. Open points

- C's value_weight 1.0: CE and MSE have different scales (MSE on [-1,1] is up to 4).
  Start at 1.0, watch the value/policy loss ratio in the first 2M positions.
- B's horizons are a guess scaled from KataGo; they are one config string.
- E uses root_q for value surprise instead of the raw net value. On the PUCT path (not
  T9, which uses Gumbel) the surprise uses raw visits and a Dirichlet-noised prior.
- The WDL head ignores value_lambda / value_gamma; the config refuses them != 1.
- Draw probability is not exported, so the engine cannot yet prefer or avoid draws.
