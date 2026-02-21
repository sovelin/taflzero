## Cmd for build
wasm-pack build --target web --release

# Current Zero Training Pipeline

Self-play + training loop (current settings):

- Network: `TaflAlphaZeroNet` with `trunk_channels=32`, `num_blocks=6`
- MCTS iterations per move: `300` (after some time switch to `600` for attackers and `300` for defenders)
- Self-play per generation: `--games-per-gen 50000` (currently counts positions, not games)
- Dataset window: `--window 500000` positions
- Training: `--steps 3000`, `--batch 512`, `--lr 1e-3` (lr could be slightly less)
- Defender reweighting: `--defender-weight 1.0`
- Policy target temperature: `POLICY_TARGET_TEMP = 0.5` (see `zero-trainer/dataset.py`)
- Temperature: first 60 moves 1.0; after 0.0
- MCTS batch size=`8`

Example:

```
node orchestrate-zero.mjs --iterations 30 --games-per-gen 50000 --window 500000 --steps 1500 --batch 512 --lr 1e-3 --defender-weight 1.0 --workers 24 --start-net .\zero-trainer\weights\random_init.onnx
```

Notes:
- MCTS and self-play settings live in `src/search/gen_train_data.rs` and `src/search/mcts/mcts.rs`.
- `--games-per-gen` is passed to Rust and currently limits positions generated, not games.

# Train net 32 channels and 6 residual blocks

```
node orchestrate-zero.mjs --iterations 30  --games-per-gen 5000 --window 500000 --steps 3000 --batch 512 --lr 1e-3 --workers 24 --start-net .\zero-trainer\weights\gen0001.onnx --start-checkpoint .\zero-trainer\checkpoints\gen0001.onxx --start-gen 1
```

# Train net 16 channels and 4 residual blocks

```
node orchestrate-zero.mjs --iterations 20 --games-per-gen 3000 --window 15000 --steps 800 --batch 256 --lr 1e-3 --workers 24
```

# Train net 8 channels and 2 residual blocks

```
node orchestrate-zero.mjs --iterations 20 --games-per-gen 3000 --window 15000 --steps 500 --batch 256 --lr 2e-3 --workers 24
```

## New pipeline with validation

`node orchestrate-zero.mjs --iterations 1000 --games-per-gen 40000 --window 200000 --steps 1600 --batch 256 --lr 7e-4 --workers 24 --defender-weight 0.3 --start-net ..\zero-trainer\weights\gen0006.onnx --start-checkpoint ..\zero-trainer\weights\gen0006.onxx --start-gen 7 --sprt-elo1 10 --sprt-nodes 200 --sprt-max-pairs 10000 --engine-bin ..\target\release\zevratafl-rust.exe`