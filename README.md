## Cmd for build
wasm-pack build --target web --release

# Current Zero Training Pipeline

Self-play + training loop (current settings):

- Network: `TaflAlphaZeroNet` with `trunk_channels=16`, `num_blocks=2`
- MCTS iterations per move: `300`
- Self-play per generation: `--games-per-gen 50000` (currently counts positions, not games)
- Dataset window: `--window 500000` positions
- Training: `--steps 1500`, `--batch 512`, `--lr 1e-3`
- Defender reweighting: `--defender-weight 1.0`
- Policy target temperature: `POLICY_TARGET_TEMP = 0.5` (see `zero-trainer/dataset.py`)
- Temperature: first 60 moves 1.0; after 0.0

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
