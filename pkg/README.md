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


Notes:
- MCTS and self-play settings live in `src/search/gen_train_data.rs` and `src/search/mcts/mcts.rs`.
- `--games-per-gen` is passed to Rust and currently limits positions generated, not games.

# Train net 128 channels and 10 residual blocks (Current RUN)

```
node .\orchestrate-zero.mjs --iterations 1000 --games-per-gen 1000000 --window 5000000 --steps 5000 --batch 512 --lr 1e-4 --workers 8 --defender-weight 1.0 --start-net ..\zero-trainer\weights\gen0033.onnx --start-checkpoint ..\zero-trainer\weights\gen0033.onxx --start-gen 34 --engine-bin ..\target\release\zevratafl-rust.exe --no-sprt --anchor-net ..\zero-trainer\weights\gen0022.onnx --anchor-pairs 1000 --sprt-nodes 100 --no-restore-best
```