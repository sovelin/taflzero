# ZevTafl

<!-- avatar/logo here -->

**ZevTafl** is an AlphaZero-based engine for 11×11 Hnefatafl (Viking chess) and currently the strongest known engine for this variant. Play it live at **[playtafl.org](https://playtafl.org/)**. It combines Monte Carlo Tree Search (MCTS) with a deep residual neural network, trained entirely through self-play — no human games or handcrafted heuristics.

## How it works

The engine follows the AlphaZero approach:

1. **Self-play** — the current network plays against itself using MCTS, generating training data
2. **Training** — a new network is trained on recent self-play positions (policy + value heads)
3. **Evaluation** — the new network is tested against an anchor network via SPRT
4. **Repeat** — the cycle continues, with the network improving each generation

Key findings from training:
- A **large replay window** (millions of positions) is critical — it was the single biggest factor in training quality
- No sharpening, no defender weighting, no curriculum — the network discovers attack/defense balance naturally (~50/50 in self-play)
- Even small networks (6×64) trained correctly outperform large networks (20×256) trained with a small window

## Architecture

- Residual CNN (configurable depth and width)
- Policy head: move probabilities over all board positions
- Value head: win probability for the current side
- ONNX export for inference (via ORT)
- WebAssembly support via tract-onnx

## Build

### Native (GPU/CPU)

```bash
# CUDA + DirectML (auto-select, default)
cargo build --release

# CUDA only
cargo build --release --no-default-features --features cuda

# DirectML only
cargo build --release --no-default-features --features directml

# CPU only
cargo build --release --no-default-features
```

### WebAssembly

```bash
wasm-pack build --target web --release
```

## Current training run

128 channels, 10 residual blocks, 400 MCTS nodes:

```
node .\orchestrate-zero.mjs --iterations 1000 --games-per-gen 1000000 --window 5000000 --steps 5000 --batch 512 --lr 1e-4 --workers 8 --defender-weight 1.0 --start-net ..\zero-trainer\weights\gen0033.onnx --start-checkpoint ..\zero-trainer\weights\gen0033.onxx --start-gen 34 --engine-bin ..\target\release\taflzero.exe --no-sprt --anchor-net ..\zero-trainer\weights\gen0022.onnx --anchor-pairs 1000 --sprt-nodes 100 --no-restore-best
```

## Project structure

- `src/` — Rust engine (MCTS, self-play, game logic)
- `zero-trainer/` — Python training scripts
- `orchestrator/` — Node.js training loop orchestration
