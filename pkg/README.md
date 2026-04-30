# TaflZero

<img width="320" alt="TaflZero logo" src="https://github.com/user-attachments/assets/a2365b08-5069-4635-84c9-61b95389a847" />
<br />
<br />

[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.com/invite/BvakpmGSTa)



**TaflZero** is an AlphaZero-based engine for 11×11 Hnefatafl (Viking chess) and currently the strongest known engine for this variant. Play it live at **[playtafl.org](https://playtafl.org/)**. It combines Monte Carlo Tree Search (MCTS) with a deep residual neural network, trained entirely through self-play — no human games or handcrafted heuristics.

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

## WebAssembly usage

The WASM build exposes a `WasmClient` that broadcasts search output as `CustomEvent`s on the global scope. It is designed to run inside a **Web Worker** — the worker thread can block during search without freezing the UI.

### go infinite / stop

Because `go infinite` blocks the worker thread, `stop` cannot be sent via `postMessage` while a search is running. Instead, use a `SharedArrayBuffer` as a shared stop flag:

```js
// Worker setup (once, before any search):
const stopBuf = new Int32Array(new SharedArrayBuffer(4));
wasmClient.set_stop_buffer(stopBuf);

// Before each new search — reset the flag:
Atomics.store(stopBuf, 0, 0);
wasmClient.run("go infinite"); // blocks the worker; broadcasts info events

// From the main thread — stop the search:
Atomics.store(stopBuf, 0, 1);
// The worker finishes the current MCTS batch (~100 ms), broadcasts bestmove, then unblocks.
```

`SharedArrayBuffer` requires the page to be cross-origin isolated. Serve it with these HTTP headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## UCI interface

TaflZero speaks a UCI-based protocol. Run the binary and communicate via stdin/stdout.

**Command-line arguments:**

```
taflzero.exe [--net <model.onnx>] [--datagen <output.bin>] [--datagen-count <N>] [--gamelog <path>]
```

- `--net <path>` — ONNX network to load (default: `./default_nn.onnx`)
- `--datagen <path>` — run self-play data generation, write to binary file
- `--datagen-count <N>` — number of games for data generation
- `--gamelog <path>` — game log output path (used with `--datagen`)

**Commands:**

| Command | Response | Description |
|---|---|---|
| `uci` | `id name ...`, `uciok`, `option ...` | Identify engine |
| `isready` | `readyok` | Check engine is ready |
| `setoption name NNFile value <path>` | — | Load a different ONNX network at runtime |
| `position startpos` | — | Set starting position |
| `position startpos moves e1e2 ...` | — | Set position with move sequence |
| `position fen <fen> moves ...` | — | Set position from FEN |
| `go nodes <N>` | `info ...`, `bestmove <move>` | Search for N MCTS nodes |
| `go movetime <ms>` | `info ...`, `bestmove <move>` | Search for given time in ms |
| `go infinite` | `info ...`, `bestmove <move>` | Search until `stop` |
| `stop` | `bestmove <move>` | Stop current search and return best move |
| `quit` | `bye` | Exit |

**FEN format:** `3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a`
- `a` — attacker, `d` — defender, `k` — king; side to move: `a` or `d`

**Move format:** algebraic `<from><to>`, e.g. `a6b6`, `f6f1`

**Output:** `info depth <N> score cp <V> nodes <N> time <ms> speed <nps> pv <moves>` → `bestmove <move>`

## Current training run

128 channels, 10 residual blocks, 400 MCTS nodes:

```
node .\orchestrate-zero.mjs --iterations 1000 --games-per-gen 1000000 --window 5000000 --steps 5000 --batch 512 --lr 1e-4 --workers 8 --defender-weight 1.0 --start-net ..\zero-trainer\weights\gen0033.onnx --start-checkpoint ..\zero-trainer\weights\gen0033.onxx --start-gen 34 --engine-bin ..\target\release\taflzero.exe --no-sprt --anchor-net ..\zero-trainer\weights\gen0022.onnx --anchor-pairs 1000 --sprt-nodes 100 --no-restore-best
```

## Project structure

- `src/` — Rust engine (MCTS, self-play, game logic)
- `zero-trainer/` — Python training scripts
- `orchestrator/` — Node.js training loop orchestration
