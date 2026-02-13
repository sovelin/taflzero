## Cmd for build
wasm-pack build --target web --release

# Train net 32 channels and 6 residual blocks

```
node orchestrate-zero.mjs --iterations 30  --games-per-gen 5000 --window 500000 --steps 3000 --batch 512 --lr 1e-3 --workers 24 --start-net .\zero-trainer\weights\gen0001.onnx --start-checkpoint .\zero-trainer\checkpoints\gen0001.onxx --start-gen 1
```