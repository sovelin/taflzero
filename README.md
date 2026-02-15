## Cmd for build
wasm-pack build --target web --release

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