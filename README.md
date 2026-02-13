## Cmd for build
wasm-pack build --target web --release

# Train net 32 channels and 6 residual blocks

```
node orchestrate-zero.mjs \
  --iterations 30 \
  --games-per-gen 5000 \
  --window 500000 \
  --steps 3000 \
  --batch 512 \
  --lr 1e-3 \
  --workers 24 \
```