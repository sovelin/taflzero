# zero-trainer: tiny Tafl AlphaZero net demo

A minimal test project implementing the neural network described in `context.md`.

- input: `6 x 11 x 11`
- output: `policy logits (4840)` + `value (-1..1)`
- policy loss: cross-entropy using `pi_target` from MCTS visit counts
- value loss: MSE
- supports `legal_mask` applied before `softmax`

## Files

- `az_micro_net.py` — model (stem + 3 residual blocks + policy/value heads)
- `training_utils.py` — masking, visit normalization, and losses
- `demo_train_step.py` — demonstration forward pass + one training step
- `test_az_micro_net.py` — basic tests

## Run

```bash
python -m pip install torch pytest
python demo_train_step.py
pytest -q
```

## Initial Weights in QNXX

```bash
python save_random_qnxx.py --out weights/random_init.qnxx --seed 42
```

The script saves a randomly initialized `TaflAlphaZeroNet` into a `.qnxx` file  
(a checkpoint containing `state_dict` and metadata), which can be used as a starting point for training.

## Loading QNXX

```bash
python load_qnxx.py --in weights/random_init.qnxx --device cpu
```

The script loads and validates the `.qnxx` file, constructs `TaflAlphaZeroNet`, and prints basic model information.
