# zero-trainer: tiny Tafl AlphaZero net demo

Минимальный тестовый проект с нейросетью из `context.md`:

- вход: `6 x 11 x 11`
- выход: `policy logits (4840)` + `value (-1..1)`
- policy loss: cross-entropy по `pi_target` от MCTS visits
- value loss: MSE
- поддержка `legal_mask` перед `softmax`

## Файлы

- `az_micro_net.py` - модель (stem + 3 residual blocks + policy/value heads)
- `training_utils.py` - маскирование, нормализация visits, лоссы
- `demo_train_step.py` - демонстрационный forward + 1 шаг обучения
- `test_az_micro_net.py` - базовые тесты

## Запуск

```bash
python -m pip install torch pytest
python demo_train_step.py
pytest -q
```

## Стартовые веса в QNXX

```bash
python save_random_qnxx.py --out weights/random_init.qnxx --seed 42
```

Скрипт сохраняет случайно инициализированную сеть `TaflAlphaZeroNet` в файл
`.qnxx` (чекпоинт с `state_dict` и метаданными), который можно использовать как
стартовую точку обучения.

## Загрузка QNXX

```bash
python load_qnxx.py --in weights/random_init.qnxx --device cpu
```

Скрипт загружает и валидирует `.qnxx`, поднимает `TaflAlphaZeroNet` и выводит
краткую информацию о модели.
