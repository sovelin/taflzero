@echo off
setlocal

call .venv\Scripts\activate.bat

set CHECKPOINT=weights\gen0437_20x256.onxx
set DATA=selfplay.bin
set WINDOW=2000000
set BATCH=256
set STEPS=4000

for %%L in (1.5e-5 3e-5 6e-5 1e-4) do (
    echo.
    echo === steps=%STEPS%  lr=%%L ===
    python .\train.py --data %DATA% --checkpoint %CHECKPOINT% --out weights\gen0438_exp_lr%%L.onnx --save-checkpoint weights\gen0438_exp_lr%%L.onxx --window %WINDOW% --steps %STEPS% --batch %BATCH% --lr %%L --defender-weight 1.0 --no-restore-best
    if errorlevel 1 (
        echo FAILED: lr=%%L
        exit /b 1
    )
)

echo.
echo === All 4 done ===
