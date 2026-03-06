@echo off
setlocal

call .venv\Scripts\activate.bat

set CHECKPOINT=weights\gen0363.onxx
set DATA=selfplay.bin
set WINDOW=500000
set BATCH=256

for %%S in (1000 2000 4000) do (
    for %%L in (1e-4 3e-5 1e-5) do (
        set NAME=weights\gen0364_exp_steps%%S_lr%%L
        echo.
        echo === steps=%%S  lr=%%L ===
        python .\train.py --data %DATA% --checkpoint %CHECKPOINT% --out weights\gen0364_exp_steps%%S_lr%%L.onnx --save-checkpoint weights\gen0364_exp_steps%%S_lr%%L.onxx --window %WINDOW% --steps %%S --batch %BATCH% --lr %%L --defender-weight 1.0 --no-restore-best
        if errorlevel 1 (
            echo FAILED: steps=%%S lr=%%L
            exit /b 1
        )
    )
)

echo.
echo === All 9 done ===
