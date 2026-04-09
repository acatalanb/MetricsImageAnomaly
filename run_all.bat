@echo off
REM Windows Batch script to run anomaly detection commands in sequence

set DATASET=ucirvine_chest_xray
set EPOCHS=20
set BATCH=32

echo 🚀 Starting Sequential Execution...

for %%M in (DenseNet121 ResNet50 EfficientNetB0) do (
    echo.
    echo --- Processing Model: %%M ---
    
    echo [1/2] Training model...
    python anomaly_detection.py --mode train --model %%M --epochs %EPOCHS% --batch %BATCH% --dataset %DATASET%
    
    echo [2/2] Evaluating model...
    python anomaly_detection.py --mode evaluate --model %%M --dataset %DATASET% --model_path cache\%%M_best.pth
)

echo.
echo ✅ All tasks completed successfully!
pause