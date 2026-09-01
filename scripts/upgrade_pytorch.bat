@echo off
chcp 65001 >nul
echo ========================================
echo CAR - Quick PyTorch CUDA Upgrade
echo ========================================
echo.
echo This script will upgrade PyTorch to CUDA version
echo without recreating the virtual environment.
echo.

REM Check if venv exists
if exist "venv\Scripts\python.exe" (
    echo [INFO] Activating existing venv...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] venv not found. Creating new one...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    echo.
)

echo.
echo [INFO] Uninstalling current PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo [INFO] Checking NVIDIA GPU...
nvidia-smi 2>nul
if errorlevel 1 (
    echo [ERROR] NVIDIA GPU not detected!
    echo Please make sure NVIDIA drivers are installed.
    pause
    exit /b 1
)

echo.
echo [INFO] Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [INFO] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo.
echo ========================================
echo PyTorch CUDA upgrade complete!
echo ========================================
echo.
echo To start training with GPU:
echo   python train.py --config configs/default.yaml
echo.
pause
