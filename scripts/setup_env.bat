@echo off
chcp 65001 >nul
echo ========================================
echo CAR Deepfake Detection - GPU 环境配置
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

echo [1/6] Checking for existing virtual environment...
if exist "venv" (
    echo [INFO] Found existing venv, removing...
    rmdir /s /q venv
)
echo.

echo [2/6] Creating virtual environment...
python -m venv venv

echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat

echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

echo [5/6] Checking NVIDIA GPU...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo [INFO] PyTorch not installed yet or no NVIDIA GPU detected
) else (
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    if errorlevel 1 (
        echo [WARNING] CUDA available but GPU not detected properly
    )
)
echo.

REM Check CUDA version
echo [6/6] Installing PyTorch with CUDA support...
echo.

REM Try to detect CUDA version using nvidia-smi
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul') do set "DRIVER_VERSION=%%i"

if defined DRIVER_VERSION (
    echo [INFO] NVIDIA Driver: %DRIVER_VERSION%
    echo [INFO] Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [WARNING] Cannot detect NVIDIA driver, installing CPU version...
    echo [INFO] To use GPU, please:
    echo   1. Install NVIDIA drivers from https://www.nvidia.com/drivers
    echo   2. Run this script again
    pip install torch torchvision torchaudio
)

echo.
echo ========================================
echo Installing other dependencies...
echo ========================================
pip install -r requirements.txt

echo.
echo ========================================
echo Environment setup complete!
echo ========================================
echo.

REM Final GPU check
call venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}');" 2>nul
if not errorlevel 1 (
    call venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>nul
    if not errorlevel 1 (
        echo.
        echo ========================================
        echo [SUCCESS] GPU environment ready!
        echo ========================================
        echo.
        echo To activate the environment, run:
        echo   venv\Scripts\activate.bat
        echo.
        echo To start training, run:
        echo   python train.py --config configs/default.yaml
        echo.
    ) else (
        echo.
        echo ========================================
        echo [WARNING] GPU not available
        echo ========================================
        echo Please ensure:
        echo   1. You have an NVIDIA GPU
        echo   2. NVIDIA drivers are installed
        echo   3. CUDA toolkit is installed
        echo.
    )
)

pause
