#!/bin/bash
echo "========================================"
echo "CAR Deepfake Detection - Environment Setup"
echo "========================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv venv

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Upgrading pip..."
pip install --upgrade pip

echo "[4/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train.py --config configs/default.yaml"
echo ""