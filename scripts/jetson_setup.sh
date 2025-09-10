#!/usr/bin/env bash
set -euo pipefail

# Jetson AGX Orin setup helper for GR00T N1.5 runtime
# - Creates venv, installs core Python deps
# - Torch install on Jetson depends on JetPack; supply a wheel URL via JETSON_TORCH_URL or install manually

echo "[1/5] Python: $(python3 --version)"

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "[2/5] Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "[3/5] Upgrading pip and wheel"
pip install --upgrade pip wheel

if python -c "import torch" >/dev/null 2>&1; then
  echo "[4/5] torch is already installed: $(python -c 'import torch; print(torch.__version__)')"
else
  echo "[4/5] torch not found."
  if [ -n "${JETSON_TORCH_URL:-}" ]; then
    echo "Installing torch from JETSON_TORCH_URL=$JETSON_TORCH_URL"
    pip install "$JETSON_TORCH_URL"
  else
    echo "Skipping torch install. Please install the Jetson-specific PyTorch wheel for your JetPack first, then re-run."
    echo "Refer to NVIDIA PyTorch for Jetson docs for the correct wheel URL."
  fi
fi

echo "[5/5] Installing Transformers runtime dependencies"
pip install -r requirements.txt || true

echo "Optional: accelerated Hub transfers (speeds up downloads)"
pip install -U hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1 || true

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
echo "Test run: python scripts/run_gr00t.py --prompt 'Describe how to grasp a mug.'"


