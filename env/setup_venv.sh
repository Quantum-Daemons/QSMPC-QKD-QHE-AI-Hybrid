#!/usr/bin/env bash
# Create the Python 3.12 virtual environment (.venv) and install the core stack.
# Heavier tiers (AI, FHE, GPU, API) are opt-in — see requirements-*.txt and the Makefile.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

PYTHON="${PYTHON:-python3.12}"
VENV="${VENV:-.venv}"

echo "==> Creating venv at $VENV using $PYTHON ($($PYTHON --version))"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> Upgrading pip/wheel/setuptools"
pip install --upgrade pip wheel setuptools

echo "==> Installing core requirements (crypto + quantum CPU)"
pip install -r requirements-core.txt

echo "==> Installing qsh package (editable)"
pip install -e .

cat <<'EOF'

==> Core environment ready. Activate with:  source .venv/bin/activate

Optional tiers:
  pip install -r requirements-ai.txt     # torch, flwr, medmnist, onnx ...
  pip install -r requirements-fhe.txt    # tenseal + openfhe (commercial-OK CKKS)
  pip install -r requirements-api.txt    # FastAPI backend
  # GPU quantum sim (Blackwell/CUDA13): see requirements-gpu.txt (CUDA-Q / lightning.gpu)

EOF
