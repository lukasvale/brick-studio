#!/bin/bash
# One-time setup: Python environment, dependencies and the app bundle.
#   ./setup.sh             Metal (GPU) engine, the default and much faster
#   ./setup.sh --cpu-only  CPU engine only (smaller install, ~10x slower cutouts)
set -euo pipefail
cd "$(dirname "$0")"

[ "$(uname -s)" = "Darwin" ] || { echo "Brick Studio runs on macOS only."; exit 1; }
[ "$(uname -m)" = "arm64" ] || echo "Warning: built and tested on Apple Silicon; Intel Macs are untested."

PYTHON="${PYTHON:-python3.13}"
command -v "$PYTHON" >/dev/null || { echo "Python 3.13 not found. Install it from python.org or with 'brew install python@3.13', or set PYTHON=/path/to/python3.13."; exit 1; }

echo "Creating .venv with $("$PYTHON" --version)…"
"$PYTHON" -m venv .venv
.venv/bin/pip install --upgrade pip >/dev/null
echo "Installing image-processing packages…"
.venv/bin/pip install -r requirements.txt
if [ "${1:-}" != "--cpu-only" ]; then
  echo "Installing the Metal engine (PyTorch, about 1 GB)…"
  .venv/bin/pip install -r requirements-mps.txt
fi

# Learned tone/detail settings from earlier production work, so suggestions work from day one.
if [ ! -e work/learning/history.sqlite3 ]; then
  mkdir -p work/learning
  cp seed/learning-history.sqlite3 work/learning/history.sqlite3
  echo "Installed the starting learning history."
fi

./build_app.sh
echo
echo "Setup complete. Open 'Brick Studio.app' in this folder."
echo "The first cutout downloads the segmentation model (about 430 MB for Metal, 930 MB for CPU)."
