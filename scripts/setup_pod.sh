#!/usr/bin/env bash
# Re-create the base Python env after a RunPod restart (container resets; /workspace persists).
# Usage:  cd /workspace/MaterialX && bash scripts/setup_pod.sh
set -e

cd "$(dirname "$0")/.."
echo ">> Installing stealth-phase1 into the base env (reuses the image's torch)..."
pip install -e . --ignore-installed blinker 2>&1 | tail -3

echo ">> Removing mismatched torchvision so matgl imports cleanly..."
pip uninstall -y torchvision 2>/dev/null || true

echo ">> Verifying..."
python -c "import stealth, matgl; print('base env ready: stealth + matgl OK')"
echo ">> Done. The GNNOpt (.venv-opt) and MatterGen (.venv-gen) envs are untouched on the volume."
