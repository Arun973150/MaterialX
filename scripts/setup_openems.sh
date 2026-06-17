#!/usr/bin/env bash
# Install openEMS (FDTD full-wave solver) on the pod by building it FROM SOURCE (GitHub).
# NO conda. The project's own update_openEMS.sh --python builds the C++ engine and
# installs the CSXCAD/openEMS Python bindings into an isolated venv it creates. Everything
# lands on /workspace (the persistent volume) so it SURVIVES pod restarts.
# Run once:  bash scripts/setup_openems.sh
set -e

SRC=/workspace/openEMS-Project
PREFIX=/workspace/openems-install      # built C++ libs/headers (on the volume)
VENV=/workspace/openems-venv           # venv that update_openEMS.sh --python creates next to PREFIX

# 1. System build dependencies (Ubuntu/Debian pod image). The PyTorch CUDA image already
#    has most of these; this is here for a fresh image. Safe to re-run.
echo ">> Ensuring build dependencies are present (apt)..."
apt-get update -qq || true
apt-get install -y -qq build-essential cmake git \
  libhdf5-dev libvtk9-dev libboost-all-dev libcgal-dev libtinyxml-dev \
  qtbase5-dev libqt5opengl5-dev python3-venv python3-dev || \
  echo ">> (apt step skipped/partial — fine if the libs are already in the image)"

# 2. Clone openEMS-Project recursively (pulls CSXCAD, fparser, AppCSXCAD, etc.)
if [ ! -d "$SRC" ]; then
  echo ">> Cloning openEMS-Project (recursive)..."
  git clone --recursive https://github.com/thliebig/openEMS-Project.git "$SRC"
fi

# 3. Build the engine and install the Python bindings into an isolated venv.
#    --python => CSXCAD + openEMS are pip-installed into $VENV (no conda involved).
cd "$SRC"
echo ">> Building openEMS + installing python bindings into a venv (this takes a while)..."
./update_openEMS.sh "$PREFIX" --python

# 4. Add the small extras the compare harness needs (numpy is already in the venv).
echo ">> Adding numpy/scipy/pandas to the openEMS venv for radar_fullwave --compare..."
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q numpy scipy pandas

# 5. Verify
echo ">> Verifying openEMS import..."
python -c "from openEMS import openEMS; from CSXCAD import ContinuousStructure; print('openEMS + CSXCAD OK')"

echo
echo ">> Done. To use it in a fresh shell:"
echo "     source $VENV/bin/activate"
echo "     cd /workspace/MaterialX && PYTHONPATH=src python -m stealth.physics.radar_fullwave --compare"
echo ">> (Source tree, install prefix, and venv all live on /workspace, so they persist across restarts.)"
