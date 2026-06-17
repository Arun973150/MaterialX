#!/usr/bin/env bash
# Install openEMS (FDTD full-wave solver) on the pod by building it FROM SOURCE.
# There is no `openems` package on conda-forge, so the verified path is the project's
# own build script. Everything lands on /workspace (the persistent volume) so it
# SURVIVES pod restarts. Run once:  bash scripts/setup_openems.sh
set -e

MF=/workspace/miniforge
SRC=/workspace/openEMS-Project
PREFIX=/workspace/openems-install      # built libs + python bindings go here (on the volume)

# 1. Miniforge (conda) on the volume ---------------------------------------------------
if [ ! -d "$MF" ]; then
  echo ">> Installing Miniforge to $MF (on the volume)..."
  wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$MF"
fi
# shellcheck disable=SC1091
source "$MF/etc/profile.d/conda.sh"

# 2. Build-dependency env (compiler + libs openEMS needs; NOT openEMS itself) ----------
if ! conda env list | grep -q "^openems "; then
  echo ">> Creating 'openems' env with the C++/Python build toolchain..."
  conda create -y -n openems -c conda-forge python=3.11 \
    cmake make gxx_linux-64 gcc_linux-64 \
    cython numpy h5py scipy matplotlib \
    boost-cpp tinyxml hdf5 cgal vtk
fi
conda activate openems

# 3. Build openEMS from source into $PREFIX (the proven install path) ------------------
if [ ! -d "$SRC" ]; then
  echo ">> Cloning openEMS-Project (recursive)..."
  git clone --recursive https://github.com/thliebig/openEMS-Project.git "$SRC"
fi
cd "$SRC"
echo ">> Building openEMS + CSXCAD + python bindings into $PREFIX (this takes a while)..."
./update_openEMS.sh "$PREFIX" --python

# 4. Verify ----------------------------------------------------------------------------
echo ">> Verifying openEMS import..."
python -c "from openEMS import openEMS; from CSXCAD import ContinuousStructure; print('openEMS + CSXCAD OK')"

echo
echo ">> Done. To use it in a fresh shell:"
echo "     source $MF/etc/profile.d/conda.sh && conda activate openems"
echo "     cd /workspace/MaterialX/phase1 && PYTHONPATH=src python -m stealth.physics.radar_fullwave --compare"
echo ">> (Both the conda env and $PREFIX live on the volume, so they persist across restarts.)"
