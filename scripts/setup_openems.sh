#!/usr/bin/env bash
# Install openEMS (FDTD full-wave solver) on the pod via conda-forge.
# Installs into /workspace (the persistent volume) so it SURVIVES pod restarts.
# Run once:  bash scripts/setup_openems.sh   then  conda activate openems
set -e

MF=/workspace/miniforge
if [ ! -d "$MF" ]; then
  echo ">> Installing Miniforge to $MF (on the volume)..."
  wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$MF"
fi

# shellcheck disable=SC1091
source "$MF/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^openems "; then
  echo ">> Creating 'openems' conda env with openEMS + CSXCAD from conda-forge..."
  conda create -y -n openems -c conda-forge python=3.11 openems matplotlib numpy h5py scipy
fi

conda activate openems
echo ">> Verifying openEMS import..."
python -c "from openEMS import openEMS; from CSXCAD import ContinuousStructure; print('openEMS + CSXCAD OK')"

echo
echo ">> Done. To use it:"
echo "     source $MF/etc/profile.d/conda.sh && conda activate openems"
echo "     cd /workspace/MaterialX && PYTHONPATH=src python -m stealth.physics.radar_fullwave --compare"
echo ">> (The conda env lives on the volume, so it persists across restarts —"
echo ">>  unlike the base pip env. If conda-forge's package name differs, adjust the create line.)"
