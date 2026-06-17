# RunPod (A100) runbook — generative discovery end-to-end

Runs the part that needs a GPU: **MatterGen** generation. Everything downstream
(screening, the n,k bridge, the physics, the stack optimizer) is CPU-fine but runs
faster here too. Target pod: **A100 80 GB, PyTorch 2.x + CUDA 12 image**.

## 1. Provision + setup
```bash
# On the pod (Linux/CUDA):
git clone <your-repo> && cd <repo>/phase1     # or upload the phase1/ folder
pip install -e .                              # our package + deps
pip install git+https://github.com/microsoft/mattergen.git   # MatterGen (downloads checkpoints on first use)

# secrets (Materials Project key, for any DB pulls)
echo "MP_API_KEY=<your_key>" > .env
```

## 2. Generate novel materials (GPU)
```bash
# Radar conductors (near-zero band gap) and dielectric spacers (wide gap):
python -m stealth.discovery.generate --role radar_conductor   --n 128 --out runs/radar --run
python -m stealth.discovery.generate --role dielectric_spacer --n 128 --out runs/diel  --run
python -m stealth.discovery.generate --role ir_phasechange    --n 128 --out runs/ir    --run
```
Each writes generated crystals as CIFs into its `runs/<role>` dir.

## 3. Screen + close the loop into physics
```bash
# Rank generated candidates by predicted stability + band-gap fit:
python -m stealth.discovery.screen   --role dielectric_spacer --cif-dir runs/diel
# Generated material -> estimated n,k -> TMM emissivity (the closed loop):
python -m stealth.discovery.pipeline --cif-dir runs/diel
```

## 4. Stack design + deliverable (CPU)
```bash
python -m stealth.optimize.problem        # NSGA-III Pareto front + candidate stacks
python -m stealth.validate.report         # buildable candidate report
pytest -q                                 # full test suite
```

## 5. openEMS full-wave radar (the trustworthy-radar anchor)
Replaces the fast ECM approximation with true 3D FDTD. openEMS has **no pip/conda package**, so
the script builds it **from source** (GitHub) via `update_openEMS.sh --python`, which installs the
Python bindings into an isolated **venv** at `/workspace/openems-venv` (no conda). Everything lives
on the **volume** (survives restarts):
```bash
bash scripts/setup_openems.sh                      # one-time: from-source build -> /workspace/openems-venv
source /workspace/openems-venv/bin/activate        # activate the openEMS venv
cd /workspace/MaterialX
PYTHONPATH=src python -m stealth.physics.radar_fullwave --compare   # openEMS vs ECM on a test design
```
`simulate()` models the absorber as a **normal-incidence TEM unit cell** (PEC x-walls ‖ E,
PMC y-walls ‖ H, PEC ground at the zmax boundary, MUR open at zmin) with a +z TEM waveguide
port reading S11 — the proven openEMS Rect_Waveguide port convention. The first bring-up fixed
three issues: patch-edge mesh lines (so the conducting sheet isn't dropped as an "unused
primitive"), dropping the redundant ground box (the zmax PEC *is* the ground), and the +z port
direction (the earlier −z/​+z mismatch blew S11 up to +100 dB).

**Validated (2026-06-17):** on the test design `--compare` gives **mean |openEMS − ECM| = 1.55 dB**
across 1–30 GHz. Both methods show the X-band absorber; the full-wave gives a deeper, slightly
higher notch (−23 dB @ 8.5 GHz vs ECM −12.6 dB @ 8.0 GHz) — the expected lumped-vs-full-wave
difference, off-resonance agreement is <2.5 dB. openEMS is now the trustworthy radar anchor and
confirms the ECM is a sound design surrogate. (Knobs if a future design needs tighter agreement at
resonance: lateral mesh resolution and the cells-through-spacer count.)

## Re-provisioning after a restart
A pod restart wipes the container but keeps `/workspace`. To rebuild the base env in one command:
```bash
cd /workspace/MaterialX && git pull && bash scripts/setup_pod.sh
```
The MatterGen (`.venv-gen`), GNNOpt (`.venv-opt`), and openEMS (`/workspace/openems-venv`, plus its source tree and install prefix) envs live on the volume and persist.

## Notes / gotchas
- **VRAM:** MatterGen fits comfortably in 80 GB; default batch 16. Raise `--batch_size` if under-utilized.
- **First run** downloads model checkpoints — allow a few minutes + disk.
- **CIF parsing:** `screen.py` skips any unreadable generated file, so a few bad structures won't stop the run.
- **Determinism:** generation is stochastic; generate a few hundred and let screening filter — that's the intended workflow (MatterGen proposes broadly, the GNN screen + physics select).
- **Windows dev box:** MatterGen does **not** run on Windows; develop/screen/optimize locally, generate on the pod.
