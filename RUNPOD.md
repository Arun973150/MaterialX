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

## 6. Full run at scale — all 3 bands, end-to-end → openEMS-confirmed design
The complete *"generate more + simulate all three attributes + confirm with full-wave"* pass.
Four envs are involved (all on the volume); each step says which to activate.

### a. Generate at scale (MatterGen env — where `mattergen-generate` works, e.g. `/workspace/.venv-gen`)
```bash
cd /workspace/MaterialX
for role in radar_conductor dielectric_spacer ir_phasechange; do
  python -m stealth.discovery.generate --role $role --n 256 --out runs/$role --run
done
mkdir -p runs/all && cp runs/*/*.cif runs/all/        # one pool spanning all three roles
```
256/role ≈ a few hundred novel crystals (raise `--n` / `--batch_size` to use more of the 80 GB).

### b. Real visible/NIR optics for the pool (GNNOpt env, e.g. `/workspace/.venv-opt`)
```bash
cd /workspace/MaterialX
PYTHONPATH=src python scripts/gnnopt_predict_nk.py --cif-dir runs/all --out runs/all/gnnopt_nk.json
```

### c. Screen → S.U.N. → fused shortlist → 3-band design (base env: where `import stealth, matgl` works)
```bash
cd /workspace/MaterialX
# Rank every generated material, gate on S.U.N., write Top-N spanning all roles.
# --prescreen hull-checks only the most-stable N (the slow MP step) so a big pool stays fast:
PYTHONPATH=src python -m stealth.discovery.select_candidates \
    --cif-dir runs/all --gnnopt-nk runs/all/gnnopt_nk.json --top 8 --prescreen 40
# Per-candidate IR + visible simulation (closed optical loop):
PYTHONPATH=src python -m stealth.discovery.pipeline \
    --cif-dir runs/all --gnnopt-nk runs/all/gnnopt_nk.json
# Assemble + geometry-optimize the multilayer over radar+IR+visible using discovered materials:
PYTHONPATH=src python -m stealth.discovery.design_stack \
    --candidates data/final_candidates.parquet --gnnopt-nk runs/all/gnnopt_nk.json
```
Outputs: `reports/final_candidates.md` (ranked novel materials, all roles), `reports/designed_coating.md`
(the chosen multilayer with simulated radar+IR+visible), and `data/designed_coating.json` (the chosen
radar layer for openEMS).

### d. Full-wave sign-off on the chosen design (openEMS venv)
```bash
source /workspace/openems-venv/bin/activate
cd /workspace/MaterialX
PYTHONPATH=src python -m stealth.physics.radar_fullwave --design data/designed_coating.json --compare
```
openEMS re-checks the **exact** radar layer the optimizer picked against the ECM — the full-wave
confirmation of the delivered design (expect the §5-style agreement, full-wave notch a bit deeper).

## 7. Property-driven discovery (the proper build: #1 dataset → #3 predictors → #4/#5 reward)
This is the research upgrade that makes generation steer on *real predicted stealth properties*
instead of a band-gap proxy. Runs in the **base/stealth env** (+ the `[train]` extra).

```bash
cd /workspace/MaterialX && git pull
pip install -e ".[train]"        # jarvis-tools, lightgbm, scikit-learn, joblib

# #1 (optional unified table) — DFT-grade EM/dielectric/moduli/magmom labels by structure:
PYTHONPATH=src python -m stealth.discovery.dataset --out data/em_dataset.parquet

# #3 train the property predictors from structure (CFID + GBT, 90:10 split, held-out MAE):
PYTHONPATH=src python -m stealth.discovery.predictors --train
#   -> models/predictors/*.joblib + reports/predictor_metrics.json
#   expect (JARVIS-ML benchmarks): refractive index ~0.5, bulk/shear modulus ~10 GPa

# #4 score any generated pool by its predicted radar+IR signature, and
# #5 reward-rank it (stealth - weight - cost - (1-durability), manufacturability-gated):
PYTHONPATH=src python -m stealth.discovery.rl_finetune --role radar_magnetic --score-dir /workspace/run_full/all/cifs

# #5 full reward-guided generation loop (generate -> score -> elite -> re-condition):
PYTHONPATH=src python -m stealth.discovery.rl_finetune --role radar_magnetic --rounds 3
```

What each piece is:
- **#1** `dataset.py` (JARVIS-DFT dft_3d) + `em_literature.py` (measured GHz ε/μ — the only real
  permeability source).
- **#3** `predictors.py` — refractive index/dielectric, bulk/shear modulus (durability), magnetic
  moment (→ permeability via the literature prior), band gap (→ σ class), all from structure.
- **#4** `objective.py` — metal-backed single-layer (Dallenbach) radar reflection from predicted
  ε,μ + IR surface emissivity → the signature to minimize.
- **#5** `rl_finetune.py` — reward-ranked generation; `finetune_on_elite` marks the heavier RAFT
  adapter-fine-tuning upgrade.
- **#6** `manufacturability.py` — abundance/cost/toxicity/density gate + RAM chemistry families
  (already wired into `select_candidates`).

**Integration (already wired, fallback-safe):** once `predictors` are trained, `select_candidates`
automatically ranks by the #4 reward (radar reflection + IR emission) and `design_stack` places the
top discovered radar material as a single-layer absorber — no flags needed; it silently uses the
proxy ranking until the predictors exist.

```bash
# #2 generator specialization — build the RAM-family fine-tuning set, then fine-tune the adapter:
PYTHONPATH=src python -m stealth.discovery.finetune_dataset --em data/em_dataset.parquet \
    --role radar_magnetic --out data/finetune/radar_magnetic
#   -> CIFs + labels.csv ; fine-tune MatterGen's adapter on it (RAFT), then generate with that ckpt

# Validation + the one-page deliverable (anchor #10 + angular stress test #8 run live):
PYTHONPATH=src python -m stealth.discovery.anchor          # measured-data anchor
PYTHONPATH=src python -m stealth.discovery.deliverable     # -> reports/DELIVERABLE.md
```

## openEMS after a restart — reinstall its apt libraries
A container restart wipes apt packages (only `/workspace` persists), so the built openEMS can't
find its system libs (`libtinyxml`, `VTK`, `Boost`, `HDF5`, …) and import fails with
`ImportError: lib*.so: cannot open shared object file`. Reinstall them in one shot:
```bash
apt-get update && apt-get install -y libboost-all-dev libvtk9.1 libhdf5-dev libcgal-dev libtinyxml2.6.2v5
```
(If a stray lib is still missing: `ldd $(find /workspace/openems-venv -name "*.so" | grep -iE "openEMS|CSXCAD" | head -3) | grep "not found"` and install it.) `radar_fullwave --design --compare` then writes `data/openems_design.json`, which the deliverable cites as the per-design full-wave confirmation.

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
