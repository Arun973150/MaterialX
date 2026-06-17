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

## Notes / gotchas
- **VRAM:** MatterGen fits comfortably in 80 GB; default batch 16. Raise `--batch_size` if under-utilized.
- **First run** downloads model checkpoints — allow a few minutes + disk.
- **CIF parsing:** `screen.py` skips any unreadable generated file, so a few bad structures won't stop the run.
- **Determinism:** generation is stochastic; generate a few hundred and let screening filter — that's the intended workflow (MatterGen proposes broadly, the GNN screen + physics select).
- **Windows dev box:** MatterGen does **not** run on Windows; develop/screen/optimize locally, generate on the pod.
