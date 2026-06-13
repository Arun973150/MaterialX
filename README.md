# stealth-phase1

Working **Phase 1** pipeline for multispectral stealth material discovery: ingest real
materials data, simulate radar + IR + visible response, and output ranked, simulation-validated
candidate multilayer stacks.

Progress is tracked in [`../PHASE_PLAN.md`](../PHASE_PLAN.md).

## Layout

```
configs/targets.yaml   # frozen numeric target spec (single source of truth)
src/stealth/
  config.py            # loads targets.yaml
  materials/           # DB queries, n,k loaders, candidate shortlist   (Phase 1)
  physics/             # optics.py (TMM, IR+visible) + radar.py (RCWA)  (Phases 2-3)
  surrogate/           # radar surrogate model                          (Phase 4)
  optimize/            # objective + NSGA-III / BoTorch                 (Phase 5)
  validate/            # high-fidelity re-check                         (Phase 6)
tests/                 # pytest
data/  reports/
```

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -e .
pytest                                             # Phase 0 smoke test
```

`meent` (GPU RCWA) and `openEMS` (FDTD) are installed on the Linux cluster, not locally —
local development covers the optics (TMM) path, which is pure Python.

## Secrets

Materials Project needs a free API key. Put it in `.env` (gitignored):

```
MP_API_KEY=your_key_here
```
