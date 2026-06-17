# MaterialX — AI-Driven Multispectral Stealth Material Discovery

An end-to-end, **open-source** pipeline that discovers and designs *smart coating materials*
that conceal objects simultaneously across **radar, infrared, and visible** detection bands —
using AI for both material discovery and multi-objective design, validated in physics simulation.

> Built entirely on open data and open tools (no commercial EM solvers, no proprietary datasets).
> Real materials data → physics models → AI optimization → ranked, buildable candidate stacks.

---

## The problem

Modern platforms face detection across the whole electromagnetic spectrum at once. The hard part is
a **physical paradox**: radar stealth wants *high* electrical conductivity (to absorb microwaves),
while IR stealth wants *low* conductivity / controlled emissivity. No single material does both — so
the solution is a **vertically integrated multilayer stack**, each layer handling one band, co-designed
by AI so the layers don't fight each other.

| Band | Detector | Stealth requirement | Layer |
|---|---|---|---|
| Radar (1–30 GHz) | microwave radar | absorb (impedance-matched, lossy) | patterned conductor + grounded spacer |
| Infrared (3–14 µm) | thermal imager | low / switchable emissivity | VO₂ (thermochromic) |
| Visible (400–700 nm) | camera / eye | match background color | electrochromic polymer |

---

## Architecture

The system is two complementary tracks plus a shared physics core:

```
                         ┌───────────────────────────────────────────────┐
   TRACK A — DISCOVERY   │  generate / pull materials  →  predict props   │
   (find the materials)  │  →  screen + classify  →  synthesizability     │
                         │     + synthesis route (formulas + pathways)    │
                         └───────────────────────┬───────────────────────┘
                                                 │  candidate materials (+ n,k bridge)
                         ┌───────────────────────▼───────────────────────┐
   SHARED PHYSICS CORE   │  Optics (TMM): IR emissivity + visible color   │
   (score any design)    │  Radar (ECM): reflection loss / RCS            │
                         └───────────────────────┬───────────────────────┘
                                                 │  fast, differentiable evaluators
                         ┌───────────────────────▼───────────────────────┐
   STACK DESIGN          │  NSGA-III multi-objective optimizer            │
   (design the recipe)   │  → Pareto front → ranked candidate STACKS      │
                         │  → high-resolution validation → report         │
                         └───────────────────────┬───────────────────────┘
                                                 ▼
   TRACK B — CONTROL      sensors → CNN threat classifier → RL controller
   (adapt in real time)   → adapt the coating   [documented next phase]
```

### Track A — Generative material discovery (`src/stealth/discovery/`)
1. **Generate** novel inorganic crystals with **MatterGen** (diffusion model), conditioned on a
   stealth-relevant proxy property per layer (`generate.py`, runs on GPU / RunPod).
2. **Screen** them locally with **matgl** GNNs — predict formation energy (stability) + band gap, then
   classify each material into the layer it best serves (`screen.py`).
3. **Bridge to physics** — estimate each material's wavelength-resolved `n,k` from its predicted band
   gap (Moss / Drude / Tauc), so a *generated* material can be dropped straight into the optics model
   (`optical_bridge.py`).
4. **Synthesizability + route** — SMACT chemical-validity + GNN stability + a first-order precursor /
   synthesis route → the gist's *"chemical formulas + manufacturing pathways"* (`synthesis.py`).

### Shared physics core (`src/stealth/physics/`)
- **`optics.py`** — Transfer Matrix Method (Abelès). Computes IR emissivity (= absorptance, Kirchhoff)
  and visible color (reflectance → CIE L\*a\*b\* → ΔE). Exact, millisecond-fast.
- **`radar.py`** — Equivalent-Circuit Model for a grounded patterned absorber → reflection loss vs
  frequency. **`radar_fullwave.py`** wraps openEMS (FDTD) for high-fidelity confirmation;
  **`radar_sweep.py`** generates the geometry→reflection dataset.

### Stack design + validation (`src/stealth/optimize/`, `src/stealth/validate/`)
- **`optimize/`** — a 6-D stack design vector → joint objective (radar + IR + visible) under a weight
  constraint, searched by **NSGA-III** (`pymoo`) for the Pareto trade-off front.
- **`validate/report.py`** — re-checks the top candidates at high resolution and writes a buildable
  candidate report.

### Materials data (`src/stealth/materials/`)
Real optical constants from **refractiveindex.info**, plus **Materials Project** / JARVIS properties; a
typed material registry (`registry.py`), optical-constant loader (`optical.py`), and the candidate
shortlist (`shortlist.py`).

### Track B — Real-time control (planned)
Sensors → CNN scene/threat classifier → RL controller adapting the coating, with the physics core as
the digital twin. Documented in `docs/RESEARCH.md`; not yet implemented.

---

## Key results (in simulation)

- **Stack optimizer:** from a 48-design Pareto front, **21 designs meet X-band radar (< −10 dB) AND
  LWIR emissivity (< 0.3) simultaneously** — best ≈ −12 dB / 0.234 at ~7 kg/m². See
  [`reports/candidates.md`](reports/candidates.md).
- **Discovery → physics loop closes:** generated/candidate materials flow all the way through to a
  *computed* signature; synthesizable formulas + routes in
  [`reports/discovery_dossier.md`](reports/discovery_dossier.md).
- **41 automated tests pass**, including physics sanity checks (e.g. the VO₂ insulating→metallic
  LWIR emissivity switch reproduced from real measured optical data).

**Honest limits:** full X+Ku radar needs a *multilayer* absorber (single-layer is narrowband); visible
ΔE is bounded by the open PEDOT:PSS/VO₂ palette (needs tunable ProDOT optical data); radar uses the fast
ECM and should be openEMS-confirmed before fabrication; generated-material `n,k` is a first-order
estimate (a trained optical GNN would refine it).

---

## Repository layout

```
src/stealth/
  config.py            # frozen target spec + secrets (.env)
  materials/           # registry, optical-constant loaders, Materials Project client, shortlist
  physics/             # optics.py (TMM) · radar.py (ECM) · radar_fullwave.py (openEMS) · radar_sweep.py
  optimize/            # objective.py · problem.py (NSGA-III)
  validate/            # report.py (high-res re-check → candidate report)
  discovery/           # targets · generate (MatterGen) · screen (matgl) · optical_bridge · synthesis · pipeline
  surrogate/           # (Phase 4 — optional ML surrogate for slow full-wave; not required)
configs/               # targets.yaml (band thresholds) · materials.yaml (candidate catalog)
reports/               # candidates.md · discovery_dossier.md
docs/                  # PHASE_PLAN.md (build tracker) · RESEARCH.md (full feasibility study)
tests/                 # 9 test modules
RUNPOD.md              # A100 / RunPod runbook for the generative step
```

---

## Quickstart

```bash
cd phase1
python -m venv .venv && . .venv/Scripts/activate     # Windows; use bin/activate on Linux
pip install -e .
pytest -q                                            # 41 tests

# Materials Project key (free at https://materialsproject.org) for DB pulls:
echo "MP_API_KEY=your_key" > .env
```

Run the pieces (all CPU-fine except MatterGen):

```bash
python -m stealth.materials.shortlist        # build the material shortlist
python -m stealth.physics.optics             # VO2 IR emissivity switch demo
python -m stealth.optimize.problem           # NSGA-III → Pareto front + candidate stacks
python -m stealth.validate.report            # buildable candidate report
python -m stealth.discovery.screen --demo    # GNN screening + role classification
python -m stealth.discovery.pipeline --demo  # discovery → n,k → physics (closed loop)
python -m stealth.discovery.synthesis --demo # formulas + synthesizability + routes
```

**Generative step on GPU / RunPod (A100):** see [`RUNPOD.md`](RUNPOD.md).

---

## Tech stack

| Layer | Tools |
|---|---|
| Materials data | Materials Project (`mp-api`), JARVIS, refractiveindex.info |
| Generative discovery | **MatterGen** (diffusion), **matgl** (M3GNet/MEGNet GNNs), SMACT |
| Physics | `tmm` (optics), Equivalent-Circuit + **openEMS** (radar) |
| Optimization | `pymoo` (NSGA-III) |
| Color science | `colour-science` (CIE L\*a\*b\*, ΔE) |
| Core | Python 3.10+, NumPy/SciPy, PyTorch, pandas, pymatgen |

---

## Documentation

- [`docs/PHASE_PLAN.md`](docs/PHASE_PLAN.md) — phase-by-phase build tracker (status of every component).
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — full feasibility study mapping the project to 2025–26 SOTA.
- [`RUNPOD.md`](RUNPOD.md) — GPU runbook for the generative step.

*Source specification PDFs are intentionally excluded from this repository.*
