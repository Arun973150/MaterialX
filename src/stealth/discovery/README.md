# Track A — Generative Material Discovery (PoC)

Demonstrates the gist's core capability: **generate novel materials → predict their
properties → screen → rank**, matching real 2025–26 SOTA.

```
 A100 (CUDA)                      local (CPU/GPU)
 ┌────────────────┐   CIFs   ┌──────────────────────────────┐
 │  MatterGen     │ ───────► │ matgl GNNs: predict E_form +  │
 │  generate.py   │          │ band gap → classify role →    │
 │ (cond. on a    │          │ filter + rank  (screen.py)    │
 │  proxy prop)   │          │      ↓                        │
 └────────────────┘          │ ranked novel candidates/layer │
                             └──────────────────────────────┘
```

## Why this split
- **Generation** = MatterGen (Microsoft diffusion model) — needs CUDA; runs on the A100.
- **Screening** = matgl pretrained M3GNet (formation energy) + MEGNet (band gap) — runs anywhere.
  Generated crystals are novel, so they're not in any database → a GNN must predict their properties.

## The proxy idea (important)
MatterGen conditions on *intrinsic* crystal properties, not "radar absorption". So per layer we
steer generation toward the intrinsic proxy that makes a good material there (see `targets.py`):

| Layer role | Proxy target | Screen rule |
|---|---|---|
| `radar_conductor` | near-zero band gap (metallic/lossy) | gap < 0.5 eV, stable |
| `ir_phasechange` | narrow-gap V–O oxide (VO₂-like) | gap 0–1 eV, stable |
| `dielectric_spacer` | wide band gap (low-loss) | gap 3–12 eV, stable |

The visible electrochromic layer is an *organic polymer* — outside MatterGen's inorganic scope — so
it isn't a generation target here (would need an organic generative model).

## Run it

**On the A100** — install + generate:
```bash
pip install git+https://github.com/microsoft/mattergen.git   # + CUDA + checkpoints
python -m stealth.discovery.generate --role radar_conductor --n 64 --out runs/radar --run
```

**Locally** — screen the generated CIFs (or the built-in demo set):
```bash
python -m stealth.discovery.screen --role radar_conductor --cif-dir runs/radar
python -m stealth.discovery.screen --demo --role dielectric_spacer   # works now, no A100
```

The demo set (Al, MgO, TiO₂, VO₂) already proves the screen correctly routes metals →
`radar_conductor`, wide-gap oxides → `dielectric_spacer`, narrow-gap oxides → `ir_phasechange`.

## Honest boundary (next link)
Screening outputs ranked **novel candidate compositions per layer**. To simulate a *specific* novel
material's device performance in the Phase 2/3 physics, we still need its wavelength-resolved n,k —
predicted by an optical GNN (TSENN/GNNOpt style) or measured. That n,k-prediction bridge is the next
step; today the physics pipeline validates *stacks* (Phase 5/6), and discovery proposes the materials
that go in them. Band gaps here are GNN estimates (a relative proxy), not final values.
```
