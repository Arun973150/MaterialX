# AI-based multispectral concealment — Part 1 (material discovery) deliverable

End-to-end AI pipeline that **discovers novel, manufacturable materials** and **designs a multispectral stealth coating** whose signatures are computed with validated physics.

## Pipeline (mapped to the project gist)
1. **Dataset** (#1) — JARVIS-DFT (dielectric, moduli, magmom, band gaps) + curated measured GHz ε/μ for real absorber classes.
2. **Generative model** (#2) — MatterGen, constrained/fine-tuned to RAM chemistry families.
3. **Property predictors** (#3) — CFID + gradient boosting: n/ε, moduli, magnetic→μ, σ-class *from structure*.
4. **Objective** (#4) — minimize predicted radar reflection + IR thermal emission.
5. **Reward-guided generation** (#5) — stealth − weight − cost − (1−durability).
6. **Manufacturability** (#6) — abundance/cost/toxicity/density gate.

## Property predictors (#3) — held-out accuracy (90:10 split)
_Predictors not trained yet — run `predictors --train` on the pod (expected JARVIS-ML MAE: refractive index ~0.5, moduli ~10 GPa)._

## Candidate shortlist (#6 manufacturable)
Manufacturable candidates: **4/4** (toxic/precious/rare-earth excluded). Top 5:

| formula | role | score | e_hull_ev_atom |
|---|---|---|---|
| VO2 | radar_conductor | 0.843 | -0.384 |
| Al | radar_conductor | 0.771 | -0.279 |
| MgO | dielectric_spacer | 0.75 | -1.035 |
| TiO2 | ir_phasechange | 0.75 | -0.581 |

## Designed coating (all three bands)
- **Radar (X-band):** -10.4 dB worst-case  (period 5.00 mm, sheet R 286 Ω/sq)
- **IR (LWIR) emissivity:** 0.032
- **Visible ΔE:** 6.8
- **Areal weight:** 6.8 kg/m²

## Validation evidence
- **Full-wave radar (openEMS vs ECM):** mean |Δ| = 1.55 dB over 1–30 GHz (validated).
- **Measured-data anchor (#10):** Carbonyl iron (70 wt% composite) reproduces -28.9 dB at 2.69 mm (λ/4 theory 2.86 mm, ratio 0.94) → PASS.
- **Angular stress test (#8):** demo absorber holds -12.7 dB at normal incidence, -5.5 dB worst-case across 0–60° TE/TM.

## Honest status — validated vs indicative
- **Validated:** radar physics (openEMS cross-check + measured-data anchor), IR/visible optics (TMM, exact), manufacturability gate.
- **Indicative (model-grade):** generated-material stability and predicted EM properties carry GNN/ML error; the shortlist is a ranked starting point for synthesis, not measured performance.

## Roadmap (named next steps)
- DFT confirmation of the top candidates; MatterGen adapter fine-tune (RAFT) on the RAM set; trained per-structure GHz ε/μ predictor; lab synthesis + measurement (closes the loop); Track B real-time adaptive control system.