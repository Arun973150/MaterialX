# Candidate shortlist — Phase 5 (ECM radar + TMM optics, NSGA-III)

Pareto front: 48 designs. Targets: radar worst <= -10.0 dB, IR emissivity < 0.3, visible deltaE < 5.0, weight <= 10.0 kg/m^2.

Designs meeting **all** targets (full X+Ku): **0**.
Designs meeting radar **X-band** + IR + visible: **0**.
Designs meeting radar **X-band** + IR (color aside): **21**.

Top 5 by targets-met then balance:

| # | radar worst X+Ku (dB) | radar worst X (dB) | IR emiss (state) | visible dE | weight (kg/m²) | bands met |
|---|---|---|---|---|---|---|
| 1 | -2.8 | -12.3 | 0.234 (met) | 22.2 | 7.0 | 2/3 |
| 2 | -2.5 | -11.5 | 0.234 (met) | 22.2 | 7.1 | 2/3 |
| 3 | -1.9 | -10.3 | 0.235 (met) | 22.2 | 7.4 | 2/3 |
| 4 | -2.8 | -12.3 | 0.277 (met) | 14.9 | 7.0 | 2/3 |
| 5 | -2.9 | -12.3 | 0.231 (met) | 25.2 | 7.0 | 2/3 |

**Honest read of the gaps:**
- *Radar:* a single-layer absorber can hit −10 dB across the X-band but not the full 8–18 GHz; full X+Ku coverage needs a multilayer (Jaumann) absorber — a concrete next step.
- *Visible:* deltaE is bounded by the open PEDOT:PSS/VO₂ palette; the spec's tunable ProDOT polymer (no open optical data) would extend color reach.
- *Radar fidelity:* ECM is fast/approximate — top candidates must be re-confirmed with openEMS full-wave (Phase 6, cluster) before trust.
- *IR:* emissivity is the best achievable VO₂ state (the material switches), per its adaptive design.