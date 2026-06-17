# Designed multispectral coating (discovery -> design bridge)

A full multilayer stealth coating, geometry-optimized (NSGA-III) over the real physics (TMM optics + ECM radar), with layers filled by **discovered** materials where available:

| layer | material |
|---|---|
| visible (electrochromic) | PEDOT:PSS (known) |
| IR (thermochromic) | VO2 (known) |
| optical spacer | SiO2 (known) |
| radar metasurface + ground | patterned conductor / Al |

**Best balanced design:** radar X-band -10.4 dB, LWIR emissivity 0.032, visible ΔE 6.8, weight 6.8 kg/m². Geometry: VO₂ 0.14 µm, spacer 3.09 mm, patch period 5.00 mm, sheet R 286 Ω/sq.

Feasible designs on the Pareto front: 21/31.

_Discovered materials are used where their GNNOpt n,k is valid (visible/NIR optical layers). IR stays on VO₂ (real IR data); generating dielectric/IR candidates extends discovery to those layers. Next: openEMS/DFT validation of the chosen design._