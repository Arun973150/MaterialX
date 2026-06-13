# Candidate shortlist — multispectral stealth stack (Phases 1-6)

From a 48-design NSGA-III Pareto front. Targets: radar worst ≤ -10.0 dB, LWIR emissivity < 0.3, visible ΔE < 5.0, weight ≤ 10.0 kg/m². Radar via fast ECM (re-confirm top picks with openEMS full-wave on the cluster).

- Meeting **X-band radar + IR**: **21** designs
- Meeting **full X+Ku radar + IR**: **0** (single-layer is narrowband — full band needs a multilayer absorber)
- Meeting **all three bands**: **0** (visible limited by the open PEDOT:PSS/VO₂ palette — needs ProDOT optical data)

## Top candidates (high-resolution confirmed)

### Candidate 1

**Stack (top → bottom):**
- Electrochromic PEDOT:PSS — 0.354 µm (visible)
- VO₂ — 0.292 µm, operate in **metallic** state (IR)
- Radar metasurface — capacitive patch, period 5.24 mm, patch 4.72 mm (fill 0.90), sheet R 268 Ω/sq
- Grounded SiO₂ spacer — 3.20 mm + Al ground

**Performance:** radar worst (X) -12.3 dB · radar worst (X+Ku) -2.8 dB · LWIR emissivity 0.234 · visible ΔE 22.2 · weight 7.0 kg/m²
**Targets:** ✅ radar X-band · ✅ IR · ⚠ visible

### Candidate 2

**Stack (top → bottom):**
- Electrochromic PEDOT:PSS — 0.353 µm (visible)
- VO₂ — 0.291 µm, operate in **metallic** state (IR)
- Radar metasurface — capacitive patch, period 6.15 mm, patch 5.17 mm (fill 0.84), sheet R 268 Ω/sq
- Grounded SiO₂ spacer — 3.23 mm + Al ground

**Performance:** radar worst (X) -11.5 dB · radar worst (X+Ku) -2.5 dB · LWIR emissivity 0.234 · visible ΔE 22.2 · weight 7.1 kg/m²
**Targets:** ✅ radar X-band · ✅ IR · ⚠ visible

### Candidate 3

**Stack (top → bottom):**
- Electrochromic PEDOT:PSS — 0.353 µm (visible)
- VO₂ — 0.289 µm, operate in **metallic** state (IR)
- Radar metasurface — capacitive patch, period 5.55 mm, patch 4.68 mm (fill 0.84), sheet R 283 Ω/sq
- Grounded SiO₂ spacer — 3.35 mm + Al ground

**Performance:** radar worst (X) -10.3 dB · radar worst (X+Ku) -1.9 dB · LWIR emissivity 0.235 · visible ΔE 22.2 · weight 7.4 kg/m²
**Targets:** ✅ radar X-band · ✅ IR · ⚠ visible

### Candidate 4

**Stack (top → bottom):**
- Electrochromic PEDOT:PSS — 0.353 µm (visible)
- VO₂ — 0.201 µm, operate in **metallic** state (IR)
- Radar metasurface — capacitive patch, period 4.50 mm, patch 4.13 mm (fill 0.92), sheet R 267 Ω/sq
- Grounded SiO₂ spacer — 3.19 mm + Al ground

**Performance:** radar worst (X) -12.3 dB · radar worst (X+Ku) -2.8 dB · LWIR emissivity 0.277 · visible ΔE 14.9 · weight 7.0 kg/m²
**Targets:** ✅ radar X-band · ✅ IR · ⚠ visible

### Candidate 5

**Stack (top → bottom):**
- Electrochromic PEDOT:PSS — 0.353 µm (visible)
- VO₂ — 0.335 µm, operate in **metallic** state (IR)
- Radar metasurface — capacitive patch, period 6.57 mm, patch 5.54 mm (fill 0.84), sheet R 261 Ω/sq
- Grounded SiO₂ spacer — 3.17 mm + Al ground

**Performance:** radar worst (X) -12.3 dB · radar worst (X+Ku) -2.9 dB · LWIR emissivity 0.231 · visible ΔE 25.2 · weight 7.0 kg/m²
**Targets:** ✅ radar X-band · ✅ IR · ⚠ visible

### Candidate 6

**Stack (top → bottom):**
- Electrochromic PEDOT:PSS — 0.353 µm (visible)
- VO₂ — 0.205 µm, operate in **metallic** state (IR)
- Radar metasurface — capacitive patch, period 7.75 mm, patch 5.85 mm (fill 0.75), sheet R 246 Ω/sq
- Grounded SiO₂ spacer — 3.07 mm + Al ground

**Performance:** radar worst (X) -12.0 dB · radar worst (X+Ku) -3.2 dB · LWIR emissivity 0.274 · visible ΔE 15.4 · weight 6.8 kg/m²
**Targets:** ✅ radar X-band · ✅ IR · ⚠ visible

## Validation status

- High-resolution re-check: max metric drift vs optimizer grid = **0.000** (small → no coarse-sampling artifact).
- Optics (IR + visible): exact TMM — already high-fidelity.
- Radar: ECM (fast/approximate). **Pending:** openEMS full-wave confirmation on the cluster before fabrication trust.

## Honest gaps & next steps
- **Full X+Ku radar** → multilayer (Jaumann) absorber (more design dimensions).
- **Visible ΔE** → tunable ProDOT electrochromic polymer (needs optical data not in open DBs).
- **Radar fidelity** → run `physics/radar_fullwave.py` (openEMS) on the cluster.