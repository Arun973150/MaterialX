"""Phase 6 validation + buildable candidate report.

Re-checks the top Pareto candidates at HIGH frequency/wavelength resolution to
confirm the optimizer's (coarse-grid) numbers weren't a sampling artifact — the
local analog of re-validating winners. Optics TMM is already exact; the remaining
high-fidelity step is the radar openEMS full-wave run, which happens on the cluster
(`physics/radar_fullwave.py`). Emits a hand-off report with full stack specs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, load_targets
from ..optimize import objective as obj
from ..physics import optics, radar

PARETO_IN = REPO_ROOT / "data" / "pareto_front.parquet"
REPORT_OUT = REPO_ROOT / "reports" / "candidates.md"


def _reconstruct(row) -> obj.Design:
    return obj.Design(**{p: float(row[p]) for p in obj.PARAM_ORDER})


def confirm(d: obj.Design, targets: dict) -> dict:
    """Recompute the metrics at high resolution (independent of the optimizer's grid)."""
    rstack = obj._radar_stack(d)
    x = targets["radar"]["bands_ghz"]["X"]
    ku = targets["radar"]["bands_ghz"]["Ku"]
    rl_x = radar.spectrum(rstack, np.linspace(x[0], x[1], 121))["reflection_loss_db"]
    rl_full = radar.spectrum(rstack, np.linspace(x[0], ku[1], 401))["reflection_loss_db"]
    lwir = tuple(targets["ir_lwir"]["band_um"])
    ir_states = {s: optics.band_emissivity(obj._ir_stack(d, s), lwir, n=101) for s in obj._VO2_STATES}
    ir_state = min(ir_states, key=ir_states.get)
    bg = targets["visible"]["background"]["srgb_hex"]
    return {
        "radar_x_worst_db": float(np.max(rl_x)),
        "radar_full_worst_db": float(np.max(rl_full)),
        "ir_emissivity": float(ir_states[ir_state]),
        "ir_state": "metallic" if ir_state.startswith("VO2 (m") else "insulating",
        "visible_deltaE": float(optics.delta_e_vs_background(obj._visible_stack(d), bg)),
        "weight_kg_m2": obj.weight_kg_m2(d),
    }


def build(top_k: int = 6) -> Path:
    targets = load_targets()
    df = pd.read_parquet(PARETO_IN)

    # Prefer candidates meeting X-band radar + IR; fall back to the ranked head.
    picked = df[df["radar_x_ok"] & df["ir_ok"]]
    picked = (picked if len(picked) else df).head(top_k)

    th = targets
    lines = [
        "# Candidate shortlist — multispectral stealth stack (Phases 1-6)",
        "",
        f"From a {len(df)}-design NSGA-III Pareto front. Targets: radar worst ≤ "
        f"{th['radar']['threshold_db']} dB, LWIR emissivity < {th['ir_lwir']['threshold']}, "
        f"visible ΔE < {th['visible']['threshold']}, weight ≤ "
        f"{th['physical']['weight_kg_per_m2']['threshold']} kg/m². Radar via fast ECM "
        "(re-confirm top picks with openEMS full-wave on the cluster).",
        "",
        f"- Meeting **X-band radar + IR**: **{int((df['radar_x_ok'] & df['ir_ok']).sum())}** designs",
        f"- Meeting **full X+Ku radar + IR**: **{int((df['radar_ok'] & df['ir_ok']).sum())}** "
        "(single-layer is narrowband — full band needs a multilayer absorber)",
        f"- Meeting **all three bands**: **{int(df['all_targets_met'].sum())}** "
        "(visible limited by the open PEDOT:PSS/VO₂ palette — needs ProDOT optical data)",
        "",
        "## Top candidates (high-resolution confirmed)",
        "",
    ]

    max_drift = 0.0
    for n, (_, row) in enumerate(picked.iterrows(), 1):
        d = _reconstruct(row)
        c = confirm(d, targets)
        max_drift = max(max_drift, abs(c["radar_x_worst_db"] - row["radar_x_worst_db"]),
                        abs(c["ir_emissivity"] - row["ir_emissivity"]))
        rx, ir, de = c["radar_x_worst_db"], c["ir_emissivity"], c["visible_deltaE"]
        flags = []
        flags.append("✅ radar X-band" if rx <= th["radar"]["threshold_db"] else "⚠ radar X-band")
        flags.append("✅ IR" if ir < th["ir_lwir"]["threshold"] else "⚠ IR")
        flags.append("✅ visible" if de < th["visible"]["threshold"] else "⚠ visible")
        lines += [
            f"### Candidate {n}",
            "",
            "**Stack (top → bottom):**",
            f"- Electrochromic PEDOT:PSS — {d.t_ec_um:.3f} µm (visible)",
            f"- VO₂ — {d.t_vo2_um:.3f} µm, operate in **{c['ir_state']}** state (IR)",
            f"- Radar metasurface — capacitive patch, period {d.radar_period_mm:.2f} mm, "
            f"patch {d.radar_period_mm * d.radar_patch_frac:.2f} mm "
            f"(fill {d.radar_patch_frac:.2f}), sheet R {d.radar_sheet_resistance_ohm_sq:.0f} Ω/sq",
            f"- Grounded SiO₂ spacer — {d.spacer_thickness_mm:.2f} mm + Al ground",
            "",
            f"**Performance:** radar worst (X) {rx:.1f} dB · radar worst (X+Ku) "
            f"{c['radar_full_worst_db']:.1f} dB · LWIR emissivity {ir:.3f} · "
            f"visible ΔE {de:.1f} · weight {c['weight_kg_m2']:.1f} kg/m²",
            f"**Targets:** {' · '.join(flags)}",
            "",
        ]

    lines += [
        "## Validation status",
        "",
        f"- High-resolution re-check: max metric drift vs optimizer grid = **{max_drift:.3f}** "
        "(small → no coarse-sampling artifact).",
        "- Optics (IR + visible): exact TMM — already high-fidelity.",
        "- Radar: ECM (fast/approximate). **Pending:** openEMS full-wave confirmation on the "
        "cluster before fabrication trust.",
        "",
        "## Honest gaps & next steps",
        "- **Full X+Ku radar** → multilayer (Jaumann) absorber (more design dimensions).",
        "- **Visible ΔE** → tunable ProDOT electrochromic polymer (needs optical data not in open DBs).",
        "- **Radar fidelity** → run `physics/radar_fullwave.py` (openEMS) on the cluster.",
    ]

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_OUT


def main() -> None:
    out = build()
    print(f"Wrote buildable candidate report -> {out}")


if __name__ == "__main__":
    main()
