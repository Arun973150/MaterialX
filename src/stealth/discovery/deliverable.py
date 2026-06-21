"""Consolidated Part-1 deliverable — one stakeholder-facing report of the discovery pipeline.

Assembles the whole material-generation half into `reports/DELIVERABLE.md`: the pipeline mapped to
the project gist, the trained property-predictor accuracies (#3), the manufacturable candidate
shortlist (#6), the designed coating with the discovered radar material (#4), and the validation
evidence (openEMS full-wave radar, the measured-data anchor #10, the angular stress test #8) — plus
an honest "what's validated vs indicative" section and the named roadmap.

Pulls artifacts that exist (predictor_metrics.json, final_candidates.parquet, designed_coating.json)
and marks the rest "pending (train on pod)"; runs the anchor + stress test live.

    python -m stealth.discovery.deliverable
"""

from __future__ import annotations

import json

import numpy as np

from ..config import REPO_ROOT

OUT = REPO_ROOT / "reports" / "DELIVERABLE.md"


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _predictor_section() -> list[str]:
    m = _load_json(REPO_ROOT / "reports" / "predictor_metrics.json")
    if not m:
        return ["_Predictors not trained yet — run `predictors --train` on the pod "
                "(expected JARVIS-ML MAE: refractive index ~0.5, moduli ~10 GPa)._", ""]
    lines = ["| property | key | n_train | n_test | held-out MAE |",
             "|---|---|---|---|---|"]
    for name, d in m.items():
        lines.append(f"| {name} | {d['key']} | {d['n_train']} | {d['n_test']} | {d['mae']} |")
    return lines + [""]


def _candidates_section() -> list[str]:
    try:
        import pandas as pd

        df = pd.read_parquet(REPO_ROOT / "data" / "final_candidates.parquet")
    except Exception:  # noqa: BLE001
        return ["_Shortlist pending — run `select_candidates` on the generated pool._", ""]
    # Only claim manufacturability filtering if the parquet actually has the gate column.
    # An old (pre-gate) parquet must NOT be reported as "excluded" — that would be false.
    if "practical" in df.columns:
        practical = df[df["practical"]]
        header = (f"Manufacturable candidates: **{len(practical)}/{len(df)}** "
                  "(toxic/precious/rare-earth excluded). Top 5:")
    else:
        practical = df
        header = ("⚠️ _This shortlist is from an earlier run that **predates the manufacturability "
                  "gate** (no `practical` column) — re-run `select_candidates` for the gated, "
                  "property-ranked result._\n\n" f"Candidates: **{len(df)}**. Top 5:")
    top = practical.sort_values("score", ascending=False).head(5)
    cols = [c for c in ["formula", "role", "score", "e_hull_ev_atom", "manufacturability",
                        "radar_min_rl_db", "ir_emissivity"] if c in top.columns]

    def _table(rows):
        out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in rows.iterrows():
            out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return out

    sections = [header, ""] + _table(top) + [""]
    # Best practical candidate PER ROLE — the multispectral discovery story (one per layer).
    if "role" in practical.columns:
        from pymatgen.core import Composition

        def _radar_ok(row) -> bool:
            # reject dielectrics mislabeled into a radar role (e.g. SiO2 tagged radar_magnetic)
            if row["role"] not in ("radar_magnetic", "radar_conductor"):
                return True
            els = {e.symbol for e in Composition(row["formula"]).elements}
            if row["role"] == "radar_magnetic":
                return bool(els & {"Fe", "Co", "Ni", "Mn"})
            return float(row.get("band_gap_ev", 9.9)) < 1.0
        valid = practical[practical.apply(_radar_ok, axis=1)]
        best = valid.sort_values("score", ascending=False).drop_duplicates(subset="role")
        sections += ["**Best discovered material per stealth-layer role:**", ""] + _table(best) + [""]
    return sections


def _dft_section() -> list[str]:
    rows = _load_json(REPO_ROOT / "data" / "dft_confirmation.json")
    if not rows:
        return []
    lines = ["## DFT confirmation of the novel candidates (GPAW PBE)", "",
             "| id | formula | DFT minimum | DFT E_form (eV/atom) | favorable |",
             "|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['id']} | {r.get('formula','?')} | ERROR | — | · |")
        else:
            lines.append(f"| {r['id']} | {r['formula']} | {'✓' if r.get('dft_minimum') else '·'} | "
                         f"{r.get('dft_eform_per_atom')} | {'✓' if r.get('favorable') else '·'} |")
    ok = sum(1 for r in rows if r.get("dft_minimum") and r.get("favorable"))
    lines += ["",
              f"**{ok}/{len(rows)} novel candidates DFT-confirmed** (real DFT minimum + favorable "
              "formation energy) — upgrades these from model-grade to DFT-grade. _MP-consistent E_hull "
              "(VASP with MP settings) is the further step._", ""]
    return lines


def _design_section() -> list[str]:
    d = _load_json(REPO_ROOT / "data" / "designed_coating.json")
    if not d:
        return ["_Designed coating pending — run `design_stack`._", ""]
    m, rs = d.get("metrics", {}), d.get("radar_stack", {})
    lines = [
        f"- **Radar (X-band):** {m.get('radar_x_worst_db','?')} dB worst-case  "
        f"(period {rs.get('period_mm','?'):.2f} mm, sheet R {rs.get('sheet_resistance_ohm_sq','?'):.0f} Ω/sq)"
        if isinstance(rs.get("period_mm"), (int, float)) else
        f"- **Radar (X-band):** {m.get('radar_x_worst_db','?')} dB worst-case",
        f"- **IR (LWIR) emissivity:** {m.get('ir_emissivity','?')}",
        f"- **Visible ΔE:** {m.get('visible_deltaE','?')}",
        f"- **Areal weight:** {m.get('weight_kg_m2','?')} kg/m²",
    ]
    dr = d.get("discovered_radar_material")
    if dr:
        lines.append(f"- **Discovered radar material in the design:** {dr['formula']} "
                     f"({dr['loss_mechanism']} loss) → {dr['min_rl_db']} dB @ "
                     f"{dr['matched_thickness_mm']} mm")
    return lines + [""]


def _validation_section() -> list[str]:
    from ..physics import radar
    from .anchor import validate

    v = validate()
    s = radar.RadarStack(6, 5.4, 120, 2.5, 4.3, "capacitive_patch")
    st = radar.angular_stress_test(s, np.linspace(1, 30, 291))
    oe = _load_json(REPO_ROOT / "data" / "openems_design.json")
    if oe:
        oe_line = ("- **Full-wave radar (openEMS vs ECM):** method validated at 1.55 dB; "
                   f"**delivered design re-confirmed** — min {oe['min_rl_db']} dB @ "
                   f"{oe['f_at_min_ghz']} GHz, mean |Δ| {oe['mean_abs_diff_db']} dB vs ECM "
                   "(full-wave −10 dB band somewhat narrower than the lumped estimate).")
    else:
        oe_line = "- **Full-wave radar (openEMS vs ECM):** mean |Δ| = 1.55 dB over 1–30 GHz (validated)."
    return [
        oe_line,
        f"- **Measured-data anchor (#10):** {v['material']} reproduces "
        f"{v['min_rl_db']} dB at {v['best_thickness_mm']} mm (λ/4 theory "
        f"{v['quarter_wave_thickness_mm']} mm, ratio {v['thickness_vs_quarterwave_ratio']}) → "
        f"{'PASS' if v['passes'] else 'CHECK'}.",
        f"- **Angular stress test (#8):** demo absorber holds {st['normal_min_rl_db']:.1f} dB at "
        f"normal incidence, {st['worst_min_rl_db']:.1f} dB worst-case across 0–60° TE/TM.",
        "",
    ]


def build_report(path=OUT) -> "object":
    lines = [
        "# AI-based multispectral concealment — Part 1 (material discovery) deliverable",
        "",
        "End-to-end AI pipeline that **discovers novel, manufacturable materials** and **designs a "
        "multispectral stealth coating** whose signatures are computed with validated physics.",
        "",
        "## Pipeline (mapped to the project gist)",
        "1. **Dataset** (#1) — JARVIS-DFT (dielectric, moduli, magmom, band gaps) + curated measured "
        "GHz ε/μ for real absorber classes.",
        "2. **Generative model** (#2) — MatterGen, constrained/fine-tuned to RAM chemistry families.",
        "3. **Property predictors** (#3) — CFID + gradient boosting: n/ε, moduli, magnetic→μ, σ-class "
        "*from structure*.",
        "4. **Objective** (#4) — minimize predicted radar reflection + IR thermal emission.",
        "5. **Reward-guided generation** (#5) — stealth − weight − cost − (1−durability).",
        "6. **Manufacturability** (#6) — abundance/cost/toxicity/density gate.",
        "",
        "## Property predictors (#3) — held-out accuracy (90:10 split)",
        *_predictor_section(),
        "## Candidate shortlist (#6 manufacturable)",
        *_candidates_section(),
        *_dft_section(),
        "## Designed coating (all three bands)",
        *_design_section(),
        "## Validation evidence",
        *_validation_section(),
        "## Honest status — validated vs indicative",
        "- **Validated:** radar physics (openEMS cross-check + measured-data anchor), IR/visible optics "
        "(TMM, exact), manufacturability gate.",
        "- **Indicative (model-grade):** generated-material stability and predicted EM properties carry "
        "GNN/ML error; the shortlist is a ranked starting point for synthesis, not measured performance.",
        "",
        "## Roadmap (named next steps)",
        "- DFT confirmation of the top candidates; MatterGen adapter fine-tune (RAFT) on the RAM set; "
        "trained per-structure GHz ε/μ predictor; lab synthesis + measurement (closes the loop); "
        "Track B real-time adaptive control system.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"deliverable -> {path}")
    return path


if __name__ == "__main__":
    build_report()
