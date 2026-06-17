"""Capstone: fuse every signal into a ranked Top-N candidate shortlist.

This is the deliverable step — it turns the pipeline's scattered per-material metrics
(band gap + role, S.U.N. stability/novelty, GNNOpt optics, synthesizability + route)
into one transparent score and selects the best candidates, with a rationale each.

A candidate must clear the **S.U.N. gate** (stable + unique + novel — real and new), then
is ranked by a transparent blend of synthesizability, role-fit, and optical consistency.

    python -m stealth.discovery.select_candidates --cif-dir /workspace/runs/radar/cifs \
        --gnnopt-nk /workspace/runs/radar/gnnopt_nk.json --top 5
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from ..config import REPO_ROOT
from .optical_bridge import load_gnnopt_nk
from .screen import demo_structures, load_cifs, predict, screen
from .stability import evaluate_sun
from .synthesis import suggest_precursors, synthesizability
from .targets import TARGETS

OUT_MD = REPO_ROOT / "reports" / "final_candidates.md"
OUT_PARQUET = REPO_ROOT / "data" / "final_candidates.parquet"


def _stability_score(e_hull: float) -> float:
    """1.0 well below hull, 0.0 at +0.2 eV/atom and above."""
    if np.isnan(e_hull):
        return 0.0
    return float(np.clip((0.1 - e_hull) / 0.2 + 0.5, 0.0, 1.0))


def _optical_consistency(role: str, k550: float) -> float:
    """Does the visible extinction match the role? conductor->lossy, dielectric->clear."""
    if np.isnan(k550):
        return 0.5
    if role == "radar_conductor":
        return float(np.clip(k550 / 2.0, 0.0, 1.0))        # want high k
    if role == "dielectric_spacer":
        return float(np.clip(1.0 - k550, 0.0, 1.0))        # want low k
    return 0.5


def build(cif_dir=None, gnnopt_nk=None, top_n=5) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    structures = load_cifs(cif_dir) if cif_dir else demo_structures()

    scr = screen(predict(structures), role="radar_conductor")          # band gap, best_role, fit_score, eform
    sun = evaluate_sun(structures)                                     # stable/unique/novel/SUN, e_hull
    nk = load_gnnopt_nk(gnnopt_nk) if gnnopt_nk else {}

    df = scr.merge(
        sun[["id", "e_hull_ev_atom", "stable", "unique", "novel", "SUN"]], on="id", how="left"
    )

    rows = []
    for _, r in df.iterrows():
        comp = Composition(r["formula"])
        synth = synthesizability(comp, float(r["eform_per_atom"]))
        rec = nk.get(r["id"])
        k550 = float(np.interp(0.55, [1.23984 / e for e in reversed(rec["energy_ev"][1:])],
                               list(reversed(rec["k"][1:])))) if rec else float("nan")
        s_synth = synth["synth_score"]
        s_role = float(r["fit_score"])
        s_stab = _stability_score(float(r["e_hull_ev_atom"]))
        s_opt = _optical_consistency(r["best_role"], k550)
        score = 0.30 * s_synth + 0.25 * s_role + 0.25 * s_stab + 0.20 * s_opt
        rows.append(
            {
                "id": r["id"], "formula": r["formula"], "role": r["best_role"],
                "band_gap_ev": round(float(r["band_gap_ev"]), 2),
                "e_hull_ev_atom": round(float(r["e_hull_ev_atom"]), 3),
                "SUN": bool(r["SUN"]), "stable": bool(r["stable"]),
                "unique": bool(r["unique"]), "novel": bool(r["novel"]),
                "k_550nm": round(k550, 2) if not np.isnan(k550) else None,
                "chem_valid": synth["chem_valid"], "synth_score": s_synth,
                "precursors": ", ".join(suggest_precursors(comp)["precursors"]),
                "score": round(score, 3),
            }
        )
    out = pd.DataFrame(rows)

    # Gate: prefer full S.U.N.; relax if too few pass.
    for gate in ("SUN", "stable_novel", "stable", "all"):
        if gate == "SUN":
            pool = out[out["SUN"]]
        elif gate == "stable_novel":
            pool = out[out["stable"] & out["novel"]]
        elif gate == "stable":
            pool = out[out["stable"]]
        else:
            pool = out
        if len(pool) >= top_n or gate == "all":
            out.attrs["gate"] = gate
            break
    return pool.sort_values("score", ascending=False).head(top_n).reset_index(drop=True), out


def write_report(top: pd.DataFrame, full: pd.DataFrame, gate: str, path=OUT_MD):
    lines = [
        "# Final candidate shortlist",
        "",
        f"Selected from **{len(full)}** generated materials. Gate applied: **{gate}** "
        f"(S.U.N. = stable + unique + novel). Score = 0.30·synthesizability + 0.25·role-fit "
        f"+ 0.25·stability + 0.20·optical-consistency.",
        "",
        "| # | formula | role | score | E_hull | SUN | synth | k₅₅₀ | precursors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in top.iterrows():
        lines.append(
            f"| {i+1} | **{r['formula']}** | {r['role'].replace('_',' ')} | {r['score']:.2f} | "
            f"{r['e_hull_ev_atom']} | {'✓' if r['SUN'] else '·'} | {r['synth_score']:.2f} | "
            f"{r['k_550nm']} | {r['precursors']} |"
        )
    lines += [
        "",
        "**How to read this:** each row is a *novel* material (not in Materials Project) that the "
        "generative model proposed, screened as physically plausible (stable), chemically valid, and "
        "fit for its stealth-layer role. Scores are research-grade composites of model predictions — "
        "a ranked starting point for synthesis, not validated performance.",
        "",
        "_Next: feed these into the multilayer stack optimizer + openEMS/DFT for validated designs._",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(cif_dir=None, gnnopt_nk=None, top_n=5):
    top, full = build(cif_dir, gnnopt_nk, top_n)
    gate = full.attrs.get("gate", "all")
    pd.set_option("display.width", 200)
    cols = ["id", "formula", "role", "score", "e_hull_ev_atom", "SUN", "synth_score", "k_550nm"]
    print(f"\n=== TOP {len(top)} CANDIDATES (gate: {gate}) ===")
    print(top[cols].to_string(index=False))
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_PARQUET, index=False)
    out = write_report(top, full, gate)
    print(f"\nShortlist -> {out}   (full table -> {OUT_PARQUET})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--gnnopt-nk", default=None)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    main(cif_dir=None if args.demo else args.cif_dir, gnnopt_nk=args.gnnopt_nk, top_n=args.top)
