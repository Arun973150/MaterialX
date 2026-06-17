"""Synthesizability scoring + precursor/route suggestion.

Closes the gist's deliverable: output **chemical formulas AND manufacturing pathways**
for the most promising candidates. For each candidate we compute:

  * chemical validity  — SMACT (charge-neutral + electronegativity-ordered), the
                         standard filter for generative-materials outputs; pymatgen
                         oxidation-state fallback if SMACT is unavailable.
  * stability          — from the GNN-predicted formation energy (Phase A screening).
  * a synthesizability score combining validity + stability + compositional simplicity.
  * a first-order synthesis route — common solid-state precursors per element + a
                         method heuristic. (A literature-mined precursor model — e.g.
                         the text-mined-recipe recommenders — would refine this.)
"""

from __future__ import annotations

from pathlib import Path

from pymatgen.core import Composition

from ..config import REPO_ROOT

DOSSIER_OUT = REPO_ROOT / "reports" / "discovery_dossier.md"

# Common solid-state precursors per element (oxides / carbonates) — first-order.
_PRECURSOR = {
    "Li": "Li2CO3", "Na": "Na2CO3", "K": "K2CO3", "Mg": "MgO", "Ca": "CaCO3",
    "Al": "Al2O3", "Si": "SiO2", "Ti": "TiO2", "V": "V2O5", "Cr": "Cr2O3",
    "Mn": "MnO2", "Fe": "Fe2O3", "Co": "Co3O4", "Ni": "NiO", "Cu": "CuO",
    "Zn": "ZnO", "Ga": "Ga2O3", "Ge": "GeO2", "Y": "Y2O3", "Zr": "ZrO2",
    "Nb": "Nb2O5", "Mo": "MoO3", "In": "In2O3", "Sn": "SnO2", "Sb": "Sb2O3",
    "Ba": "BaCO3", "La": "La2O3", "W": "WO3", "Ta": "Ta2O5", "Hf": "HfO2",
}
_ANION_SOURCE = {"O": "(oxygen from oxide precursors / air)", "N": "N2 / NH3 atmosphere", "C": "C (graphite)"}


def chemical_validity(comp: Composition) -> bool:
    """SMACT validity (charge-neutral + electronegativity), pymatgen fallback."""
    try:
        from smact.screening import smact_validity

        return bool(smact_validity(comp))
    except Exception:  # noqa: BLE001 - SMACT optional; fall back to pymatgen
        try:
            return len(comp.oxi_state_guesses()) > 0
        except Exception:  # noqa: BLE001
            return False


def _stability_score(eform_per_atom: float) -> float:
    """More-negative formation energy -> more stable -> higher score (0..1)."""
    return float(min(1.0, max(0.0, -eform_per_atom / 2.0)))


def synthesizability(comp: Composition, eform_per_atom: float) -> dict:
    valid = chemical_validity(comp)
    stab = _stability_score(eform_per_atom)
    n_el = len(comp.elements)
    simplicity = {1: 1.0, 2: 1.0, 3: 0.8, 4: 0.6}.get(n_el, 0.4)
    score = 0.3 * float(valid) + 0.5 * stab + 0.2 * simplicity
    return {
        "chem_valid": valid,
        "stability_score": round(stab, 3),
        "n_elements": n_el,
        "synth_score": round(score, 3),
    }


def suggest_precursors(comp: Composition) -> dict:
    """First-order synthesis route: precursors per element + a method heuristic."""
    els = [e.symbol for e in comp.elements]
    precursors, notes = [], []
    for el in els:
        if el in _PRECURSOR:
            precursors.append(_PRECURSOR[el])
        elif el in _ANION_SOURCE:
            notes.append(_ANION_SOURCE[el])
        else:
            precursors.append(f"{el}-oxide")
    if "C" in els:
        method = "carbothermal reduction (high-temp, inert/reducing atmosphere)"
    elif "O" in els:
        method = "solid-state calcination of mixed oxide precursors (or sol-gel)"
    elif "N" in els:
        method = "nitridation under N2/NH3"
    else:
        method = "arc-melting / solid-state reaction of elemental precursors"
    return {
        "precursors": sorted(set(precursors)),
        "method": method,
        "atmosphere_notes": "; ".join(sorted(set(notes))) or "-",
        "route_confidence": "heuristic (first-order; literature-mined model would refine)",
    }


def dossier(cid: str, comp: Composition, eform_per_atom: float) -> dict:
    """Full per-candidate record: formula + synthesizability + synthesis route."""
    synth = synthesizability(comp, eform_per_atom)
    route = suggest_precursors(comp)
    return {
        "id": cid,
        "formula": comp.reduced_formula,
        "eform_per_atom": round(float(eform_per_atom), 3),
        **synth,
        "precursors": ", ".join(route["precursors"]),
        "method": route["method"],
        "route_confidence": route["route_confidence"],
    }


def build_dossier(cif_dir: str | None = None):
    """Screen structures, then attach synthesizability + synthesis route to each."""
    import warnings

    import pandas as pd

    warnings.filterwarnings("ignore")
    from .screen import demo_structures, load_cifs, predict

    structures = load_cifs(cif_dir) if cif_dir else demo_structures()
    eform = dict(zip(*[predict(structures)[c] for c in ("id", "eform_per_atom")]))
    rows = [dossier(sid, s.composition, eform[sid]) for sid, s in structures]
    return pd.DataFrame(rows).sort_values("synth_score", ascending=False).reset_index(drop=True)


def write_dossier_report(ddf, path: Path = DOSSIER_OUT) -> Path:
    lines = [
        "# Discovery dossier — chemical formulas + manufacturing pathways",
        "",
        "Each candidate from generative discovery, with a synthesizability score (SMACT validity +",
        "GNN stability + compositional simplicity) and a first-order solid-state synthesis route.",
        "",
        "| # | formula | synth score | chem valid | stability | precursors | method |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in ddf.iterrows():
        lines.append(
            f"| {i+1} | {r['formula']} | {r['synth_score']:.2f} | "
            f"{'yes' if r['chem_valid'] else 'no'} | {r['stability_score']:.2f} | "
            f"{r['precursors']} | {r['method']} |"
        )
    lines += [
        "",
        "_Routes are heuristic (first-order). A literature-mined precursor recommender (e.g. "
        "text-mined-recipe models) would refine precursor choice, temperatures and times._",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(cif_dir: str | None = None) -> None:
    import pandas as pd

    ddf = build_dossier(cif_dir)
    pd.set_option("display.width", 180)
    cols = ["id", "formula", "synth_score", "chem_valid", "precursors", "method"]
    print(ddf[cols].to_string(index=False))
    out = write_dossier_report(ddf)
    print(f"\nDossier ({len(ddf)} candidates) -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", default=None, help="dir of MatterGen CIFs (default: demo structures)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    main(cif_dir=None if args.demo else args.cif_dir)
