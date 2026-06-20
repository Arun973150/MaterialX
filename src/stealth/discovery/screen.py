"""Screen generated (or candidate) crystal structures with matgl GNNs — runs locally.

For each structure: predict formation energy (stability proxy) and band gap (the
conductor/dielectric proxy), classify which stack layer it best serves, and filter
against the per-role targets. The same function ingests MatterGen's generated CIFs
(from the A100) or a built-in demo set of real structures.

    python -m stealth.discovery.screen --demo
    python -m stealth.discovery.screen --role radar_conductor --cif-dir <mattergen_out>
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from .targets import TARGETS, get_target

OUTPUT = REPO_ROOT / "data" / "discovery_candidates.parquet"

_MODELS: dict = {}


def _models():
    """Lazy-load + cache the pretrained GNN predictors."""
    if not _MODELS:
        warnings.filterwarnings("ignore")
        import matgl

        _MODELS["eform"] = matgl.load_model("M3GNet-Eform-MP-2018.6.1")
        _MODELS["bandgap"] = matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")
    return _MODELS


def _band_gap(model, structure) -> float:
    """Band gap from the multi-fidelity MEGNet model.

    The model has 4 fidelity heads (PBE/GLLB-SC/HSE/SCAN). Some heads return ~0 for
    wide-gap insulators, and the model's failure mode is *under*-estimation, so we
    take the max across fidelities as the most physical estimate. Absolute values are
    approximate (a GNN proxy) — used only to separate conductors from dielectrics;
    final properties are confirmed by physics/DFT downstream.
    """
    import torch

    vals = []
    for fid in range(4):
        try:
            vals.append(float(model.predict_structure(structure, state_attr=torch.tensor([fid]))))
        except TypeError:
            vals.append(float(model.predict_structure(structure, state_feats=torch.tensor([fid]))))
    return max(vals)


def predict(structures: list[tuple[str, "object"]]) -> pd.DataFrame:
    """structures: list of (id, pymatgen Structure) -> properties DataFrame."""
    m = _models()
    rows = []
    for sid, s in structures:
        rows.append(
            {
                "id": sid,
                "formula": s.composition.reduced_formula,
                "n_sites": len(s),
                "eform_per_atom": float(m["eform"].predict_structure(s)),
                "band_gap_ev": max(0.0, _band_gap(m["bandgap"], s)),
            }
        )
    return pd.DataFrame(rows)


def _chem_match(els: set[str], t) -> float:
    """Fraction of the composition's elements that belong to the role's chemical family."""
    if not t.chemical_system:
        return 1.0
    allowed = set(t.chemical_system)
    return sum(1 for e in els if e in allowed) / max(1, len(els))


def _role_fit(bg: float, eform: float, els: set[str], t) -> float:
    """0..1 fitness for a role: band gap in range + stability margin + chemistry match."""
    lo, hi = t.band_gap_ev
    in_gap = 1.0 if lo <= bg <= hi else max(0.0, 1.0 - min(abs(bg - lo), abs(bg - hi)))
    stable = 1.0 if eform <= t.max_eform_per_atom else max(0.0, 1.0 + (t.max_eform_per_atom - eform))
    chem = _chem_match(els, t)
    return float(0.45 * in_gap + 0.3 * stable + 0.25 * chem)


def screen(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Filter + rank candidates for one layer role."""
    from pymatgen.core import Composition

    t = get_target(role)
    out = df.copy()
    out["target_role"] = role
    els = [set(Composition(f).get_el_amt_dict()) for f in out["formula"]]
    out["fit_score"] = [_role_fit(bg, ef, e, t)
                        for bg, ef, e in zip(out["band_gap_ev"], out["eform_per_atom"], els)]
    out["gap_ok"] = out["band_gap_ev"].between(*t.band_gap_ev)
    out["stable_ok"] = out["eform_per_atom"] <= t.max_eform_per_atom
    out["passes"] = out["gap_ok"] & out["stable_ok"]
    # best-fitting role across all targets (the classification view)
    out["best_role"] = [
        max(TARGETS, key=lambda r: _role_fit(bg, ef, e, TARGETS[r]))
        for bg, ef, e in zip(out["band_gap_ev"], out["eform_per_atom"], els)
    ]
    return out.sort_values(["passes", "fit_score"], ascending=[False, False]).reset_index(drop=True)


def demo_structures() -> list[tuple[str, "object"]]:
    """A few real structures spanning the roles (proves screening end-to-end now)."""
    from pymatgen.core import Lattice, Structure

    sg = Structure.from_spacegroup
    return [
        ("demo-Al", sg(225, Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])),                       # metal/conductor
        ("demo-MgO", sg(225, Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])),  # wide-gap dielectric
        ("demo-TiO2", sg(136, Lattice.tetragonal(4.59, 2.96), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]])),
        ("demo-VO2", sg(136, Lattice.tetragonal(4.55, 2.85), ["V", "O"], [[0, 0, 0], [0.30, 0.30, 0]])),  # phase-change
    ]


def load_cifs(cif_dir: str | Path) -> list[tuple[str, "object"]]:
    from pymatgen.core import Structure

    cif_dir = Path(cif_dir)
    out = []
    for f in sorted(cif_dir.glob("*.cif")):
        try:
            out.append((f.stem, Structure.from_file(f)))
        except Exception as exc:  # noqa: BLE001 - skip unreadable generated files
            print(f"  skip {f.name}: {exc}")
    return out


def main(role: str = "radar_conductor", cif_dir: str | None = None) -> None:
    structures = load_cifs(cif_dir) if cif_dir else demo_structures()
    print(f"Screening {len(structures)} structures for role '{role}'...")
    df = predict(structures)
    ranked = screen(df, role)

    cols = ["id", "formula", "band_gap_ev", "eform_per_atom", "best_role", "fit_score", "passes"]
    pd.set_option("display.width", 160)
    print(ranked[cols].to_string(index=False))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(OUTPUT, index=False)
    n_pass = int(ranked["passes"].sum())
    print(f"\n{n_pass}/{len(ranked)} pass the '{role}' target -> {OUTPUT}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="radar_conductor", choices=list(TARGETS))
    ap.add_argument("--cif-dir", default=None, help="dir of MatterGen-generated CIFs (default: demo set)")
    ap.add_argument("--demo", action="store_true", help="use the built-in demo structures")
    args = ap.parse_args()
    main(role=args.role, cif_dir=None if args.demo else args.cif_dir)
