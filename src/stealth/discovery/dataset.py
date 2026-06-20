"""#1 dataset (Tier A): DFT-grade EM / dielectric / mechanical / magnetic labels from structure.

Pulls the JARVIS-DFT `dft_3d` database (~75k DFT entries) and extracts, per structure, the
labels the property predictors (#3) and the durability objective (#5) are trained on:

  * dielectric constant (OptB88vdW + MBJ static eps)  -> dielectric / dielectric-loss predictor
  * DFPT dielectric (incl. ionic lattice response)    -> IR / thermal dielectric response
  * band gaps (OptB88vdW, MBJ)                         -> DFT-grade conductor/dielectric label
  * bulk + shear modulus                              -> durability objective (#5)
  * total magnetic moment                             -> permeability / magnetic-loss features (#3 mu)
  * formation energy per atom                         -> stability cross-check

Frequency-dependent optical n,k spectra stay with GNNOpt (already validated vis/NIR); this
supplies the scalar/tensor DFT labels the rest of #3 needs. The companion Tier-A2 table
(`em_literature.py`) supplies measured GHz epsilon(f)/mu(f), which DFT does not give directly.

Install (on the pod):  pip install jarvis-tools
Run:
    python -m stealth.discovery.dataset --out data/em_dataset.parquet              # full (~75k)
    python -m stealth.discovery.dataset --out data/em_dataset_small.parquet --max 3000
"""

from __future__ import annotations

import json
import math
import warnings

from ..config import REPO_ROOT

OUTPUT = REPO_ROOT / "data" / "em_dataset.parquet"

# JARVIS key aliases (the DB has shifted names across releases; try each in order).
_KEYS = {
    "eform_per_atom": ["formation_energy_peratom"],
    "bandgap_optb88": ["optb88vdw_bandgap"],
    "bandgap_mbj": ["mbj_bandgap"],
    "dfpt_dielectric": ["dfpt_piezo_max_dielectric"],
    "bulk_modulus": ["bulk_modulus_kv"],
    "shear_modulus": ["shear_modulus_gv"],
    "magmom": ["magmom_outcar", "magmom_oszicar"],
}


def _num(v) -> float | None:
    """JARVIS encodes missing values as 'na'/None/'' — return a clean float or None."""
    try:
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in ("na", "", "none"):
            return None
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _first(entry: dict, names: list[str]) -> float | None:
    for n in names:
        v = _num(entry.get(n))
        if v is not None:
            return v
    return None


def _mean(*vals) -> float | None:
    xs = [x for x in (_num(v) for v in vals) if x is not None]
    return sum(xs) / len(xs) if xs else None


def to_structure(atoms_json: str):
    """Rebuild a pymatgen Structure from the stored JARVIS atoms dict."""
    from jarvis.core.atoms import Atoms

    return Atoms.from_dict(json.loads(atoms_json)).pymatgen_converter()


def build_dataset(max_n: int | None = None, out=OUTPUT):
    import pandas as pd
    from jarvis.db.figshare import data as jdata

    warnings.filterwarnings("ignore")
    print("Downloading JARVIS-DFT dft_3d (first run fetches a few hundred MB)...")
    db = jdata("dft_3d")
    if max_n:
        db = db[:max_n]
    print(f"  {len(db)} entries; extracting labels...")

    rows = []
    for e in db:
        atoms = e.get("atoms")
        if not atoms:
            continue
        rows.append(
            {
                "jid": e.get("jid"),
                "formula": e.get("formula"),
                "atoms": json.dumps(atoms),  # serialized structure (rebuild via to_structure)
                "eform_per_atom": _first(e, _KEYS["eform_per_atom"]),
                "bandgap_optb88": _first(e, _KEYS["bandgap_optb88"]),
                "bandgap_mbj": _first(e, _KEYS["bandgap_mbj"]),
                "eps_optb88": _mean(e.get("epsx"), e.get("epsy"), e.get("epsz")),
                "eps_mbj": _mean(e.get("mepsx"), e.get("mepsy"), e.get("mepsz")),
                "dfpt_dielectric": _first(e, _KEYS["dfpt_dielectric"]),
                "bulk_modulus_kv": _first(e, _KEYS["bulk_modulus"]),
                "shear_modulus_gv": _first(e, _KEYS["shear_modulus"]),
                "magmom": _first(e, _KEYS["magmom"]),
            }
        )

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    n = len(df)
    print(f"\nSaved {n} structures -> {out}\nLabel coverage (non-null):")
    for col in ["eps_optb88", "eps_mbj", "dfpt_dielectric", "bulk_modulus_kv",
                "shear_modulus_gv", "magmom", "bandgap_optb88", "bandgap_mbj"]:
        c = int(df[col].notna().sum())
        print(f"  {col:18}: {c:6d}  ({100*c/max(n,1):4.1f}%)")
    return df


def main(max_n: int | None = None, out=OUTPUT):
    build_dataset(max_n=max_n, out=out)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--max", type=int, default=None, help="limit entries (for a quick test run)")
    args = ap.parse_args()
    main(max_n=args.max, out=Path(args.out))
