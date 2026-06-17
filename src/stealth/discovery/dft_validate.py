"""GAP #2: validate the pipeline's predictors against DFT ground truth.

The whole discovery chain rests on ML predictions; this anchors them in DFT. We take a
sample of known Materials Project entries (which carry DFT-computed band gap + formation
energy), run the *same* predictors the pipeline uses (matgl M3GNet-Eform + MEGNet-BandGap),
and report the error vs DFT (MAE, R^2). That is the honest error budget on every candidate
score — "our formation energy is within X eV/atom of DFT, band gap within Y."

    python -m stealth.discovery.dft_validate --n 40

(The ultimate anchor — running Quantum ESPRESSO/VASP on a *novel* candidate — is a separate,
compute-heavy cluster step; this validates the predictors against DFT on known materials.)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from .screen import _band_gap, _models

OUTPUT = REPO_ROOT / "data" / "dft_validation.parquet"


def _mae(y, yp):
    return float(np.mean(np.abs(np.array(y) - np.array(yp))))


def _r2(y, yp):
    y, yp = np.array(y), np.array(yp)
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def validate(n: int = 40) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    from tqdm import tqdm

    from ..materials.sources import mp_client

    with mp_client() as mpr:
        docs = mpr.materials.summary.search(
            num_elements=(2, 3),
            is_stable=True,
            fields=["material_id", "formula_pretty", "band_gap", "formation_energy_per_atom", "structure"],
            num_chunks=1,
            chunk_size=n,
        )

    m = _models()
    rows = []
    for d in tqdm(docs, desc="predict vs DFT", unit="mat"):
        if d.structure is None or d.formation_energy_per_atom is None or d.band_gap is None:
            continue
        s = d.structure
        rows.append(
            {
                "material_id": str(d.material_id),
                "formula": d.formula_pretty,
                "eform_dft": float(d.formation_energy_per_atom),
                "eform_pred": float(m["eform"].predict_structure(s)),
                "band_gap_dft": float(d.band_gap),
                "band_gap_pred": float(_band_gap(m["bandgap"], s)),
            }
        )
    return pd.DataFrame(rows)


def main(n: int = 40) -> None:
    df = validate(n)
    if df.empty:
        print("No DFT entries returned.")
        return
    ef_mae, ef_r2 = _mae(df["eform_dft"], df["eform_pred"]), _r2(df["eform_dft"], df["eform_pred"])
    bg_mae, bg_r2 = _mae(df["band_gap_dft"], df["band_gap_pred"]), _r2(df["band_gap_dft"], df["band_gap_pred"])
    print(f"\nValidated {len(df)} predictors against Materials Project DFT:")
    print(f"  Formation energy:  MAE = {ef_mae:.3f} eV/atom   R^2 = {ef_r2:.2f}")
    print(f"  Band gap:          MAE = {bg_mae:.3f} eV        R^2 = {bg_r2:.2f}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"-> {OUTPUT}")
    print("This is the error budget on candidate scores: formation energy is the tight, trustworthy "
          "predictor; band gap is looser (and worst for metals) — weight decisions accordingly.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    main(n=args.n)
