"""GAP #3: uncertainty quantification for the discovery predictors.

Every prediction the pipeline makes is otherwise a bare point estimate. Here we attach
a confidence to each material so the ranking can flag shaky predictions:

  * band gap  -> spread across the MEGNet model's 4 fidelity heads (PBE/GLLB-SC/HSE/SCAN)
  * formation energy -> disagreement between two independent models (M3GNet vs MEGNet)

Large spread / model disagreement = low confidence. Cheap (reuses models already loaded).

    python -m stealth.discovery.uncertainty --demo
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from .screen import demo_structures, load_cifs

OUTPUT = REPO_ROOT / "data" / "discovery_uncertainty.parquet"

_M = {}


def _models():
    if not _M:
        warnings.filterwarnings("ignore")
        import matgl

        _M["bg"] = matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")
        _M["ef_m3"] = matgl.load_model("M3GNet-Eform-MP-2018.6.1")
        _M["ef_meg"] = matgl.load_model("MEGNet-Eform-MP-2018.6.1")
    return _M


def band_gap_uncertainty(structure):
    """(mean, std, max) of band gap across the 4 multi-fidelity heads."""
    import torch

    m = _models()["bg"]
    vals = []
    for fid in range(4):
        try:
            vals.append(float(m.predict_structure(structure, state_attr=torch.tensor([fid]))))
        except TypeError:
            vals.append(float(m.predict_structure(structure, state_feats=torch.tensor([fid]))))
    vals = np.clip(vals, 0.0, None)
    return float(np.mean(vals)), float(np.std(vals)), float(np.max(vals))


def eform_uncertainty(structure):
    """(mean, |model disagreement|) of formation energy from two independent GNNs."""
    m = _models()
    e1 = float(m["ef_m3"].predict_structure(structure))
    e2 = float(m["ef_meg"].predict_structure(structure))
    return float(np.mean([e1, e2])), float(abs(e1 - e2))


def _confidence_level(bg_std: float, ef_spread: float) -> str:
    """High when both predictors agree tightly; low when either spreads widely."""
    if bg_std < 0.4 and ef_spread < 0.15:
        return "high"
    if bg_std < 1.0 and ef_spread < 0.40:
        return "medium"
    return "low"


def confidence(structure) -> dict:
    bg_mean, bg_std, bg_max = band_gap_uncertainty(structure)
    ef_mean, ef_spread = eform_uncertainty(structure)
    level = _confidence_level(bg_std, ef_spread)
    return {
        "band_gap_mean": round(bg_mean, 3),
        "band_gap_std": round(bg_std, 3),
        "eform_mean": round(ef_mean, 3),
        "eform_model_spread": round(ef_spread, 3),
        "confidence": level,
    }


def evaluate(structures) -> pd.DataFrame:
    from tqdm import tqdm

    rows = []
    for sid, s in tqdm(structures, desc="uncertainty", unit="mat"):
        rows.append({"id": sid, "formula": s.composition.reduced_formula, **confidence(s)})
    return pd.DataFrame(rows)


def main(cif_dir: str | None = None) -> None:
    structures = load_cifs(cif_dir) if cif_dir else demo_structures()
    df = evaluate(structures)
    print(df.to_string(index=False))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    counts = df["confidence"].value_counts().to_dict()
    print(f"\nConfidence: {counts}  -> {OUTPUT}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    main(cif_dir=None if args.demo else args.cif_dir)
