"""#3 property predictors — EM / optical / mechanical / magnetic properties FROM STRUCTURE.

The missing scientific link: predict the intrinsic properties that decide stealth + durability
directly from a generated crystal, using the proven JARVIS-ML workflow — CFID descriptors
(1557 features/structure) + gradient-boosted trees, trained on JARVIS-DFT with a 90:10 split
(the gist's "9:1") and reported by held-out MAE.

Predicted properties and their JARVIS-ML benchmark accuracy:
  * refractive index n / dielectric eps   (MAE ~0.5)   -> optical + dielectric-loss response
  * bulk & shear modulus                  (MAE ~10 GPa) -> durability objective (#5)
  * magnetic moment / magnetic class      (AUC ~0.96)  -> permeability mu (gates the literature
                                                          mu prior in em_literature.py)
  * band gap (OptB88vdW)                  (MAE ~0.3)   -> conductor/dielectric + sigma class

For the radar band, optical conductivity sigma is taken from the metallic/small-gap class and
the magnetic permeability from the magnetic class x the measured-literature prior (Tier A2) —
DFT alone does not give GHz mu, so this is the honest, calibrated stand-in.

Train (on the pod, needs `pip install -e .[train]`):
    python -m stealth.discovery.predictors --train            # all models, prints held-out MAE
Predict:
    from stealth.discovery.predictors import predict_properties
    predict_properties(structure)  # pymatgen Structure -> dict of properties
"""

from __future__ import annotations

import json
import warnings

from ..config import REPO_ROOT
from .em_literature import class_mu_prior

MODEL_DIR = REPO_ROOT / "models" / "predictors"
METRICS = REPO_ROOT / "reports" / "predictor_metrics.json"

# property -> JARVIS get_ml_data key(s). Trained independently; each saved to MODEL_DIR.
# `key` may be a list of fallbacks (JARVIS key names vary across releases / property availability).
PROPS = {
    "bandgap": {"key": "optb88vdw_bandgap"},
    "eps": {"key": "epsx"},                                          # static dielectric (n = sqrt(eps))
    "bulk_modulus": {"key": "bulk_modulus_kv"},
    "shear_modulus": {"key": "shear_modulus_gv"},
    # magmom is OPTIONAL: JARVIS-ML treats magnetism as a classification, not a regression target,
    # so get_ml_data often has no magmom key. The magnetic GATE is composition-based (Fe/Co/Ni/Mn)
    # in predict_properties regardless; if a key works this just adds a refining estimate.
    "magmom": {"key": ["magmom_outcar", "magmom_oszicar"], "optional": True},
}


def _regressor():
    """LightGBM if available (the JARVIS-ML choice); else sklearn HistGradientBoosting."""
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=800, learning_rate=0.08, num_leaves=128, n_jobs=-1)
    except Exception:  # noqa: BLE001
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08)


def _pmg_to_jarvis(structure):
    from jarvis.core.atoms import Atoms

    return Atoms(
        lattice_mat=structure.lattice.matrix,
        coords=structure.frac_coords,
        elements=[str(s.specie) for s in structure],
        cartesian=False,
    )


def cfid_features(structure):
    """1557 CFID descriptors for a pymatgen Structure (JARVIS featurization)."""
    from jarvis.ai.descriptors.cfid import CFID

    return CFID(_pmg_to_jarvis(structure)).get_comp_descp()


def train(props=tuple(PROPS), test_size: float = 0.1, seed: int = 1) -> dict:
    """Train CFID + GBT models on JARVIS-DFT; report held-out MAE; save models."""
    import joblib
    import numpy as np
    from jarvis.ai.pkgs.utils import get_ml_data, regr_scores
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name in props:
        spec = PROPS[name]
        keys = spec["key"] if isinstance(spec["key"], list) else [spec["key"]]
        data = err = None
        for k in keys:                                # try fallback keys in order
            try:
                print(f"[{name}] loading JARVIS CFID data for '{k}'...")
                X, y, _ = get_ml_data(ml_property=k, dataset="cfid_3d")
                data, used_key = (np.asarray(X, dtype=float), np.asarray(y, dtype=float)), k
                break
            except Exception as exc:  # noqa: BLE001
                err = exc
        if data is None:
            msg = f"  [skip] {name}: no usable JARVIS key in {keys} ({err})"
            if spec.get("optional"):
                print(msg + "  -- optional, continuing")
                continue
            raise
        X, y = data
        ok = np.isfinite(y)                           # drop entries without this label
        X, y = X[ok], y[ok]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
        model = _regressor()
        model.fit(Xtr, ytr)
        mae = float(regr_scores(yte, model.predict(Xte))["mae"])
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        metrics[name] = {"key": used_key, "n_train": int(len(ytr)), "n_test": int(len(yte)),
                         "mae": round(mae, 4)}
        print(f"  {name}: n={len(y)}  held-out MAE={mae:.3f}  -> {name}.joblib")

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nmetrics -> {METRICS}  ({len(metrics)} models)")
    return metrics


_CACHE: dict = {}


def _load(name):
    if name not in _CACHE:
        import joblib

        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — train first: python -m stealth.discovery.predictors --train")
        _CACHE[name] = joblib.load(path)
    return _CACHE[name]


def predict_properties(structure) -> dict:
    """Predict EM/optical/mechanical/magnetic properties for one structure -> dict.

    Returns refractive index n, dielectric eps, band gap, conductivity class, complex
    permeability mu (magnetic class x literature prior), elastic moduli + a 0..1 durability
    score — the inputs the signature objective (#4) and the durability term (#5) consume.
    """
    import numpy as np

    x = np.asarray(cfid_features(structure), dtype=float).reshape(1, -1)
    eps = max(1.0, float(_load("eps").predict(x)[0]))
    bandgap = max(0.0, float(_load("bandgap").predict(x)[0]))
    kv = float(_load("bulk_modulus").predict(x)[0])
    gv = float(_load("shear_modulus").predict(x)[0])

    els = {str(s.specie.symbol) for s in structure}
    # magnetic gate is composition-based (ferrite/magnetic-absorber chemistry); the literature
    # mu prior returns mu>1 only for Fe/Co/Ni/Mn-bearing compositions. Refine with the optional
    # magmom model if it was trained.
    mu = class_mu_prior(els)
    is_magnetic = mu.imag > 0
    magmom = None
    try:
        magmom = abs(float(_load("magmom").predict(x)[0]))
        is_magnetic = is_magnetic or magmom > 0.5
    except FileNotFoundError:
        pass
    if not is_magnetic:
        mu = complex(1.0, 0.0)
    # conductivity class from the DFT-grade band gap (metallic/small-gap -> conductive loss)
    sigma_class = "conductive" if bandgap < 0.1 else ("semiconducting" if bandgap < 1.0 else "insulating")
    durability = float(np.clip((kv - 50.0) / 250.0, 0.0, 1.0))  # ~50->0, ~300 GPa->1

    return {
        "refractive_index_n": round(eps ** 0.5, 3),
        "dielectric_eps": round(eps, 3),
        "band_gap_ev": round(bandgap, 3),
        "sigma_class": sigma_class,
        "magnetic": is_magnetic,
        "magmom": round(magmom, 3) if magmom is not None else None,
        "mu_real": round(mu.real, 3),
        "mu_imag": round(mu.imag, 3),
        "bulk_modulus_gpa": round(kv, 1),
        "shear_modulus_gpa": round(gv, 1),
        "durability_score": round(durability, 3),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train all predictors on JARVIS-DFT (pod)")
    args = ap.parse_args()
    if args.train:
        train()
    else:
        print("Use --train to fit the predictors (needs jarvis-tools + the [train] extra on the pod).")
