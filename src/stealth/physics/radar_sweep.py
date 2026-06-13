"""Phase 3 radar dataset generation: sweep the absorber design space.

Samples circuit-analog absorber designs (Latin hypercube over a unit cube),
evaluates each with the ECM forward model, and stores design parameters + the
reflection-loss spectrum + summary metrics. The unit-cube <-> design mapping
(`DESIGN_BOUNDS`, `stack_from_unit`) is shared with the Phase 5 optimizer.

ECM is analytic-fast, so this runs in seconds locally; the cluster generates the
larger sweep and the openEMS ground-truth subset (see `radar_fullwave.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from .radar import RadarStack, metrics, spectrum

OUTPUT = REPO_ROOT / "data" / "radar_dataset.parquet"

# Fixed frequency grid the spectra are sampled on (GHz) — 1-30 GHz at 0.5 GHz.
FREQ_GHZ = np.linspace(1.0, 30.0, 59)

# Design space (unit cube -> physical params). Reused by the Phase 5 optimizer.
DESIGN_BOUNDS = {
    "period_mm": (3.0, 8.0),
    "patch_frac": (0.70, 0.97),            # patch_mm = period_mm * patch_frac
    "sheet_resistance_ohm_sq": (30.0, 400.0),
    "spacer_thickness_mm": (0.5, 5.0),
    "spacer_eps_r": (2.2, 4.5),            # PDMS(2.7) / SiO2(3.9) / FR4(4.3) range
}
PARAM_ORDER = tuple(DESIGN_BOUNDS)


def stack_from_unit(u: np.ndarray) -> RadarStack:
    """Map a point in [0,1]^5 (PARAM_ORDER) to a RadarStack."""
    u = np.asarray(u, dtype=float).ravel()
    lo_hi = np.array([DESIGN_BOUNDS[p] for p in PARAM_ORDER])
    vals = lo_hi[:, 0] + u * (lo_hi[:, 1] - lo_hi[:, 0])
    period, patch_frac, rs, d, eps = vals
    return RadarStack(
        period_mm=float(period),
        patch_mm=float(period * patch_frac),
        sheet_resistance_ohm_sq=float(rs),
        spacer_thickness_mm=float(d),
        spacer_eps_r=float(eps),
        pattern="capacitive_patch",
    )


def generate_dataset(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Latin-hypercube sweep -> DataFrame of params, metrics, and RL spectra."""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=len(PARAM_ORDER), seed=seed)
    points = sampler.random(n)

    rows: list[dict] = []
    for u in points:
        stack = stack_from_unit(u)
        m = metrics(stack, FREQ_GHZ)
        rl = spectrum(stack, FREQ_GHZ)["reflection_loss_db"]
        rows.append(
            {
                "period_mm": stack.period_mm,
                "patch_mm": stack.patch_mm,
                "sheet_resistance_ohm_sq": stack.sheet_resistance_ohm_sq,
                "spacer_thickness_mm": stack.spacer_thickness_mm,
                "spacer_eps_r": stack.spacer_eps_r,
                **m,
                "rl_spectrum_db": rl.tolist(),
            }
        )
    return pd.DataFrame(rows)


def export(df: pd.DataFrame, path: str | Path = OUTPUT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main(n: int = 2000) -> None:
    df = generate_dataset(n)
    out = export(df)
    good = df[df["frac_band_below_thresh"] > 0]
    print(f"Generated {len(df)} designs -> {out}")
    print(f"  designs with any -10 dB absorption band: {len(good)} ({100*len(good)/len(df):.0f}%)")
    print(f"  best min RL: {df['min_rl_db'].min():.1f} dB")
    print(f"  widest -10 dB band: {(df['bw10_hi_ghz'] - df['bw10_lo_ghz']).max():.1f} GHz")


if __name__ == "__main__":
    main()
