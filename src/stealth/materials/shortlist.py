"""Assemble the per-layer candidate shortlist and export it.

Combines the registry, the optical-constants loader, and a few computed summary
metrics (band coverage, n/k at band centers, an LWIR emissivity proxy) into one
table that downstream phases (optics model, radar engine) consume.

Run:  python -m stealth.materials.shortlist
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from .optical import (
    BAND_CENTERS_UM,
    NoOpticalData,
    optical_constants,
    single_surface_emissivity,
    wl_range_um,
)
from .registry import Material, load_registry

OUTPUT = REPO_ROOT / "data" / "materials_shortlist.parquet"

# Scalar properties promoted to their own columns when present.
_SCALAR_PROPS = (
    "conductivity_S_per_cm",
    "sheet_resistance_ohm_sq",
    "eps_r_microwave",
    "loss_tangent",
    "transition_temp_C",
    "density_g_cm3",
)


def _nk_at(material: Material, wl_um: float) -> tuple[float, float]:
    """(n, k) at one wavelength; NaN if unavailable or out of range."""
    try:
        N = optical_constants(material, wl_um)[0]
    except NoOpticalData:
        return float("nan"), float("nan")
    return float(np.real(N)), float(np.imag(N))


def build_shortlist(materials: list[Material] | None = None) -> pd.DataFrame:
    """Build the candidate shortlist DataFrame from the registry."""
    materials = materials or load_registry()
    rows: list[dict] = []
    for m in materials:
        rng = wl_range_um(m)
        row: dict = {
            "name": m.name,
            "layer_role": m.layer_role,
            "state": m.state,
            "source": m.source,
            "wl_min_um": rng[0] if rng else np.nan,
            "wl_max_um": rng[1] if rng else np.nan,
        }
        for prop in _SCALAR_PROPS:
            row[prop] = m.properties.get(prop, np.nan)

        for band, wl in BAND_CENTERS_UM.items():
            n, k = _nk_at(m, wl)
            row[f"n_{band}"] = n
            row[f"k_{band}"] = k

        # LWIR emissivity proxy — only meaningful where optical data covers 10 um.
        row["emissivity_proxy_lwir"] = single_surface_emissivity(m, BAND_CENTERS_UM["lwir"])
        row["provenance"] = m.provenance
        rows.append(row)

    return pd.DataFrame(rows)


def export(df: pd.DataFrame, path: str | Path = OUTPUT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    df = build_shortlist()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)

    cols = ["name", "layer_role", "state", "wl_min_um", "wl_max_um",
            "k_lwir", "emissivity_proxy_lwir", "conductivity_S_per_cm"]
    print(df[cols].to_string(index=False))

    out = export(df)
    print(f"\nWrote {len(df)} materials -> {out}")

    # Highlight the thermochromic switch the whole IR design hinges on.
    vo2 = df[df["layer_role"] == "ir_thermochromic"].set_index("state")
    if {"insulating", "metallic"} <= set(vo2.index):
        ei = vo2.loc["insulating", "emissivity_proxy_lwir"]
        em = vo2.loc["metallic", "emissivity_proxy_lwir"]
        print(f"\nVO2 LWIR emissivity proxy: insulating={ei:.3f}  metallic={em:.3f}  (delta={em - ei:+.3f})")


if __name__ == "__main__":
    main()
