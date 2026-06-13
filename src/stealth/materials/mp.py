"""Optional Materials Project enrichment for screening / cross-checking.

The shortlist's optical constants come from refractiveindex.info; Materials Project
adds DFT-computed scalar properties (band gap, density, stability, dielectric) useful
for screening alternative candidates. Requires the API key + network, so this is
opt-in and never imported on the offline shortlist path.
"""

from __future__ import annotations

import pandas as pd

from .sources import mp_client

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "band_gap",
    "density",
    "is_stable",
    "energy_above_hull",
]


def fetch_summary(formulas: list[str]) -> pd.DataFrame:
    """Pull core summary properties for each formula (most stable polymorph first)."""
    rows: list[dict] = []
    with mp_client() as mpr:
        for formula in formulas:
            docs = mpr.materials.summary.search(formula=formula, fields=SUMMARY_FIELDS)
            docs = sorted(docs, key=lambda d: (d.energy_above_hull or 1e9))
            for d in docs[:1]:  # keep the most stable polymorph
                rows.append(
                    {
                        "query_formula": formula,
                        "material_id": str(d.material_id),
                        "formula": d.formula_pretty,
                        "band_gap_eV": d.band_gap,
                        "density_g_cm3": d.density,
                        "is_stable": d.is_stable,
                        "e_above_hull_eV": d.energy_above_hull,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Cross-check a few of our registry materials against DFT data.
    df = fetch_summary(["VO2", "SiO2", "Al2O3"])
    print(df.to_string(index=False))
