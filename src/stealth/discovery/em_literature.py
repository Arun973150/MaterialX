"""#1 dataset (Tier A2): measured GHz complex permittivity & permeability of real absorbers.

DFT (Tier A1, `dataset.py`) does not give microwave epsilon(f)/mu(f) — that data only exists
in the experimental literature. This is a curated table of **representative X-band (~8-12 GHz)
complex permittivity (eps' - j eps'') and permeability (mu' - j mu'')** for the real radar-absorber
material classes, compiled from review/measurement papers:

  * magnetic-loss: carbonyl iron, spinel ferrites (Fe3O4, NiZn, MnZn, CoFe2O4), hexaferrites
  * dielectric/conductive-loss: carbon black, CNT, graphene/RGO, MXene (Ti3C2), SiC, PANI

Values are representative (order-of-magnitude / typical of the class), suitable for:
  * (#3) calibrating the magnetic-permeability prior — the *only* source of real mu;
  * (#10) anchoring the radar physics against a known absorber (reproduce its reflection);
  * supplying class-level eps/mu when per-structure prediction is uncertain.

Sources (representative, X-band): RAM reviews [Kim, Adv. Sci. 2023; ScienceDirect RAM overview],
ferrite-polymer composites [J. Magn. Magn. Mater.; IEEE Trans. Magn.], MnZn-ferrite RAM
[arXiv:1105.5969], MXene microwave-absorption review [Adv. Compos. Hybrid Mater. 2024]. Each row
is class-typical, not a single-sample reproduction; exact per-paper provenance is added per entry
as the dataset grows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EMRecord:
    material: str
    mat_class: str          # "magnetic" | "dielectric" | "conductive"
    freq_ghz: float         # representative measurement frequency
    eps_real: float
    eps_imag: float
    mu_real: float
    mu_imag: float
    note: str = ""

    @property
    def loss_tan_e(self) -> float:
        return self.eps_imag / self.eps_real if self.eps_real else 0.0

    @property
    def loss_tan_m(self) -> float:
        return self.mu_imag / self.mu_real if self.mu_real else 0.0


# Representative X-band values (eps', eps'', mu', mu'') from the absorber literature.
LITERATURE: list[EMRecord] = [
    # --- magnetic-loss (mu'' significant; the low-GHz RAM workhorses) ---
    EMRecord("Carbonyl iron (70 wt% composite)", "magnetic", 10.0, 15.0, 1.5, 2.0, 1.0,
             "classic magnetic RAM; broadband mu'' loss"),
    EMRecord("Fe3O4 (magnetite composite)", "magnetic", 10.0, 10.0, 2.0, 1.3, 0.4,
             "spinel ferrite; dielectric+magnetic loss"),
    EMRecord("NiZn ferrite (Ni0.5Zn0.5Fe2O4)", "magnetic", 4.0, 11.0, 0.4, 2.2, 1.0,
             "best below X-band (Snoek limit)"),
    EMRecord("MnZn ferrite", "magnetic", 2.0, 13.0, 0.8, 2.5, 1.2,
             "very high mu at MHz; rolls off by GHz"),
    EMRecord("CoFe2O4", "magnetic", 10.0, 9.0, 1.0, 1.4, 0.6, "high anisotropy spinel"),
    EMRecord("Ba hexaferrite (BaFe12O19, substituted)", "magnetic", 10.0, 7.0, 1.5, 1.6, 1.2,
             "natural FMR tunable into X-band"),
    EMRecord("FeCo alloy flakes", "magnetic", 6.0, 30.0, 8.0, 3.0, 2.0,
             "high-Ms metal; high eps and mu loss"),
    # --- dielectric / conductive-loss (mu ~ 1; loss via conductivity/polarization) ---
    EMRecord("Carbon black (composite)", "conductive", 10.0, 22.0, 10.0, 1.0, 0.0,
             "dielectric/conductive loss, non-magnetic"),
    EMRecord("MWCNT (polymer composite)", "conductive", 10.0, 30.0, 18.0, 1.0, 0.0,
             "high conductive loss"),
    EMRecord("Reduced graphene oxide (RGO)", "conductive", 10.0, 11.0, 6.0, 1.0, 0.0,
             "tunable dielectric loss"),
    EMRecord("MXene Ti3C2Tx (film/composite)", "conductive", 10.0, 45.0, 25.0, 1.0, 0.0,
             "metallic 2D; very high eps'' — needs impedance matching"),
    EMRecord("SiC", "dielectric", 10.0, 8.0, 1.5, 1.0, 0.0,
             "high-temperature dielectric absorber"),
    EMRecord("Polyaniline (PANI)", "conductive", 10.0, 18.0, 8.0, 1.0, 0.0,
             "conductive polymer dielectric loss"),
    EMRecord("BaTiO3 (dielectric)", "dielectric", 10.0, 35.0, 2.0, 1.0, 0.0,
             "high-permittivity dielectric"),
]

# magnetic elements whose presence implies a magnetic-loss (mu > 1) material class
_MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Mn"}


def as_dataframe():
    import pandas as pd

    return pd.DataFrame(
        [{**r.__dict__, "loss_tan_e": round(r.loss_tan_e, 3), "loss_tan_m": round(r.loss_tan_m, 3)}
         for r in LITERATURE]
    )


def class_mu_prior(elements: set[str]) -> complex:
    """Class-level complex permeability prior for a composition's element set.

    Magnetic (Fe/Co/Ni/Mn-bearing) compositions get the median magnetic-class mu; everything
    else is non-magnetic (mu = 1). This is the honest stand-in used by the radar objective
    until the structure-level permeability predictor (#3) is trained on this table.
    """
    mag = [r for r in LITERATURE if r.mat_class == "magnetic"]
    if elements & _MAGNETIC_ELEMENTS and mag:
        mu_r = sorted(r.mu_real for r in mag)[len(mag) // 2]
        mu_i = sorted(r.mu_imag for r in mag)[len(mag) // 2]
        return complex(mu_r, mu_i)
    return complex(1.0, 0.0)


def reference_absorber(name: str = "Carbonyl iron (70 wt% composite)") -> EMRecord:
    """Return a known absorber record for the validation anchor (#10)."""
    for r in LITERATURE:
        if r.material.startswith(name) or name in r.material:
            return r
    raise KeyError(name)


if __name__ == "__main__":
    df = as_dataframe()
    import pandas as pd

    pd.set_option("display.width", 200)
    print(df[["material", "mat_class", "freq_ghz", "eps_real", "eps_imag",
              "mu_real", "mu_imag", "loss_tan_m"]].to_string(index=False))
    print(f"\n{len(df)} records | magnetic mu prior for Fe-O: {class_mu_prior({'Fe', 'O'})}")
