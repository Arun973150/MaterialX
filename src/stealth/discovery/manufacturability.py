"""Manufacturability screening — keep candidates that could actually become a coating.

The generator optimizes intrinsic physics and will happily propose rhodium / thulium
intermetallics that are stable but unusable in a real, deployable, affordable coating.
This module scores a composition on the engineering constraints the physics ignores:

  * earth-abundance   (crustal ppm)      — is the element actually available?
  * cost              (USD/kg, bulk)     — is it affordable at coating scale?
  * toxicity          (element-level)    — is it safe / non-regulated?
  * density           (g/cm^3)           — coatings must stay light (kg/m^2 cap)

It exposes a 0..1 `manufacturability_score`, a hard `is_practical` gate (no toxic / no
precious / not too heavy), and `practical_elements(role)` — the abundant, cheap, non-toxic
element families used to *constrain MatterGen generation* to real stealth chemistry.

Data are order-of-magnitude reference values (crustal abundance, bulk metal prices,
solid densities); they're for *relative screening*, not accounting. Unknown elements are
treated conservatively rather than silently passed.
"""

from __future__ import annotations

import math

from pymatgen.core import Composition, Element

# crustal abundance, ppm by mass (CRC / USGS reference values)
_ABUNDANCE_PPM = {
    "O": 461000, "Si": 282000, "Al": 82300, "Fe": 56300, "Ca": 41500, "Na": 23600,
    "Mg": 23300, "K": 20900, "Ti": 5650, "H": 1400, "P": 1050, "Mn": 950, "F": 585,
    "Ba": 425, "Sr": 370, "S": 350, "C": 200, "Zr": 165, "Cl": 145, "V": 120, "Cr": 102,
    "Ni": 84, "Zn": 70, "Cu": 60, "Co": 25, "Li": 20, "N": 19, "Nb": 20, "Ga": 19,
    "Pb": 14, "B": 10, "Th": 9.6, "Sc": 22, "Y": 33, "La": 39, "Ce": 66.5, "Nd": 41.5,
    "Pr": 9.2, "Sm": 7.05, "Gd": 6.2, "Dy": 5.2, "Er": 3.5, "Yb": 3.2, "Ho": 1.3,
    "Tm": 0.52, "Lu": 0.8, "Eu": 2.0, "Tb": 1.2, "Sn": 2.3, "As": 1.8, "Mo": 1.2,
    "W": 1.25, "Ge": 1.5, "Hf": 3.0, "Ta": 2.0, "Be": 2.8, "Cd": 0.15, "Sb": 0.2,
    "Bi": 0.009, "Ag": 0.075, "Hg": 0.085, "Se": 0.05, "In": 0.25, "Te": 0.001,
    "Au": 0.004, "Pt": 0.005, "Pd": 0.015, "Rh": 0.001, "Ru": 0.001, "Ir": 0.001,
    "Os": 0.0015, "Re": 0.0007,
}

# bulk price, USD/kg (rough 2020s order of magnitude)
_PRICE_USD_KG = {
    "O": 0.1, "N": 0.1, "C": 0.5, "H": 2, "Fe": 0.5, "Al": 2, "Si": 2, "Ca": 4,
    "Na": 3, "Mg": 4, "K": 12, "Ti": 12, "Mn": 2, "Ba": 5, "Sr": 6, "Zr": 35, "V": 25,
    "Cr": 9, "Ni": 18, "Zn": 3, "Cu": 9, "Co": 50, "Li": 80, "Nb": 45, "Ga": 300,
    "Pb": 2, "B": 4, "Sn": 25, "As": 2, "Mo": 40, "W": 35, "Ge": 1200, "Hf": 900,
    "Ta": 200, "Be": 850, "Cd": 3, "Sb": 6, "Bi": 6, "Ag": 600, "Hg": 30, "Se": 30,
    "In": 250, "Te": 60, "Au": 60000, "Pt": 30000, "Pd": 50000, "Rh": 300000,
    "Ru": 15000, "Ir": 150000, "Os": 100000, "Re": 3000, "P": 3, "S": 0.1, "F": 2,
    "Cl": 0.2, "Sc": 15000, "Y": 50, "La": 5, "Ce": 5, "Nd": 80, "Pr": 100, "Sm": 20,
    "Gd": 40, "Dy": 400, "Er": 40, "Yb": 25, "Ho": 60, "Tm": 3000, "Lu": 1500,
    "Eu": 30, "Tb": 3000, "Th": 200,
}

# elements with significant element-level toxicity / heavy regulatory burden
_TOXIC = {"Pb", "Cd", "Hg", "As", "Tl", "Be", "Os"}
_MODERATE_TOX = {"Se", "Sb", "Te", "Ba", "Cr"}  # context-dependent; soft penalty only

# precious + supply-critical elements (avoid for a scalable coating)
_PRECIOUS = {"Au", "Pt", "Pd", "Rh", "Ru", "Ir", "Os", "Ag", "Re"}
_RARE_EARTH = {"Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
               "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
_CRITICAL = {"Ga", "In", "Te", "Hf", "Ta", "Ge"}  # supply-constrained
AVOID = _PRECIOUS | _RARE_EARTH | _CRITICAL

# RAM-relevant abundant/cheap element families, by stealth-layer role. These constrain
# MatterGen generation (chemical_system) to chemistries real absorbers are made from.
_ROLE_ELEMENTS = {
    # magnetic-loss absorbers: spinel/hex ferrites + carbonyl-iron-like (Fe/Co/Ni oxides)
    "radar_magnetic": ["Fe", "Co", "Ni", "Mn", "Zn", "Mg", "Ba", "Sr", "O"],
    # dielectric/conductive-loss absorbers: carbides, nitrides, doped oxides (MXene/SiC-like)
    "radar_conductor": ["Ti", "V", "Cr", "Fe", "Nb", "Si", "Al", "C", "N"],
    # wide-gap, light, abundant dielectric spacers
    "dielectric_spacer": ["Si", "Al", "Mg", "Ca", "O", "N"],
    # narrow-gap transition-metal oxides for switchable IR (VO2-like)
    "ir_phasechange": ["V", "Ti", "Nb", "Mn", "Fe", "O"],
}


def _density(sym: str) -> float:
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = Element(sym).density_of_solid  # kg/m^3 in pymatgen
        if d:
            return float(d) / 1000.0       # -> g/cm^3
    except Exception:  # noqa: BLE001
        pass
    return 8.0  # conservative default for unknown (and gases like O/N)


def element_penalty(sym: str) -> float:
    """0 (great for a coating) .. 1 (bad): abundance + price + toxicity + density."""
    ppm = _ABUNDANCE_PPM.get(sym)
    price = _PRICE_USD_KG.get(sym)
    # abundance: log10(ppm) >= 2 (100 ppm) good -> 0 ; <= -1 (0.1 ppm) bad -> 1
    p_ab = 0.6 if ppm is None else min(1.0, max(0.0, (2.0 - math.log10(ppm)) / 3.0))
    # price: <= 10 USD/kg good -> 0 ; >= 10000 bad -> 1
    p_pr = 0.6 if price is None else min(1.0, max(0.0, (math.log10(price) - 1.0) / 3.0))
    tox = 1.0 if sym in _TOXIC else (0.4 if sym in _MODERATE_TOX else 0.0)
    p_de = min(1.0, max(0.0, (_density(sym) - 5.0) / 10.0))  # light (<5) good, heavy (>15) bad
    return float(0.35 * p_ab + 0.35 * p_pr + 0.20 * tox + 0.10 * p_de)


def manufacturability_score(comp: Composition) -> float:
    """0..1 (higher = more manufacturable), molar-fraction-weighted over elements."""
    fracs = comp.fractional_composition.get_el_amt_dict()
    if not fracs:
        return 0.0
    pen = sum(f * element_penalty(sym) for sym, f in fracs.items())
    return float(round(1.0 - pen, 3))


def practicality(comp: Composition, min_score: float = 0.45,
                 max_avoid_frac: float = 0.0, max_density: float = 12.0) -> dict:
    """Hard gate + reasons. By default: no toxic, no precious/REE/critical, light enough."""
    fracs = comp.fractional_composition.get_el_amt_dict()
    els = set(fracs)
    toxic = sorted(els & _TOXIC)
    avoid = sorted(els & AVOID)
    avoid_frac = sum(f for s, f in fracs.items() if s in AVOID)
    density = sum(f * _density(s) for s, f in fracs.items())
    score = manufacturability_score(comp)

    reasons = []
    if toxic:
        reasons.append(f"toxic element(s): {', '.join(toxic)}")
    if avoid_frac > max_avoid_frac:
        reasons.append(f"precious/rare-earth/critical: {', '.join(avoid)} ({avoid_frac:.0%})")
    if density > max_density:
        reasons.append(f"too dense ({density:.1f} > {max_density} g/cm^3)")
    if score < min_score:
        reasons.append(f"low manufacturability score ({score:.2f} < {min_score})")

    return {
        "practical": len(reasons) == 0,
        "manufacturability_score": score,
        "avg_density_g_cm3": round(density, 2),
        "avoid_fraction": round(avoid_frac, 3),
        "reasons": reasons,
    }


def is_practical(comp: Composition, **kw) -> bool:
    return practicality(comp, **kw)["practical"]


def practical_elements(role: str) -> list[str]:
    """Abundant/cheap/non-toxic element family for a role, to constrain generation."""
    return _ROLE_ELEMENTS.get(role, sorted(
        s for s in _ABUNDANCE_PPM if s not in AVOID and s not in _TOXIC
    ))
