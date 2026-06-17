"""Bridge: turn a generated material's predicted band gap into an n,k spectrum.

This closes the loop between Track A (generative discovery) and the physics pipeline.
A freshly generated crystal isn't in any optical database, so we estimate its complex
refractive index N(lambda) = n + i*k from its GNN-predicted band gap using standard
first-order relations:

  * refractive index   -> Moss relation  n^4 * Eg ~= 95 eV
  * absorption (k)     -> transparent below the gap (E < Eg), Tauc-like rise above it
  * metals (Eg < ~0.3) -> a Drude dielectric (high reflectivity / loss)

These are deliberately first-order (a trained optical GNN such as TSENN/GNNOpt would
replace them for accuracy). They are enough to (a) give physically correct *trends* —
wide-gap materials read as low-IR-emissivity, metals as reflective — and (b) let a
generated material be dropped straight into the TMM stack model.
"""

from __future__ import annotations

import numpy as np

from ..materials.registry import Material

_HC_EV_UM = 1.239841984  # photon energy (eV) = _HC_EV_UM / wavelength (um)

# Default wavelength grid (um) spanning visible -> LWIR.
DEFAULT_GRID_UM = np.concatenate([
    np.linspace(0.40, 1.40, 30),   # visible + NIR
    np.linspace(1.5, 15.0, 40),    # MWIR + LWIR
])


def estimate_nk(band_gap_ev: float, wl_um: np.ndarray) -> np.ndarray:
    """First-order complex refractive index from a band gap, over wl_um (um)."""
    wl = np.atleast_1d(np.asarray(wl_um, dtype=float))
    E = _HC_EV_UM / wl                      # photon energy (eV)
    eg = max(float(band_gap_ev), 0.0)

    if eg < 0.3:
        # Metallic / lossy: Drude dielectric (representative metal scale, eV units).
        eps_inf, wp, gamma = 1.0, 9.0, 0.1
        eps = eps_inf - wp**2 / (E**2 + 1j * E * gamma)
        N = np.sqrt(eps.astype(complex))
        return np.abs(N.real) + 1j * np.abs(N.imag)

    # Semiconductor / dielectric: Moss n0, transparent below gap, Tauc rise above.
    n0 = float(np.clip((95.0 / eg) ** 0.25, 1.3, 5.0))
    n = np.full_like(wl, n0)
    k = np.zeros_like(wl)
    above = E > eg
    k[above] = 0.3 * np.sqrt(np.maximum(E[above] - eg, 0.0))
    return n + 1j * k


def load_gnnopt_nk(path: str) -> dict:
    """Load a GNNOpt n,k JSON ({id: {energy_ev, n, k}}) produced on the GPU pod."""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def material_from_gnnopt(cid: str, record: dict, layer_role: str = "dielectric_spacer") -> Material:
    """Physics-ready Material from a GNNOpt n,k record (energy eV -> wavelength um).

    GNNOpt's 0-50 eV electronic spectrum is trustworthy in the visible/NIR; it has no
    far-IR resolution, so the table spans ~0.025-6.2 um and interpolation clamps beyond
    that (LWIR stays approximate — use the band-gap estimate there).
    """
    e = np.asarray(record["energy_ev"], dtype=float)
    n = np.asarray(record["n"], dtype=float)
    k = np.asarray(record["k"], dtype=float)
    mask = e > 1e-6
    wl = _HC_EV_UM / e[mask]
    order = np.argsort(wl)
    role = layer_role if layer_role in {"ir_thermochromic", "dielectric_spacer"} else "dielectric_spacer"
    return Material(
        name=cid,
        layer_role=role,
        source="tabulated",
        provenance="GNNOpt-predicted electronic n,k (0-50 eV); trustworthy visible/NIR",
        nk_table={"wl_um": wl[order].tolist(), "n": n[mask][order].tolist(), "k": k[mask][order].tolist()},
    )


def candidate_material(
    name: str,
    band_gap_ev: float,
    layer_role: str,
    *,
    grid_um: np.ndarray = DEFAULT_GRID_UM,
) -> Material:
    """Build a physics-ready (tabulated) Material from a generated candidate."""
    nk = estimate_nk(band_gap_ev, grid_um)
    return Material(
        name=name,
        layer_role=layer_role if layer_role in {"ir_thermochromic", "dielectric_spacer"} else "dielectric_spacer",
        source="tabulated",
        provenance=f"estimated n,k from predicted band gap {band_gap_ev:.2f} eV (Moss/Drude/Tauc first-order)",
        nk_table={"wl_um": grid_um.tolist(), "n": nk.real.tolist(), "k": nk.imag.tolist()},
        properties={"band_gap_ev": float(band_gap_ev)},
    )
