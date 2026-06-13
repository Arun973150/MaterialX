"""Optical constants n(lambda), k(lambda) for registry materials.

Returns the complex refractive index ``N = n + i*k`` for any material over a
requested wavelength grid (micrometers). refractiveindex.info-backed materials are
looked up in the local database; literature materials return their documented
constant. This is the primary input consumed by the Phase 2 optics (TMM) model.
"""

from __future__ import annotations

import numpy as np

from .registry import Material

# Band centers (um) used for quick summaries.
BAND_CENTERS_UM = {"visible": 0.55, "nir": 1.0, "mwir": 4.0, "lwir": 10.0}


class NoOpticalData(Exception):
    """Raised when a material has no optical n,k (e.g. a pure radar conductor)."""


def wl_range_um(material: Material) -> tuple[float, float] | None:
    """Valid wavelength range (um) for a refractiveindex material; None otherwise."""
    if material.source != "refractiveindex":
        return None
    from refractiveindex import RefractiveIndexMaterial

    return RefractiveIndexMaterial(**material.ri).get_wl_range(unit="um")


def optical_constants(
    material: Material,
    wl_um: float | np.ndarray,
    *,
    clip: bool = False,
) -> np.ndarray:
    """Complex refractive index N = n + i*k at the given wavelength(s) in um.

    Out-of-range wavelengths return NaN unless ``clip=True``, which holds the
    nearest in-range endpoint (use deliberately — extrapolation is unphysical).
    """
    wl = np.atleast_1d(np.asarray(wl_um, dtype=float))

    if material.source == "refractiveindex":
        from refractiveindex import RefractiveIndexMaterial
        from refractiveindex.refractiveindex import NoExtinctionCoefficient

        m = RefractiveIndexMaterial(**material.ri)
        rng = m.get_wl_range(unit="um")
        q = np.clip(wl, rng[0], rng[1]) if (clip and rng) else wl

        n = np.asarray(m.get_refractive_index(q, unit="um"), dtype=float)
        try:
            k = np.asarray(m.get_extinction_coefficient(q, unit="um"), dtype=float)
        except NoExtinctionCoefficient:
            k = np.zeros_like(n)
        return n + 1j * k

    if material.source == "literature":
        if material.literature_nk:
            n = float(material.literature_nk.get("n", float("nan")))
            k = float(material.literature_nk.get("k", 0.0))
            return np.full(wl.shape, n + 1j * k, dtype=complex)
        raise NoOpticalData(
            f"{material.name}: no optical n,k (radar conductor - use properties.conductivity)"
        )

    raise ValueError(f"{material.name}: unknown source {material.source!r}")


def single_surface_emissivity(material: Material, wl_um: float) -> float:
    """Opaque single-interface emissivity proxy, e ~= 1 - R (normal incidence).

    A characterization shortcut for Phase 1 only; the real layered emissivity is
    computed by the Phase 2 TMM model. Returns NaN if out of range or if the
    material has no optical data (e.g. a pure radar conductor).
    """
    try:
        N = optical_constants(material, wl_um)[0]
    except NoOpticalData:
        return float("nan")
    if np.isnan(N):
        return float("nan")
    R = np.abs((N - 1.0) / (N + 1.0)) ** 2
    return float(1.0 - R)
