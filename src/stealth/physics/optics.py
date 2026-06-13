"""Phase 2 optics forward model (Transfer Matrix Method).

Given a planar multilayer ``Stack``, *computes* its IR emissivity and visible color
from the real material constants gathered in Phase 1. This is the first place the
pipeline generates device-level performance data rather than looking it up.

Physics:
  * Reflectance/transmittance from the Abeles transfer-matrix method (`tmm` pkg).
  * Absorptance A = 1 - R - T. With an opaque metal ground T -> 0, so the surface
    emissivity equals the absorptance (Kirchhoff's law).
  * Visible color: reflectance over 400-700 nm -> CIE XYZ (D65 / 2 deg observer)
    -> L*a*b* -> deltaE vs a target background.

Wavelengths and thicknesses are both in micrometers (TMM only needs them consistent).
Normal incidence by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..materials.optical import NoOpticalData, optical_constants
from ..materials.registry import load_registry

# Visible sampling grid (nm) for color.
_VIS_NM = np.arange(400, 701, 10)


class CoverageError(Exception):
    """A layer material lacks optical data over the requested band."""


@dataclass(frozen=True)
class Layer:
    """One planar film: a registry material name + physical thickness (um)."""

    material: str
    thickness_um: float


@dataclass(frozen=True)
class Stack:
    """Ordered planar layers, top (air-side) to bottom. Air is added on both sides."""

    layers: tuple[Layer, ...]

    @classmethod
    def of(cls, *layers: Layer) -> "Stack":
        return cls(tuple(layers))


@lru_cache(maxsize=1)
def _registry_index() -> dict:
    return {m.name: m for m in load_registry()}


def _layer_nk(layer: Layer, wl_um: np.ndarray, clip: bool) -> np.ndarray:
    index = _registry_index()
    if layer.material not in index:
        raise KeyError(f"unknown material {layer.material!r} (not in registry)")
    try:
        return optical_constants(index[layer.material], wl_um, clip=clip)
    except NoOpticalData as exc:
        raise CoverageError(
            f"{layer.material} has no optical n,k (it's a radar conductor) — "
            f"it can't be used as an optical layer"
        ) from exc


def stack_spectrum(
    stack: Stack,
    wl_um,
    *,
    theta: float = 0.0,
    pol: str = "s",
    clip: bool = False,
) -> dict:
    """Reflectance, transmittance and absorptance spectra over ``wl_um``.

    Returns dict of arrays: ``wl_um, R, T, A`` (A = 1 - R - T).
    Raises :class:`CoverageError` if any layer lacks data in-band (unless ``clip``).
    """
    from tmm import coh_tmm

    wl = np.atleast_1d(np.asarray(wl_um, dtype=float))
    n_layers = [_layer_nk(layer, wl, clip) for layer in stack.layers]

    for layer, N in zip(stack.layers, n_layers):
        if not clip and np.isnan(N).any():
            bad = wl[np.isnan(N)]
            raise CoverageError(
                f"{layer.material}: no optical data at {bad.min():.3g}-{bad.max():.3g} um. "
                f"Restrict the band or pass clip=True."
            )

    d_inner = [layer.thickness_um for layer in stack.layers]
    R = np.empty(wl.size)
    T = np.empty(wl.size)
    for i, lam in enumerate(wl):
        n_list = [1.0] + [n_layers[j][i] for j in range(len(stack.layers))] + [1.0]
        d_list = [np.inf] + d_inner + [np.inf]
        res = coh_tmm(pol, n_list, d_list, theta, lam)
        R[i] = res["R"]
        T[i] = res["T"]

    return {"wl_um": wl, "R": R, "T": T, "A": 1.0 - R - T}


def band_emissivity(stack: Stack, band_um: tuple[float, float], *, n: int = 51, **kw) -> float:
    """Band-averaged emissivity (= absorptance) over ``band_um`` at normal incidence."""
    wl = np.linspace(band_um[0], band_um[1], n)
    return float(np.mean(stack_spectrum(stack, wl, **kw)["A"]))


# ---- Visible color -----------------------------------------------------------

@lru_cache(maxsize=1)
def _color_ctx():
    import colour

    return (
        colour,
        colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
        colour.SDS_ILLUMINANTS["D65"],
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
    )


def visible_lab(stack: Stack, **kw) -> np.ndarray:
    """CIE L*a*b* of the stack's reflected color under D65 (clips band edges)."""
    colour, cmfs, illum, d65_xy = _color_ctx()
    R = stack_spectrum(stack, _VIS_NM / 1000.0, clip=True, **kw)["R"]
    sd = colour.SpectralDistribution(dict(zip(_VIS_NM.tolist(), R.tolist())))
    xyz = colour.sd_to_XYZ(sd, cmfs, illum) / 100.0
    return colour.XYZ_to_Lab(xyz, d65_xy)


def hex_to_lab(hex_color: str) -> np.ndarray:
    """CIE L*a*b* of an sRGB hex color (e.g. '#228B22')."""
    colour, *_ = _color_ctx()
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(colour.notation.HEX_to_RGB(hex_color)))


def delta_e_vs_background(stack: Stack, background_hex: str, *, method: str = "CIE 1976", **kw) -> float:
    """Color difference deltaE between the stack and a background hex color."""
    colour, *_ = _color_ctx()
    return float(colour.delta_E(visible_lab(stack, **kw), hex_to_lab(background_hex), method=method))


def nir_reflectance(stack: Stack, band_um: tuple[float, float] = (0.7, 1.4), *, n: int = 31, **kw) -> float:
    """Band-averaged NIR reflectance (for comparison against a terrain background)."""
    wl = np.linspace(band_um[0], band_um[1], n)
    return float(np.mean(stack_spectrum(stack, wl, clip=True, **kw)["R"]))


# ---- Demo --------------------------------------------------------------------

def demo() -> None:
    """Generate the headline Phase 2 result: the VO2 emissivity switch in a real stack."""
    from ..config import load_targets

    t = load_targets()
    lwir = tuple(t["ir_lwir"]["band_um"])
    mwir = tuple(t["ir_mwir"]["band_um"])
    spacer = Layer("SiO2 (IR dielectric)", 1.0)
    ground = Layer("Aluminum (ground plane)", 0.3)

    print("Stack: VO2(0.2um) / SiO2(1um) / Al(0.3um)   [IR emissivity, normal incidence]")
    print(f"{'VO2 state':<12}{'MWIR e':>10}{'LWIR e':>10}")
    for state in ("VO2 (insulating, <Tc)", "VO2 (metallic, >Tc)"):
        s = Stack.of(Layer(state, 0.2), spacer, ground)
        em = band_emissivity(s, mwir)
        el = band_emissivity(s, lwir)
        print(f"{state.split('(')[1][:10]:<12}{em:>10.3f}{el:>10.3f}")
    print(f"(LWIR target: emissivity < {t['ir_lwir']['threshold']})")

    # Visible color of an electrochromic top stack vs the forest-green target.
    vis = Stack.of(Layer("PEDOT:PSS", 0.1), Layer("VO2 (insulating, <Tc)", 0.2), ground)
    bg = t["visible"]["background"]["srgb_hex"]
    lab = visible_lab(vis)
    de = delta_e_vs_background(vis, bg)
    print(f"\nVisible: PEDOT:PSS/VO2/Al  Lab=({lab[0]:.1f},{lab[1]:.1f},{lab[2]:.1f})  "
          f"deltaE vs {bg} = {de:.1f}  (target < {t['visible']['threshold']})")


if __name__ == "__main__":
    demo()
