"""Phase 3 radar forward model — transmission-line / equivalent-circuit model.

Computes the reflection-loss spectrum of a circuit-analog absorber: a patterned
resistive metasurface (shunt sheet impedance) over a grounded dielectric spacer.
This is the standard, fast, differentiable method for radar-absorbing structures
(Salisbury / Jaumann / circuit-analog absorbers) and the microwave analog of the
Phase 2 optics TMM. The PEC ground is a short-circuited transmission-line stub.

Geometry -> sheet impedance:
  * capacitive patch array: Zs = Rs + 1/(jwC), C from period/gap (Luukkonen 2008).
  * inductive strip grid:    Zs = Rs + jwL, L from period/strip width.
  * resistive_only:          Zs = Rs (a plain Salisbury resistive sheet).

Reflection:
  Z_stub = j (Z0/sqrt(eps_r)) tan(beta d)      # shorted dielectric stub
  Zin    = Zs || Z_stub                        # FSS shunt + stub
  Gamma  = (Zin - Z0)/(Zin + Z0)
  RL(dB) = 20 log10|Gamma|,   absorption = 1 - |Gamma|^2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C0 = 299_792_458.0          # speed of light, m/s
EPS0 = 8.8541878128e-12     # vacuum permittivity, F/m
MU0 = 1.25663706212e-6      # vacuum permeability, H/m
Z0 = float(np.sqrt(MU0 / EPS0))  # ~376.73 ohm, free-space impedance

PATTERNS = ("capacitive_patch", "inductive_grid", "resistive_only")


@dataclass(frozen=True)
class RadarStack:
    """A circuit-analog absorber unit cell (one resistive FSS over grounded spacer)."""

    period_mm: float                 # unit-cell period D
    patch_mm: float                  # patch size (capacitive) or strip width (inductive)
    sheet_resistance_ohm_sq: float   # effective sheet resistance Rs of the pattern
    spacer_thickness_mm: float       # grounded dielectric thickness d
    spacer_eps_r: float              # spacer relative permittivity
    pattern: str = "capacitive_patch"

    def __post_init__(self) -> None:
        if self.pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}, got {self.pattern!r}")
        if self.pattern != "resistive_only" and not (0 < self.patch_mm < self.period_mm):
            raise ValueError(f"need 0 < patch ({self.patch_mm}) < period ({self.period_mm}) mm")


def _eps_eff(stack: RadarStack) -> float:
    """Effective permittivity seen by the grid (dielectric + air average)."""
    return (stack.spacer_eps_r + 1.0) / 2.0


def grid_capacitance_F(stack: RadarStack) -> float:
    """Effective capacitance of a periodic patch array (Luukkonen et al., 2008)."""
    D = stack.period_mm * 1e-3
    g = (stack.period_mm - stack.patch_mm) * 1e-3   # gap between patches
    return EPS0 * _eps_eff(stack) * (2 * D / np.pi) * np.log(1.0 / np.sin(np.pi * g / (2 * D)))


def grid_inductance_H(stack: RadarStack) -> float:
    """Effective inductance of a periodic strip grid (dual of the patch array)."""
    D = stack.period_mm * 1e-3
    w = stack.patch_mm * 1e-3   # strip width
    return MU0 * (D / (2 * np.pi)) * np.log(1.0 / np.sin(np.pi * w / (2 * D)))


def sheet_impedance(stack: RadarStack, f_hz: np.ndarray) -> np.ndarray:
    """Complex shunt sheet impedance Zs(f) of the patterned layer."""
    w = 2 * np.pi * f_hz
    Rs = stack.sheet_resistance_ohm_sq
    if stack.pattern == "resistive_only":
        return np.full_like(f_hz, Rs, dtype=complex)
    if stack.pattern == "capacitive_patch":
        C = grid_capacitance_F(stack)
        return Rs + 1.0 / (1j * w * C)
    L = grid_inductance_H(stack)              # inductive_grid
    return Rs + 1j * w * L


def _angle_factors(eps_r: float, theta_deg: float, pol: str):
    """Oblique-incidence wave impedances + dielectric refraction factor (TE/TM).

    Returns (Z_free, Z_dielectric, cos_theta_t): the polarization-dependent free-space and
    in-dielectric wave impedances and the refracted-angle cosine (Snell). theta=0 reduces to
    normal incidence (Z_free=Z0, Z_d=Z0/sqrt(eps_r), cos_t=1).
    """
    theta = np.radians(theta_deg)
    cos_i = np.cos(theta)
    sin_t = np.sin(theta) / np.sqrt(eps_r)            # Snell into the spacer
    cos_t = np.sqrt(max(0.0, 1.0 - float(sin_t) ** 2))
    if pol.upper() == "TM":
        return Z0 * cos_i, (Z0 / np.sqrt(eps_r)) * cos_t, cos_t
    return Z0 / cos_i, (Z0 / np.sqrt(eps_r)) / cos_t, cos_t   # TE (default)


def input_impedance(stack: RadarStack, f_hz: np.ndarray, theta_deg: float = 0.0,
                    pol: str = "TE") -> np.ndarray:
    """Input impedance of FSS sheet in parallel with the shorted dielectric stub.

    At oblique incidence the stub uses the z-directed propagation constant and the
    polarization-dependent dielectric wave impedance.
    """
    _, Zd, cos_t = _angle_factors(stack.spacer_eps_r, theta_deg, pol)
    beta_z = (2 * np.pi * f_hz / C0) * np.sqrt(stack.spacer_eps_r) * cos_t
    Z_stub = 1j * Zd * np.tan(beta_z * stack.spacer_thickness_mm * 1e-3)
    Zs = sheet_impedance(stack, f_hz)
    return Zs * Z_stub / (Zs + Z_stub)


def reflection_coefficient(stack: RadarStack, f_hz: np.ndarray, theta_deg: float = 0.0,
                           pol: str = "TE") -> np.ndarray:
    Z_free, _, _ = _angle_factors(stack.spacer_eps_r, theta_deg, pol)
    Zin = input_impedance(stack, f_hz, theta_deg, pol)
    return (Zin - Z_free) / (Zin + Z_free)


def spectrum(stack: RadarStack, f_ghz: np.ndarray, theta_deg: float = 0.0,
             pol: str = "TE") -> dict:
    """Reflection loss (dB) and absorption over a frequency grid (GHz), at incidence angle/pol."""
    f_hz = np.atleast_1d(np.asarray(f_ghz, dtype=float)) * 1e9
    gamma = reflection_coefficient(stack, f_hz, theta_deg, pol)
    mag = np.clip(np.abs(gamma), 1e-6, None)        # floor avoids log10(0)
    return {
        "f_ghz": f_hz / 1e9,
        "reflection_loss_db": 20 * np.log10(mag),
        "absorption": 1.0 - np.abs(gamma) ** 2,
    }


def angular_stress_test(stack: RadarStack, f_ghz: np.ndarray,
                        angles=(0, 15, 30, 45, 60), pols=("TE", "TM"),
                        threshold_db: float = -10.0) -> dict:
    """Worst-case reflection across incidence angles + polarizations — the radar stress test.

    Real detection is not at normal incidence; a deployable absorber must hold up over a range
    of angles and both polarizations. Returns the per-(angle,pol) min RL + band coverage, plus
    the worst case (least absorption, least coverage) across all of them.
    """
    rows = []
    for th in angles:
        for pol in pols:
            rl = spectrum(stack, f_ghz, th, pol)["reflection_loss_db"]
            rows.append({"angle_deg": th, "pol": pol, "min_rl_db": float(rl.min()),
                         "frac_band_below_thresh": float(np.mean(rl <= threshold_db))})
    return {
        "per_angle": rows,
        "worst_min_rl_db": max(r["min_rl_db"] for r in rows),       # least absorption
        "worst_coverage": min(r["frac_band_below_thresh"] for r in rows),
        "normal_min_rl_db": next(r["min_rl_db"] for r in rows if r["angle_deg"] == 0 and r["pol"] == "TE"),
    }


def metrics(stack: RadarStack, f_ghz: np.ndarray, threshold_db: float = -10.0) -> dict:
    """Summary labels for one design: peak null, -10 dB bandwidth, band coverage."""
    sp = spectrum(stack, f_ghz)
    rl = sp["reflection_loss_db"]
    f = sp["f_ghz"]
    below = rl <= threshold_db
    if below.any():
        bw_lo, bw_hi = float(f[below].min()), float(f[below].max())
        frac = float(np.mean(below))
    else:
        bw_lo = bw_hi = frac = 0.0
    return {
        "min_rl_db": float(rl.min()),
        "f_at_min_ghz": float(f[np.argmin(rl)]),
        "bw10_lo_ghz": bw_lo,
        "bw10_hi_ghz": bw_hi,
        "frac_band_below_thresh": frac,
    }


def demo() -> None:
    f = np.linspace(1, 30, 581)

    # 1) Salisbury screen: Rs=Z0 resistive sheet, quarter-wave (7.5 mm) air gap.
    #    Analytic expectation: near-total absorption at 10 GHz.
    salisbury = RadarStack(period_mm=3, patch_mm=1, sheet_resistance_ohm_sq=Z0,
                           spacer_thickness_mm=7.5, spacer_eps_r=1.0,
                           pattern="resistive_only")
    m = metrics(salisbury, f)
    print(f"Salisbury (Rs={Z0:.0f}, d=7.5mm air): min RL {m['min_rl_db']:.1f} dB "
          f"at {m['f_at_min_ghz']:.2f} GHz  (expect deep null ~10 GHz)")

    # 2) Circuit-analog absorber: capacitive patch on grounded FR4.
    caa = RadarStack(period_mm=6, patch_mm=5.4, sheet_resistance_ohm_sq=120,
                     spacer_thickness_mm=2.5, spacer_eps_r=4.3,
                     pattern="capacitive_patch")
    m = metrics(caa, f)
    print(f"Circuit-analog (patch on FR4): min RL {m['min_rl_db']:.1f} dB at {m['f_at_min_ghz']:.2f} GHz, "
          f"-10 dB band {m['bw10_lo_ghz']:.1f}-{m['bw10_hi_ghz']:.1f} GHz")


if __name__ == "__main__":
    demo()
