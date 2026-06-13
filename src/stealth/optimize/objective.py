"""Phase 5 joint objective: one design vector -> radar + IR + visible performance.

Ties the Phase 2 optics (TMM) and Phase 3 radar (ECM) models into a single
evaluation over the full stack design vector, plus a weight constraint. This is
what the multi-objective optimizer searches.

Integration model (documented assumption — scale separation):
  The device is sub-micron optical films (EC polymer, VO2) sitting on a millimeter
  radar absorber (patterned sheet + grounded dielectric spacer). Because the radar
  wavelength (cm) >> the optical films (sub-um), the films are electrically
  negligible to the radar, and the mm spacer is treated as decoupled from the thin
  optical interference. The bands couple through (a) the shared spacer material and
  (b) the global weight budget. Full thick-spacer IR coupling and metasurface
  optical blocking are noted refinements (see PHASE_PLAN Phase 5 notes).

Objectives (all minimized):
  f_radar : mean reflection loss (dB) over X+Ku (8-18 GHz)   -> want <= -10
  f_ir    : LWIR emissivity (8-14 um), VO2 insulating state  -> want < 0.3
  f_vis   : deltaE vs target background                      -> want < 5
Constraint:
  weight per area <= cap (default 10 kg/m^2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import load_targets
from ..physics import optics, radar

# Fixed material choices for this design family (spacer = SiO2: eps_r for radar, n,k for optics).
_SPACER_EPS_R = 3.9
_OPTICAL_SPACER_UM = 1.0      # thin optical interface beneath VO2 (see scale-separation note)
_GROUND_UM = 0.3

# Mass densities (kg/m^3) for the weight budget.
_DENSITY = {
    "ec": 1000.0,        # PEDOT:PSS ~1.0 g/cc
    "vo2": 4600.0,       # VO2
    "spacer": 2200.0,    # SiO2
    "ground": 2700.0,    # Al
}

# Joint design vector: unit cube [0,1]^6 -> physical params.
DESIGN_BOUNDS = {
    "t_ec_um": (0.05, 0.50),            # electrochromic polymer thickness (visible)
    "t_vo2_um": (0.05, 0.50),           # VO2 thickness (IR)
    "spacer_thickness_mm": (0.5, 5.0),  # radar grounded dielectric
    "radar_period_mm": (3.0, 8.0),
    "radar_patch_frac": (0.70, 0.97),
    "radar_sheet_resistance_ohm_sq": (30.0, 400.0),
}
PARAM_ORDER = tuple(DESIGN_BOUNDS)
N_DIM = len(PARAM_ORDER)


@dataclass(frozen=True)
class Design:
    t_ec_um: float
    t_vo2_um: float
    spacer_thickness_mm: float
    radar_period_mm: float
    radar_patch_frac: float
    radar_sheet_resistance_ohm_sq: float


def design_from_unit(u: np.ndarray) -> Design:
    """Map [0,1]^6 (PARAM_ORDER) to a physical Design."""
    u = np.asarray(u, dtype=float).ravel()
    lo_hi = np.array([DESIGN_BOUNDS[p] for p in PARAM_ORDER])
    vals = lo_hi[:, 0] + u * (lo_hi[:, 1] - lo_hi[:, 0])
    return Design(*(float(v) for v in vals))


def _radar_stack(d: Design) -> radar.RadarStack:
    return radar.RadarStack(
        period_mm=d.radar_period_mm,
        patch_mm=d.radar_period_mm * d.radar_patch_frac,
        sheet_resistance_ohm_sq=d.radar_sheet_resistance_ohm_sq,
        spacer_thickness_mm=d.spacer_thickness_mm,
        spacer_eps_r=_SPACER_EPS_R,
        pattern="capacitive_patch",
    )


_VO2_STATES = ("VO2 (insulating, <Tc)", "VO2 (metallic, >Tc)")


def _ir_stack(d: Design, vo2_state: str) -> optics.Stack:
    # Thin EC polymer treated as IR-transparent (data gap); IR set by VO2/spacer/ground.
    return optics.Stack.of(
        optics.Layer(vo2_state, d.t_vo2_um),
        optics.Layer("SiO2 (IR dielectric)", _OPTICAL_SPACER_UM),
        optics.Layer("Aluminum (ground plane)", _GROUND_UM),
    )


def _visible_stack(d: Design) -> optics.Stack:
    return optics.Stack.of(
        optics.Layer("PEDOT:PSS", d.t_ec_um),
        optics.Layer("VO2 (insulating, <Tc)", d.t_vo2_um),
        optics.Layer("SiO2 (IR dielectric)", _OPTICAL_SPACER_UM),
        optics.Layer("Aluminum (ground plane)", _GROUND_UM),
    )


def weight_kg_m2(d: Design) -> float:
    return (
        d.t_ec_um * 1e-6 * _DENSITY["ec"]
        + d.t_vo2_um * 1e-6 * _DENSITY["vo2"]
        + _OPTICAL_SPACER_UM * 1e-6 * _DENSITY["spacer"]
        + d.spacer_thickness_mm * 1e-3 * _DENSITY["spacer"]
        + _GROUND_UM * 1e-6 * _DENSITY["ground"]
    )


def evaluate(u: np.ndarray, targets: dict | None = None) -> dict:
    """Evaluate one unit-cube design -> objectives, constraint, and diagnostics."""
    t = targets or load_targets()
    d = design_from_unit(u)

    # Radar: mean reflection loss over X + Ku (the worst/least-absorbing matters too).
    rstack = _radar_stack(d)
    x_band, ku_band = t["radar"]["bands_ghz"]["X"], t["radar"]["bands_ghz"]["Ku"]
    f_ghz = np.linspace(x_band[0], ku_band[1], 41)
    rl = radar.spectrum(rstack, f_ghz)["reflection_loss_db"]
    f_radar = float(np.mean(rl))
    # X-band specifically (the primary threat a single-layer absorber can realistically cover).
    rl_x = radar.spectrum(rstack, np.linspace(x_band[0], x_band[1], 21))["reflection_loss_db"]
    radar_x_worst = float(np.max(rl_x))

    # IR: LWIR emissivity = best achievable VO2 state (material is reconfigurable).
    lwir = tuple(t["ir_lwir"]["band_um"])
    ir_by_state = {s: optics.band_emissivity(_ir_stack(d, s), lwir) for s in _VO2_STATES}
    ir_state = min(ir_by_state, key=ir_by_state.get)
    f_ir = ir_by_state[ir_state]

    # Visible: deltaE vs background.
    bg = t["visible"]["background"]["srgb_hex"]
    f_vis = optics.delta_e_vs_background(_visible_stack(d), bg)

    w = weight_kg_m2(d)
    return {
        "f_radar_meandb": f_radar,
        "f_ir_emissivity": float(f_ir),
        "f_vis_deltaE": float(f_vis),
        "weight_kg_m2": w,
        "radar_worst_db": float(np.max(rl)),
        "radar_best_db": float(np.min(rl)),
        "radar_x_worst_db": radar_x_worst,
        "ir_state": ir_state,
        "weight_ok": w <= t["physical"]["weight_kg_per_m2"]["threshold"],
        "design": d,
    }


def meets_all_targets(metrics: dict, targets: dict | None = None) -> bool:
    """True if a design meets every band threshold and the weight cap."""
    t = targets or load_targets()
    return (
        metrics["radar_worst_db"] <= t["radar"]["threshold_db"]
        and metrics["f_ir_emissivity"] < t["ir_lwir"]["threshold"]
        and metrics["f_vis_deltaE"] < t["visible"]["threshold"]
        and metrics["weight_ok"]
    )
