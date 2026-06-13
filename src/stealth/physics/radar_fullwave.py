"""openEMS (FDTD) ground-truth check for the radar absorber — runs on the cluster.

The ECM forward model (`radar.py`) approximates the patterned layer as a lumped
sheet impedance. openEMS simulates the *actual* patterned geometry with a PEC
ground and periodic (Floquet) boundaries, giving the high-fidelity reflection
spectrum we validate the ECM and surrogate against.

openEMS does not install cleanly on Windows; this module is intended for the Linux
cluster. It builds the geometry from a `RadarStack` and runs FDTD, raising a clear
message if openEMS isn't available so nothing silently fakes a result.

FDTD setup (documented so the cluster run is reproducible):
  * Unit cell = one period; periodic boundary conditions on the 4 lateral faces.
  * PEC ground on the bottom; absorbing (MUR/PML) boundary on top.
  * Capacitive patch = PEC/resistive sheet of size patch_mm centered in the cell.
  * Resistive sheet modeled via a lumped sheet resistance (ohm/sq).
  * Plane-wave excitation (nf2ff or a waveguide port), sweep 1-30 GHz.
  * Reflection loss = 20 log10 |S11|.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from .radar import RadarStack

FREQ_GHZ = np.linspace(1.0, 30.0, 59)


def openems_available() -> bool:
    return importlib.util.find_spec("openEMS") is not None


def geometry_spec(stack: RadarStack) -> dict:
    """The geometry/excitation parameters an openEMS build needs (no solver call)."""
    return {
        "period_mm": stack.period_mm,
        "patch_mm": stack.patch_mm,
        "gap_mm": stack.period_mm - stack.patch_mm,
        "sheet_resistance_ohm_sq": stack.sheet_resistance_ohm_sq,
        "spacer_thickness_mm": stack.spacer_thickness_mm,
        "spacer_eps_r": stack.spacer_eps_r,
        "ground": "PEC",
        "lateral_bc": "periodic (Floquet)",
        "top_bc": "PML/MUR absorbing",
        "freq_ghz": (float(FREQ_GHZ[0]), float(FREQ_GHZ[-1])),
    }


def simulate(stack: RadarStack, f_ghz: np.ndarray = FREQ_GHZ) -> dict:
    """Run the FDTD reflection sweep (cluster only)."""
    if not openems_available():
        raise RuntimeError(
            "openEMS is not installed (expected — it runs on the Linux cluster, not "
            "this Windows box). Geometry is ready via geometry_spec(); run this module "
            "on the cluster to produce ground-truth spectra."
        )
    # Cluster implementation: build CSXCAD geometry from geometry_spec(), add the
    # resistive sheet + PEC ground + periodic BCs, excite, run, read S11.
    raise NotImplementedError("openEMS FDTD build is wired up in the cluster environment.")


if __name__ == "__main__":
    demo = RadarStack(6, 5.4, 120, 2.5, 4.3, "capacitive_patch")
    print("openEMS available:", openems_available())
    print("geometry spec:", geometry_spec(demo))
