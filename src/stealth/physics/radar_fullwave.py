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


C0_MMHZ = 299_792_458.0 * 1e3  # speed of light in mm*Hz (CSX unit is mm)
_SHEET_T_MM = 0.035            # modeled thickness of the resistive sheet (typical foil)
_NOT_AVAIL = (
    "openEMS is not installed (expected on Windows — it runs on the Linux pod). "
    "Install with scripts/setup_openems.sh, then run inside the 'openems' conda env."
)


def simulate(stack: RadarStack, f_ghz: np.ndarray = FREQ_GHZ) -> dict:
    """Full-wave FDTD reflection spectrum of the absorber unit cell (pod/cluster only).

    Method: normal-incidence reflection off the periodic grounded absorber, modeled as a
    parallel-plate TEM unit cell — E along x with PEC walls (x), PMC walls (y), PEC ground
    (z-) and an absorbing top (z+). A TEM waveguide port at the top excites the plane wave
    and reads S11. Returns the same dict shape as `radar.spectrum`.

    Untested on Windows (no openEMS) — three spots to validate/tune on the pod are flagged
    [TUNE]: the boundary polarization, the port mode functions, and the mesh density.
    """
    if not openems_available():
        raise RuntimeError(_NOT_AVAIL)

    import tempfile

    from CSXCAD import ContinuousStructure
    from openEMS import openEMS

    f = np.atleast_1d(np.asarray(f_ghz, dtype=float))
    f0 = float(f.mean()) * 1e9
    fc = max(float(f.max() - f.min()), 5.0) / 2.0 * 1.2 * 1e9
    D, d, p = stack.period_mm, stack.spacer_thickness_mm, stack.patch_mm
    air = 40.0
    z_port = d + 0.9 * air

    FDTD = openEMS(EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    # [TUNE] E-field along x -> PEC on x-walls, PMC on y-walls; PEC ground (z-), MUR top (z+).
    FDTD.SetBoundaryCond(["PEC", "PEC", "PMC", "PMC", "PEC", "MUR"])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)  # mm
    res = (C0_MMHZ / (f0 + fc)) / 20.0          # ~lambda/20 (mm)  [TUNE]
    mesh.AddLine("x", [-D / 2, D / 2])
    mesh.AddLine("y", [-D / 2, D / 2])
    mesh.AddLine("z", [0, d, z_port, d + air])
    for ax in "xy":
        mesh.SmoothMeshLines(ax, res)
    mesh.SmoothMeshLines("z", min(res, d / 10.0))   # >= ~10 cells through the spacer

    # grounded dielectric spacer
    sp = CSX.AddMaterial("spacer", epsilon=stack.spacer_eps_r)
    sp.AddBox([-D / 2, -D / 2, 0], [D / 2, D / 2, d])
    # resistive patch as a conducting sheet (sigma from target sheet resistance)
    sigma = 1.0 / (stack.sheet_resistance_ohm_sq * _SHEET_T_MM * 1e-3)
    patch = CSX.AddConductingSheet("patch", conductivity=sigma, thickness=_SHEET_T_MM * 1e-3)
    patch.AddBox([-p / 2, -p / 2, d], [p / 2, p / 2, d])
    gnd = CSX.AddMetal("gnd")
    gnd.AddBox([-D / 2, -D / 2, 0], [D / 2, D / 2, 0])

    # [TUNE] TEM port at the top: E along x, H along -y; kc=0 (TEM), unit amplitude.
    port = FDTD.AddWaveGuidePort(
        0, [-D / 2, -D / 2, z_port], [D / 2, D / 2, z_port], "z",
        ["1", "0", "0"], ["0", "-1", "0"], 0, 1.0,
    )

    sim_path = tempfile.mkdtemp(prefix="oems_")
    FDTD.Run(sim_path, cleanup=True)

    fr = f * 1e9
    port.CalcPort(sim_path, fr)
    s11 = np.asarray(port.uf_ref) / np.asarray(port.uf_inc)
    mag = np.clip(np.abs(s11), 1e-6, None)
    return {
        "f_ghz": f,
        "reflection_loss_db": 20 * np.log10(mag),
        "absorption": 1.0 - mag ** 2,
    }


def compare(stack: RadarStack | None = None, f_ghz: np.ndarray | None = None) -> "object":
    """Run openEMS and the ECM on the same design and tabulate the difference."""
    import pandas as pd

    from . import radar

    stack = stack or RadarStack(6.0, 5.4, 120.0, 2.5, 4.3, "capacitive_patch")
    f = FREQ_GHZ if f_ghz is None else np.asarray(f_ghz, dtype=float)
    ems = simulate(stack, f)["reflection_loss_db"]
    ecm = radar.spectrum(stack, f)["reflection_loss_db"]
    df = pd.DataFrame({"f_ghz": f, "openems_db": np.round(ems, 2), "ecm_db": np.round(ecm, 2),
                       "diff_db": np.round(ems - ecm, 2)})
    print(df.to_string(index=False))
    print(f"\nMean |openEMS - ECM| = {np.mean(np.abs(ems - ecm)):.2f} dB   "
          f"(min openEMS {ems.min():.1f} dB @ {f[np.argmin(ems)]:.1f} GHz)")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="run openEMS vs ECM on a test design")
    args = ap.parse_args()
    demo = RadarStack(6, 5.4, 120, 2.5, 4.3, "capacitive_patch")
    print("openEMS available:", openems_available())
    print("geometry spec:", geometry_spec(demo))
    if args.compare:
        compare(demo)
