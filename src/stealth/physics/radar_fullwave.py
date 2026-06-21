"""openEMS (FDTD) ground-truth check for the radar absorber — runs on the cluster.

The ECM forward model (`radar.py`) approximates the patterned layer as a lumped
sheet impedance. openEMS simulates the *actual* patterned geometry with a PEC
ground and periodic (Floquet) boundaries, giving the high-fidelity reflection
spectrum we validate the ECM and surrogate against.

openEMS does not install cleanly on Windows; this module is intended for the Linux
cluster. It builds the geometry from a `RadarStack` and runs FDTD, raising a clear
message if openEMS isn't available so nothing silently fakes a result.

FDTD setup (documented so the cluster run is reproducible):
  * Unit cell = one period; normal-incidence TEM walls (PEC on the x-walls ‖ E,
    PMC on the y-walls ‖ H) make one period behave like an infinite periodic array.
  * PEC ground = the zmax boundary behind the spacer; MUR absorbing boundary at zmin.
  * Capacitive patch = resistive sheet of size patch_mm centered in the cell.
  * Resistive sheet modeled via a conducting sheet whose conductivity gives the target
    sheet resistance (ohm/sq).
  * TEM waveguide port fires +z toward the absorber and reads S11 (sweep 1-30 GHz).
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
        "ground": "PEC (zmax boundary, behind the spacer)",
        "lateral_bc": "PEC x-walls / PMC y-walls (normal-incidence TEM unit cell)",
        "open_bc": "MUR absorbing (zmin, below the +z TEM port)",
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
    parallel-plate **TEM** unit cell. For a linearly-polarized normal-incidence plane wave
    with E along x and H along y propagating +z, image theory makes the cell walls:
      * x-walls (normal ‖ E)  -> PEC   (n×E = 0, n·H = 0 both satisfied)
      * y-walls (normal ‖ H)  -> PMC
    so the unit cell is a square parallel-plate TEM waveguide. Layout along +z (matching the
    proven openEMS Rect_Waveguide port convention — excite at low z, propagate +z toward the
    structure):

        z = 0           MUR open boundary (absorbs the reflected wave below the port)
        z = port (slab) TEM waveguide port, fires +z, reads S11
        ...air gap...
        z = z_patch     resistive capacitive patch (conducting sheet)
        z = z_patch+d   dielectric spacer up to here
        z = z_top       PEC ground  ==  the zmax boundary (no separate metal box needed)

    For a square D×D cell the port's measured V/I equals the analytic TEM impedance
    ZL = Z0, so S11 = uf_ref/uf_inc is the true reflection coefficient with no manual
    reference impedance. Returns the same dict shape as `radar.spectrum`.

    Validated against the ECM (`radar.spectrum`) via `compare()`; if they disagree, the
    knobs are the lateral mesh resolution and the cells-through-spacer count below.
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

    res = (C0_MMHZ / (f0 + fc)) / 20.0          # ~lambda/20 (mm) at the highest excited freq
    res_z = min(res, d / 12.0)                  # >= ~12 cells through the spacer
    air = max(30.0, 8.0 * res)                  # air column between the open boundary and patch
    z_patch = air                               # patch sits at the top of the air column
    z_top = air + d                             # spacer top == PEC ground (zmax boundary)
    port_z0 = 0.35 * air                        # TEM port slab, well inside the air region
    port_z1 = port_z0 + max(5.0 * res_z, 1.0)

    FDTD = openEMS(EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    # E ‖ x  ->  PEC x-walls, PMC y-walls; MUR open below the port, PEC ground at the top.
    FDTD.SetBoundaryCond(["PEC", "PEC", "PMC", "PMC", "MUR", "PEC"])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)  # mm
    # Fixed lines at the patch edges (±p/2) so the conducting sheet is actually meshed
    # (otherwise openEMS reports "Unused primitive ... patch!" and drops the absorber).
    mesh.AddLine("x", [-D / 2, -p / 2, p / 2, D / 2])
    mesh.AddLine("y", [-D / 2, -p / 2, p / 2, D / 2])
    mesh.AddLine("z", [0, port_z0, port_z1, z_patch, z_top])
    mesh.SmoothMeshLines("x", res)
    mesh.SmoothMeshLines("y", res)
    mesh.SmoothMeshLines("z", res_z)

    # grounded dielectric spacer (ground = the PEC zmax boundary at z_top)
    sp = CSX.AddMaterial("spacer", epsilon=stack.spacer_eps_r)
    sp.AddBox([-D / 2, -D / 2, z_patch], [D / 2, D / 2, z_top], priority=1)
    # resistive capacitive patch as a conducting sheet (sigma from target sheet resistance)
    sigma = 1.0 / (stack.sheet_resistance_ohm_sq * _SHEET_T_MM * 1e-3)
    patch = CSX.AddConductingSheet("patch", conductivity=sigma, thickness=_SHEET_T_MM * 1e-3)
    patch.AddBox([-p / 2, -p / 2, z_patch], [p / 2, p / 2, z_patch], priority=10)

    # TEM port firing +z toward the absorber: E along +x, H along +y (E×H ‖ +z), kc=0.
    port = FDTD.AddWaveGuidePort(
        0, [-D / 2, -D / 2, port_z0], [D / 2, D / 2, port_z1], "z",
        ["1", "0", "0"], ["0", "1", "0"], 0, 1.0,
    )

    sim_path = tempfile.mkdtemp(prefix="oems_")
    FDTD.Run(sim_path, cleanup=True)

    fr = f * 1e9
    port.CalcPort(sim_path, fr)             # kc=0 -> analytic ZL = Z0 used for the inc/ref split
    s11 = np.asarray(port.uf_ref) / np.asarray(port.uf_inc)
    mag = np.clip(np.abs(s11), 1e-6, None)
    return {
        "f_ghz": f,
        "reflection_loss_db": 20 * np.log10(mag),
        "absorption": 1.0 - mag ** 2,
    }


def stack_from_design(path: str) -> RadarStack:
    """Build the RadarStack from a design JSON saved by discovery.design_stack."""
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text(encoding="utf-8"))["radar_stack"]
    return RadarStack(
        period_mm=d["period_mm"], patch_mm=d["patch_mm"],
        sheet_resistance_ohm_sq=d["sheet_resistance_ohm_sq"],
        spacer_thickness_mm=d["spacer_thickness_mm"],
        spacer_eps_r=d.get("spacer_eps_r", 3.9),
        pattern=d.get("pattern", "capacitive_patch"),
    )


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

    # Save a summary so the deliverable can cite the per-design full-wave confirmation.
    # (Local REPO_ROOT — avoids importing stealth.config, which may need deps absent in the
    # openEMS venv.)
    import json
    from pathlib import Path

    out = Path(__file__).resolve().parents[3] / "data" / "openems_design.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "mean_abs_diff_db": round(float(np.mean(np.abs(ems - ecm))), 2),
        "min_rl_db": round(float(ems.min()), 1),
        "f_at_min_ghz": round(float(f[np.argmin(ems)]), 1),
    }, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="run openEMS vs ECM on a test design")
    ap.add_argument("--design", default=None,
                    help="design JSON from discovery.design_stack; openEMS-checks that exact radar layer")
    args = ap.parse_args()
    stack = stack_from_design(args.design) if args.design else RadarStack(6, 5.4, 120, 2.5, 4.3, "capacitive_patch")
    print("openEMS available:", openems_available())
    print("geometry spec:", geometry_spec(stack))
    if args.design and not args.compare:
        print(f"(loaded design from {args.design}; add --compare to run openEMS vs ECM)")
    if args.compare or args.design:
        compare(stack)
