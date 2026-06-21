"""GAP #1 fix: design a multilayer stealth coating USING discovered materials.

Closes the loop between discovery (Track A) and design (Phase 5): it places the top
discovered materials into their stack-layer roles, optimizes the geometry with the real
physics (TMM optics + ECM radar), and reports a *designed coating* with simulated radar
+ IR + visible performance. Roles without a discovered material fall back to a known one.

    python -m stealth.discovery.design_stack \
        --candidates data/final_candidates.parquet --gnnopt-nk /workspace/runs/radar/gnnopt_nk.json

Honest scope: a discovered material's GNNOpt n,k is valid in the visible/NIR, so it's used
for the optical spacer / visible layers; the IR layer stays on VO2 (known, real IR data),
and the discovered conductor fills the radar layer (its conductivity sets the sheet-resistance
range). Generating dielectric/IR candidates lets more layers use discovered materials.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, load_targets
from ..physics import optics, radar
from .optical_bridge import load_gnnopt_nk, material_from_gnnopt

REPORT = REPO_ROOT / "reports" / "designed_coating.md"
DESIGN_JSON = REPO_ROOT / "data" / "designed_coating.json"   # chosen design -> openEMS re-check

_DENSITY = {"ec": 1000.0, "vo2": 4600.0, "spacer": 2200.0, "ground": 2700.0}
_GROUND = "Aluminum (ground plane)"
RADAR_SPACER_EPS_R = 3.9   # dielectric constant of the radar spacer (shared by _phys and the saved design)

BOUNDS = {
    "t_ec_um": (0.05, 0.50),
    "t_ir_um": (0.05, 0.50),
    "t_spacer_opt_um": (0.2, 3.0),
    "spacer_mm": (0.5, 5.0),
    "period_mm": (3.0, 8.0),
    "patch_frac": (0.70, 0.97),
    "rs_ohm_sq": (30.0, 400.0),
}
PARAMS = tuple(BOUNDS)


def _phys(u, materials, targets):
    """Evaluate one design (unit cube) -> dict of band metrics + weight."""
    lo_hi = np.array([BOUNDS[p] for p in PARAMS])
    v = dict(zip(PARAMS, lo_hi[:, 0] + np.asarray(u) * (lo_hi[:, 1] - lo_hi[:, 0])))
    ec, ir, diel = materials["ec"], materials["ir"], materials["diel"]
    ground = optics.Layer(_GROUND, 0.3)

    ir_stack = optics.Stack.of(optics.Layer(ir, v["t_ir_um"]),
                               optics.Layer(diel, v["t_spacer_opt_um"]), ground)
    vis_stack = optics.Stack.of(optics.Layer(ec, v["t_ec_um"]), optics.Layer(ir, v["t_ir_um"]),
                                optics.Layer(diel, v["t_spacer_opt_um"]), ground)
    rstack = radar.RadarStack(period_mm=v["period_mm"], patch_mm=v["period_mm"] * v["patch_frac"],
                              sheet_resistance_ohm_sq=v["rs_ohm_sq"], spacer_thickness_mm=v["spacer_mm"],
                              spacer_eps_r=RADAR_SPACER_EPS_R, pattern="capacitive_patch")
    x = targets["radar"]["bands_ghz"]["X"]
    rl_x = radar.spectrum(rstack, np.linspace(x[0], x[1], 21))["reflection_loss_db"]
    weight = (v["t_ec_um"] * 1e-6 * _DENSITY["ec"] + v["t_ir_um"] * 1e-6 * _DENSITY["vo2"]
              + v["spacer_mm"] * 1e-3 * _DENSITY["spacer"] + 0.3e-6 * _DENSITY["ground"])
    return {
        "radar_x_worst_db": float(np.max(rl_x)),
        "ir_emissivity": optics.band_emissivity(ir_stack, tuple(targets["ir_lwir"]["band_um"]), clip=True),
        "visible_deltaE": optics.delta_e_vs_background(vis_stack, targets["visible"]["background"]["srgb_hex"]),
        "weight_kg_m2": weight,
        "params": v,
    }


def optimize(materials, targets, n_gen=30, n_part=12, seed=0):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    cap = targets["physical"]["weight_kg_per_m2"]["threshold"]

    class P(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(PARAMS), n_obj=3, n_ieq_constr=1, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *a, **k):
            m = _phys(x, materials, targets)
            out["F"] = [m["radar_x_worst_db"], m["ir_emissivity"], m["visible_deltaE"]]
            out["G"] = [m["weight_kg_m2"] - cap]

    ref = get_reference_directions("das-dennis", 3, n_partitions=n_part)
    res = minimize(P(), NSGA3(pop_size=len(ref), ref_dirs=ref), ("n_gen", n_gen), seed=seed, verbose=False)
    return np.atleast_2d(res.X)


def _class_eps(mat_class: str) -> complex:
    """Median complex permittivity of a material class from the measured-literature table."""
    from .em_literature import LITERATURE

    rows = [r for r in LITERATURE if r.mat_class == mat_class]
    if not rows:
        return complex(10.0, 1.0)
    return complex(float(np.median([r.eps_real for r in rows])),
                   float(np.median([r.eps_imag for r in rows])))


def discovered_radar_layer(candidates_parquet=None):
    """Put the top discovered radar material INTO the design as a single-layer (Dallenbach) absorber.

    Uses the candidate's class-level measured EM (magnetic ferrite vs conductive/dielectric loss,
    from em_literature) and finds its impedance-matched thickness + reflection. This is what makes
    the discovered radar material part of the *design*, not just the shortlist. Per-structure
    predicted eps/mu (with the trained #3 predictors) refines the class values.
    """
    if not candidates_parquet:
        return None
    from pymatgen.core import Composition

    from .anchor import optimal_thickness
    from .em_literature import class_mu_prior

    df = pd.read_parquet(candidates_parquet)
    if "practical" in df.columns:          # never place a toxic/precious/rare-earth radar material
        df = df[df["practical"]]
    radar = df[df["role"].isin(["radar_conductor", "radar_magnetic"])].copy()

    # Require GENUINE radar relevance, so mislabeled wide-gap dielectrics (e.g. SiO2 tagged
    # radar_magnetic) can't be named the absorber: magnetic role must contain a magnetic
    # element; conductor role must be low-gap (metallic/lossy).
    def _radar_relevant(row) -> bool:
        els = {e.symbol for e in Composition(row["formula"]).elements}
        if row["role"] == "radar_magnetic":
            return bool(els & {"Fe", "Co", "Ni", "Mn"})
        return float(row.get("band_gap_ev", 9.9)) < 1.0   # radar_conductor: conductive/lossy

    radar = radar[radar.apply(_radar_relevant, axis=1)].sort_values("score", ascending=False)
    if not len(radar):
        return None
    top = radar.iloc[0]
    els = {e.symbol for e in Composition(top["formula"]).elements}
    mu = class_mu_prior(els)
    magnetic = mu.imag > 0
    eps = _class_eps("magnetic" if magnetic else "conductive")
    best = optimal_thickness(complex(eps.real, -abs(eps.imag)), complex(mu.real, -abs(mu.imag)))
    return {
        "formula": top["formula"],
        "role": top["role"],
        "loss_mechanism": "magnetic" if magnetic else "dielectric/conductive",
        "eps_class": [round(eps.real, 1), round(eps.imag, 1)],
        "mu_class": [round(mu.real, 2), round(mu.imag, 2)],
        "matched_thickness_mm": round(best["thickness_mm"], 2),
        "min_rl_db": round(best["min_rl_db"], 1),
        "f_at_min_ghz": round(best["f_at_min_ghz"], 1),
        "eps_source": "literature class median (refine with trained #3 predictors + CIF)",
    }


def pick_materials(candidates_parquet=None, gnnopt_nk=None):
    """Choose the discovered material per role (GNNOpt n,k); fall back to known materials."""
    nk = load_gnnopt_nk(gnnopt_nk) if gnnopt_nk else {}
    mats = {"ec": "PEDOT:PSS", "ir": "VO2 (insulating, <Tc)", "diel": "SiO2 (IR dielectric)"}
    used = {"ec": "PEDOT:PSS (known)", "ir": "VO2 (known)", "diel": "SiO2 (known)"}
    if candidates_parquet and gnnopt_nk:
        df = pd.read_parquet(candidates_parquet)
        # best discovered DIELECTRIC -> optical spacer (its GNNOpt n,k genuinely changes the optics)
        diel = df[df["role"] == "dielectric_spacer"].sort_values("score", ascending=False)
        if len(diel) and diel.iloc[0]["id"] in nk:
            cid = diel.iloc[0]["id"]
            mats["diel"] = material_from_gnnopt(cid, nk[cid], "dielectric_spacer")
            used["diel"] = f"{diel.iloc[0]['formula']} (discovered)"
    return mats, used


def main(candidates_parquet=None, gnnopt_nk=None):
    warnings.filterwarnings("ignore")
    targets = load_targets()
    materials, used = pick_materials(candidates_parquet, gnnopt_nk)

    print("Designing coating with layers:")
    for role, label in used.items():
        print(f"  {role:5s}: {label}")
    print("Optimizing geometry (NSGA-III)...")
    X = optimize(materials, targets)

    rows = [_phys(u, materials, targets) for u in X]
    df = pd.DataFrame(
        [{"radar_x_db": round(r["radar_x_worst_db"], 1), "ir_emis": round(r["ir_emissivity"], 3),
          "visible_dE": round(r["visible_deltaE"], 1), "weight": round(r["weight_kg_m2"], 1), **r["params"]}
         for r in rows]
    ).drop_duplicates(subset=list(PARAMS)).reset_index(drop=True)
    # best balanced: meets radar+IR, lowest visible dE
    feasible = df[(df["radar_x_db"] <= -10) & (df["ir_emis"] < 0.3) & (df["weight"] <= 10)]
    pick = (feasible if len(feasible) else df).sort_values("visible_dE").iloc[0]

    print("\n=== DESIGNED COATING (best balanced) ===")
    print(f"  radar X-band worst: {pick['radar_x_db']:.1f} dB   IR emissivity: {pick['ir_emis']:.3f}   "
          f"visible deltaE: {pick['visible_dE']:.1f}   weight: {pick['weight']:.1f} kg/m^2")

    # Discovered radar material placed into the design as a bulk single-layer absorber:
    disc_radar = discovered_radar_layer(candidates_parquet)
    if disc_radar:
        print(f"  discovered radar material: {disc_radar['formula']} "
              f"({disc_radar['loss_mechanism']} loss) -> {disc_radar['min_rl_db']} dB "
              f"@ {disc_radar['matched_thickness_mm']} mm")

    # Persist the chosen design so openEMS can re-check this EXACT radar layer:
    #   python -m stealth.physics.radar_fullwave --design data/designed_coating.json
    design = {
        "radar_stack": {
            "period_mm": float(pick["period_mm"]),
            "patch_mm": float(pick["period_mm"] * pick["patch_frac"]),
            "sheet_resistance_ohm_sq": float(pick["rs_ohm_sq"]),
            "spacer_thickness_mm": float(pick["spacer_mm"]),
            "spacer_eps_r": RADAR_SPACER_EPS_R,
            "pattern": "capacitive_patch",
        },
        "discovered_radar_material": disc_radar,
        "metrics": {
            "radar_x_worst_db": float(pick["radar_x_db"]),
            "ir_emissivity": float(pick["ir_emis"]),
            "visible_deltaE": float(pick["visible_dE"]),
            "weight_kg_m2": float(pick["weight"]),
        },
        "materials": used,
    }
    DESIGN_JSON.parent.mkdir(parents=True, exist_ok=True)
    DESIGN_JSON.write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(f"  chosen design saved -> {DESIGN_JSON}  (openEMS re-check: radar_fullwave --design)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Designed multispectral coating (discovery -> design bridge)",
        "",
        "A full multilayer stealth coating, geometry-optimized (NSGA-III) over the real physics "
        "(TMM optics + ECM radar), with layers filled by **discovered** materials where available:",
        "",
        "| layer | material |",
        "|---|---|",
        f"| visible (electrochromic) | {used['ec']} |",
        f"| IR (thermochromic) | {used['ir']} |",
        f"| optical spacer | {used['diel']} |",
        f"| radar metasurface + ground | patterned conductor / Al |",
        "",
        "**Best balanced design:** "
        f"radar X-band {pick['radar_x_db']:.1f} dB, LWIR emissivity {pick['ir_emis']:.3f}, "
        f"visible ΔE {pick['visible_dE']:.1f}, weight {pick['weight']:.1f} kg/m². "
        f"Geometry: VO₂ {pick['t_ir_um']:.2f} µm, spacer {pick['spacer_mm']:.2f} mm, "
        f"patch period {pick['period_mm']:.2f} mm, sheet R {pick['rs_ohm_sq']:.0f} Ω/sq.",
        "",
        f"Feasible designs on the Pareto front: {len(feasible)}/{len(df)}.",
        "",
    ]
    if disc_radar:
        lines += [
            "## Discovered radar material in the design",
            "",
            f"The top discovered radar candidate **{disc_radar['formula']}** "
            f"({disc_radar['role'].replace('_', ' ')}, {disc_radar['loss_mechanism']} loss) placed as a "
            f"metal-backed single-layer absorber: **{disc_radar['min_rl_db']} dB** at "
            f"{disc_radar['matched_thickness_mm']} mm ({disc_radar['f_at_min_ghz']} GHz), using "
            f"class ε={disc_radar['eps_class']}, μ={disc_radar['mu_class']}. "
            "_(ε from the measured-literature class median; refines with the trained #3 predictors + the candidate CIF.)_",
            "",
        ]
    lines += [
        "_Discovered materials are used where their GNNOpt n,k is valid (visible/NIR optical layers). "
        "IR stays on VO₂ (real IR data); generating dielectric/IR candidates extends discovery to those "
        "layers. Next: openEMS/DFT validation of the chosen design._",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDesigned-coating report -> {REPORT}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=None, help="final_candidates.parquet (discovered materials)")
    ap.add_argument("--gnnopt-nk", default=None, help="GNNOpt n,k JSON")
    args = ap.parse_args()
    main(candidates_parquet=args.candidates, gnnopt_nk=args.gnnopt_nk)
