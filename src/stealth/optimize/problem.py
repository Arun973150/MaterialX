"""Phase 5 multi-objective optimization (NSGA-III over the joint objective).

Searches the 6-D stack design space for the Pareto front trading radar absorption
vs IR emissivity vs visible color, subject to the weight cap. Outputs the full
front plus a ranked candidate shortlist.

The physics evaluators (optics TMM + radar ECM) are millisecond-fast, so NSGA-III
searches them directly — no ML surrogate needed here. (A learned surrogate becomes
worthwhile only to emulate slow openEMS full-wave on the cluster; see Phase 4.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, load_targets
from .objective import N_DIM, PARAM_ORDER, design_from_unit, evaluate, meets_all_targets

PARETO_OUT = REPO_ROOT / "data" / "pareto_front.parquet"
REPORT_OUT = REPO_ROOT / "reports" / "candidates.md"


def _problem(targets: dict):
    from pymoo.core.problem import ElementwiseProblem

    cap = targets["physical"]["weight_kg_per_m2"]["threshold"]

    class StealthProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=N_DIM, n_obj=3, n_ieq_constr=1, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            m = evaluate(x, targets)
            # Optimize X-band worst-case (drives a real absorber); Ku is reported, not chased
            # (full X+Ku is a multilayer problem). IR = best VO2 state, visible = deltaE.
            out["F"] = [m["radar_x_worst_db"], m["f_ir_emissivity"], m["f_vis_deltaE"]]
            out["G"] = [m["weight_kg_m2"] - cap]   # <= 0 is feasible

    return StealthProblem()


def run_optimization(n_gen: int = 40, n_partitions: int = 12, seed: int = 0):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    targets = load_targets()
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    algorithm = NSGA3(pop_size=len(ref_dirs), ref_dirs=ref_dirs)
    res = minimize(_problem(targets), algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    return res, targets


def front_to_frame(res, targets: dict) -> pd.DataFrame:
    """Re-evaluate every Pareto design to a full metrics table, ranked."""
    X = np.atleast_2d(res.X)
    rows = []
    for u in X:
        m = evaluate(u, targets)
        d = m["design"]
        rows.append(
            {
                **{p: getattr(d, p) for p in PARAM_ORDER},
                "radar_worst_db": m["radar_worst_db"],
                "radar_x_worst_db": m["radar_x_worst_db"],
                "radar_mean_db": m["f_radar_meandb"],
                "ir_emissivity": m["f_ir_emissivity"],
                "ir_state": m["ir_state"],
                "visible_deltaE": m["f_vis_deltaE"],
                "weight_kg_m2": m["weight_kg_m2"],
                "radar_ok": m["radar_worst_db"] <= targets["radar"]["threshold_db"],
                "radar_x_ok": m["radar_x_worst_db"] <= targets["radar"]["threshold_db"],
                "ir_ok": m["f_ir_emissivity"] < targets["ir_lwir"]["threshold"],
                "vis_ok": m["f_vis_deltaE"] < targets["visible"]["threshold"],
                "all_targets_met": meets_all_targets(m, targets),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=list(PARAM_ORDER)).reset_index(drop=True)
    # Realistic compliance counts X-band radar (the optimized objective), not full X+Ku.
    df["targets_met"] = df[["radar_x_ok", "ir_ok", "vis_ok"]].sum(axis=1)
    # Rank: most targets met, then best (lowest) normalized objective sum.
    for col in ("radar_x_worst_db", "ir_emissivity", "visible_deltaE"):
        df[f"_n_{col}"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
    df["_score"] = df[["_n_radar_x_worst_db", "_n_ir_emissivity", "_n_visible_deltaE"]].sum(axis=1)
    df = df.sort_values(["targets_met", "_score"], ascending=[False, True]).reset_index(drop=True)
    return df.drop(columns=[c for c in df.columns if c.startswith("_n_") or c == "_score"])


def write_report(df: pd.DataFrame, targets: dict, path: Path = REPORT_OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = df.head(5)
    lines = [
        "# Candidate shortlist — Phase 5 (ECM radar + TMM optics, NSGA-III)",
        "",
        f"Pareto front: {len(df)} designs. Targets: radar worst <= "
        f"{targets['radar']['threshold_db']} dB, IR emissivity < {targets['ir_lwir']['threshold']}, "
        f"visible deltaE < {targets['visible']['threshold']}, weight <= "
        f"{targets['physical']['weight_kg_per_m2']['threshold']} kg/m^2.",
        "",
        f"Designs meeting **all** targets (full X+Ku): **{int(df['all_targets_met'].sum())}**.",
        f"Designs meeting radar **X-band** + IR + visible: **{int((df['radar_x_ok'] & df['ir_ok'] & df['vis_ok']).sum())}**.",
        f"Designs meeting radar **X-band** + IR (color aside): **{int((df['radar_x_ok'] & df['ir_ok']).sum())}**.",
        "",
        "Top 5 by targets-met then balance:",
        "",
        "| # | radar worst X+Ku (dB) | radar worst X (dB) | IR emiss (state) | visible dE | weight (kg/m²) | bands met |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in top.iterrows():
        st = "met" if r["ir_state"].startswith("VO2 (m") else "ins"
        lines.append(
            f"| {i+1} | {r['radar_worst_db']:.1f} | {r['radar_x_worst_db']:.1f} | "
            f"{r['ir_emissivity']:.3f} ({st}) | {r['visible_deltaE']:.1f} | "
            f"{r['weight_kg_m2']:.1f} | {int(r['targets_met'])}/3 |"
        )
    lines += [
        "",
        "**Honest read of the gaps:**",
        "- *Radar:* a single-layer absorber can hit −10 dB across the X-band but not the full "
        "8–18 GHz; full X+Ku coverage needs a multilayer (Jaumann) absorber — a concrete next step.",
        "- *Visible:* deltaE is bounded by the open PEDOT:PSS/VO₂ palette; the spec's tunable ProDOT "
        "polymer (no open optical data) would extend color reach.",
        "- *Radar fidelity:* ECM is fast/approximate — top candidates must be re-confirmed with "
        "openEMS full-wave (Phase 6, cluster) before trust.",
        "- *IR:* emissivity is the best achievable VO₂ state (the material switches), per its adaptive design.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(n_gen: int = 40) -> None:
    res, targets = run_optimization(n_gen=n_gen)
    df = front_to_frame(res, targets)
    df.to_parquet(PARETO_OUT, index=False)
    report = write_report(df, targets)

    n_all = int(df["all_targets_met"].sum())
    n_x_ir = int((df["radar_x_ok"] & df["ir_ok"]).sum())
    n_x_ir_vis = int((df["radar_x_ok"] & df["ir_ok"] & df["vis_ok"]).sum())
    best = df.iloc[0]
    print(f"Pareto front: {len(df)} designs -> {PARETO_OUT}")
    print(f"Meeting ALL targets (full X+Ku): {n_all}")
    print(f"Meeting radar X-band + IR: {n_x_ir}   (+ visible: {n_x_ir_vis})")
    print(f"Best by ranking: radar_worst={best['radar_worst_db']:.1f}dB "
          f"radar_X={best['radar_x_worst_db']:.1f}dB ir={best['ir_emissivity']:.3f} "
          f"dE={best['visible_deltaE']:.1f} wt={best['weight_kg_m2']:.1f}")
    print(f"Report -> {report}")


if __name__ == "__main__":
    main()
