"""Phase 5: joint objective + a small NSGA-III smoke run."""

import numpy as np

from stealth.optimize.objective import (
    DESIGN_BOUNDS,
    N_DIM,
    design_from_unit,
    evaluate,
    weight_kg_m2,
)


def test_design_from_unit_hits_bounds():
    lo = design_from_unit(np.zeros(N_DIM))
    hi = design_from_unit(np.ones(N_DIM))
    assert lo.spacer_thickness_mm == DESIGN_BOUNDS["spacer_thickness_mm"][0]
    assert hi.spacer_thickness_mm == DESIGN_BOUNDS["spacer_thickness_mm"][1]


def test_evaluate_returns_physical_ranges():
    m = evaluate(np.full(N_DIM, 0.5))
    assert m["radar_x_worst_db"] <= 0.0           # reflection loss is never positive
    assert 0.0 <= m["f_ir_emissivity"] <= 1.0
    assert m["f_vis_deltaE"] >= 0.0
    assert m["weight_kg_m2"] > 0.0
    assert m["ir_state"] in ("VO2 (insulating, <Tc)", "VO2 (metallic, >Tc)")


def test_weight_grows_with_spacer():
    thin = design_from_unit(np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.5]))
    thick = design_from_unit(np.array([0.5, 0.5, 1.0, 0.5, 0.5, 0.5]))
    assert weight_kg_m2(thick) > weight_kg_m2(thin)


def test_ir_uses_best_achievable_state():
    """Reported IR emissivity must be <= either single state (it's the min over states)."""
    from stealth.optimize.objective import _ir_stack
    from stealth.physics.optics import band_emissivity

    d = design_from_unit(np.full(N_DIM, 0.5))
    m = evaluate(np.full(N_DIM, 0.5))
    ins = band_emissivity(_ir_stack(d, "VO2 (insulating, <Tc)"), (8.0, 14.0))
    met = band_emissivity(_ir_stack(d, "VO2 (metallic, >Tc)"), (8.0, 14.0))
    assert m["f_ir_emissivity"] <= min(ins, met) + 1e-9


def test_nsga_smoke_run_produces_front():
    from stealth.optimize.problem import front_to_frame, run_optimization

    res, targets = run_optimization(n_gen=3, n_partitions=4, seed=1)
    df = front_to_frame(res, targets)
    assert len(df) > 0
    for col in ("radar_x_worst_db", "ir_emissivity", "visible_deltaE", "targets_met"):
        assert col in df.columns
