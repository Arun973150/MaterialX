"""Discovery->design bridge: material selection + single-design physics eval."""

import numpy as np

from stealth.config import load_targets
from stealth.discovery.design_stack import BOUNDS, PARAMS, _phys, pick_materials


def test_params_match_bounds():
    assert set(PARAMS) == set(BOUNDS)


def test_pick_materials_falls_back_to_known():
    mats, used = pick_materials(None, None)
    assert mats["ec"] == "PEDOT:PSS"
    assert "known" in used["diel"]


def test_phys_returns_sane_metrics():
    mats, _ = pick_materials(None, None)
    m = _phys(np.full(len(PARAMS), 0.5), mats, load_targets())
    assert m["radar_x_worst_db"] <= 0.0
    assert 0.0 <= m["ir_emissivity"] <= 1.0
    assert m["visible_deltaE"] >= 0.0
    assert m["weight_kg_m2"] > 0.0
