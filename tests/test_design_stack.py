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


def test_discovered_radar_layer_places_magnetic_candidate(tmp_path):
    """The top discovered radar candidate becomes a single-layer absorber in the design."""
    import pandas as pd

    from stealth.discovery.design_stack import discovered_radar_layer

    p = tmp_path / "fc.parquet"
    pd.DataFrame([
        {"id": "g1", "formula": "CoFe2O4", "role": "radar_magnetic", "score": 0.8},
        {"id": "g2", "formula": "TiC", "role": "radar_conductor", "score": 0.6},
    ]).to_parquet(p)
    r = discovered_radar_layer(str(p))
    assert r["formula"] == "CoFe2O4" and r["loss_mechanism"] == "magnetic"
    assert r["min_rl_db"] < -10.0 and r["matched_thickness_mm"] > 0
    assert discovered_radar_layer(None) is None
