"""Phase 3: ECM radar forward model — passivity, the Salisbury null, and the sweep."""

import numpy as np
import pytest

from stealth.physics.radar import (
    Z0,
    RadarStack,
    grid_capacitance_F,
    metrics,
    spectrum,
)
from stealth.physics.radar_sweep import DESIGN_BOUNDS, generate_dataset, stack_from_unit


def test_passivity_no_gain():
    """A passive absorber can never reflect more power than it receives."""
    stack = RadarStack(6, 5.4, 120, 2.5, 4.3, "capacitive_patch")
    A = spectrum(stack, np.linspace(1, 30, 200))["absorption"]
    assert np.all(A <= 1.0 + 1e-9) and np.all(A >= -1e-9)


def test_salisbury_screen_null_at_design_frequency():
    """Rs=Z0 resistive sheet at quarter-wave (7.5 mm) air gap -> deep null at 10 GHz."""
    s = RadarStack(3, 1, Z0, 7.5, 1.0, "resistive_only")
    m = metrics(s, np.linspace(1, 30, 581))
    assert m["min_rl_db"] < -30.0
    assert abs(m["f_at_min_ghz"] - 10.0) < 0.3


def test_capacitance_positive_and_grows_with_patch():
    """Smaller gap (bigger patch) -> larger capacitance."""
    big_patch = RadarStack(6, 5.7, 100, 2, 4.3)
    small_patch = RadarStack(6, 4.5, 100, 2, 4.3)
    assert grid_capacitance_F(big_patch) > grid_capacitance_F(small_patch) > 0


def test_invalid_patch_rejected():
    with pytest.raises(ValueError):
        RadarStack(6, 6.5, 100, 2, 4.3)  # patch >= period


def test_stack_from_unit_respects_bounds():
    lo = stack_from_unit(np.zeros(5))
    hi = stack_from_unit(np.ones(5))
    assert lo.period_mm == DESIGN_BOUNDS["period_mm"][0]
    assert hi.period_mm == DESIGN_BOUNDS["period_mm"][1]
    assert lo.spacer_eps_r == DESIGN_BOUNDS["spacer_eps_r"][0]


def test_sweep_generates_valid_dataset():
    df = generate_dataset(n=64, seed=1)
    assert len(df) == 64
    assert (df["min_rl_db"] <= 0).all()                  # RL is never positive
    assert df["rl_spectrum_db"].iloc[0].__len__() == 59  # full spectrum stored


def test_stack_from_design_roundtrip(tmp_path):
    """The design JSON saved by design_stack rebuilds into the same RadarStack (openEMS --design)."""
    import json

    from stealth.physics.radar_fullwave import stack_from_design

    design = {"radar_stack": {
        "period_mm": 5.0, "patch_mm": 4.3, "sheet_resistance_ohm_sq": 286.0,
        "spacer_thickness_mm": 3.1, "spacer_eps_r": 3.9, "pattern": "capacitive_patch",
    }}
    p = tmp_path / "designed_coating.json"
    p.write_text(json.dumps(design), encoding="utf-8")
    s = stack_from_design(str(p))
    assert s.period_mm == 5.0 and s.patch_mm == 4.3
    assert s.sheet_resistance_ohm_sq == 286.0 and s.spacer_thickness_mm == 3.1
    assert s.spacer_eps_r == 3.9
    # and the rebuilt stack is physically valid in the ECM (RL never positive)
    assert (spectrum(s, np.linspace(8, 12, 9))["reflection_loss_db"] <= 1e-9).all()
