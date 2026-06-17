"""Capstone scoring helpers (no models/network)."""

import numpy as np

from stealth.discovery.select_candidates import _optical_consistency, _stability_score


def test_stability_score_monotonic():
    assert _stability_score(-0.5) == 1.0          # well below hull -> max
    assert _stability_score(0.3) == 0.0           # well above hull -> min
    assert _stability_score(float("nan")) == 0.0  # unknown -> 0
    assert 0.0 < _stability_score(0.05) < 1.0


def test_optical_consistency_role_aware():
    # conductor should be lossy (high k), dielectric clear (low k)
    assert _optical_consistency("radar_conductor", 2.0) > _optical_consistency("radar_conductor", 0.1)
    assert _optical_consistency("dielectric_spacer", 0.0) > _optical_consistency("dielectric_spacer", 1.0)
    assert _optical_consistency("radar_conductor", float("nan")) == 0.5
