"""Phase 0 smoke test: the frozen target spec loads and is well-formed."""

from stealth.config import REQUIRED_BANDS, load_targets


def test_targets_load_and_have_required_bands():
    targets = load_targets()
    for band in REQUIRED_BANDS:
        assert band in targets, f"missing band: {band}"


def test_thresholds_present_and_numeric():
    targets = load_targets()
    assert targets["radar"]["threshold_db"] == -10.0
    assert targets["ir_lwir"]["threshold"] == 0.3
    assert targets["visible"]["threshold"] == 5.0
    assert targets["physical"]["weight_kg_per_m2"]["threshold"] == 10.0
