"""GAP #2/#3 helpers: confidence levels + DFT-validation metrics (no models/network)."""

from stealth.discovery.dft_validate import _mae, _r2
from stealth.discovery.uncertainty import _confidence_level


def test_confidence_levels():
    assert _confidence_level(0.1, 0.05) == "high"      # both tight
    assert _confidence_level(0.6, 0.30) == "medium"
    assert _confidence_level(1.6, 0.05) == "low"        # band gap spreads wide
    assert _confidence_level(0.1, 0.9) == "low"         # models disagree on eform


def test_mae_and_r2():
    y = [0.0, 1.0, 2.0, 3.0]
    assert _mae(y, y) == 0.0
    assert _r2(y, y) == 1.0
    assert _mae([0.0, 0.0], [1.0, 1.0]) == 1.0
    assert _r2(y, [v + 1 for v in y]) < 1.0   # constant offset -> imperfect
