"""#10 experimental anchor — measured eps,mu reproduce known absorption + lambda/4 matching."""

from stealth.discovery.anchor import quarter_wave_thickness, reproduce, validate
from stealth.discovery.em_literature import reference_absorber


def test_carbonyl_iron_reproduces_strong_absorption():
    v = validate()
    assert v["passes"]
    assert v["min_rl_db"] <= -15.0
    # scanned optimum thickness is close to the quarter-wave-in-medium theory
    assert 0.5 <= v["thickness_vs_quarterwave_ratio"] <= 2.0


def test_quarter_wave_thickness_physical():
    # higher refractive index (eps*mu) -> thinner matching layer
    thin = quarter_wave_thickness(30 - 5j, 3 - 2j, 10.0)
    thick = quarter_wave_thickness(8 - 1j, 1 + 0j, 10.0)
    assert 0 < thin < thick


def test_magnetic_absorbers_all_match_well():
    # every magnetic reference reproduces a deep, well-matched minimum
    from stealth.discovery.em_literature import LITERATURE

    for rec in [r for r in LITERATURE if r.mat_class == "magnetic"]:
        r = reproduce(rec)
        assert r["min_rl_db"] < -10.0
        ratio = r["best_thickness_mm"] / r["quarter_wave_thickness_mm"]
        assert 0.4 <= ratio <= 2.5, f"{rec.material}: d_opt/lambda4 = {ratio:.2f}"
