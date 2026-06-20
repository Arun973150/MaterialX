"""#6 manufacturability screening — exotic candidates rejected, real RAM materials pass."""

from pymatgen.core import Composition

from stealth.discovery.manufacturability import (
    is_practical,
    manufacturability_score,
    practical_elements,
    practicality,
)


def test_real_ram_materials_are_practical():
    for f in ["Fe3O4", "CoFe2O4", "SiO2", "TiC", "VO2", "MnZnFe2O4", "Al2O3"]:
        assert is_practical(Composition(f)), f"{f} should be practical"


def test_exotic_and_toxic_rejected():
    # rare-earth / precious intermetallics and toxic elements must be filtered out
    for f in ["Ce3Si5Rh", "Ho5Rh2", "NdMgSn2", "PbTiO3", "BeO"]:
        assert not is_practical(Composition(f)), f"{f} should be rejected"


def test_score_orders_sensibly():
    assert manufacturability_score(Composition("SiO2")) > manufacturability_score(Composition("Ho5Rh2"))
    assert 0.0 <= manufacturability_score(Composition("Fe3O4")) <= 1.0


def test_reasons_name_the_problem():
    r = practicality(Composition("Ho5Rh2"))
    assert not r["practical"]
    assert any("rare-earth" in reason or "precious" in reason for reason in r["reasons"])


def test_practical_elements_are_abundant_and_clean():
    for role in ["radar_magnetic", "radar_conductor", "dielectric_spacer", "ir_phasechange"]:
        els = practical_elements(role)
        assert len(els) >= 4
        assert "Rh" not in els and "Pb" not in els  # no precious / toxic in the generation family
