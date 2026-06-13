"""Phase 1: registry loads, optical data resolves, and the VO2 physics holds."""

import numpy as np
import pytest

from stealth.materials.optical import NoOpticalData, optical_constants
from stealth.materials.registry import by_role, load_registry


def _get(materials, name):
    return next(m for m in materials if m.name == name)


def test_registry_loads_all_roles():
    mats = load_registry()
    assert len(mats) >= 8
    for role in ("ir_thermochromic", "visible_ec", "radar_conductor", "substrate", "ground"):
        assert by_role(mats, role), f"no material for role {role}"


def test_vo2_emissivity_switch_is_physical():
    """The core IR mechanism: metallic VO2 is far lossier in LWIR than insulating."""
    mats = load_registry()
    insulating = optical_constants(_get(mats, "VO2 (insulating, <Tc)"), 10.0)[0]
    metallic = optical_constants(_get(mats, "VO2 (metallic, >Tc)"), 10.0)[0]
    assert metallic.imag > insulating.imag, "metallic phase must have higher LWIR k"
    assert metallic.imag > 1.0 and insulating.imag < 1.0


def test_literature_substrate_returns_constant_nk():
    mats = load_registry()
    pdms = _get(mats, "PDMS (flexible substrate)")
    N = optical_constants(pdms, np.array([0.5, 1.0, 2.0]))
    assert np.allclose(N.real, 1.40) and np.allclose(N.imag, 0.0)


def test_pure_conductor_has_no_optical_data():
    mats = load_registry()
    mxene = _get(mats, "MXene Ti3C2Tx")
    with pytest.raises(NoOpticalData):
        optical_constants(mxene, 10.0)


def test_out_of_range_returns_nan_without_clip():
    mats = load_registry()
    ito = _get(mats, "ITO")  # valid 0.25-1.0 um
    assert np.isnan(optical_constants(ito, 10.0)[0])  # LWIR is out of range
    assert not np.isnan(optical_constants(ito, 10.0, clip=True)[0])
