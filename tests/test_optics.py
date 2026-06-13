"""Phase 2: TMM optics forward model — conservation, coverage, and the VO2 switch."""

import numpy as np
import pytest

from stealth.physics.optics import (
    CoverageError,
    Layer,
    Stack,
    band_emissivity,
    delta_e_vs_background,
    stack_spectrum,
    visible_lab,
)

GROUND = Layer("Aluminum (ground plane)", 0.3)
SPACER = Layer("SiO2 (IR dielectric)", 1.0)


def _vo2_stack(state: str) -> Stack:
    return Stack.of(Layer(state, 0.2), SPACER, GROUND)


def test_energy_is_conserved():
    sp = stack_spectrum(_vo2_stack("VO2 (insulating, <Tc)"), np.linspace(8, 14, 25))
    total = sp["R"] + sp["T"] + sp["A"]
    assert np.allclose(total, 1.0, atol=1e-6)


def test_opaque_ground_blocks_transmission():
    sp = stack_spectrum(_vo2_stack("VO2 (metallic, >Tc)"), np.linspace(8, 14, 11))
    assert np.all(sp["T"] < 1e-3)


def test_emissivity_in_unit_interval():
    e = band_emissivity(_vo2_stack("VO2 (insulating, <Tc)"), (8.0, 14.0))
    assert 0.0 <= e <= 1.0


def test_vo2_switch_changes_lwir_emissivity():
    """Switching VO2 phase must produce a significant, valid emissivity change."""
    e_ins = band_emissivity(_vo2_stack("VO2 (insulating, <Tc)"), (8.0, 14.0))
    e_met = band_emissivity(_vo2_stack("VO2 (metallic, >Tc)"), (8.0, 14.0))
    assert 0.0 <= e_ins <= 1.0 and 0.0 <= e_met <= 1.0
    assert abs(e_met - e_ins) > 0.05  # the layer must actually do something


def test_coverage_error_when_material_out_of_band():
    # ITO has no data beyond 1 um; using it in an LWIR calc must fail loudly.
    bad = Stack.of(Layer("ITO", 0.05), SPACER, GROUND)
    with pytest.raises(CoverageError):
        stack_spectrum(bad, np.linspace(8, 14, 5))


def test_visible_lab_and_delta_e_are_sane():
    vis = Stack.of(Layer("PEDOT:PSS", 0.1), Layer("VO2 (insulating, <Tc)", 0.2), GROUND)
    lab = visible_lab(vis)
    assert 0.0 <= lab[0] <= 100.0
    de = delta_e_vs_background(vis, "#228B22")
    assert de >= 0.0 and np.isfinite(de)
