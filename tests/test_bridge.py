"""Discovery->physics bridge: band gap -> n,k -> flows into TMM."""

import numpy as np

from stealth.discovery.optical_bridge import candidate_material, estimate_nk
from stealth.physics import optics


def test_wide_gap_is_ir_transparent():
    # A 4 eV gap material is transparent in LWIR (photon energy << gap): k ~ 0.
    N = estimate_nk(4.0, np.array([10.0]))[0]
    assert N.imag < 1e-6
    assert 1.3 <= N.real <= 5.0


def test_metal_is_lossy_in_ir():
    # Near-zero gap -> Drude metal -> large extinction in the IR.
    N = estimate_nk(0.0, np.array([10.0]))[0]
    assert N.imag > 1.0


def test_absorption_turns_on_above_gap():
    # For a 2 eV gap: transparent at 1 eV (1.24 um), absorbing at 3 eV (0.41 um).
    nk = estimate_nk(2.0, np.array([1.24, 0.41]))
    assert nk[0].imag < 1e-6      # below gap
    assert nk[1].imag > 0.0       # above gap


def test_candidate_material_flows_into_physics():
    mat = candidate_material("gen-001", band_gap_ev=3.5, layer_role="dielectric_spacer")
    assert mat.source == "tabulated"
    stack = optics.Stack.of(optics.Layer(mat, 2.0), optics.Layer("Aluminum (ground plane)", 0.3))
    e = optics.band_emissivity(stack, (8.0, 14.0), clip=True)
    assert 0.0 <= e <= 1.0
