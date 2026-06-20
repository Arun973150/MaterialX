"""#4 signature objective — physics of single-layer reflection + surface emissivity."""

import numpy as np

from stealth.discovery.objective import (
    _class_loss_tangent,
    dallenbach_reflection,
    surface_emissivity,
)


def test_reflection_is_passive():
    """A metal-backed lossy layer can never reflect more than it receives (RL <= 0 dB)."""
    rl = dallenbach_reflection(15 - 1.5j, 2 - 1j, 2.0)
    assert np.all(rl <= 1e-9)


def test_good_absorber_reaches_strong_absorption():
    """A carbonyl-iron-like layer at a matched thickness gives a deep reflection minimum."""
    best = min(dallenbach_reflection(15 - 1.5j, 2 - 1j, d).min() for d in (1.0, 1.5, 2.0))
    assert best < -15.0


def test_lossless_layer_barely_absorbs():
    """No loss (real eps, mu=1) -> near-total reflection (RL ~ 0 dB)."""
    rl = dallenbach_reflection(4 + 0j, 1 + 0j, 2.0)
    assert rl.max() > -1.0


def test_surface_emissivity_metal_below_dielectric():
    """High-loss/metallic surfaces reflect (low IR emissivity); dielectrics emit more."""
    assert surface_emissivity(10 - 30j) < surface_emissivity(4 + 0j)
    assert 0.0 <= surface_emissivity(10 - 30j) <= 1.0


def test_class_loss_tangents_positive_and_ordered():
    # conductive materials are the lossiest dielectrically
    assert _class_loss_tangent("conductive") > _class_loss_tangent("dielectric") > 0
