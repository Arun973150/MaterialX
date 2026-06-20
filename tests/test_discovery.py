"""Track A discovery: targets, screening logic, generation command (no GNN load)."""

import pytest

from stealth.discovery.generate import generation_command
from stealth.discovery.screen import _role_fit, demo_structures
from stealth.discovery.targets import TARGETS, get_target


def test_targets_well_formed():
    for role, t in TARGETS.items():
        assert t.role == role
        lo, hi = t.band_gap_ev
        assert 0.0 <= lo <= hi


def test_get_target_rejects_unknown():
    with pytest.raises(ValueError):
        get_target("not_a_role")


def test_role_fit_routes_metal_to_conductor():
    # conductive-loss chemistry (Ti-C), band gap ~0, stable -> best fit is the radar conductor
    fits = {r: _role_fit(0.0, -0.5, {"Ti", "C"}, TARGETS[r]) for r in TARGETS}
    assert max(fits, key=fits.get) == "radar_conductor"


def test_role_fit_routes_wide_gap_to_dielectric():
    fits = {r: _role_fit(5.0, -2.0, {"Si", "O"}, TARGETS[r]) for r in TARGETS}
    assert max(fits, key=fits.get) == "dielectric_spacer"


def test_role_fit_routes_ferrite_to_magnetic():
    # Fe-O ferrite chemistry, moderate gap, stable -> magnetic-loss radar role
    fits = {r: _role_fit(0.5, -1.5, {"Fe", "O"}, TARGETS[r]) for r in TARGETS}
    assert max(fits, key=fits.get) == "radar_magnetic"


def test_demo_structures_build():
    structs = demo_structures()
    assert len(structs) == 4
    ids = {sid for sid, _ in structs}
    assert {"demo-Al", "demo-MgO", "demo-TiO2", "demo-VO2"} <= ids


def test_generation_command_encodes_conditioning():
    cmd = generation_command("radar_conductor", n=64, out_dir="runs/x", batch_size=16)
    joined = " ".join(cmd)
    assert "mattergen-generate" in cmd[0]
    assert "dft_band_gap" in joined
    assert "--num_batches=4" in joined  # ceil(64/16)
