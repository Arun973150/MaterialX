"""DFT confirmation — formation-energy math (GPAW calls are not exercised here)."""

from pymatgen.core import Composition

from stealth.discovery.dft_confirm import formation_energy_per_atom


def test_formation_energy_binary():
    # MgO: E_tot=-12, refs Mg=-1.5, O=-4.0 -> (-12 - (-5.5))/2 = -3.25 eV/atom
    ef = formation_energy_per_atom(-12.0, Composition("MgO"), {"Mg": -1.5, "O": -4.0})
    assert abs(ef + 3.25) < 1e-9


def test_formation_energy_negative_for_favorable():
    ef = formation_energy_per_atom(-90.0, Composition("MnV2MoO6"),
                                   {"Mn": -9, "V": -9, "Mo": -11, "O": -4.9})
    assert ef < 0


def test_formation_energy_per_atom_normalization():
    # energy exactly at the references -> zero formation energy
    refs = {"Ca": -2.0, "F": -3.0}
    e_tot = 1 * -2.0 + 2 * -3.0  # CaF2 at reference
    assert abs(formation_energy_per_atom(e_tot, Composition("CaF2"), refs)) < 1e-9
