"""S.U.N. helpers: uniqueness dedup (no model/network needed)."""

from pymatgen.core import Lattice, Structure

from stealth.discovery.stability import _unique_flags


def test_unique_flags_marks_duplicates():
    al = Structure.from_spacegroup(225, Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    al_dup = al.copy()
    mgo = Structure.from_spacegroup(225, Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    flags = _unique_flags([al, al_dup, mgo])
    assert flags == [True, False, True]  # second Al is a duplicate


def test_all_distinct_are_unique():
    al = Structure.from_spacegroup(225, Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    mgo = Structure.from_spacegroup(225, Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert _unique_flags([al, mgo]) == [True, True]
