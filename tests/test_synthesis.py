"""Synthesizability scoring + precursor/route suggestion."""

from pymatgen.core import Composition

from stealth.discovery.synthesis import (
    chemical_validity,
    dossier,
    suggest_precursors,
    synthesizability,
)


def test_chemical_validity_rejects_implausible():
    assert chemical_validity(Composition("TiO2")) is True
    assert chemical_validity(Composition("NaCl3")) is False


def test_synthesizability_scores_stable_valid_high():
    s = synthesizability(Composition("TiO2"), eform_per_atom=-2.7)
    assert s["chem_valid"] is True
    assert 0.0 <= s["synth_score"] <= 1.0
    assert s["synth_score"] > 0.8  # stable + valid + simple


def test_precursors_for_oxide_are_solid_state():
    route = suggest_precursors(Composition("VO2"))
    assert "V2O5" in route["precursors"]
    assert "solid-state" in route["method"] or "sol-gel" in route["method"]


def test_dossier_record_has_formula_and_route():
    d = dossier("c1", Composition("In2O3"), -2.0)
    assert d["formula"] == "In2O3"
    assert d["precursors"]
    assert d["method"]
