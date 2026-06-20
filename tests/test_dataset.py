"""#1 dataset (Tier A1 ingestion helpers) + Tier A2 literature EM table."""

from stealth.discovery import em_literature
from stealth.discovery.dataset import _first, _mean, _num


def test_num_handles_jarvis_missing_values():
    assert _num("na") is None and _num(None) is None and _num("") is None
    assert _num("3.5") == 3.5 and _num(2) == 2.0
    assert _num(float("nan")) is None


def test_mean_and_first_skip_missing():
    assert _mean("na", 2.0, 4.0) == 3.0
    assert _mean("na", "na") is None
    assert _first({"a": "na", "b": 2.0}, ["a", "b"]) == 2.0
    assert _first({"a": "na"}, ["a", "x"]) is None


def test_literature_table_has_magnetic_and_dielectric_classes():
    df = em_literature.as_dataframe()
    assert len(df) >= 10
    assert set(df["mat_class"]) >= {"magnetic", "conductive", "dielectric"}
    # magnetic rows must carry non-zero mu'' (the magnetic loss); non-magnetic ~ 1+0j
    mag = df[df["mat_class"] == "magnetic"]
    assert (mag["mu_imag"] > 0).all()


def test_class_mu_prior_only_magnetic_elements_get_mu():
    assert em_literature.class_mu_prior({"Fe", "O"}).imag > 0      # ferrite -> magnetic loss
    assert em_literature.class_mu_prior({"Si", "O"}) == complex(1.0, 0.0)  # dielectric -> mu=1


def test_reference_absorber_for_validation_anchor():
    r = em_literature.reference_absorber()
    assert r.mat_class == "magnetic" and r.mu_imag > 0 and r.eps_real > 0


def test_finetune_dataset_filters_to_role_chemistry():
    import pandas as pd

    from stealth.discovery.finetune_dataset import select_for_role

    df = pd.DataFrame([
        {"jid": "j1", "formula": "CoFe2O4", "magmom": 3.0, "bandgap_optb88": 0.5, "atoms": "{}"},
        {"jid": "j2", "formula": "Fe3O4", "magmom": 4.0, "bandgap_optb88": 0.0, "atoms": "{}"},
        {"jid": "j3", "formula": "SiO2", "magmom": None, "bandgap_optb88": 8.0, "atoms": "{}"},
        {"jid": "j4", "formula": "Ce3Si5Rh", "magmom": 1.0, "bandgap_optb88": 0.2, "atoms": "{}"},
    ])
    mag = set(select_for_role(df, "radar_magnetic")["formula"])
    assert mag == {"CoFe2O4", "Fe3O4"}                 # magnetic family + has magmom label
    diel = set(select_for_role(df, "dielectric_spacer")["formula"])
    assert diel == {"SiO2"}                              # Si-Al-Mg-O family, off-family rejected
