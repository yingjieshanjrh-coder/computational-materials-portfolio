import json
from pathlib import Path

import numpy as np
import pytest

import features as ft
import train

HERE = Path(__file__).parent


def test_parse_formula():
    assert ft.parse_formula("GaAs") == {"Ga": 1, "As": 1}
    assert ft.parse_formula("Al2O3") == {"Al": 2, "O": 3}
    assert ft.parse_formula("Ba(TiO3)") == {"Ba": 1, "Ti": 1, "O": 3}
    assert ft.parse_formula("Ca3(PO4)2") == {"Ca": 3, "P": 2, "O": 8}
    assert ft.parse_formula("Mg(Al(OH)4)2") == {"Mg": 1, "Al": 2, "O": 8, "H": 8}
    assert ft.parse_formula("Ga0.5In0.5As") == pytest.approx({"Ga": 0.5, "In": 0.5, "As": 1})


@pytest.mark.parametrize("bad", ["", "Ga)As", "(GaAs", "2GaAs", "Ga?As", "Xx2O"])
def test_bad_formulas_rejected(bad):
    with pytest.raises((ValueError, KeyError)):
        ft.parse_formula(bad)


@pytest.mark.parametrize("sym,expected", [
    ("H", (1, 1)), ("C", (2, 14)), ("Na", (3, 1)), ("Cl", (3, 17)), ("Ti", (4, 4)),
    ("Zn", (4, 12)), ("Ga", (4, 13)), ("Te", (5, 16)), ("Ba", (6, 2)), ("La", (6, 3)),
    ("Lu", (6, 3)), ("Hf", (6, 4)), ("Au", (6, 11)), ("Pb", (6, 14)), ("Bi", (6, 15))])
def test_period_group(sym, expected):
    assert ft.period_group(ft.ELEMENTS[sym][0]) == expected


def test_feature_values_nacl():
    f = dict(zip(ft.FEATURE_NAMES, ft.featurize("NaCl")))
    assert len(ft.FEATURE_NAMES) == 33
    assert f["Z_mean"] == pytest.approx(14.0)
    assert f["Z_std"] == pytest.approx(3.0)
    assert f["electronegativity_range"] == pytest.approx(3.16 - 0.93)
    assert f["valence_mean"] == pytest.approx(4.0)        # (1 + 7) / 2
    assert f["n_elements"] == 2 and f["max_fraction"] == pytest.approx(0.5)
    assert f["entropy_mixing"] == pytest.approx(np.log(2))


def test_features_depend_only_on_composition():
    assert np.allclose(ft.featurize("TiO2"), ft.featurize("Ti2O4"))
    assert np.allclose(ft.featurize("Ba(TiO3)"), ft.featurize("O3TiBa"))
    assert ft.reduced_formula_key("TiO2") == ft.reduced_formula_key("Ti3O6")


def test_duplicates_merged_and_bad_rows_reported(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("formula,target\nTiO2,3.0\nTi2O4,3.4\nGaAs,1.4\nXxO,1.0\nSi,\n")
    data, bad = train.load_dataset(csv)
    assert len(data) == 2 and len(bad) == 1
    assert data.loc[data.formula.isin(["TiO2", "Ti2O4"]), "target"].item() == pytest.approx(3.2)


def test_pipeline_end_to_end(tmp_path):
    train.main([str(HERE / "data" / "demo_gaps.csv"), "--target", "gap_eV",
                "--out", str(tmp_path)])
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["n_compounds"] == 46 and m["n_features"] == 33
    for name in ("mean_baseline", "ridge", "random_forest"):
        assert np.isfinite(m["metrics"][name]["MAE"])
    # a learned model must beat predicting the mean, even on this tiny set
    assert m["metrics"]["random_forest"]["MAE"] < m["metrics"]["mean_baseline"]["MAE"]
    assert (tmp_path / "parity.png").stat().st_size > 5000
