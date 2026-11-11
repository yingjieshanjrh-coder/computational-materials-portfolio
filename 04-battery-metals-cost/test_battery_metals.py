from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chemistry as ch
import cost
import market
import report

HERE = Path(__file__).parent
C = ch.CATHODES


# ------------------------------------------------------------------ chemistry
@pytest.mark.parametrize("name", list(C))
def test_mass_fractions_sum_to_one(name):
    assert sum(C[name].mass_fractions.values()) == pytest.approx(1.0)


def test_nmc811_intensity_by_hand():
    c = C["NMC811"]
    assert c.molar_mass == pytest.approx(97.27, abs=0.01)
    assert c.cam_per_kwh == pytest.approx(1000 / (200 * 3.7))
    it = c.intensity()
    assert it["Li"] == pytest.approx(0.0964, abs=5e-4)
    assert it["Ni"] == pytest.approx(0.6524, abs=5e-4)
    assert "O" not in it


def test_lfp_has_no_nickel_or_cobalt_and_sodium_cell_has_no_lithium():
    assert set(C["LFP"].intensity()) == {"Li", "Fe", "P"}
    assert C["LFP"].intensity()["Li"] == pytest.approx(0.0859, abs=5e-4)
    assert "Li" not in C["Na-NFM111"].intensity() and C["Na-NFM111"].carrier == "Na"


def test_cobalt_falls_along_the_nmc_series():
    co = [C[n].intensity()["Co"] for n in ("NMC111", "NMC622", "NMC811")]
    assert co[0] > co[1] > co[2]


def test_compound_price_conversions():
    assert ch.LI_IN_LI2CO3 == pytest.approx(0.1879, abs=2e-4)
    assert ch.NA_IN_NA2CO3 == pytest.approx(0.4338, abs=2e-4)
    assert ch.lithium_price_from_carbonate(10.0) == pytest.approx(53.2, abs=0.1)


# ------------------------------------------------------------------ cost
BASE, LOW, HIGH = report.read_scenario(HERE / "data" / "prices_scenario.csv")


def test_cost_is_sum_of_intensity_times_price():
    c = C["NMC622"]
    by_hand = sum(kg * BASE[m] for m, kg in c.intensity().items())
    assert cost.total_cost(c, BASE) == pytest.approx(by_hand)


def test_missing_price_is_an_error_unless_allowed():
    with pytest.raises(KeyError):
        cost.total_cost(C["LFP"], BASE)                     # no phosphorus price
    assert cost.total_cost(C["LFP"], BASE, "zero") > 0


def test_cost_is_linear_in_each_price():
    c = C["NMC811"]
    doubled = {**BASE, "Ni": 2 * BASE["Ni"]}
    extra = cost.total_cost(c, doubled) - cost.total_cost(c, BASE)
    assert extra == pytest.approx(c.intensity()["Ni"] * BASE["Ni"])


def test_tornado_sorted_and_bracketing_base():
    rows = cost.tornado(C["NMC811"], BASE, LOW, HIGH)
    swings = [hi - lo for _, lo, hi in rows]
    assert swings == sorted(swings, reverse=True)
    centre = cost.total_cost(C["NMC811"], BASE)
    assert all(lo <= centre <= hi for _, lo, hi in rows)


def test_breakeven_equalises_costs():
    a, b = C["LFP"], C["Na-NFM111"]
    p = cost.breakeven_price(a, b, BASE, "Li", "zero")
    at = {**BASE, "Li": p}
    assert cost.total_cost(a, at, "zero") == pytest.approx(cost.total_cost(b, at, "zero"))
    assert cost.breakeven_price(a, a, BASE, "Li", "zero") is None


def test_cost_history_matches_point_calculation():
    dates = pd.date_range("2024-01-01", periods=3, freq="MS")
    frame = pd.DataFrame({"Ni": [10.0, 20.0, 30.0]}, index=dates)
    fixed = {k: v for k, v in BASE.items() if k != "Ni"}
    series = cost.cost_history(C["NMC811"], frame, fixed)
    assert series.iloc[1] == pytest.approx(cost.total_cost(C["NMC811"], {**BASE, "Ni": 20.0}))


# ------------------------------------------------------------------ market
def test_hhi_reference_cases():
    assert market.hhi([1, 1, 1, 1]) == pytest.approx(2500)
    assert market.hhi({"A": 5}) == pytest.approx(10000)
    assert market.hhi({"A": 70, "B": 20, "C": 10}) == pytest.approx(4900 + 400 + 100)
    with pytest.raises(ValueError):
        market.hhi([0, 0])


def test_volatility_and_drawdown_on_known_series():
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    p = pd.Series([100.0, 120.0, 60.0, 90.0, 80.0], index=idx)
    assert market.max_drawdown(p) == pytest.approx(0.5)
    r = np.diff(np.log(p.values))
    assert float(market.annualised_volatility(p)) == pytest.approx(r.std(ddof=1) * np.sqrt(12))
    flat = pd.Series([5.0] * 6, index=pd.date_range("2020-01-01", periods=6, freq="MS"))
    assert float(market.annualised_volatility(flat)) == 0.0 and market.max_drawdown(flat) == 0.0


def _fake_pink_sheet(path):
    """Workbook laid out like the World Bank monthly file (synthetic numbers)."""
    rows = [["World Bank Commodity Price Data (The Pink Sheet)", None, None, None],
            ["monthly prices in nominal US dollars, 1960 to present", None, None, None],
            ["Updated on (synthetic test file)", None, None, None],
            [None, None, None, None],
            [None, "Aluminum", "Nickel", "Tin"],
            [None, "($/mt)", "($/mt)", "($/mt)"]]
    months = pd.period_range("2019-11", periods=14, freq="M")
    for i, m in enumerate(months):
        rows.append([f"{m.year}M{m.month:02d}", 2000 + 10 * i, 15000 + 500 * i,
                     ".." if i < 2 else 30000.0])
    pd.DataFrame(rows).to_excel(path, sheet_name="Monthly Prices", header=False, index=False)


def test_pink_sheet_parser(tmp_path):
    f = tmp_path / "pink.xlsx"
    _fake_pink_sheet(f)
    frame = market.load_pink_sheet(f)
    assert list(frame.columns) == ["Aluminum", "Nickel", "Tin"]
    assert frame.index[0] == pd.Timestamp("2019-11-01") and len(frame) == 14
    assert frame["Nickel"].iloc[-1] == 15000 + 500 * 13
    assert frame["Tin"].isna().sum() == 2                    # '..' becomes missing
    assert frame.attrs["units"]["Nickel"] == "$/mt"


def test_report_end_to_end(tmp_path):
    f = tmp_path / "pink.xlsx"
    _fake_pink_sheet(f)
    brief = report.build(HERE / "data" / "prices_scenario.csv", tmp_path / "out", f, since=2020)
    text = brief.read_text()
    assert "| NMC811 |" in text and "Price history (2020-01" in text
    for name in ("intensity.png", "cost.png", "tornado_nmc811.png", "price_history.png"):
        assert (tmp_path / "out" / name).stat().st_size > 5000
