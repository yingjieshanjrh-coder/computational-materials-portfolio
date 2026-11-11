import numpy as np
import pytest

import backtest_stats as bs


def test_sharpe_ratio_known_value():
    r = np.array([0.01, 0.03, -0.01, 0.02])
    expected = r.mean() / r.std(ddof=1)
    assert bs.sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_ratio_columns_independent():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, (500, 3))
    out = bs.sharpe_ratio(x)
    for j in range(3):
        assert out[j] == pytest.approx(bs.sharpe_ratio(x[:, j]))


def test_annualise():
    assert bs.annualise(0.1, 252) == pytest.approx(0.1 * np.sqrt(252))


def test_std_error_normal_matches_textbook():
    # Normal returns: Var(SR) = (1 + SR^2 / 2) / (n - 1)
    sr, n = 0.1, 1000
    se = bs.sharpe_std_error(sr, n, skewness=0.0, kurt=3.0)
    assert se == pytest.approx(np.sqrt((1 + sr**2 / 2) / (n - 1)))


def test_psr_is_half_at_benchmark():
    assert bs.probabilistic_sharpe(0.05, 0.05, 500) == pytest.approx(0.5)


def test_psr_monotone_in_sharpe_and_sample_size():
    assert bs.probabilistic_sharpe(0.08, 0, 500) > bs.probabilistic_sharpe(0.04, 0, 500)
    assert bs.probabilistic_sharpe(0.05, 0, 2000) > bs.probabilistic_sharpe(0.05, 0, 500)


def test_negative_skew_lowers_psr():
    base = bs.probabilistic_sharpe(0.1, 0, 500, skewness=0.0)
    skewed = bs.probabilistic_sharpe(0.1, 0, 500, skewness=-1.0)
    assert skewed < base


def test_expected_max_single_trial_is_zero():
    assert bs.expected_max_sharpe(1, 0.01) == 0.0


def test_expected_max_grows_with_trials_and_variance():
    a = bs.expected_max_sharpe(10, 0.01)
    b = bs.expected_max_sharpe(100, 0.01)
    c = bs.expected_max_sharpe(100, 0.04)
    assert 0 < a < b < c


def test_expected_max_close_to_monte_carlo():
    rng = np.random.default_rng(2)
    n, reps, var = 50, 4000, 0.0004
    sims = rng.normal(0, np.sqrt(var), (reps, n)).max(axis=1).mean()
    assert bs.expected_max_sharpe(n, var) == pytest.approx(sims, rel=0.05)


def test_deflated_below_probabilistic():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0008, 0.01, 1260)
    sr = bs.sharpe_ratio(r)
    psr = bs.probabilistic_sharpe(sr, 0, len(r))
    dsr = bs.deflated_sharpe(r, n_trials=100, sr_var_trials=1 / 1259)
    assert dsr < psr


def test_deflated_equals_probabilistic_for_one_trial():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0008, 0.01, 1260)
    from scipy.stats import kurtosis, skew
    psr = bs.probabilistic_sharpe(
        bs.sharpe_ratio(r), 0, len(r), skew(r), kurtosis(r, fisher=False)
    )
    assert bs.deflated_sharpe(r, 1, 0.01) == pytest.approx(psr)


def test_null_max_sharpe_annual_grows_slowly():
    n_obs = 1260
    vals = [bs.null_max_sharpe_annual(n, n_obs) for n in (1, 10, 100, 1000)]
    assert vals[0] == 0.0
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 1.5  # five years, 1000 trials: about 0.7 to 0.8


def test_pbo_cscv_combination_count_and_range():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.01, (800, 20))
    out = bs.pbo_cscv(x, n_blocks=8)
    assert len(out["logits"]) == 70  # C(8, 4)
    assert 0.0 <= out["pbo"] <= 1.0


def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(6)
    vals = [bs.pbo_cscv(rng.normal(0, 0.01, (800, 50)), 8)["pbo"] for _ in range(20)]
    assert np.mean(vals) == pytest.approx(0.5, abs=0.12)


def test_pbo_low_when_one_strategy_has_real_edge():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, (1600, 30))
    x[:, 0] += 0.0015  # strong persistent edge
    assert bs.pbo_cscv(x, 8)["pbo"] < 0.1


def test_pbo_in_sample_exceeds_out_of_sample_on_noise():
    rng = np.random.default_rng(8)
    out = bs.pbo_cscv(rng.normal(0, 0.01, (800, 100)), 8)
    assert out["sr_in"].mean() > out["sr_out"].mean() + 0.05


def test_pbo_rejects_odd_blocks():
    with pytest.raises(ValueError):
        bs.pbo_cscv(np.zeros((100, 3)), n_blocks=5)
