"""Selection-bias statistics for backtests.

Implements the probabilistic and deflated Sharpe ratios (Bailey and Lopez de
Prado, 2012 and 2014) and the probability of backtest overfitting through
combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado
and Zhu, 2017) with numpy and scipy only.

All Sharpe ratios in this module are per-period unless a function says
otherwise. Use `annualise` to convert.
"""
from itertools import combinations

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns, axis=0):
    """Per-period Sharpe ratio (mean over sample standard deviation)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return r.mean(axis=axis) / sd


def annualise(sr, periods_per_year=252):
    return sr * np.sqrt(periods_per_year)


def sharpe_std_error(sr, n_obs, skewness=0.0, kurt=3.0):
    """Standard error of an estimated per-period Sharpe ratio.

    `kurt` is the ordinary (non-excess) kurtosis, 3 for a normal sample.
    """
    var = (1.0 - skewness * sr + (kurt - 1.0) / 4.0 * sr**2) / (n_obs - 1)
    return np.sqrt(np.maximum(var, 1e-18))


def probabilistic_sharpe(sr, sr_benchmark, n_obs, skewness=0.0, kurt=3.0):
    """Probability that the true Sharpe ratio exceeds `sr_benchmark`."""
    z = (sr - sr_benchmark) / sharpe_std_error(sr, n_obs, skewness, kurt)
    return norm.cdf(z)


def expected_max_sharpe(n_trials, sr_var):
    """Expected maximum of `n_trials` independent null Sharpe estimates.

    `sr_var` is the variance of the per-period Sharpe estimates across
    trials. Returns 0 for a single trial (nothing to select from).
    """
    if n_trials <= 1:
        return 0.0
    a = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
    b = EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_var) * (a + b))


def null_max_sharpe_annual(n_trials, n_obs, periods_per_year=252):
    """Annualised expected best Sharpe among `n_trials` zero-skill strategies.

    Uses Var(SR_hat) = 1 / (n_obs - 1) for a normal sample with zero edge.
    """
    per_period = expected_max_sharpe(n_trials, 1.0 / (n_obs - 1))
    return annualise(per_period, periods_per_year)


def deflated_sharpe(returns, n_trials, sr_var_trials):
    """Deflated Sharpe ratio of the selected strategy's return series.

    The benchmark is the Sharpe ratio one would expect from the best of
    `n_trials` strategies with no skill, so the result answers: how likely
    is it that this strategy's edge is real after selection?
    """
    r = np.asarray(returns, dtype=float)
    sr = sharpe_ratio(r)
    benchmark = expected_max_sharpe(n_trials, sr_var_trials)
    return probabilistic_sharpe(
        sr, benchmark, len(r), _skew(r), _kurtosis(r, fisher=False)
    )


def _block_sums(returns, n_blocks):
    t = returns.shape[0] - returns.shape[0] % n_blocks
    x = returns[:t].reshape(n_blocks, t // n_blocks, -1)
    return x.sum(axis=1), (x**2).sum(axis=1), t // n_blocks


def _sharpe_from_blocks(select, s1, s2, size):
    """Sharpe ratio of the union of blocks picked by each row of `select`."""
    n = select.sum(axis=1, keepdims=True) * size
    m1 = select @ s1
    m2 = select @ s2
    mean = m1 / n
    var = (m2 - n * mean**2) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mean / np.sqrt(var)


def pbo_cscv(returns, n_blocks=16):
    """Probability of backtest overfitting by CSCV.

    `returns` has shape (T, N): T periods for N candidate strategies. The
    sample is cut into `n_blocks` blocks, and every way of assigning half of
    them to a training set is tried. The strategy that is best in training
    is ranked among all strategies on the remaining half.

    Returns a dict with `pbo`, the `logits` of the out-of-sample relative
    rank, and the paired in-sample and out-of-sample Sharpe ratios of the
    selected strategy (per-period).
    """
    returns = np.asarray(returns, dtype=float)
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    s1, s2, size = _block_sums(returns, n_blocks)
    n_strat = returns.shape[1]

    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    train = np.zeros((len(combos), n_blocks))
    np.put_along_axis(train, combos, 1.0, axis=1)
    test = 1.0 - train

    sr_in = _sharpe_from_blocks(train, s1, s2, size)
    sr_out = _sharpe_from_blocks(test, s1, s2, size)
    best = np.nanargmax(sr_in, axis=1)
    rows = np.arange(len(combos))
    chosen_out = sr_out[rows, best]
    rank = (sr_out <= chosen_out[:, None]).sum(axis=1)
    omega = rank / (n_strat + 1.0)
    logits = np.log(omega / (1.0 - omega))
    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "logits": logits,
        "sr_in": sr_in[rows, best],
        "sr_out": chosen_out,
    }
