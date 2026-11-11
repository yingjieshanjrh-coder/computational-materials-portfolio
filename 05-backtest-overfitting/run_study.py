"""Simulation study: how much does selecting a winner inflate a backtest?

Everything here is synthetic. Daily returns are drawn from a normal
distribution so that the true Sharpe ratio of each strategy is known, which is
the only way to say how often a procedure is wrong. Run with
`python run_study.py` (about a minute on two cores).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew

import backtest_stats as bs

OUT = "example_results"
T = 1260  # five years of daily returns
SIGMA = 0.01
PPY = 252
N_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def simulate_null(rng, n):
    return rng.normal(0.0, SIGMA, (T, n))


def study_null_max(reps=300, seed=1):
    """Best annualised Sharpe among N strategies that all have zero skill."""
    rng = np.random.default_rng(seed)
    best = {n: [] for n in N_GRID}
    naive_psr, dsr_pass, sr_above_1 = ({n: [] for n in N_GRID} for _ in range(3))
    for _ in range(reps):
        x = simulate_null(rng, N_GRID[-1])
        sr_all = bs.sharpe_ratio(x)
        for n in N_GRID:
            srs = sr_all[:n]
            k = int(np.argmax(srs))
            best[n].append(bs.annualise(srs[k], PPY))
            r = x[:, k]
            sk, ku = skew(r), kurtosis(r, fisher=False)
            naive = bs.probabilistic_sharpe(srs[k], 0.0, T, sk, ku)
            var = srs.var(ddof=1) if n > 1 else 1.0 / (T - 1)
            dsr = bs.probabilistic_sharpe(
                srs[k], bs.expected_max_sharpe(n, var), T, sk, ku
            )
            naive_psr[n].append(naive > 0.95)
            dsr_pass[n].append(dsr > 0.95)
            sr_above_1[n].append(bs.annualise(srs[k], PPY) > 1.0)
    rows = []
    for n in N_GRID:
        rows.append(
            {
                "n_trials": n,
                "mean_best_sharpe": float(np.mean(best[n])),
                "theory_best_sharpe": float(bs.null_max_sharpe_annual(n, T, PPY)),
                "p_best_sharpe_above_1": float(np.mean(sr_above_1[n])),
                "false_positive_psr95": float(np.mean(naive_psr[n])),
                "false_positive_dsr95": float(np.mean(dsr_pass[n])),
            }
        )
    return rows


def study_pbo(reps=30, n_strat=100, seed=2):
    """PBO as a growing share of the candidates carries a real edge."""
    rng = np.random.default_rng(seed)
    mu_daily = 1.0 * SIGMA / np.sqrt(PPY)  # true annual Sharpe of 1
    rows = []
    for share in [0.0, 0.02, 0.05, 0.10, 0.25, 0.50]:
        n_skilled = int(round(share * n_strat))
        vals, deg = [], []
        for _ in range(reps):
            x = simulate_null(rng, n_strat)
            x[:, :n_skilled] += mu_daily
            out = bs.pbo_cscv(x, 16)
            vals.append(out["pbo"])
            deg.append(
                bs.annualise(out["sr_in"].mean(), PPY)
                - bs.annualise(out["sr_out"].mean(), PPY)
            )
        rows.append(
            {
                "skilled_share": share,
                "pbo_mean": float(np.mean(vals)),
                "pbo_sd": float(np.std(vals, ddof=1)),
                "sharpe_drop_in_to_out": float(np.mean(deg)),
            }
        )
    return rows


def study_power(reps=200, n_trials=100, seed=3):
    """How often DSR confirms one real strategy hidden among null ones."""
    rng = np.random.default_rng(seed)
    rows = []
    for true_sr in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        mu = true_sr * SIGMA / np.sqrt(PPY)
        hit_psr, hit_dsr, picked = 0, 0, 0
        for _ in range(reps):
            x = simulate_null(rng, n_trials)
            x[:, 0] += mu
            srs = bs.sharpe_ratio(x)
            k = int(np.argmax(srs))
            r = x[:, k]
            sk, ku = skew(r), kurtosis(r, fisher=False)
            hit_psr += bs.probabilistic_sharpe(srs[k], 0.0, T, sk, ku) > 0.95
            bench = bs.expected_max_sharpe(n_trials, srs.var(ddof=1))
            hit_dsr += bs.probabilistic_sharpe(srs[k], bench, T, sk, ku) > 0.95
            picked += k == 0
        rows.append(
            {
                "true_sharpe": true_sr,
                "p_selected_is_the_real_one": picked / reps,
                "psr95_flags": hit_psr / reps,
                "dsr95_flags": hit_dsr / reps,
            }
        )
    return rows


def make_figures(null_rows, pbo_rows, power_rows):
    n = [r["n_trials"] for r in null_rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].plot(n, [r["mean_best_sharpe"] for r in null_rows], "o-", label="simulated")
    ax[0].plot(n, [r["theory_best_sharpe"] for r in null_rows], "k--", label="formula")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("strategies tried (all zero skill)")
    ax[0].set_ylabel("best annualised Sharpe")
    ax[0].set_title("Luck alone produces a good-looking winner")
    ax[0].legend()

    ax[1].plot(n, [r["false_positive_psr95"] for r in null_rows], "o-", label="PSR > 0.95 (no deflation)")
    ax[1].plot(n, [r["false_positive_dsr95"] for r in null_rows], "s-", label="DSR > 0.95")
    ax[1].axhline(0.05, color="grey", lw=0.8, ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("strategies tried (all zero skill)")
    ax[1].set_ylabel("share of runs that declare a winner")
    ax[1].set_title("False discoveries")
    ax[1].legend()

    ax[2].errorbar(
        [r["skilled_share"] for r in pbo_rows],
        [r["pbo_mean"] for r in pbo_rows],
        yerr=[r["pbo_sd"] for r in pbo_rows],
        fmt="o-",
        capsize=3,
    )
    ax[2].axhline(0.5, color="grey", lw=0.8, ls=":")
    ax[2].set_xlabel("share of 100 candidates with true Sharpe 1")
    ax[2].set_ylabel("probability of backtest overfitting")
    ax[2].set_title("PBO vs share of skilled candidates")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "study_summary.png"), dpi=140)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(5.5, 4))
    s = [r["true_sharpe"] for r in power_rows]
    a.plot(s, [r["psr95_flags"] for r in power_rows], "o-", label="PSR > 0.95")
    a.plot(s, [r["dsr95_flags"] for r in power_rows], "s-", label="DSR > 0.95")
    a.plot(s, [r["p_selected_is_the_real_one"] for r in power_rows], "^--", label="real strategy is the best one")
    a.set_xlabel("true annual Sharpe of the one real strategy (99 others: zero)")
    a.set_ylabel("share of runs")
    a.set_title("Detecting one real edge among 100 trials, 5 years")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "study_power.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    null_rows = study_null_max()
    pbo_rows = study_pbo()
    power_rows = study_power()
    with open(os.path.join(OUT, "study_results.json"), "w") as f:
        json.dump(
            {
                "setup": {"periods": T, "daily_sigma": SIGMA, "periods_per_year": PPY},
                "null_max": null_rows,
                "pbo": pbo_rows,
                "power": power_rows,
            },
            f,
            indent=2,
        )
    make_figures(null_rows, pbo_rows, power_rows)
    print("null max Sharpe (5y, zero skill)")
    for r in null_rows:
        print(
            f"  N={r['n_trials']:5d} best={r['mean_best_sharpe']:.2f} "
            f"theory={r['theory_best_sharpe']:.2f} P(best>1)={r['p_best_sharpe_above_1']:.2f} "
            f"FP(PSR)={r['false_positive_psr95']:.2f} FP(DSR)={r['false_positive_dsr95']:.2f}"
        )
    print("PBO vs share of skilled candidates")
    for r in pbo_rows:
        print(
            f"  share={r['skilled_share']:.2f} PBO={r['pbo_mean']:.2f}"
            f"+-{r['pbo_sd']:.2f} drop={r['sharpe_drop_in_to_out']:.2f}"
        )
    print("power with one real strategy among 100")
    for r in power_rows:
        print(
            f"  SR={r['true_sharpe']:.1f} picked={r['p_selected_is_the_real_one']:.2f} "
            f"PSR={r['psr95_flags']:.2f} DSR={r['dsr95_flags']:.2f}"
        )


if __name__ == "__main__":
    main()
