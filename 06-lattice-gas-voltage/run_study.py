"""Validation and voltage-curve study for the lattice-gas model.

    python run_study.py --part order     # order parameter vs Onsager, voltage curves
    python run_study.py --part binder    # critical temperature from Binder crossings

The two parts are independent, so they can run side by side.
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lgas

OUT = "example_results"


def part_order():
    rng_seed = 100
    temps = np.round(np.arange(0.20, 0.801, 0.04), 3)
    size = 32
    rows = []
    for k, kT in enumerate(temps):
        init = "checker" if kT < lgas.KTC_EXACT else "random"
        g = lgas.LatticeGas(size, kT, seed=rng_seed + k, init=init)
        s = g.run(3000, 6000)
        m = np.abs(s[:, 1])
        mean, err = lgas.jackknife(m, np.mean)
        rows.append(
            {
                "kT": float(kT),
                "m_mc": float(mean),
                "m_err": float(err),
                "m_exact_infinite_lattice": float(lgas.onsager_order_parameter(kT)),
                "coverage": float(s[:, 0].mean()),
            }
        )
        print(f"order kT={kT:.2f} m={mean:.3f}+-{err:.3f} exact={rows[-1]['m_exact_infinite_lattice']:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    fine = np.linspace(0.2, 0.8, 300)
    ax.plot(fine, lgas.onsager_order_parameter(fine), "k-", label="Onsager (infinite lattice)")
    ax.errorbar(
        [r["kT"] for r in rows],
        [r["m_mc"] for r in rows],
        yerr=[r["m_err"] for r in rows],
        fmt="o",
        ms=4,
        label=f"Monte Carlo, L={size}",
    )
    ax.axvline(lgas.KTC_EXACT, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("$k_B T / J$")
    ax.set_ylabel("staggered order $|x_A - x_B|$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "order_parameter.png"), dpi=140)
    plt.close(fig)

    # Voltage-composition curves: V = (eps0 - mu) / e, plotted as -mu / J
    # with an arbitrary reference. Sweep mu up and then down at fixed kT.
    mus = np.round(np.arange(-1.0, 5.0001, 0.1), 3)
    curves = {}
    for kT in (0.25, 0.45, 0.80):
        g = lgas.LatticeGas(size, kT, mu=mus[0], seed=7, init="empty")
        up, down = [], []
        for mu in mus:
            g.mu = float(mu)
            up.append(g.run(1500, 1500)[:, 0].mean())
        for mu in mus[::-1]:
            g.mu = float(mu)
            down.append(g.run(1500, 1500)[:, 0].mean())
        curves[kT] = {"mu": mus.tolist(), "x_up": up, "x_down": down[::-1]}
        print(f"voltage curve kT={kT} done", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for kT, c in curves.items():
        mu = np.array(c["mu"])
        x = 0.5 * (np.array(c["x_up"]) + np.array(c["x_down"]))
        label = f"$k_BT/J$ = {kT}" + (" (below $T_c$)" if kT < lgas.KTC_EXACT else " (above $T_c$)")
        ax[0].plot(x, -mu, "-", label=label)
        ax[1].plot(mu, np.gradient(x, mu), "-", label=label)
        if kT == 0.25:
            ax[0].plot(c["x_up"], -mu, ":", color="C0", lw=0.8)
            ax[0].plot(c["x_down"], -mu, ":", color="C0", lw=0.8)
    ax[0].set_xlabel("ion fraction x")
    ax[0].set_ylabel(r"voltage $\propto -\mu / J$ (arbitrary offset)")
    ax[0].set_title("Voltage vs composition")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel(r"$\mu / J$")
    ax[1].set_ylabel("dx / d(mu/J)  (differential capacity)")
    ax[1].set_title("Differential capacity")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "voltage_curves.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(OUT, "order_and_voltage.json"), "w") as f:
        json.dump({"size": size, "order": rows, "voltage": {str(k): v for k, v in curves.items()}}, f, indent=2)


def part_binder():
    sizes = [16, 32, 64]
    temps = np.round(np.linspace(0.52, 0.62, 11), 4)
    data = {s: {"U": [], "err": []} for s in sizes}
    for s in sizes:
        t0 = time.time()
        for k, kT in enumerate(temps):
            g = lgas.LatticeGas(s, kT, seed=1000 * s + k, init="random")
            m = g.run(5000, 30000)[:, 1]
            u, err = lgas.jackknife(m, lgas.binder_cumulant)
            data[s]["U"].append(float(u))
            data[s]["err"].append(float(err))
        print(f"binder L={s} done in {time.time() - t0:.0f}s", flush=True)

    tc = {}
    for a, b in [(16, 32), (32, 64)]:
        tc[f"{a}-{b}"] = float(lgas.crossing(temps, data[a]["U"], data[b]["U"]))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for s in sizes:
        ax.errorbar(temps, data[s]["U"], yerr=data[s]["err"], fmt="o-", ms=3, capsize=2, label=f"L = {s}")
    ax.axvline(lgas.KTC_EXACT, color="k", ls=":", lw=0.9, label=f"exact $T_c$ = {lgas.KTC_EXACT:.4f}")
    ax.set_xlabel("$k_B T / J$")
    ax.set_ylabel("Binder cumulant of staggered order")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "binder_crossing.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(OUT, "binder.json"), "w") as f:
        json.dump(
            {"temps": temps.tolist(), "data": {str(k): v for k, v in data.items()}, "crossings": tc, "exact_kTc": float(lgas.KTC_EXACT)},
            f,
            indent=2,
        )
    print("crossings", tc, "exact", lgas.KTC_EXACT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["order", "binder"], required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"order": part_order, "binder": part_binder}[args.part]()
