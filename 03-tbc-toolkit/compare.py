"""Collect finished NEMD runs and set them against the mismatch models and experiment.

    python compare.py                       # table of every runs/*/result.json
    python compare.py --plot g_vs_T.png

Runs whose thermostat energies disagree by more than --max-mismatch (default
5 %) are listed but flagged as not converged and left out of the plot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mismatch as mm

HERE = Path(__file__).resolve().parent


def collect(runs_dir):
    """Read every result.json below runs_dir, sorted by preset then temperature."""
    rows = []
    for f in sorted(Path(runs_dir).glob("*/result.json")):
        r = json.loads(f.read_text())
        rows.append({"run": f.parent.name, "preset": r.get("preset", "?"),
                     "T": float(r["temperature"]), "G": r["G"] / 1e6,
                     "sem": r.get("G_sem", float("nan")) / 1e6,
                     "mismatch": r.get("thermostat_mismatch", float("nan")),
                     "atoms": r.get("n_atoms"), "steps": r.get("md_steps")})
    return sorted(rows, key=lambda r: (r["preset"], r["T"]))


def table(rows, pair=("Al", "Si"), max_mismatch=0.05):
    a, b = mm.MATERIALS[pair[0]], mm.MATERIALS[pair[1]]
    lines = [f"{'run':<16}{'T (K)':>7}{'G_MD':>9}{'+/-':>7}{'DMM':>8}{'AMM':>8}{'expt':>7}  status"]
    for r in rows:
        expt = mm.EXPERIMENT_300K.get(pair) if abs(r["T"] - 300) < 1e-9 else None
        ok = np.isfinite(r["mismatch"]) and r["mismatch"] <= max_mismatch
        lines.append(f"{r['run']:<16}{r['T']:>7.0f}{r['G']:>9.1f}{r['sem']:>7.1f}"
                     f"{mm.dmm_conductance(a, b, r['T']) / 1e6:>8.0f}"
                     f"{mm.amm_conductance(a, b, r['T']) / 1e6:>8.0f}"
                     f"{(f'{expt / 1e6:.0f}' if expt else '-'):>7}  "
                     f"{'ok' if ok else 'not steady (mismatch %.0f %%)' % (100 * r['mismatch'])}")
    return "\n".join(lines)


def plot(rows, out, pair=("Al", "Si"), max_mismatch=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = mm.MATERIALS[pair[0]], mm.MATERIALS[pair[1]]
    ts = np.linspace(50, 700, 66)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(ts, [mm.dmm_conductance(a, b, t) / 1e6 for t in ts], color="#1f4e79", label="DMM")
    ax.plot(ts, [mm.amm_conductance(a, b, t) / 1e6 for t in ts], color="#1f4e79", ls="--",
            label="AMM")
    good = [r for r in rows if np.isfinite(r["mismatch"]) and r["mismatch"] <= max_mismatch]
    for preset, marker in (("quick", "s"), ("full", "o")):
        sel = [r for r in good if r["preset"] == preset]
        if sel:
            ax.errorbar([r["T"] for r in sel], [r["G"] for r in sel], yerr=[r["sem"] for r in sel],
                        fmt=marker, color="#c0392b", capsize=3, label=f"NEMD ({preset})")
    if pair in mm.EXPERIMENT_300K:
        ax.plot(300, mm.EXPERIMENT_300K[pair] / 1e6, "*", color="black", ms=11, label="measured")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$G$ (MW m$^{-2}$ K$^{-1}$)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--max-mismatch", type=float, default=0.05)
    ap.add_argument("--plot")
    args = ap.parse_args(argv)
    rows = collect(args.runs)
    if not rows:
        print(f"no result.json files under {args.runs}")
        return
    print(table(rows, max_mismatch=args.max_mismatch))
    if args.plot:
        plot(rows, args.plot, max_mismatch=args.max_mismatch)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
