"""Build the figures and a short written brief.

    python report.py --scenario data/prices_scenario.csv --out report/
    python report.py --scenario data/prices_scenario.csv \
        --pink-sheet CMO-Historical-Data-Monthly.xlsx --since 2015 --out report/

The scenario file gives low / base / high prices per kg of contained metal.
The optional World Bank workbook adds a price-history section.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import cost
import market
from chemistry import CATHODES

COLOURS = {"Li": "#1f4e79", "Na": "#6a9fca", "Ni": "#2e7d32", "Co": "#c0392b",
           "Mn": "#8e6bbf", "Fe": "#8d6e63", "Al": "#9e9e9e", "P": "#d4a017"}
# World Bank series name -> (metal symbol, factor to USD/kg)
PINK_SHEET_METALS = {"Nickel": ("Ni", 1e-3), "Aluminum": ("Al", 1e-3), "Copper": ("Cu", 1e-3)}


def read_scenario(path):
    table = pd.read_csv(path, comment="#").set_index("metal")
    return (table["base"].to_dict(), table["low"].to_dict(), table["high"].to_dict())


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_intensity(out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = list(CATHODES)
    bottom = [0.0] * len(names)
    for metal in COLOURS:
        vals = [CATHODES[n].intensity().get(metal, 0.0) for n in names]
        if any(vals):
            ax.bar(names, vals, bottom=bottom, color=COLOURS[metal], label=metal)
            bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("contained element (kg per kWh)")
    ax.legend(ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_cost(base, low, high, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = list(CATHODES)
    bottom = [0.0] * len(names)
    for metal in COLOURS:
        vals = [cost.metal_cost(CATHODES[n], base, "zero").get(metal, 0.0) for n in names]
        if any(vals):
            ax.bar(names, vals, bottom=bottom, color=COLOURS[metal], label=metal)
            bottom = [b + v for b, v in zip(bottom, vals)]
    lo = [cost.total_cost(CATHODES[n], low, "zero") for n in names]
    hi = [cost.total_cost(CATHODES[n], high, "zero") for n in names]
    ax.errorbar(names, bottom, yerr=[[b - l for b, l in zip(bottom, lo)],
                                     [h - b for b, h in zip(bottom, hi)]],
                fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_ylabel("contained-metal cost (USD per kWh)")
    ax.legend(ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_tornado(name, base, low, high, out):
    plt = _plt()
    rows = cost.tornado(CATHODES[name], base, low, high, "zero")[::-1]
    centre = cost.total_cost(CATHODES[name], base, "zero")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for i, (metal, lo, hi) in enumerate(rows):
        ax.barh(i, hi - lo, left=lo, color=COLOURS.get(metal, "0.5"))
    ax.axvline(centre, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(f"{name} contained-metal cost (USD per kWh)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def build(scenario, out, pink_sheet=None, since=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base, low, high = read_scenario(scenario)
    plot_intensity(out / "intensity.png")
    plot_cost(base, low, high, out / "cost.png")
    plot_tornado("NMC811", base, low, high, out / "tornado_nmc811.png")

    lines = ["# Battery cathode metals: intensity and price exposure", "",
             f"Price scenario: `{scenario}` (USD per kg of contained metal).", "",
             "| Cathode | CAM kg/kWh | Low | Base | High | Largest driver |",
             "|---|---|---|---|---|---|"]
    for name, c in CATHODES.items():
        driver = cost.tornado(c, base, low, high, "zero")[0][0]
        lines.append(f"| {name} | {c.cam_per_kwh:.2f} | {cost.total_cost(c, low, 'zero'):.1f} | "
                     f"{cost.total_cost(c, base, 'zero'):.1f} | "
                     f"{cost.total_cost(c, high, 'zero'):.1f} | {driver} |")
    lines += ["", "Costs are USD per kWh of contained metal at the active-material level.", ""]
    be = cost.breakeven_price(CATHODES["LFP"], CATHODES["Na-NFM111"], base, "Li", "zero")
    if be is not None:
        verdict = (f"{be:.1f} USD/kg Li, i.e. about "
                   f"{be * 0.1879:.1f} USD/kg of lithium carbonate" if be > 0 else
                   "no positive lithium price (LFP is cheaper on metals throughout)")
        lines += [f"LFP and Na-NFM111 have equal contained-metal cost at {verdict}, with the "
                  "other prices at their base values.", ""]

    if pink_sheet:
        prices = market.load_pink_sheet(pink_sheet)
        cols = [c for c in prices.columns if c in PINK_SHEET_METALS]
        hist = prices[cols].rename(columns={c: PINK_SHEET_METALS[c][0] for c in cols})
        for c in cols:
            hist[PINK_SHEET_METALS[c][0]] *= PINK_SHEET_METALS[c][1]
        if since:
            hist = hist.loc[str(since):]
        stats = market.summary(hist)
        lines += [f"## Price history ({hist.index[0]:%Y-%m} to {hist.index[-1]:%Y-%m}, World Bank)",
                  "", "| Metal | Last USD/kg | Change % | Annualised vol % | Max drawdown % |",
                  "|---|---|---|---|---|"]
        for metal, r in stats.iterrows():
            lines.append(f"| {metal} | {r['last']:.2f} | {r['change_%']:.0f} | "
                         f"{r['ann_vol_%']:.0f} | {r['max_drawdown_%']:.0f} |")
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        for metal in hist.columns:
            ax.plot(hist.index, hist[metal] / hist[metal].dropna().iloc[0] * 100,
                    label=metal, color=COLOURS.get(metal, "#b87333"))
        ax.set_ylabel("price index (start = 100)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "price_history.png", dpi=300)
        plt.close(fig)
    (out / "brief.md").write_text("\n".join(lines) + "\n")
    return out / "brief.md"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="data/prices_scenario.csv")
    ap.add_argument("--pink-sheet")
    ap.add_argument("--since", type=int)
    ap.add_argument("--out", default="report")
    args = ap.parse_args(argv)
    path = build(args.scenario, args.out, args.pink_sheet, args.since)
    print(path.read_text())


if __name__ == "__main__":
    main()
