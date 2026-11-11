"""Train and cross-validate composition-only regressors for a materials property.

    python train.py data/demo_gaps.csv --target gap_eV --out results/

Input: a CSV with a formula column and a numeric target column.
Output: metrics.json, cv_predictions.csv, parity.png, importances.png.

Leakage control: rows are grouped by reduced composition, so polymorphs or
repeated measurements of one compound never sit on both sides of a split, and
duplicates are collapsed to their median before training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, featurize_many, reduced_formula_key


def load_dataset(path, formula_col="formula", target_col="target"):
    """Read the CSV, drop unparsable formulas, merge duplicate compositions."""
    df = pd.read_csv(path, comment="#")
    keys, bad = [], []
    for f in df[formula_col]:
        try:
            keys.append(reduced_formula_key(str(f)))
        except (KeyError, ValueError) as exc:
            keys.append(None)
            bad.append((f, str(exc)))
    df["_key"] = keys
    df = df.dropna(subset=["_key", target_col])
    merged = (df.groupby("_key", sort=True)
                .agg(formula=(formula_col, "first"), target=(target_col, "median"))
                .reset_index(drop=True))
    return merged, bad


def models(seed=0):
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=1,
                                               n_jobs=-1, random_state=seed),
    }


def evaluate(X, y, n_splits=5, seed=0):
    """K-fold cross-validated predictions and metrics for every model."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds, metrics = {}, {}
    for name, model in models(seed).items():
        p = cross_val_predict(model, X, y, cv=cv)
        preds[name] = p
        metrics[name] = {"MAE": float(mean_absolute_error(y, p)),
                         "RMSE": float(root_mean_squared_error(y, p)),
                         "R2": float(r2_score(y, p))}
    return preds, metrics


def make_plots(y, preds, X, outdir, unit="eV", seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    p = preds["random_forest"]
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lim, lim, color="0.5", lw=0.8)
    ax.scatter(y, p, s=14, color="#1f4e79", alpha=0.7, lw=0)
    ax.set_xlabel(f"reference ({unit})")
    ax.set_ylabel(f"cross-validated prediction ({unit})")
    ax.set_title(f"random forest, MAE = {mean_absolute_error(y, p):.2f} {unit}", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "parity.png", dpi=300)
    plt.close(fig)

    rf = models(seed)["random_forest"].fit(X, y)
    order = np.argsort(rf.feature_importances_)[-10:]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.barh([FEATURE_NAMES[i] for i in order], rf.feature_importances_[order], color="#1f4e79")
    ax.set_xlabel("impurity-based importance")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "importances.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--formula", default="formula")
    ap.add_argument("--target", default="target")
    ap.add_argument("--out", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    data, bad = load_dataset(args.csv, args.formula, args.target)
    if bad:
        print(f"skipped {len(bad)} rows with unsupported formulas, e.g. {bad[0]}")
    X = featurize_many(data["formula"])
    y = data["target"].to_numpy(float)
    preds, metrics = evaluate(X, y, args.folds, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"n_compounds": int(len(y)), "n_features": int(X.shape[1]),
               "cv_folds": args.folds, "seed": args.seed, "metrics": metrics}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"formula": data["formula"], "reference": y, **preds}).to_csv(
        out / "cv_predictions.csv", index=False)
    make_plots(y, preds, X, out, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
