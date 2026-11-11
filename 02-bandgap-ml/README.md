# bandgap-ml

Composition-only machine learning for materials properties, written to be read:
a formula parser, 33 descriptors built from a small built-in element table, and
a cross-validation pipeline that compares a mean baseline, ridge regression and
a random forest.

```bash
python train.py data/demo_gaps.csv --target gap_eV --out results/
```

Outputs `metrics.json`, `cv_predictions.csv`, `parity.png`, `importances.png`.

## Design choices

- **Descriptors** (`features.py`): fraction-weighted mean, standard deviation,
  minimum, maximum and range of atomic number, mass, Pauling electronegativity,
  group, period and valence-electron count, plus element count, largest
  fraction and mixing entropy. Similar in spirit to Magpie
  (Ward et al., npj Comput. Mater. 2, 16028 (2016)) but far smaller.
- **Leakage control** (`train.py`): formulas are reduced to a canonical
  composition and duplicates merged to their median before splitting, so the
  same compound cannot appear in both training and test folds.
- **Baselines first**: every model is reported next to "predict the mean".

## Demo numbers (not a benchmark)

`data/demo_gaps.csv` holds 46 common compounds with approximate textbook band
gaps, enough to show the pipeline runs. Five-fold cross-validation on it gives:

| Model | MAE (eV) | RMSE (eV) | R² |
|---|---|---|---|
| mean baseline | 2.40 | 3.15 | −0.01 |
| ridge | 1.32 | 1.79 | 0.67 |
| random forest | 1.09 | 1.66 | 0.72 |

With 46 points these figures have large uncertainty and say nothing about
performance on real data.

## Running on a real dataset

The standard benchmark is `matbench_expt_gap` (4,604 experimental gaps; columns
`composition` and `gap expt`). With `pip install matminer`:

```python
from matminer.datasets import load_dataset
load_dataset("matbench_expt_gap").to_csv("data/matbench_expt_gap.csv", index=False)
```

```bash
python train.py data/matbench_expt_gap.csv --formula composition --target "gap expt"
```

Reference points on that task (MAE in eV, from the Matbench leaderboard and
De Breuck et al., arXiv:2102.02263): mean-value dummy 1.14, random forest with
Magpie features 0.45, CrabNet 0.35, MODNet 0.33, current best 0.29. A minimal
descriptor set like this one should land near the random-forest line, not the
top. Note the official protocol uses fixed nested folds; `train.py` uses a plain
shuffled 5-fold, so numbers are comparable only approximately.

## Related work

Composition-only property prediction is a well-developed field: Magpie/matminer
descriptors, ElemNet, Roost, CrabNet and MODNet all have open implementations.
Nothing here is new methodology; the value is a compact, readable baseline with
explicit leakage control.

## Tests

```bash
pytest -q        # 26 tests
```

## Known limits

- Element table covers 73 elements; noble gases and some f-block elements are
  excluded and rows containing them are skipped with a message.
- Electronegativities and masses were entered by hand; spot-check them.
- Composition alone cannot distinguish polymorphs.
