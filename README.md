# Computational materials portfolio

Six small, tested Python projects in computational materials science,
materials markets and quantitative finance.
Each folder is self-contained and can become its own GitHub repository.

| Folder | What it does | Status |
|---|---|---|
| `01-vasp-postkit` | Parses VASP `OUTCAR` / `EIGENVAL` / `DOSCAR`, reports band gaps and relaxation convergence, draws band-structure and DOS figures | 9 tests pass; parsers exercised on synthetic files written in VASP format, not yet on real calculations |
| `02-bandgap-ml` | Composition descriptors + cross-validated regression for materials properties | 26 tests pass; shipped numbers come from a 46-compound demo table and are not a benchmark |
| `03-tbc-toolkit` | Thermal boundary conductance: AMM/DMM baselines, Al(111)/Si(111) interface builder, NEMD post-processing, LAMMPS template | 26 tests pass; baselines are real calculations, the NEMD workflow is smoke-tested, production runs pending |
| `04-battery-metals-cost` | Metal intensity of battery cathodes from stoichiometry, price sensitivity, break-even prices, market statistics | 21 tests pass; intensities are computed, shipped prices are an illustrative scenario |
| `05-backtest-overfitting` | Probabilistic and deflated Sharpe ratio, probability of backtest overfitting (CSCV), and a simulation study of how much selecting a winner inflates a backtest | 18 tests pass; simulated data only, formulas checked numerically against the study |
| `06-lattice-gas-voltage` | Lattice-gas Monte Carlo of ion ordering; voltage steps and differential capacity | 13 tests pass; order parameter and critical temperature checked against the exact Onsager solution; 2D model, no specific material |

Requirements: Python 3.10+, `numpy`, `matplotlib`; project 05 also needs `scipy`; projects 02 and 04 also need
`pandas`; 02 needs `scikit-learn`, 04 needs `openpyxl` for Excel input. Run `pytest` inside any folder.

## Before publishing

1. Run project 01 on one of your own VASP calculations and compare the gap
   and energies with a tool you trust (e.g. pymatgen or vaspkit).
2. Replace the demo table in project 02 with a real dataset (instructions in
   its README) and report those numbers instead.
3. Verify every material constant in `03-tbc-toolkit/mismatch.py` against a
   primary source.
4. Replace the price scenario in project 04 with dated, sourced prices.
