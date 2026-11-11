# Computational materials portfolio

Three small, tested Python projects for computational materials science.
Each folder is self-contained and can become its own GitHub repository.

| Folder | What it does | Status |
|---|---|---|
| `01-vasp-postkit` | Parses VASP `OUTCAR` / `EIGENVAL` / `DOSCAR`, reports band gaps and relaxation convergence, draws band-structure and DOS figures | 9 tests pass; parsers exercised on synthetic files written in VASP format, not yet on real calculations |
| `02-bandgap-ml` | Composition descriptors + cross-validated regression for materials properties | 26 tests pass; shipped numbers come from a 46-compound demo table and are not a benchmark |
| `03-tbc-toolkit` | Thermal boundary conductance: AMM/DMM baselines, Al(111)/Si(111) interface builder, NEMD post-processing, LAMMPS template | 25 tests pass; baselines are real calculations, the NEMD workflow is smoke-tested, production runs pending |

Requirements: Python 3.10+, `numpy`, `matplotlib`; project 02 also needs
`pandas` and `scikit-learn`. Run `pytest` inside any folder.

## Before publishing

1. Run project 01 on one of your own VASP calculations and compare the gap
   and energies with a tool you trust (e.g. pymatgen or vaspkit).
2. Replace the demo table in project 02 with a real dataset (instructions in
   its README) and report those numbers instead.
3. Verify every material constant in `03-tbc-toolkit/mismatch.py` against a
   primary source.
