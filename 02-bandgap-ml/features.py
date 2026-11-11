"""Composition-based descriptors for inorganic compounds (no external data files).

A chemical formula is turned into a fixed-length vector of statistics over
elemental properties, in the spirit of the Magpie descriptor set
(Ward et al., npj Comput. Mater. 2, 16028 (2016)), but with a deliberately
small built-in element table so the package has no dependencies beyond numpy.
"""
from __future__ import annotations

import re

import numpy as np

# symbol: (atomic number, standard atomic weight, Pauling electronegativity)
# Noble gases and a few f-block elements are omitted (no reliable Pauling value).
ELEMENTS = {
    "H": (1, 1.008, 2.20), "Li": (3, 6.94, 0.98), "Be": (4, 9.012, 1.57),
    "B": (5, 10.81, 2.04), "C": (6, 12.011, 2.55), "N": (7, 14.007, 3.04),
    "O": (8, 15.999, 3.44), "F": (9, 18.998, 3.98), "Na": (11, 22.990, 0.93),
    "Mg": (12, 24.305, 1.31), "Al": (13, 26.982, 1.61), "Si": (14, 28.085, 1.90),
    "P": (15, 30.974, 2.19), "S": (16, 32.06, 2.58), "Cl": (17, 35.45, 3.16),
    "K": (19, 39.098, 0.82), "Ca": (20, 40.078, 1.00), "Sc": (21, 44.956, 1.36),
    "Ti": (22, 47.867, 1.54), "V": (23, 50.942, 1.63), "Cr": (24, 51.996, 1.66),
    "Mn": (25, 54.938, 1.55), "Fe": (26, 55.845, 1.83), "Co": (27, 58.933, 1.88),
    "Ni": (28, 58.693, 1.91), "Cu": (29, 63.546, 1.90), "Zn": (30, 65.38, 1.65),
    "Ga": (31, 69.723, 1.81), "Ge": (32, 72.630, 2.01), "As": (33, 74.922, 2.18),
    "Se": (34, 78.971, 2.55), "Br": (35, 79.904, 2.96), "Rb": (37, 85.468, 0.82),
    "Sr": (38, 87.62, 0.95), "Y": (39, 88.906, 1.22), "Zr": (40, 91.224, 1.33),
    "Nb": (41, 92.906, 1.60), "Mo": (42, 95.95, 2.16), "Ru": (44, 101.07, 2.20),
    "Rh": (45, 102.906, 2.28), "Pd": (46, 106.42, 2.20), "Ag": (47, 107.868, 1.93),
    "Cd": (48, 112.414, 1.69), "In": (49, 114.818, 1.78), "Sn": (50, 118.710, 1.96),
    "Sb": (51, 121.760, 2.05), "Te": (52, 127.60, 2.10), "I": (53, 126.904, 2.66),
    "Cs": (55, 132.905, 0.79), "Ba": (56, 137.327, 0.89), "La": (57, 138.905, 1.10),
    "Ce": (58, 140.116, 1.12), "Pr": (59, 140.908, 1.13), "Nd": (60, 144.242, 1.14),
    "Sm": (62, 150.36, 1.17), "Gd": (64, 157.25, 1.20), "Dy": (66, 162.500, 1.22),
    "Ho": (67, 164.930, 1.23), "Er": (68, 167.259, 1.24), "Tm": (69, 168.934, 1.25),
    "Lu": (71, 174.967, 1.27), "Hf": (72, 178.49, 1.30), "Ta": (73, 180.948, 1.50),
    "W": (74, 183.84, 2.36), "Re": (75, 186.207, 1.90), "Os": (76, 190.23, 2.20),
    "Ir": (77, 192.217, 2.20), "Pt": (78, 195.084, 2.28), "Au": (79, 196.967, 2.54),
    "Hg": (80, 200.592, 2.00), "Tl": (81, 204.38, 1.62), "Pb": (82, 207.2, 2.33),
    "Bi": (83, 208.980, 2.02),
}

_NOBLE = (0, 2, 10, 18, 36, 54, 86)


def period_group(z: int) -> tuple[int, int]:
    """Period and IUPAC group (1-18) from the atomic number; lanthanides -> group 3."""
    period = next(i for i in range(1, 7) if z <= _NOBLE[i])
    n = z - _NOBLE[period - 1]                 # position within the period
    if period == 1:
        group = 1 if n == 1 else 18
    elif period in (2, 3):
        group = n if n <= 2 else n + 10
    elif period in (4, 5):
        group = n
    else:                                       # period 6
        if n <= 2:
            group = n
        elif n <= 17:                           # La ... Lu
            group = 3
        else:
            group = n - 14
    return period, group


def valence_electrons(group: int) -> int:
    return group if group <= 12 else group - 10


_TOKEN = re.compile(r"([A-Z][a-z]?)|(\()|(\))|(\d+\.?\d*|\.\d+)")


def parse_formula(formula: str) -> dict[str, float]:
    """'Ba(TiO3)' -> {'Ba': 1, 'Ti': 1, 'O': 3}. Supports nested brackets and decimals."""
    text = formula.replace(" ", "").replace("[", "(").replace("]", ")")
    stack: list[dict[str, float]] = [{}]
    pos, last = 0, None                         # last = most recent element or group
    for m in _TOKEN.finditer(text):
        if m.start() != pos:
            raise ValueError(f"Cannot parse formula {formula!r} at position {pos}")
        pos = m.end()
        el, lpar, rpar, num = m.groups()
        if el:
            if el not in ELEMENTS:
                raise KeyError(f"Element {el!r} in {formula!r} is not in the built-in table")
            stack[-1][el] = stack[-1].get(el, 0) + 1
            last = {el: 1.0}
        elif lpar:
            stack.append({})
            last = None
        elif rpar:
            if len(stack) == 1:
                raise ValueError(f"Unbalanced ')' in {formula!r}")
            last = stack.pop()
            for k, v in last.items():
                stack[-1][k] = stack[-1].get(k, 0) + v
        else:
            if last is None:
                raise ValueError(f"Number without element in {formula!r}")
            for k, v in last.items():           # we already added x1; add the rest
                stack[-1][k] += v * (float(num) - 1)
            last = None
    if pos != len(text) or len(stack) != 1 or not stack[0]:
        raise ValueError(f"Cannot parse formula {formula!r}")
    return {k: v for k, v in stack[0].items() if v > 0}


def reduced_formula_key(formula: str) -> str:
    """Canonical key for de-duplication: fractions rounded, elements sorted."""
    comp = parse_formula(formula)
    total = sum(comp.values())
    return " ".join(f"{el}{comp[el] / total:.4f}" for el in sorted(comp))


PROPERTIES = ("Z", "mass", "electronegativity", "group", "period", "valence")
STATS = ("mean", "std", "min", "max", "range")
FEATURE_NAMES = [f"{p}_{s}" for p in PROPERTIES for s in STATS] + [
    "n_elements", "max_fraction", "entropy_mixing"]


def featurize(formula: str) -> np.ndarray:
    """Fraction-weighted statistics of elemental properties (33 features)."""
    comp = parse_formula(formula)
    frac = np.array(list(comp.values()), dtype=float)
    frac /= frac.sum()
    rows = []
    for el in comp:
        z, mass, en = ELEMENTS[el]
        period, group = period_group(z)
        rows.append((z, mass, en, group, period, valence_electrons(group)))
    props = np.array(rows, dtype=float)         # (n_elements, n_properties)
    feats = []
    for j in range(props.shape[1]):
        x = props[:, j]
        mean = float(frac @ x)
        feats += [mean, float(np.sqrt(frac @ (x - mean) ** 2)), x.min(), x.max(),
                  x.max() - x.min()]
    feats += [len(comp), frac.max(), float(-(frac * np.log(frac)).sum())]
    return np.array(feats)


def featurize_many(formulas) -> np.ndarray:
    return np.vstack([featurize(f) for f in formulas])
