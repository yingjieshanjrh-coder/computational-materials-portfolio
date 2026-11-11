"""Generate SYNTHETIC files in VASP output formats for testing and demos.

These are NOT results of a real DFT calculation. They follow the layout of
OUTCAR / EIGENVAL / DOSCAR / POSCAR closely enough to exercise the parsers,
and encode a model band structure whose gaps are known exactly:

    fundamental gap = 0.60 eV (indirect, VBM at Gamma, CBM at (0.4, 0, 0.4))
    direct gap      = 0.985 eV (at (0.25, 0, 0.25) on the sampled path)
"""
from pathlib import Path

import numpy as np

GAP, K0, CURV = 0.60, np.array([0.4, 0.0, 0.4]), 3.0
VBM_ABS = 5.0            # absolute energy of the VBM in the fake files (eV)
A_SI = 5.431


def kpath():
    seg1 = np.linspace([0.5, 0.5, 0.5], [0, 0, 0], 20)       # L -> Gamma
    seg2 = np.linspace([0, 0, 0], [0.5, 0, 0.5], 21)         # Gamma -> X
    return np.vstack([seg1, seg2])


def model_bands(k):
    k2 = (k ** 2).sum(axis=1)
    valence = [VBM_ABS - 1.5 * n - (2.0 + n) * k2 for n in range(4)][::-1]
    dk2 = ((k - K0) ** 2).sum(axis=1)
    conduction = [VBM_ABS + GAP + 2.0 * n + CURV * dk2 for n in range(4)]
    return np.array(valence + conduction).T                  # (nk, 8)


def write_all(outdir):
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    k = kpath()
    e = model_bands(k)
    nk, nb = e.shape

    # POSCAR (fcc primitive cell of silicon)
    h = A_SI / 2
    (d / "POSCAR").write_text(
        "Si2 synthetic\n1.0\n"
        f"0.0 {h} {h}\n{h} 0.0 {h}\n{h} {h} 0.0\n"
        "Si\n2\nDirect\n0.00 0.00 0.00\n0.25 0.25 0.25\n")

    # EIGENVAL
    lines = ["    2    2    1    1", "  0.1 0.1 0.1 0.1 0.5E-15", "  1.0E-04",
             "  CAR", " synthetic", f"      8   {nk}    {nb}"]
    for i in range(nk):
        lines.append("")
        lines.append("  %.7E  %.7E  %.7E  %.7E" % (*k[i], 1.0 / nk))
        for b in range(nb):
            lines.append("  %3d     %12.6f   %8.6f" % (b + 1, e[i, b], 1.0 if b < 4 else 0.0))
    (d / "EIGENVAL").write_text("\n".join(lines) + "\n")

    # DOSCAR (Gaussian-broadened histogram of the model bands)
    grid = np.linspace(-5, 15, 301)
    sigma = 0.15
    dos = np.exp(-0.5 * ((grid[:, None] - e.ravel()[None, :]) / sigma) ** 2).sum(1)
    dos *= 2.0 / (nk * sigma * np.sqrt(2 * np.pi))           # 2 electrons per band
    idos = np.cumsum(dos) * (grid[1] - grid[0])
    fermi = VBM_ABS + GAP / 2
    lines = ["   2   2   0   0", "  0.1 0.1 0.1 0.1 0.5E-15", "  1.0E-04", "  CAR",
             " synthetic", "  %.8f  %.8f  %d  %.8f  1.0" % (grid[-1], grid[0], len(grid), fermi)]
    lines += ["  %10.4f  %.4E  %.4E" % row for row in zip(grid, dos, idos)]
    (d / "DOSCAR").write_text("\n".join(lines) + "\n")

    # OUTCAR (five ionic steps of a fake relaxation)
    energies = [-10.60, -10.79, -10.835, -10.8440, -10.8456]
    fmax = [0.80, 0.30, 0.09, 0.02, 0.004]
    txt = ["   number of dos      NEDOS =    301   number of ions     NIONS =      2", ""]
    for en, f in zip(energies, fmax):
        txt += [f" E-fermi :   {fermi:.4f}     XC(G=0): -10.0000     alpha+bet :-10.0000", "",
                " POSITION                                       TOTAL-FORCE (eV/Angst)",
                " " + "-" * 83,
                "      0.00000      0.00000      0.00000    %12.6f %12.6f %12.6f" % (f, 0.0, 0.0),
                "      1.35775      1.35775      1.35775    %12.6f %12.6f %12.6f" % (-f, 0.0, 0.0),
                " " + "-" * 83,
                "    total drift:                                0.000000      0.000000      0.000000",
                "", "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
                "  ---------------------------------------------------",
                f"  free  energy   TOTEN  =       {en:.8f} eV", ""]
    txt.append(" reached required accuracy - stopping structural energy minimisation")
    (d / "OUTCAR").write_text("\n".join(txt) + "\n")
    return d


if __name__ == "__main__":
    print("wrote synthetic files to", write_all(Path(__file__).parent / "example_synthetic"))
