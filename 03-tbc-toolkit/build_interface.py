"""Build an Al(111)/Si(111) interface supercell for NEMD and write a LAMMPS data file.

Pure numpy. The lattice match is 4 x 4 Al(111) cells on 3 x 3 Si(111) cells
(orthorhombic setting); Al is strained in-plane by about +0.57 % to fit Si.

    python build_interface.py --si-bilayers 40 --al-layers 60 -o alsi.data

The script also writes <output>.regions.lmp with the z-boundaries of the fixed
walls and thermostat regions, which the LAMMPS template reads.

The interface separation is a starting guess: relax the structure (and check
the registry of Al on Si) with your interatomic potential before production.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

A_AL, A_SI = 4.0495, 5.4310          # lattice constants, Angstrom
MASS = {"Si": 28.085, "Al": 26.982}


def fcc111_layers(a, nx, ny, nlayers, ax=None, ay=None):
    """fcc(111) slab in an orthorhombic cell. Returns (positions, (Lx, Ly), d111).

    Per layer the cell x = a/sqrt(2), y = sqrt(3/2) a holds two atoms; successive
    layers are shifted by one third of the cell along y (ABC stacking).
    ax, ay override the in-plane cell lengths to apply epitaxial strain.
    """
    ax = a / np.sqrt(2) if ax is None else ax
    ay = a * np.sqrt(1.5) if ay is None else ay
    d = a / np.sqrt(3)
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cells = np.stack([ix.ravel(), iy.ravel()], axis=1).astype(float)
    basis = np.array([[0.0, 0.0], [0.5, 0.5]])
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 2)
    pos = []
    for k in range(nlayers):
        f = frac + np.array([0.0, (k % 3) / 3.0])
        f[:, 1] %= ny
        pos.append(np.column_stack([f[:, 0] * ax, f[:, 1] * ay, np.full(len(f), k * d)]))
    return np.vstack(pos), (nx * ax, ny * ay), d


def diamond111_bilayers(a, nx, ny, nbilayers):
    """Diamond-cubic (111) slab terminated between bilayers (one dangling bond per
    surface atom on both faces). Returns (positions, (Lx, Ly))."""
    base, cell, d = fcc111_layers(a, nx, ny, nbilayers + 1)
    partner = base + np.array([0.0, 0.0, a * np.sqrt(3) / 4])
    per_layer = 2 * nx * ny
    pos = np.vstack([partner[: nbilayers * per_layer],     # drop top lone sublayer
                     base[per_layer:]])                    # drop bottom lone sublayer
    pos[:, 2] -= pos[:, 2].min()
    return pos, cell


def build_al_si(si_bilayers=40, al_layers=60, nrep=1, gap=2.4, pad=5.0):
    """Si slab at the bottom, Al slab on top. Returns dict with positions, types, box.

    nrep multiplies the in-plane size (cross-section convergence tests).
    Types: 1 = Si, 2 = Al.
    """
    si, (lx, ly) = diamond111_bilayers(A_SI, 3 * nrep, 3 * nrep, si_bilayers)
    al, _, _ = fcc111_layers(A_AL, 4 * nrep, 4 * nrep, al_layers,
                             ax=lx / (4 * nrep), ay=ly / (4 * nrep))
    strain = lx / (4 * nrep) / (A_AL / np.sqrt(2)) - 1.0
    z_int = si[:, 2].max() + gap / 2
    al[:, 2] += si[:, 2].max() + gap
    pos = np.vstack([si, al])
    pos[:, 2] += pad
    z_int += pad
    types = np.concatenate([np.ones(len(si), int), np.full(len(al), 2)])
    box = np.array([lx, ly, pos[:, 2].max() + pad])
    return {"positions": pos, "types": types, "box": box, "z_interface": z_int,
            "al_strain": strain, "n_si": len(si), "n_al": len(al)}


def write_lammps_data(path, s, comment="Al(111)/Si(111) interface"):
    pos, types, box = s["positions"], s["types"], s["box"]
    lines = [comment, "", f"{len(pos)} atoms", "2 atom types", "",
             f"0.0 {box[0]:.8f} xlo xhi", f"0.0 {box[1]:.8f} ylo yhi",
             f"0.0 {box[2]:.8f} zlo zhi", "", "Masses", "",
             f"1 {MASS['Si']}", f"2 {MASS['Al']}", "", "Atoms # atomic", ""]
    lines += [f"{i + 1} {t} {x:.6f} {y:.6f} {z:.6f}"
              for i, (t, (x, y, z)) in enumerate(zip(types, pos))]
    Path(path).write_text("\n".join(lines) + "\n")


def write_regions(path, s, wall=6.0, bath=20.0):
    """z-limits (Angstrom) of fixed walls and thermostat slabs as LAMMPS variables."""
    z = s["positions"][:, 2]
    lo, hi = z.min(), z.max()
    v = {"z_wall_lo": lo + wall, "z_cold_hi": lo + wall + bath,
         "z_hot_lo": hi - wall - bath, "z_wall_hi": hi - wall,
         "z_interface": s["z_interface"], "area": s["box"][0] * s["box"][1]}
    if not v["z_cold_hi"] < v["z_interface"] < v["z_hot_lo"]:
        raise ValueError("slabs are too thin for the requested wall and bath thicknesses")
    Path(path).write_text("".join(f"variable {k} equal {val:.6f}\n" for k, val in v.items()))
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--si-bilayers", type=int, default=40)
    ap.add_argument("--al-layers", type=int, default=60)
    ap.add_argument("--nrep", type=int, default=1)
    ap.add_argument("--gap", type=float, default=2.4, help="initial Si-Al separation (A)")
    ap.add_argument("-o", "--output", default="alsi.data")
    args = ap.parse_args(argv)
    s = build_al_si(args.si_bilayers, args.al_layers, args.nrep, args.gap)
    write_lammps_data(args.output, s)
    v = write_regions(args.output + ".regions.lmp", s)
    print(f"{len(s['positions'])} atoms ({s['n_si']} Si, {s['n_al']} Al), "
          f"box {s['box'][0]:.3f} x {s['box'][1]:.3f} x {s['box'][2]:.3f} A")
    print(f"Al in-plane strain {100 * s['al_strain']:+.2f} %, interface at z = "
          f"{v['z_interface']:.2f} A, cross-section {v['area']:.1f} A^2")


if __name__ == "__main__":
    main()
