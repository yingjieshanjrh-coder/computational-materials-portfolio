"""vasppost - lightweight post-processing for VASP output files.

Depends only on numpy and matplotlib. Parses OUTCAR, EIGENVAL, DOSCAR and
(optionally) POSCAR, then reports band gaps, relaxation convergence and
produces publication-style figures.

Usage
-----
    python vasppost.py summary  <calc_dir>            # JSON summary to stdout
    python vasppost.py bands    <calc_dir> -o bands.png
    python vasppost.py dos      <calc_dir> -o dos.png
    python vasppost.py relax    <calc_dir> -o relax.png
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "parse_outcar", "parse_eigenval", "parse_doscar", "parse_poscar_lattice",
    "band_gap", "kpath_distance", "plot_bands", "plot_dos", "plot_relaxation",
    "summarise",
]


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
@dataclass
class Outcar:
    energies: np.ndarray          # free energy TOTEN per ionic step (eV)
    max_forces: np.ndarray        # largest |F| on any atom per ionic step (eV/A)
    forces: list                  # list of (nions, 3) arrays
    fermi: float | None           # last reported Fermi level (eV)
    nions: int | None
    converged: bool               # "reached required accuracy" was printed


def parse_outcar(path) -> Outcar:
    """Read energies, forces, Fermi level and convergence flag from an OUTCAR."""
    lines = Path(path).read_text(errors="replace").splitlines()
    energies, forces = [], []
    fermi, nions, converged = None, None, False
    i = 0
    while i < len(lines):
        line = lines[i]
        if "free  energy   TOTEN" in line:
            energies.append(float(line.split("=")[1].split()[0]))
        elif "E-fermi" in line:
            fermi = float(line.split(":")[1].split()[0])
        elif "NIONS" in line:
            m = re.search(r"NIONS\s*=\s*(\d+)", line)
            if m:
                nions = int(m.group(1))
        elif "TOTAL-FORCE" in line:
            i += 2                                   # skip header and dashed line
            block = []
            while i < len(lines) and not lines[i].lstrip().startswith("---"):
                block.append([float(x) for x in lines[i].split()[3:6]])
                i += 1
            forces.append(np.array(block))
        elif "reached required accuracy" in line:
            converged = True
        i += 1
    max_forces = np.array([np.linalg.norm(f, axis=1).max() for f in forces])
    return Outcar(np.array(energies), max_forces, forces, fermi, nions, converged)


@dataclass
class Eigenval:
    kpoints: np.ndarray           # (nk, 3) fractional coordinates
    weights: np.ndarray           # (nk,)
    energies: np.ndarray          # (nspin, nk, nbands) in eV
    occupations: np.ndarray | None  # same shape, or None for old VASP versions
    nelect: int

    @property
    def nspin(self) -> int:
        return self.energies.shape[0]


def parse_eigenval(path) -> Eigenval:
    """Read an EIGENVAL file (ISPIN = 1 or 2, with or without occupations)."""
    lines = Path(path).read_text().splitlines()
    nspin = int(lines[0].split()[3])
    nelect, nk, nb = (int(float(x)) for x in lines[5].split()[:3])
    body = [ln.split() for ln in lines[6:] if ln.strip()]
    if len(body) != nk * (nb + 1):
        raise ValueError(
            f"EIGENVAL: expected {nk * (nb + 1)} data lines, found {len(body)}")
    kpts = np.empty((nk, 3))
    wts = np.empty(nk)
    en = np.empty((nspin, nk, nb))
    has_occ = len(body[1]) == 1 + 2 * nspin
    occ = np.empty((nspin, nk, nb)) if has_occ else None
    row = 0
    for k in range(nk):
        vals = [float(x) for x in body[row]]
        kpts[k], wts[k] = vals[:3], vals[3]
        row += 1
        for b in range(nb):
            vals = [float(x) for x in body[row][1:]]
            en[:, k, b] = vals[:nspin]
            if has_occ:
                occ[:, k, b] = vals[nspin:2 * nspin]
            row += 1
    return Eigenval(kpts, wts, en, occ, nelect)


@dataclass
class Doscar:
    energies: np.ndarray          # (nedos,)
    dos: np.ndarray               # (nspin, nedos) states/eV
    integrated: np.ndarray        # (nspin, nedos)
    fermi: float


def parse_doscar(path) -> Doscar:
    """Read the total DOS block of a DOSCAR."""
    lines = Path(path).read_text().splitlines()
    head = lines[5].split()
    nedos, fermi = int(head[2]), float(head[3])
    data = np.array([[float(x) for x in ln.split()] for ln in lines[6:6 + nedos]])
    nspin = (data.shape[1] - 1) // 2
    return Doscar(data[:, 0], data[:, 1:1 + nspin].T,
                  data[:, 1 + nspin:1 + 2 * nspin].T, fermi)


def parse_poscar_lattice(path) -> np.ndarray:
    """Return the 3x3 lattice (rows = vectors, Angstrom) from POSCAR/CONTCAR."""
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                                    # negative scale = target volume
        scale = (-scale / abs(np.linalg.det(lat))) ** (1 / 3)
    return lat * scale


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _occupied_mask(eig: Eigenval, fermi: float | None) -> np.ndarray:
    if eig.occupations is not None:
        return eig.occupations > 0.5
    if fermi is not None:
        return eig.energies <= fermi
    if eig.nspin == 1:
        mask = np.zeros_like(eig.energies, dtype=bool)
        mask[:, :, : eig.nelect // 2] = True
        return mask
    raise ValueError("Spin-polarised EIGENVAL without occupations needs a Fermi level")


def band_gap(eig: Eigenval, fermi: float | None = None) -> dict:
    """Fundamental and direct band gap.

    Occupied states are identified from occupations if present, otherwise from
    the Fermi level, otherwise from the electron count (non-spin-polarised).
    A system is reported as a metal (gap = 0) when any band is occupied at
    some k-points and empty at others, i.e. it crosses the Fermi level.
    """
    occ = _occupied_mask(eig, fermi)
    e = eig.energies
    if occ.all() or not occ.any():
        raise ValueError("Cannot define a gap: all states occupied or all empty")
    e_occ = np.where(occ, e, -np.inf)
    e_emp = np.where(occ, np.inf, e)
    vbm_idx = np.unravel_index(np.argmax(e_occ), e.shape)
    cbm_idx = np.unravel_index(np.argmin(e_emp), e.shape)
    vbm, cbm = float(e[vbm_idx]), float(e[cbm_idx])
    # direct gap: smallest (lowest empty - highest occupied) at the same k and spin
    per_k = e_emp.min(axis=2) - e_occ.max(axis=2)
    direct = float(per_k.min())
    crossing = occ.any(axis=1) & ~occ.all(axis=1)      # band partly filled over k
    is_metal = bool(crossing.any()) or cbm - vbm <= 0
    return {
        "is_metal": bool(is_metal),
        "gap": 0.0 if is_metal else cbm - vbm,
        "direct_gap": 0.0 if is_metal else direct,
        "is_direct": bool(not is_metal and vbm_idx[:2] == cbm_idx[:2]),
        "vbm": vbm, "cbm": cbm,
        "vbm_kpoint": eig.kpoints[vbm_idx[1]].tolist(),
        "cbm_kpoint": eig.kpoints[cbm_idx[1]].tolist(),
    }


def kpath_distance(kpoints: np.ndarray, lattice: np.ndarray | None = None):
    """Cumulative path length and positions of segment boundaries.

    With a lattice the length is in 1/Angstrom (Cartesian); without one it is
    measured in fractional reciprocal coordinates, which distorts segment
    lengths but keeps the topology of the plot.
    """
    k = np.asarray(kpoints, dtype=float)
    if lattice is not None:
        k = k @ (2 * np.pi * np.linalg.inv(lattice).T)
    steps = np.linalg.norm(np.diff(k, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(steps)])
    breaks = [0.0] + [float(dist[i + 1]) for i, s in enumerate(steps) if s < 1e-10]
    breaks.append(float(dist[-1]))
    return dist, sorted(set(breaks))


def summarise(calc_dir) -> dict:
    """Collect everything that can be read from a calculation directory."""
    d = Path(calc_dir)
    out: dict = {"directory": str(d)}
    fermi = None
    if (d / "OUTCAR").exists():
        oc = parse_outcar(d / "OUTCAR")
        fermi = oc.fermi
        out["outcar"] = {
            "ionic_steps": int(len(oc.energies)),
            "final_energy_eV": float(oc.energies[-1]) if len(oc.energies) else None,
            "final_max_force_eV_per_A": float(oc.max_forces[-1]) if len(oc.max_forces) else None,
            "fermi_eV": oc.fermi, "nions": oc.nions, "converged": oc.converged,
        }
        if oc.nions and len(oc.energies):
            out["outcar"]["final_energy_per_atom_eV"] = float(oc.energies[-1]) / oc.nions
    if (d / "EIGENVAL").exists():
        out["band_gap"] = band_gap(parse_eigenval(d / "EIGENVAL"), fermi)
    return out


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0,
                         "xtick.direction": "in", "ytick.direction": "in"})
    return plt


def plot_bands(eig: Eigenval, out, fermi=None, lattice=None, labels=None,
               ylim=(-6, 6)):
    """Band structure with the VBM (or Fermi level for metals) at zero."""
    plt = _pyplot()
    gap = band_gap(eig, fermi)
    ref = fermi if (gap["is_metal"] and fermi is not None) else gap["vbm"]
    dist, breaks = kpath_distance(eig.kpoints, lattice)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    colours = ["#1f4e79", "#c0392b"]
    for s in range(eig.nspin):
        ax.plot(dist, eig.energies[s] - ref, color=colours[s], lw=1.2)
    for b in breaks:
        ax.axvline(b, color="0.6", lw=0.6)
    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xlim(dist[0], dist[-1])
    ax.set_ylim(*ylim)
    ax.set_xticks(breaks)
    ax.set_xticklabels(labels if labels and len(labels) == len(breaks)
                       else [""] * len(breaks))
    ax.set_ylabel(r"$E - E_\mathrm{VBM}$ (eV)" if ref == gap["vbm"]
                  else r"$E - E_\mathrm{F}$ (eV)")
    title = "metal" if gap["is_metal"] else (
        f"gap = {gap['gap']:.2f} eV ({'direct' if gap['is_direct'] else 'indirect'})")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_dos(dos: Doscar, out, xlim=(-8, 8)):
    """Total DOS relative to the Fermi level (spin-down plotted negative)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = dos.energies - dos.fermi
    ax.fill_between(x, dos.dos[0], color="#1f4e79", alpha=0.35, lw=0)
    ax.plot(x, dos.dos[0], color="#1f4e79", lw=1.0)
    if dos.dos.shape[0] == 2:
        ax.fill_between(x, -dos.dos[1], color="#c0392b", alpha=0.35, lw=0)
        ax.plot(x, -dos.dos[1], color="#c0392b", lw=1.0)
    ax.axvline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"$E - E_\mathrm{F}$ (eV)")
    ax.set_ylabel("DOS (states/eV)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_relaxation(oc: Outcar, out):
    """Energy and maximum force against ionic step."""
    plt = _pyplot()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.5, 4.5), sharex=True)
    steps = np.arange(1, len(oc.energies) + 1)
    a1.plot(steps, oc.energies - oc.energies[-1], "o-", color="#1f4e79", ms=3)
    a1.set_ylabel(r"$E - E_\mathrm{final}$ (eV)")
    if len(oc.max_forces):
        a2.semilogy(np.arange(1, len(oc.max_forces) + 1), oc.max_forces, "s-",
                    color="#c0392b", ms=3)
    a2.set_ylabel(r"max $|F|$ (eV/$\mathrm{\AA}$)")
    a2.set_xlabel("ionic step")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Post-process a VASP calculation directory.")
    p.add_argument("command", choices=["summary", "bands", "dos", "relax"])
    p.add_argument("calc_dir")
    p.add_argument("-o", "--output", help="output image path")
    p.add_argument("--labels", help="comma-separated k-point labels, e.g. L,G,X")
    args = p.parse_args(argv)
    d = Path(args.calc_dir)

    if args.command == "summary":
        print(json.dumps(summarise(d), indent=2))
        return
    out = args.output or f"{args.command}.png"
    if args.command == "bands":
        fermi = parse_outcar(d / "OUTCAR").fermi if (d / "OUTCAR").exists() else None
        lattice = None
        for name in ("CONTCAR", "POSCAR"):
            if (d / name).exists() and (d / name).stat().st_size:
                lattice = parse_poscar_lattice(d / name)
                break
        labels = args.labels.split(",") if args.labels else None
        plot_bands(parse_eigenval(d / "EIGENVAL"), out, fermi, lattice, labels)
    elif args.command == "dos":
        plot_dos(parse_doscar(d / "DOSCAR"), out)
    else:
        plot_relaxation(parse_outcar(d / "OUTCAR"), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
