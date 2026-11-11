"""Post-process a steady-state NEMD run into a thermal boundary conductance.

    G = J / dT
    J  : heat flux from the energy exchanged by the two Langevin thermostats
    dT : temperature jump at the interface, from linear fits to the temperature
         profile on each side, extrapolated to the interface plane

    python nemd_post.py temp_profile.dat heat_tally.dat \
        --area 229.9 --interface 130.5 --left 40 120 --right 141 250

Inputs (as written by in.nemd.lammps):
  temp_profile.dat  output of `fix ave/chunk ... temp`
  heat_tally.dat    columns: step, cumulative E_hot, cumulative E_cold (eV)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

EV_PER_PS_A2_TO_W_M2 = 1.602176634e-19 / 1e-12 / 1e-20


def read_chunk_profile(path):
    """Parse a LAMMPS fix ave/chunk file with columns: chunk, coord, ncount, temp.

    Returns (z, T) with T of shape (n_frames, n_chunks). Empty chunks become NaN.
    """
    rows = [ln.split() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    frames, i, z = [], 0, None
    while i < len(rows):
        nchunk = int(rows[i][1])
        block = np.array(rows[i + 1:i + 1 + nchunk], dtype=float)
        z = block[:, 1]
        frames.append(np.where(block[:, 2] > 0, block[:, 3], np.nan))
        i += 1 + nchunk
    return z, np.array(frames)


def heat_flux(time_ps, e_hot, e_cold, area_A2):
    """Steady-state flux (W/m^2) from the slopes of the cumulative thermostat energies.

    The hot bath adds energy and the cold bath removes it, so the two slopes
    have opposite signs; their magnitudes should agree, and the relative
    difference is returned as a check on steady state / energy conservation.
    """
    s_hot = abs(np.polyfit(time_ps, e_hot, 1)[0])
    s_cold = abs(np.polyfit(time_ps, e_cold, 1)[0])
    mean = 0.5 * (s_hot + s_cold)
    return mean / area_A2 * EV_PER_PS_A2_TO_W_M2, abs(s_hot - s_cold) / mean


def temperature_jump(z, T, z_interface, left, right):
    """Extrapolate linear fits on both sides to the interface plane.

    left / right are (zmin, zmax) fit windows; keep them clear of the
    thermostats and of the first few atomic layers next to the interface.
    Returns (dT, slope_left, slope_right) with dT = T_left - T_right at z_interface.
    """
    out = []
    for lo, hi in (left, right):
        m = (z >= lo) & (z <= hi) & np.isfinite(T)
        if m.sum() < 3:
            raise ValueError(f"fit window {lo}-{hi} contains fewer than 3 bins")
        out.append(np.polyfit(z[m], T[m], 1))
    (sl, il), (sr, ir) = out
    return (sl * z_interface + il) - (sr * z_interface + ir), sl, sr


def conductance(z, frames, flux, z_interface, left, right, n_blocks=5):
    """G in W m^-2 K^-1 with a block-average standard error over time."""
    def mean_profile(f):          # bins inside the rigid walls hold no mobile atoms
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(f, axis=0)

    dT, sl, sr = temperature_jump(z, mean_profile(frames), z_interface, left, right)
    g = flux / abs(dT)
    blocks = [flux / abs(temperature_jump(z, mean_profile(b), z_interface, left, right)[0])
              for b in np.array_split(frames, n_blocks) if len(b)]
    sem = float(np.std(blocks, ddof=1) / np.sqrt(len(blocks))) if len(blocks) > 1 else float("nan")
    return {"G": float(g), "G_sem": sem, "dT": float(dT), "flux": float(flux),
            "grad_left": float(sl), "grad_right": float(sr)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("profile")
    ap.add_argument("tally")
    ap.add_argument("--area", type=float, required=True, help="cross-section, A^2")
    ap.add_argument("--interface", type=float, required=True, help="interface z, A")
    ap.add_argument("--left", type=float, nargs=2, required=True)
    ap.add_argument("--right", type=float, nargs=2, required=True)
    ap.add_argument("--timestep", type=float, default=0.001, help="ps")
    ap.add_argument("--discard", type=float, default=0.3,
                    help="leading fraction of the run discarded as transient")
    ap.add_argument("--plot")
    args = ap.parse_args(argv)

    z, frames = read_chunk_profile(args.profile)
    frames = frames[int(args.discard * len(frames)):]
    tally = np.loadtxt(args.tally, comments="#")
    tally = tally[int(args.discard * len(tally)):]
    flux, mismatch = heat_flux(tally[:, 0] * args.timestep, tally[:, 1], tally[:, 2], args.area)
    r = conductance(z, frames, flux, args.interface, args.left, args.right)
    print(f"heat flux      {r['flux'] / 1e9:.3f} GW/m^2  (hot/cold mismatch {100 * mismatch:.1f} %)")
    print(f"interface dT   {r['dT']:.2f} K")
    print(f"G              {r['G'] / 1e6:.1f} +/- {r['G_sem'] / 1e6:.1f} MW m^-2 K^-1")
    if mismatch > 0.05:
        print("warning: thermostat energies differ by > 5 % - run is not in steady state")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Tm = np.nanmean(frames, axis=0)
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.plot(z, Tm, ".", color="0.4", ms=3)
        for (lo, hi), col in ((args.left, "#1f4e79"), (args.right, "#c0392b")):
            m = (z >= lo) & (z <= hi) & np.isfinite(Tm)
            c = np.polyfit(z[m], Tm[m], 1)
            zz = np.array([min(lo, args.interface), max(hi, args.interface)])
            ax.plot(zz, np.polyval(c, zz), color=col)
        ax.axvline(args.interface, color="0.6", lw=0.6, ls="--")
        ax.set_xlabel(r"$z$ ($\mathrm{\AA}$)")
        ax.set_ylabel("temperature (K)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=300)


if __name__ == "__main__":
    main()
