"""One-command NEMD workflow for the Al(111)/Si(111) thermal boundary conductance.

    python run_nemd.py --preset smoke          # minutes: checks that everything runs
    python run_nemd.py --preset quick -T 300   # hours:   first usable number
    python run_nemd.py --preset full  -T 300   # ~a day:  production length

Steps: build the interface -> write the potential file -> run LAMMPS ->
extract G from the temperature profile and thermostat energies.
Results go to runs/<preset>_T<temperature>/ (result.json, log.lammps, screen.out).

The default potential is the Al-Si-Mg-Cu-Fe 2NN-MEAM set distributed with
LAMMPS (library.meam + AlSiMgCuFe.meam; Jelinek et al., Phys. Rev. B 85,
245102 (2012)).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import build_interface as bi
import nemd_post as post

HERE = Path(__file__).resolve().parent

PRESETS = {
    #         Si bilayers, Al layers, equilibration, steady, production, frame
    "smoke": dict(si=12, al=18, nequil=500, nsteady=500, nprod=2000, nfreq=500,
                  ntally=100, wall=4.0, bath=8.0),
    "quick": dict(si=30, al=45, nequil=50_000, nsteady=200_000, nprod=500_000,
                  nfreq=50_000, ntally=1000, wall=6.0, bath=20.0),
    "full": dict(si=40, al=60, nequil=200_000, nsteady=500_000, nprod=2_000_000,
                 nfreq=100_000, ntally=1000, wall=6.0, bath=20.0),
}

MEAM_FILES = ("library.meam", "AlSiMgCuFe.meam")
# Atom types in the data file: 1 = Si, 2 = Al. The alloy file indexes its five
# elements by position, so all five must be listed in this order.
MEAM_POTENTIAL = (
    "pair_style      meam\n"
    "pair_coeff      * * library.meam AlS SiS MgS CuS FeS AlSiMgCuFe.meam SiS AlS\n"
)


def find_lammps(explicit=None):
    for name in ([explicit] if explicit else ["lmp", "lmp_serial", "lmp_mpi", "lammps"]):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("LAMMPS executable not found. Install it (see README) or pass --lmp PATH.")


def find_potentials(extra=None):
    """Directory that holds the MEAM files shipped with LAMMPS."""
    candidates = [extra, os.environ.get("LAMMPS_POTENTIALS"), HERE / "potentials",
                  "/usr/share/lammps/potentials", "/usr/local/share/lammps/potentials",
                  Path(sys.prefix) / "share" / "lammps" / "potentials"]
    for c in candidates:
        if c and all((Path(c) / f).exists() for f in MEAM_FILES):
            return Path(c)
    raise SystemExit(
        "Could not find library.meam and AlSiMgCuFe.meam. Copy both from the 'potentials' "
        f"folder of the LAMMPS source into {HERE / 'potentials'} or pass --potentials DIR.")


def prepare(run_dir: Path, cfg: dict, potentials: Path | None, potential_file: Path | None):
    run_dir.mkdir(parents=True, exist_ok=True)
    s = bi.build_al_si(cfg["si"], cfg["al"])
    bi.write_lammps_data(run_dir / "alsi.data", s)
    regions = bi.write_regions(run_dir / "alsi.data.regions.lmp", s, cfg["wall"], cfg["bath"])
    shutil.copy(HERE / "in.nemd.lammps", run_dir / "in.nemd.lammps")
    if potential_file:
        shutil.copy(potential_file, run_dir / "potential.lmp")
    else:
        (run_dir / "potential.lmp").write_text(MEAM_POTENTIAL)
        if potentials:
            for f in MEAM_FILES:
                shutil.copy(potentials / f, run_dir / f)
    return s, regions


def fit_windows(regions: dict, margin_bath=5.0, margin_interface=8.0):
    """Keep the linear fits away from the thermostats and from the interface."""
    zi = regions["z_interface"]
    left = (regions["z_cold_hi"] + margin_bath, zi - margin_interface)
    right = (zi + margin_interface, regions["z_hot_lo"] - margin_bath)
    if left[1] - left[0] < 6 or right[1] - right[0] < 6:      # thin test systems
        left = (regions["z_cold_hi"], zi - 3.0)
        right = (zi + 3.0, regions["z_hot_lo"])
    return left, right


def analyse(run_dir: Path, regions: dict, timestep=0.001, discard=0.0):
    z, frames = post.read_chunk_profile(run_dir / "temp_profile.dat")
    tally = np.loadtxt(run_dir / "heat_tally.dat", comments="#")
    frames, tally = frames[int(discard * len(frames)):], tally[int(discard * len(tally)):]
    flux, mismatch = post.heat_flux(tally[:, 0] * timestep, tally[:, 1], tally[:, 2],
                                    regions["area"])
    left, right = fit_windows(regions)
    r = post.conductance(z, frames, flux, regions["z_interface"], left, right,
                         n_blocks=min(5, len(frames)))
    r.update(thermostat_mismatch=float(mismatch), fit_left=left, fit_right=right,
             n_frames=int(len(frames)))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=PRESETS, default="smoke")
    ap.add_argument("-T", "--temperature", type=float, default=300.0)
    ap.add_argument("--dT", type=float, default=30.0, help="thermostat offset (K)")
    ap.add_argument("--lmp", help="LAMMPS executable")
    ap.add_argument("--np", type=int, default=1, help="MPI processes (needs mpirun)")
    ap.add_argument("--potentials", help="directory with the MEAM files")
    ap.add_argument("--potential-file", type=Path,
                    help="LAMMPS snippet defining pair_style/pair_coeff (replaces MEAM)")
    ap.add_argument("--dry-run", action="store_true", help="write inputs, do not run")
    args = ap.parse_args(argv)

    cfg = PRESETS[args.preset]
    run_dir = HERE / "runs" / f"{args.preset}_T{args.temperature:g}"
    need_meam = args.potential_file is None
    potentials = None
    if need_meam and not args.dry_run:
        potentials = find_potentials(args.potentials)
    s, regions = prepare(run_dir, cfg, potentials, args.potential_file)
    steps = cfg["nequil"] + cfg["nsteady"] + cfg["nprod"]
    print(f"{len(s['positions'])} atoms, {steps:,} MD steps, run directory {run_dir}")

    variables = {"T": args.temperature, "dT": args.dT, "nequil": cfg["nequil"],
                 "nsteady": cfg["nsteady"], "nprod": cfg["nprod"], "nfreq": cfg["nfreq"],
                 "ntally": cfg["ntally"]}
    cmd = ["LMP", "-in", "in.nemd.lammps", "-log", "log.lammps"]
    for k, v in variables.items():
        cmd += ["-var", k, str(int(v)) if float(v).is_integer() else repr(float(v))]
    if args.dry_run:
        print("dry run; command would be:", " ".join(cmd))
        return run_dir
    cmd[0] = find_lammps(args.lmp)
    if args.np > 1:
        cmd = ["mpirun", "-np", str(args.np)] + cmd

    t0 = time.time()
    with open(run_dir / "screen.out", "w") as out:
        code = subprocess.run(cmd, cwd=run_dir, stdout=out, stderr=subprocess.STDOUT).returncode
    if code != 0:
        tail = (run_dir / "screen.out").read_text().splitlines()[-15:]
        raise SystemExit("LAMMPS failed. Last lines of screen.out:\n" + "\n".join(tail))
    print(f"LAMMPS finished in {(time.time() - t0) / 60:.1f} min")

    r = analyse(run_dir, regions)
    r.update(preset=args.preset, temperature=args.temperature, n_atoms=int(len(s["positions"])),
             md_steps=steps, wall_minutes=(time.time() - t0) / 60)
    (run_dir / "result.json").write_text(json.dumps(r, indent=2))
    print(f"G = {r['G'] / 1e6:.1f} +/- {r['G_sem'] / 1e6:.1f} MW m^-2 K^-1   "
          f"(dT = {r['dT']:.2f} K, thermostat mismatch {100 * r['thermostat_mismatch']:.1f} %)")
    if args.preset == "smoke":
        print("Smoke preset: this only shows that the pipeline runs. The number is not physical.")
    return run_dir


if __name__ == "__main__":
    main()
