# tbc-toolkit

Tools for predicting thermal boundary conductance (TBC) at solid–solid
interfaces, starting from the Al(111)/Si(111) interface.

| File | Purpose | Status |
|---|---|---|
| `mismatch.py` | Acoustic and diffuse mismatch models (Debye) | computed, tested |
| `build_interface.py` | Al(111)/Si(111) supercell → LAMMPS data file | tested (geometry) |
| `nemd_post.py` | NEMD output → flux, temperature jump, G ± error | tested on synthetic output |
| `in.nemd.lammps` | NEMD input template | **not yet run** |

## 1. Classical baselines

```bash
python mismatch.py --plot amm_dmm_baseline.png
```

Computed with this code (MW m⁻² K⁻¹), measured values as quoted from
Li et al., Nat. Commun. 13, 4901 (2022):

| Interface | T (K) | DMM | AMM | Measured |
|---|---|---|---|---|
| Al/Si | 300 | 794 | 752 | ≈379 |
| Al/GaN | 300 | 938 | 791 | ≈423 |
| Cu/diamond | 300 | 329 | 114 | – |

Both Debye-based models overestimate the two measured interfaces by roughly a
factor of two, and both saturate above about 500 K, whereas the measurements
are reported to keep rising. The absolute values are sensitive to model
conventions: counting Si phonon states per primitive cell instead of per atom
lowers the Al/Si DMM value from 794 to 680. Full-dispersion DMM variants in
the literature give different numbers again. Treat these as *this
implementation's* baselines.

Checks in the test suite: analytic high-temperature limit for identical solids
(`G = k_B n Σv / 8` for DMM, `/4` for AMM), detailed balance (`G_ab = G_ba`),
`T³` law at low temperature, and grid convergence.

![baseline](amm_dmm_baseline.png)

## 2. Interface structure

```bash
python build_interface.py --si-bilayers 40 --al-layers 60 -o alsi.data
```

4×4 Al(111) on 3×3 Si(111) in an orthorhombic cell, Al strained +0.59 %
in-plane. The default gives 3,360 atoms, a 229.9 Å² cross-section and a
273 Å long cell. The Si slab is cut between bilayers (one dangling bond per
surface atom). The 2.4 Å interface separation and the lateral registry are
starting guesses and must be relaxed.

## 3. NEMD and analysis

```bash
lmp -in in.nemd.lammps -var T 300
python nemd_post.py temp_profile.dat heat_tally.dat --area 229.9 \
    --interface 129.3 --left 40 120 --right 140 240 --plot profile.png
```

`G = J / ΔT`, with `J` from the thermostat energy slopes and `ΔT` from linear
fits on each side extrapolated to the interface; the error bar is a block
average over time. The analysis recovers a planted value of 379 MW m⁻² K⁻¹
from synthetic LAMMPS-format output to within 3 %.

## Roadmap

1. Short test run of the template with a classical Al–Si potential.
2. Convergence in length and cross-section; relax the interface.
3. Swap in a machine-learned potential and compare against the baselines and
   experiment over 80–700 K.

Things the MD will not capture without extra modelling: electron–phonon
coupling in the metal, quantum statistics below the Debye temperatures
(Si ≈ 640 K), and real interfacial disorder.

## Related work to read first

- Phonon local non-equilibrium at the Al/Si interface from machine-learning MD,
  J. Appl. Phys. 137, 115301 (2025) – a neural-network potential for this exact
  interface.
- Thermal boundary conductance of metal–diamond interfaces predicted by MLIPs,
  arXiv:2404.15465.
- GaN/AlN TBC with moment tensor potentials, arXiv:2503.05084.

## Tests

```bash
pytest -q        # 20 tests
```
