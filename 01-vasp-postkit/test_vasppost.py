import json

import numpy as np
import pytest

import make_fixtures as mf
import vasppost as vp


@pytest.fixture(scope="module")
def calc(tmp_path_factory):
    return mf.write_all(tmp_path_factory.mktemp("calc"))


def test_outcar(calc):
    oc = vp.parse_outcar(calc / "OUTCAR")
    assert oc.nions == 2 and oc.converged
    assert len(oc.energies) == 5 and oc.energies[-1] == pytest.approx(-10.8456)
    assert oc.max_forces == pytest.approx([0.80, 0.30, 0.09, 0.02, 0.004])
    assert oc.fermi == pytest.approx(5.3)


def test_eigenval_roundtrip(calc):
    eig = vp.parse_eigenval(calc / "EIGENVAL")
    k = mf.kpath()
    assert eig.energies.shape == (1, len(k), 8)
    assert np.allclose(eig.kpoints, k, atol=1e-6)
    assert np.allclose(eig.energies[0], mf.model_bands(k), atol=1e-6)
    assert eig.occupations is not None and eig.nelect == 8


def test_band_gap_known_values(calc):
    gap = vp.band_gap(vp.parse_eigenval(calc / "EIGENVAL"))
    assert not gap["is_metal"] and not gap["is_direct"]
    assert gap["gap"] == pytest.approx(0.60, abs=1e-5)
    assert gap["direct_gap"] == pytest.approx(0.985, abs=1e-5)   # at (0.25, 0, 0.25)
    assert gap["vbm_kpoint"] == pytest.approx([0, 0, 0])
    assert gap["cbm_kpoint"] == pytest.approx([0.4, 0, 0.4])


def test_gap_without_occupations_agrees(calc):
    eig = vp.parse_eigenval(calc / "EIGENVAL")
    ref = vp.band_gap(eig)["gap"]
    eig.occupations = None
    assert vp.band_gap(eig)["gap"] == pytest.approx(ref)            # electron count
    assert vp.band_gap(eig, fermi=5.3)["gap"] == pytest.approx(ref)  # Fermi level


def test_metal_detected(calc):
    eig = vp.parse_eigenval(calc / "EIGENVAL")
    eig.occupations = None
    assert vp.band_gap(eig, fermi=4.0)["is_metal"]   # Fermi level inside valence bands


def test_spin_polarised_eigenval(tmp_path):
    lines = ["  1 1 1 2", "x", "x", "x", "x", "  2  1  2", "",
             "  0.0 0.0 0.0 1.0", "  1  -1.0  -0.5  1.0  1.0", "  2   2.0   1.5  0.0  0.0"]
    (tmp_path / "EIGENVAL").write_text("\n".join(lines) + "\n")
    eig = vp.parse_eigenval(tmp_path / "EIGENVAL")
    assert eig.nspin == 2
    assert vp.band_gap(eig)["gap"] == pytest.approx(2.0)   # VBM -0.5 (down), CBM 1.5 (down)


def test_doscar_integrates_to_electron_count(calc):
    dos = vp.parse_doscar(calc / "DOSCAR")
    assert dos.dos.shape == (1, 301) and dos.fermi == pytest.approx(5.3)
    below = dos.energies <= dos.fermi
    n = np.trapezoid(dos.dos[0][below], dos.energies[below])
    assert n == pytest.approx(8.0, rel=0.02)               # four filled bands x 2


def test_kpath_uses_reciprocal_lattice(calc):
    lat = vp.parse_poscar_lattice(calc / "POSCAR")
    dist, breaks = vp.kpath_distance(mf.kpath(), lat)
    b = 2 * np.pi / mf.A_SI
    assert breaks[1] == pytest.approx(b * np.sqrt(3) / 2, rel=1e-6)   # |Gamma-L|
    assert dist[-1] - breaks[1] == pytest.approx(b, rel=1e-6)         # |Gamma-X|
    assert len(breaks) == 3


def test_cli_and_plots(calc, tmp_path, capsys):
    vp.main(["summary", str(calc)])
    out = json.loads(capsys.readouterr().out)
    assert out["band_gap"]["gap"] == pytest.approx(0.60, abs=1e-5)
    assert out["outcar"]["final_energy_per_atom_eV"] == pytest.approx(-5.4228)
    for cmd in ("bands", "dos", "relax"):
        png = tmp_path / f"{cmd}.png"
        vp.main([cmd, str(calc), "-o", str(png)])
        assert png.stat().st_size > 5000
