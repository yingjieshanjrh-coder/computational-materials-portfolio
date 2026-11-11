import numpy as np
import pytest

import build_interface as bi
import compare
import mismatch as mm
import nemd_post as npost
import run_nemd

AL, SI, DIA = mm.MATERIALS["Al"], mm.MATERIALS["Si"], mm.MATERIALS["diamond"]


# ---------------------------------------------------------------- AMM / DMM
def test_mode_heat_capacity_limits():
    assert mm.mode_heat_capacity(np.array([1e9]), 300.0)[0] == pytest.approx(mm.KB, rel=1e-6)
    assert mm.mode_heat_capacity(np.array([1e15]), 10.0)[0] < 1e-200


def test_dmm_identical_solids_classical_limit():
    # alpha = 1/2 and every mode carries k_B:  G = k_B n (v_L + 2 v_T) / 8
    exact = mm.KB * SI.n * SI.velocities.sum() / 8
    assert mm.dmm_conductance(SI, SI, 1e6) == pytest.approx(exact, rel=1e-3)


def test_amm_identical_solids_classical_limit():
    # perfect transmission, Gamma = 1/2:  G = k_B n (v_L + 2 v_T) / 4
    assert mm.amm_transmission(SI, SI, 0) == pytest.approx(0.5, rel=1e-6)
    exact = mm.KB * SI.n * SI.velocities.sum() / 4
    assert mm.amm_conductance(SI, SI, 1e6) == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize("T", [20.0, 300.0, 700.0])
@pytest.mark.parametrize("pair", [(AL, SI), (AL, DIA), (SI, DIA)])
def test_detailed_balance(pair, T):
    a, b = pair
    assert mm.dmm_conductance(a, b, T) == pytest.approx(mm.dmm_conductance(b, a, T), rel=1e-9)
    assert mm.amm_conductance(a, b, T) == pytest.approx(mm.amm_conductance(b, a, T), rel=2e-3)


def test_low_temperature_cubic_law():
    for model in (mm.dmm_conductance, mm.amm_conductance):
        assert model(AL, SI, 2.0) / model(AL, SI, 1.0) == pytest.approx(8.0, rel=1e-2)


def test_monotonic_and_saturating():
    g = [mm.dmm_conductance(AL, SI, T) for T in (50, 100, 300, 700, 5000)]
    assert all(x < y for x, y in zip(g, g[1:]))
    assert g[4] / g[3] < 1.1                      # elastic models saturate at high T


def test_grid_converged():
    assert mm.dmm_conductance(AL, SI, 300, n_omega=20000) == pytest.approx(
        mm.dmm_conductance(AL, SI, 300, n_omega=80000), rel=1e-4)


# ---------------------------------------------------------------- structure
def _neighbours(pos, cell, rcut):
    d = pos[:, None, :] - pos[None, :, :]
    for k in range(2):
        d[:, :, k] -= cell[k] * np.round(d[:, :, k] / cell[k])
    r = np.linalg.norm(d, axis=2)
    return ((r > 1e-6) & (r < rcut)).sum(axis=1), r


def test_fcc111_bulk_coordination():
    pos, cell, d = bi.fcc111_layers(bi.A_AL, 4, 4, 9)
    n, r = _neighbours(pos, cell, 1.1 * bi.A_AL / np.sqrt(2))
    interior = (pos[:, 2] > 1.5 * d) & (pos[:, 2] < pos[:, 2].max() - 1.5 * d)
    assert set(n[interior]) == {12}
    assert r[r > 1e-6].min() == pytest.approx(bi.A_AL / np.sqrt(2), rel=1e-9)


def test_diamond111_coordination_and_termination():
    pos, cell = bi.diamond111_bilayers(bi.A_SI, 3, 3, 6)
    bond = bi.A_SI * np.sqrt(3) / 4
    n, r = _neighbours(pos, cell, 1.1 * bond)
    assert len(pos) == 6 * 2 * 18
    assert r[r > 1e-6].min() == pytest.approx(bond, rel=1e-9)
    top = pos[:, 2] > pos[:, 2].max() - 1e-6
    bottom = pos[:, 2] < 1e-6
    assert set(n[top]) == {3} and set(n[bottom]) == {3}     # one dangling bond each
    assert set(n[~top & ~bottom]) == {4}


def test_interface_supercell(tmp_path):
    s = bi.build_al_si(si_bilayers=6, al_layers=9)
    assert s["n_si"] == 6 * 36 and s["n_al"] == 9 * 32
    assert s["al_strain"] == pytest.approx(0.0059, abs=2e-4)
    si_top = s["positions"][s["types"] == 1][:, 2].max()
    al_bot = s["positions"][s["types"] == 2][:, 2].min()
    assert al_bot - si_top == pytest.approx(2.4)
    assert si_top < s["z_interface"] < al_bot
    _, r = _neighbours(s["positions"], s["box"], 3.0)
    assert r[r > 1e-6].min() > 2.2                          # no overlapping atoms
    out = tmp_path / "alsi.data"
    bi.write_lammps_data(out, s)
    with pytest.raises(ValueError):                         # default baths do not fit
        bi.write_regions(str(out) + ".regions.lmp", s)
    v = bi.write_regions(str(out) + ".regions.lmp", s, wall=2.0, bath=6.0)
    text = out.read_text().splitlines()
    assert text[2] == f"{len(s['positions'])} atoms"
    assert len(text) == 16 + len(s["positions"])
    assert v["z_wall_lo"] < v["z_cold_hi"] < v["z_interface"] < v["z_hot_lo"] < v["z_wall_hi"]


# ---------------------------------------------------------------- NEMD analysis
def _synthetic_run(tmp_path, g_true=379e6, dT=4.0, area=229.9, seed=0):
    """Fake LAMMPS output with a known conductance (not a simulation result)."""
    rng = np.random.default_rng(seed)
    z = np.arange(1.0, 260.0, 2.0)
    z_int = 130.0
    ideal = np.where(z < z_int, 290.0 + 0.02 * (z - z_int),
                     290.0 + dT + 0.05 * (z - z_int))
    lines = ["# Chunk-averaged data for fix profile", "# Timestep Number-of-chunks Total-count",
             "# Chunk Coord1 Ncount v_temp"]
    for f in range(40):
        lines.append(f"{(f + 1) * 100000} {len(z)} {30 * len(z)}")
        noisy = ideal + rng.normal(0, 0.15, len(z))
        lines += [f"  {i + 1} {zi:.3f} 30.0 {t:.4f}" for i, (zi, t) in enumerate(zip(z, noisy))]
    (tmp_path / "temp_profile.dat").write_text("\n".join(lines) + "\n")
    flux = g_true * dT                                       # W / m^2
    rate = flux * area / npost.EV_PER_PS_A2_TO_W_M2          # eV / ps
    steps = np.arange(0, 4_000_000, 1000)
    t = steps * 0.001
    tally = np.column_stack([steps, -rate * t + rng.normal(0, 0.5, len(t)),
                             rate * t + rng.normal(0, 0.5, len(t))])
    np.savetxt(tmp_path / "heat_tally.dat", tally, header="step E_hot E_cold")
    return z_int, area


def test_nemd_postprocessing_recovers_known_conductance(tmp_path, capsys):
    z_int, area = _synthetic_run(tmp_path)
    z, frames = npost.read_chunk_profile(tmp_path / "temp_profile.dat")
    assert frames.shape == (40, len(z))
    tally = np.loadtxt(tmp_path / "heat_tally.dat")
    flux, mismatch = npost.heat_flux(tally[:, 0] * 0.001, tally[:, 1], tally[:, 2], area)
    assert mismatch < 1e-3
    r = npost.conductance(z, frames, flux, z_int, (30, 120), (140, 230))
    assert r["dT"] == pytest.approx(-4.0, abs=0.1)
    assert r["G"] == pytest.approx(379e6, rel=0.03)
    assert 0 < r["G_sem"] < 0.05 * r["G"]
    npost.main([str(tmp_path / "temp_profile.dat"), str(tmp_path / "heat_tally.dat"),
                "--area", str(area), "--interface", str(z_int),
                "--left", "30", "120", "--right", "140", "230",
                "--plot", str(tmp_path / "p.png")])
    assert "MW m^-2 K^-1" in capsys.readouterr().out


def test_unit_conversion():
    assert npost.EV_PER_PS_A2_TO_W_M2 == pytest.approx(1.602176634e13)


# ---------------------------------------------------------------- driver
def test_driver_dry_run_writes_consistent_inputs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_nemd, "HERE", tmp_path)
    (tmp_path / "in.nemd.lammps").write_text((run_nemd.Path(__file__).parent / "in.nemd.lammps").read_text())
    run_dir = run_nemd.main(["--preset", "full", "--dry-run"])
    out = capsys.readouterr().out
    assert "3360 atoms" in out and "2,700,000 MD steps" in out
    assert "-var nprod 2000000" in out and "e+" not in out      # LAMMPS needs plain integers
    for name in ("alsi.data", "alsi.data.regions.lmp", "potential.lmp", "in.nemd.lammps"):
        assert (run_dir / name).exists()
    assert "SiS AlS" in (run_dir / "potential.lmp").read_text()
    text = (run_dir / "in.nemd.lammps").read_text()
    for var in ("z_wall_lo", "z_cold_hi", "z_hot_lo", "z_wall_hi"):
        assert "${%s}" % var in text and var in (run_dir / "alsi.data.regions.lmp").read_text()


@pytest.mark.parametrize("preset", list(run_nemd.PRESETS))
def test_presets_are_self_consistent(preset, tmp_path):
    cfg = run_nemd.PRESETS[preset]
    assert cfg["nprod"] % cfg["nfreq"] == 0 and cfg["nfreq"] % 10 == 0
    s = bi.build_al_si(cfg["si"], cfg["al"])
    regions = bi.write_regions(tmp_path / "r.lmp", s, cfg["wall"], cfg["bath"])
    left, right = run_nemd.fit_windows(regions)
    assert regions["z_cold_hi"] <= left[0] < left[1] < regions["z_interface"]
    assert regions["z_interface"] < right[0] < right[1] <= regions["z_hot_lo"]
    assert left[1] - left[0] > 6 and right[1] - right[0] > 6


def test_driver_analysis_on_synthetic_output(tmp_path):
    z_int, area = _synthetic_run(tmp_path)
    regions = {"z_interface": z_int, "area": area, "z_cold_hi": 25.0, "z_hot_lo": 235.0}
    r = run_nemd.analyse(tmp_path, regions)
    assert r["G"] == pytest.approx(379e6, rel=0.03) and r["n_frames"] == 40


def test_compare_collects_and_flags_runs(tmp_path):
    import json
    for name, T, g, mis in (("quick_T300", 300, 5.1e8, 0.02), ("quick_T500", 500, 5.6e8, 0.30)):
        d = tmp_path / name
        d.mkdir()
        (d / "result.json").write_text(json.dumps({"preset": "quick", "temperature": T, "G": g,
                                                   "G_sem": 3e7, "thermostat_mismatch": mis}))
    rows = compare.collect(tmp_path)
    assert [r["T"] for r in rows] == [300.0, 500.0] and rows[0]["G"] == pytest.approx(510.0)
    text = compare.table(rows)
    assert "379" in text.splitlines()[1] and text.splitlines()[1].endswith("ok")
    assert "not steady" in text.splitlines()[2]
    compare.plot(rows, tmp_path / "g.png")
    assert (tmp_path / "g.png").stat().st_size > 5000
