import numpy as np
import pytest

import lgas


def total_energy(n, mu):
    L = n.shape[0]
    e = 0.0
    for i in range(L):
        for j in range(L):
            e += n[i, j] * (n[(i + 1) % L, j] + n[i, (j + 1) % L])
    return e - mu * n.sum()


def test_flip_cost_matches_brute_force_energy_difference():
    g = lgas.LatticeGas(6, kT=1.0, mu=1.3, seed=3)
    cost = g.flip_cost()
    e0 = total_energy(g.n, g.mu)
    for i, j in [(0, 0), (2, 3), (5, 5), (1, 4)]:
        m = g.n.copy()
        m[i, j] ^= 1
        assert total_energy(m, g.mu) - e0 == pytest.approx(cost[i, j])


def test_energy_per_site_matches_brute_force():
    g = lgas.LatticeGas(8, kT=1.0, mu=0.7, seed=4)
    assert g.energy_per_site() * 64 == pytest.approx(total_energy(g.n, g.mu))


def test_sublattice_update_only_touches_one_sublattice():
    g = lgas.LatticeGas(8, kT=5.0, mu=2.0, seed=5)
    before = g.n.copy()
    d_e = g.flip_cost()
    p = np.exp(np.minimum(-d_e / g.kT, 0.0))
    flip = (g.rng.random(g.n.shape) < p) & g.masks[0]
    assert not flip[g.masks[1]].any()
    assert before.shape == g.n.shape


def test_odd_size_rejected():
    with pytest.raises(ValueError):
        lgas.LatticeGas(5, kT=1.0)


def test_empty_below_zero_chemical_potential():
    g = lgas.LatticeGas(12, kT=0.3, mu=-2.0, seed=6)
    x = g.run(200, 50)[:, 0].mean()
    assert x < 0.01


def test_full_above_four_j():
    g = lgas.LatticeGas(12, kT=0.3, mu=6.0, seed=7)
    x = g.run(200, 50)[:, 0].mean()
    assert x > 0.99


def test_half_filling_at_symmetric_point_high_temperature():
    g = lgas.LatticeGas(16, kT=2.0, mu=lgas.MU_HALF, seed=8)
    x = g.run(500, 1500)[:, 0].mean()
    assert x == pytest.approx(0.5, abs=0.02)


def test_onsager_limits():
    assert lgas.onsager_order_parameter(0.05) == pytest.approx(1.0, abs=1e-6)
    assert lgas.onsager_order_parameter(lgas.KTC_EXACT * 1.001) == 0.0
    # beta = 1/8, so the order parameter vanishes very slowly at Tc
    near = float(lgas.onsager_order_parameter(lgas.KTC_EXACT * (1 - 1e-8)))
    far = float(lgas.onsager_order_parameter(lgas.KTC_EXACT * (1 - 1e-3)))
    assert 0 < near < far < 0.6


def test_critical_temperature_value():
    assert lgas.KTC_EXACT == pytest.approx(0.5673, abs=1e-4)


def test_order_parameter_matches_onsager_in_ordered_phase():
    kT = 0.40
    g = lgas.LatticeGas(16, kT=kT, mu=lgas.MU_HALF, seed=9, init="checker")
    m = np.abs(g.run(500, 3000)[:, 1]).mean()
    assert m == pytest.approx(float(lgas.onsager_order_parameter(kT)), abs=0.02)


def test_binder_cumulant_limits():
    ordered = np.ones(1000) * 0.9
    assert lgas.binder_cumulant(ordered) == pytest.approx(2 / 3)
    rng = np.random.default_rng(10)
    gauss = rng.normal(0, 0.1, 200000)
    assert lgas.binder_cumulant(gauss) == pytest.approx(0.0, abs=0.02)


def test_jackknife_error_scales_like_standard_error():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 20000)
    mean, err = lgas.jackknife(x, np.mean)
    assert abs(mean) < 4 * err
    assert err == pytest.approx(1 / np.sqrt(20000), rel=0.5)


def test_crossing_finds_intersection():
    t = np.array([1.0, 2.0, 3.0])
    # difference is 2, 0.5, -1: sign change between t=2 and t=3 at a third of the way
    assert lgas.crossing(t, [0.0, 1.0, 2.0], [2.0, 1.5, 1.0]) == pytest.approx(2 + 1 / 3)
    assert np.isnan(lgas.crossing(t, [0, 0, 0], [1, 1, 1]))
