"""Lattice-gas Monte Carlo for ion ordering in an intercalation host.

A square lattice of sites holds mobile ions (occupation n = 0 or 1) with a
nearest-neighbour repulsion J and a chemical potential mu:

    H = J * sum_<ij> n_i n_j - mu * sum_i n_i

Because the square lattice is bipartite, one sublattice can be updated in a
single vectorised Metropolis step. Mapping n = (s + 1) / 2 turns the model
into an antiferromagnetic Ising model with coupling J / 4 at the
particle-hole symmetric point mu = 2 J, so the ordering transition at half
filling can be checked against the exact Onsager result.

Energies are in units of J (set J = 1), temperatures are k_B T / J.
"""
import numpy as np

KTC_EXACT = 0.5 / np.log(1.0 + np.sqrt(2.0))  # k_B T_c / J, about 0.5673
MU_HALF = 2.0  # mu / J at half filling


def onsager_order_parameter(kT):
    """Exact staggered order |x_A - x_B| of the model at mu = 2 J."""
    kT = np.asarray(kT, dtype=float)
    k = 0.5 / kT  # 2 |J_ising| / kT with |J_ising| = J / 4
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        m = (1.0 - np.sinh(k) ** -4.0) ** 0.125
    return np.where(kT < KTC_EXACT, m, 0.0)


class LatticeGas:
    def __init__(self, size, kT, mu=MU_HALF, seed=0, init="random"):
        if size % 2:
            raise ValueError("size must be even for the two-sublattice update")
        self.L = size
        self.kT = float(kT)
        self.mu = float(mu)
        self.rng = np.random.default_rng(seed)
        i, j = np.indices((size, size))
        self.parity = (i + j) % 2
        self.masks = [self.parity == 0, self.parity == 1]
        if init == "random":
            self.n = (self.rng.random((size, size)) < 0.5).astype(np.int8)
        elif init == "checker":
            self.n = (self.parity == 0).astype(np.int8)
        elif init == "empty":
            self.n = np.zeros((size, size), dtype=np.int8)
        elif init == "full":
            self.n = np.ones((size, size), dtype=np.int8)
        else:
            raise ValueError(f"unknown init {init!r}")

    def neighbour_sum(self):
        n = self.n
        return (
            np.roll(n, 1, 0) + np.roll(n, -1, 0) + np.roll(n, 1, 1) + np.roll(n, -1, 1)
        )

    def flip_cost(self):
        """Energy change of flipping every site on its own, shape (L, L)."""
        return (1 - 2 * self.n) * (self.neighbour_sum() - self.mu)

    def sweep(self):
        for mask in self.masks:
            d_e = self.flip_cost()
            p = np.exp(np.minimum(-d_e / self.kT, 0.0))
            flip = (self.rng.random(self.n.shape) < p) & mask
            self.n[flip] ^= 1

    def energy_per_site(self):
        n = self.n
        bonds = (n * np.roll(n, -1, 0)).sum() + (n * np.roll(n, -1, 1)).sum()
        return (bonds - self.mu * n.sum()) / n.size

    def observables(self):
        n = self.n
        xa = n[self.masks[0]].mean()
        xb = n[self.masks[1]].mean()
        return n.mean(), xa - xb, self.energy_per_site()

    def run(self, n_equil, n_sample, every=1):
        for _ in range(n_equil):
            self.sweep()
        out = np.empty((n_sample // every, 3))
        for k in range(out.shape[0]):
            for _ in range(every):
                self.sweep()
            out[k] = self.observables()
        return out  # columns: coverage, staggered order, energy per site


def binder_cumulant(m):
    m2 = np.mean(m**2)
    return 1.0 - np.mean(m**4) / (3.0 * m2**2)


def jackknife(samples, statistic, n_blocks=20):
    """Jackknife estimate and error of `statistic(samples)` over blocks."""
    samples = np.asarray(samples)
    t = len(samples) - len(samples) % n_blocks
    blocks = samples[:t].reshape(n_blocks, -1)
    full = statistic(blocks.ravel())
    loo = np.array(
        [statistic(np.delete(blocks, b, axis=0).ravel()) for b in range(n_blocks)]
    )
    err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((loo - loo.mean()) ** 2))
    return full, err


def crossing(temps, u_small, u_large):
    """Temperature where two Binder curves cross (linear interpolation)."""
    d = np.asarray(u_large) - np.asarray(u_small)
    for k in range(len(d) - 1):
        if d[k] * d[k + 1] < 0:
            frac = d[k] / (d[k] - d[k + 1])
            return temps[k] + frac * (temps[k + 1] - temps[k])
    return float("nan")
