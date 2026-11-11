"""Acoustic and diffuse mismatch models (AMM / DMM) for thermal boundary conductance.

These are the two classical baselines that any new predictive method for
interfacial heat transport has to beat (Swartz & Pohl, Rev. Mod. Phys. 61, 605
(1989)). Both are implemented here with a Debye (linear-dispersion) description
of each solid, three acoustic branches (L, T, T) and branch-resolved cut-offs.

    python mismatch.py                 # table for the built-in interfaces
    python mismatch.py --plot g_vs_T.png

Limitations (by construction of the models, not of this code):
  * elastic, phonon-only transport - no inelastic scattering, no electrons;
  * Debye dispersion overestimates the group velocity of short-wavelength modes;
  * AMM is implemented without mode conversion at the interface;
  * no information about interfacial bonding, roughness or interlayers enters.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

HBAR = 1.054571817e-34      # J s
KB = 1.380649e-23           # J / K


@dataclass(frozen=True)
class Solid:
    name: str
    density: float           # kg / m^3
    v_l: float               # longitudinal sound velocity, m / s
    v_t: float               # transverse sound velocity, m / s
    n: float                 # atomic number density, 1 / m^3

    @property
    def velocities(self) -> np.ndarray:
        return np.array([self.v_l, self.v_t, self.v_t])

    @property
    def cutoffs(self) -> np.ndarray:
        """Debye cut-off angular frequency of each branch, rad / s."""
        return self.velocities * (6 * np.pi ** 2 * self.n) ** (1 / 3)


# Approximate room-temperature literature values. Sound velocities for Al, Si,
# Cu and diamond follow the table in Swartz & Pohl (1989); GaN values are for
# propagation along the c axis. Re-check every number against a primary source
# before using results in a publication.
MATERIALS = {
    "Al": Solid("Al", 2700.0, 6240.0, 3040.0, 6.02e28),
    "Si": Solid("Si", 2330.0, 8970.0, 5332.0, 4.99e28),
    "GaN": Solid("GaN", 6150.0, 7960.0, 4130.0, 8.76e28),
    "Cu": Solid("Cu", 8960.0, 4910.0, 2500.0, 8.47e28),
    "diamond": Solid("diamond", 3512.0, 17500.0, 12800.0, 1.763e29),
}


def mode_heat_capacity(omega: np.ndarray, T: float) -> np.ndarray:
    """d(hbar*omega*n_BE)/dT for one mode, J / K. Tends to k_B when hbar*omega << k_B*T."""
    x = HBAR * np.asarray(omega, dtype=float) / (KB * T)
    out = np.full_like(x, KB)
    big = x > 1e-6
    xb = x[big]                                   # overflow-safe form of x^2 e^x/(e^x-1)^2
    out[big] = KB * xb ** 2 * np.exp(-xb) / np.expm1(-xb) ** 2
    return out


def _midpoint_grid(omega_max: float, n: int):
    d = omega_max / n
    return (np.arange(n) + 0.5) * d, d


def dmm_conductance(a: Solid, b: Solid, T: float, n_omega: int = 20000) -> float:
    """Diffuse mismatch model, W m^-2 K^-1.

    A phonon reaching the interface forgets where it came from and is emitted
    into either side in proportion to the available density of outgoing states:
        alpha_ab(w) = S_b(w) / (S_a(w) + S_b(w)),  S_x(w) = sum_j theta(w_xj - w) / v_xj^2
        G = 1/(8 pi^2) * integral w^2 c(w, T) S_a(w) alpha_ab(w) dw
    """
    w, dw = _midpoint_grid(max(a.cutoffs.max(), b.cutoffs.max()), n_omega)
    s_a = ((w[:, None] < a.cutoffs) / a.velocities ** 2).sum(axis=1)
    s_b = ((w[:, None] < b.cutoffs) / b.velocities ** 2).sum(axis=1)
    total = s_a + s_b
    kernel = np.divide(s_a * s_b, total, out=np.zeros_like(total), where=total > 0)
    return float((w ** 2 * mode_heat_capacity(w, T) * kernel).sum() * dw / (8 * np.pi ** 2))


def amm_transmission(a: Solid, b: Solid, branch: int, n_theta: int = 4000) -> float:
    """Angle-averaged AMM transmission  Gamma = int_0^{pi/2} t(theta) cos(theta) sin(theta) dtheta."""
    v1, v2 = a.velocities[branch], b.velocities[branch]
    z1, z2 = a.density * v1, b.density * v2
    theta = (np.arange(n_theta) + 0.5) * (np.pi / 2) / n_theta
    sin2 = v2 / v1 * np.sin(theta)
    ok = sin2 < 1.0                                # beyond this: total internal reflection
    c1, c2 = np.cos(theta[ok]), np.sqrt(1.0 - sin2[ok] ** 2)
    t = 4 * z1 * z2 * c1 * c2 / (z1 * c2 + z2 * c1) ** 2
    return float((t * c1 * np.sin(theta[ok])).sum() * (np.pi / 2) / n_theta)


def amm_conductance(a: Solid, b: Solid, T: float, n_omega: int = 20000) -> float:
    """Acoustic mismatch model (specular, no mode conversion), W m^-2 K^-1.

        G = 1/2 * sum_j v_aj Gamma_j * integral D_aj(w) c(w, T) dw,   D = w^2 / (2 pi^2 v^3)
    Each branch is integrated up to the lower of the two Debye cut-offs,
    because the transmitted mode has to exist on the other side.
    """
    g = 0.0
    for j in range(3):
        w, dw = _midpoint_grid(min(a.cutoffs[j], b.cutoffs[j]), n_omega)
        v = a.velocities[j]
        integral = (w ** 2 * mode_heat_capacity(w, T)).sum() * dw / (2 * np.pi ** 2 * v ** 3)
        g += 0.5 * v * amm_transmission(a, b, j) * integral
    return float(g)


# Experimental targets quoted on the Discovered Materials challenge page
# (Problem 05), originally from Li et al., Nat. Commun. 13, 4901 (2022).
EXPERIMENT_300K = {("Al", "Si"): 379e6, ("Al", "GaN"): 423e6}


def main(argv=None):
    ap = argparse.ArgumentParser(description="AMM / DMM thermal boundary conductance")
    ap.add_argument("--pairs", nargs="*", default=["Al/Si", "Al/GaN", "Cu/diamond"])
    ap.add_argument("--temperatures", nargs="*", type=float,
                    default=[80, 150, 300, 500, 700])
    ap.add_argument("--plot")
    args = ap.parse_args(argv)

    print(f"{'interface':<12}{'T (K)':>7}{'DMM':>9}{'AMM':>9}{'expt':>8}   (MW m^-2 K^-1)")
    curves = {}
    for pair in args.pairs:
        a, b = (MATERIALS[x] for x in pair.split("/"))
        for T in args.temperatures:
            dmm, amm = dmm_conductance(a, b, T) / 1e6, amm_conductance(a, b, T) / 1e6
            ex = EXPERIMENT_300K.get((a.name, b.name)) if abs(T - 300) < 1e-9 else None
            print(f"{pair:<12}{T:>7.0f}{dmm:>9.1f}{amm:>9.1f}"
                  f"{(f'{ex / 1e6:.0f}' if ex else '-'):>8}")
        ts = np.linspace(50, 700, 66)
        curves[pair] = (ts, [dmm_conductance(a, b, t) / 1e6 for t in ts],
                        [amm_conductance(a, b, t) / 1e6 for t in ts])
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for (pair, (ts, dmm, amm)), col in zip(curves.items(),
                                               ["#1f4e79", "#c0392b", "#2e7d32", "#7b1fa2"]):
            ax.plot(ts, dmm, color=col, label=f"{pair} DMM")
            ax.plot(ts, amm, color=col, ls="--", label=f"{pair} AMM")
            a, b = pair.split("/")
            if (a, b) in EXPERIMENT_300K:
                ax.plot(300, EXPERIMENT_300K[(a, b)] / 1e6, "o", color=col, mfc="white",
                        label=f"{pair} expt")
        ax.set_xlabel("temperature (K)")
        ax.set_ylabel(r"$G$ (MW m$^{-2}$ K$^{-1}$)")
        ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=300)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
