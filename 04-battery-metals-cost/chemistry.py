"""Metal intensity of battery cathode chemistries from stoichiometry.

For a cathode active material (CAM) with molar composition {element: moles},
practical specific capacity Q (mAh/g) and average cell voltage V (V):

    energy per kg of CAM   = Q * V              [Wh/kg]
    CAM needed per kWh     = 1000 / (Q * V)     [kg/kWh]
    metal needed per kWh   = mass fraction * 1000 / (Q * V)

This is the contained-metal requirement at the active-material level. It leaves
out processing yield, the anode, electrolyte, current collectors and pack
overheads, so it is a lower bound on what a real cell consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

ATOMIC_MASS = {
    "Li": 6.94, "Na": 22.990, "Ni": 58.693, "Mn": 54.938, "Co": 58.933,
    "Al": 26.982, "Fe": 55.845, "P": 30.974, "O": 15.999,
}


@dataclass(frozen=True)
class Cathode:
    name: str
    composition: dict          # element -> moles per formula unit
    capacity: float            # practical specific capacity, mAh/g
    voltage: float             # average full-cell voltage, V
    carrier: str = "Li"        # charge-carrying ion

    @property
    def molar_mass(self) -> float:
        return sum(ATOMIC_MASS[el] * n for el, n in self.composition.items())

    @property
    def mass_fractions(self) -> dict:
        m = self.molar_mass
        return {el: ATOMIC_MASS[el] * n / m for el, n in self.composition.items()}

    @property
    def cam_per_kwh(self) -> float:
        """kg of active material per kWh."""
        return 1000.0 / (self.capacity * self.voltage)

    def intensity(self, skip=("O",)) -> dict:
        """kg of each contained element per kWh (oxygen omitted by default)."""
        k = self.cam_per_kwh
        return {el: w * k for el, w in self.mass_fractions.items() if el not in skip}


# Capacities and voltages are typical practical values of the kind quoted in
# review articles; they vary with cut-off voltage, rate and cell design.
# Change them here (or build your own Cathode) before drawing conclusions.
CATHODES = {
    c.name: c for c in [
        Cathode("NMC111", {"Li": 1, "Ni": 1 / 3, "Mn": 1 / 3, "Co": 1 / 3, "O": 2}, 160, 3.7),
        Cathode("NMC622", {"Li": 1, "Ni": 0.6, "Mn": 0.2, "Co": 0.2, "O": 2}, 180, 3.7),
        Cathode("NMC811", {"Li": 1, "Ni": 0.8, "Mn": 0.1, "Co": 0.1, "O": 2}, 200, 3.7),
        Cathode("NCA", {"Li": 1, "Ni": 0.8, "Co": 0.15, "Al": 0.05, "O": 2}, 200, 3.6),
        Cathode("LFP", {"Li": 1, "Fe": 1, "P": 1, "O": 4}, 160, 3.2),
        Cathode("LMO", {"Li": 1, "Mn": 2, "O": 4}, 110, 3.8),
        Cathode("Na-NFM111", {"Na": 1, "Ni": 1 / 3, "Fe": 1 / 3, "Mn": 1 / 3, "O": 2}, 130, 3.1,
                carrier="Na"),
    ]
}

# Contained-metal content of the traded compounds, for converting quoted prices.
LI_IN_LI2CO3 = 2 * ATOMIC_MASS["Li"] / (2 * ATOMIC_MASS["Li"] + 12.011 + 3 * ATOMIC_MASS["O"])
NA_IN_NA2CO3 = 2 * ATOMIC_MASS["Na"] / (2 * ATOMIC_MASS["Na"] + 12.011 + 3 * ATOMIC_MASS["O"])


def lithium_price_from_carbonate(usd_per_kg_li2co3: float) -> float:
    """USD per kg of contained lithium from a lithium carbonate price."""
    return usd_per_kg_li2co3 / LI_IN_LI2CO3


def sodium_price_from_soda_ash(usd_per_kg_na2co3: float) -> float:
    """USD per kg of contained sodium from a soda ash (Na2CO3) price."""
    return usd_per_kg_na2co3 / NA_IN_NA2CO3
