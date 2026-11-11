"""Contained-metal cost of cathodes and its sensitivity to metal prices."""
from __future__ import annotations

from chemistry import Cathode


def metal_cost(cathode: Cathode, prices: dict, missing="raise") -> dict:
    """USD per kWh contributed by each metal. prices are USD per kg of contained metal.

    missing = "raise" stops on a metal without a price; "zero" treats it as free
    (useful for phosphorus, which is rarely the cost driver).
    """
    out = {}
    for metal, kg in cathode.intensity().items():
        if metal not in prices:
            if missing == "zero":
                out[metal] = 0.0
                continue
            raise KeyError(f"no price given for {metal} (needed by {cathode.name})")
        out[metal] = kg * prices[metal]
    return out


def total_cost(cathode: Cathode, prices: dict, missing="raise") -> float:
    return sum(metal_cost(cathode, prices, missing).values())


def tornado(cathode: Cathode, base: dict, low: dict, high: dict, missing="raise"):
    """One-at-a-time sensitivity: move each metal between its low and high price.

    Returns rows (metal, cost_at_low, cost_at_high) sorted by swing, largest first.
    """
    rows = []
    for metal in cathode.intensity():
        if metal not in base:
            if missing == "zero":
                continue
            raise KeyError(f"no price given for {metal}")
        lo = total_cost(cathode, {**base, metal: low[metal]}, missing)
        hi = total_cost(cathode, {**base, metal: high[metal]}, missing)
        rows.append((metal, lo, hi))
    return sorted(rows, key=lambda r: abs(r[2] - r[1]), reverse=True)


def breakeven_price(a: Cathode, b: Cathode, prices: dict, metal="Li", missing="raise"):
    """Price of `metal` at which cathodes a and b have equal contained-metal cost.

    Cost is linear in every price, so the answer is exact. Returns None when the
    two chemistries use the same amount of the metal per kWh (parallel lines).
    A negative result means one chemistry is cheaper at every positive price.
    """
    zero = {**prices, metal: 0.0}
    slope = a.intensity().get(metal, 0.0) - b.intensity().get(metal, 0.0)
    if abs(slope) < 1e-12:
        return None
    return (total_cost(b, zero, missing) - total_cost(a, zero, missing)) / slope


def cost_history(cathode: Cathode, price_frame, fixed: dict | None = None, missing="raise"):
    """Contained-metal cost through time.

    price_frame: DataFrame indexed by date, one column per metal (USD/kg).
    fixed: constant prices for metals that have no time series.
    """
    fixed = fixed or {}
    total = 0.0
    for metal, kg in cathode.intensity().items():
        if metal in price_frame.columns:
            total = total + kg * price_frame[metal]
        elif metal in fixed:
            total = total + kg * fixed[metal]
        elif missing != "zero":
            raise KeyError(f"no price series or fixed price for {metal}")
    return total.dropna()
