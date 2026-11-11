"""Price-series loading and basic market statistics.

Data sources this module can read:
  * World Bank Commodity Price Data ("Pink Sheet") monthly workbook, downloaded
    from https://www.worldbank.org/commodities  (Monthly prices, XLS)
  * any CSV with a date column and one column per series
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_MONTH = re.compile(r"^\d{4}M\d{2}$")


def load_pink_sheet(path, sheet="Monthly Prices") -> pd.DataFrame:
    """Monthly nominal prices from the World Bank workbook.

    The sheet has a few title rows, a row of series names, a row of units, then
    rows keyed '1960M01', '1960M02', ...; missing values are written as '..'.
    Returns a DataFrame indexed by month start with one column per series; units
    are kept in `frame.attrs["units"]`.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    first = raw[0].astype(str).str.strip()
    data_rows = first[first.str.match(_MONTH)].index
    if len(data_rows) == 0:
        raise ValueError("no rows of the form 1960M01 found - is this the monthly workbook?")
    start = data_rows[0]
    # the name row is the nearest row above the data with the most text cells
    above = raw.iloc[max(0, start - 6):start, 1:]
    counts = above.apply(lambda r: sum(isinstance(v, str) and v.strip() != "" for v in r), axis=1)
    name_row = counts[counts == counts.max()].index[0]
    names = [str(v).strip() if isinstance(v, str) else None for v in raw.iloc[name_row, 1:]]
    units = {}
    if name_row + 1 < start:
        for n, u in zip(names, raw.iloc[name_row + 1, 1:]):
            if n and isinstance(u, str):
                units[n] = u.strip().strip("()")
    body = raw.loc[data_rows]
    index = pd.to_datetime(body[0].astype(str).str.strip().str.replace("M", "-", regex=False),
                           format="%Y-%m")
    frame = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    frame.index = index
    frame.columns = names
    frame = frame.loc[:, [c for c in frame.columns if c]]
    frame.index.name = "date"
    frame.attrs["units"] = units
    return frame


def load_price_csv(path, date_col="date") -> pd.DataFrame:
    frame = pd.read_csv(path, comment="#", parse_dates=[date_col]).set_index(date_col)
    return frame.apply(pd.to_numeric, errors="coerce").sort_index()


def log_returns(prices: pd.DataFrame | pd.Series):
    return np.log(prices).diff().dropna(how="all")


def annualised_volatility(prices, periods_per_year=12):
    """Standard deviation of log returns scaled to one year."""
    return log_returns(prices).std() * np.sqrt(periods_per_year)


def max_drawdown(prices: pd.Series) -> float:
    """Largest peak-to-trough fall as a fraction of the peak (0.35 = down 35 %)."""
    p = prices.dropna()
    return float((1.0 - p / p.cummax()).max())


def summary(prices: pd.DataFrame, periods_per_year=12) -> pd.DataFrame:
    """Per-series table: last price, change over the window, volatility, drawdown."""
    rows = {}
    for name in prices.columns:
        p = prices[name].dropna()
        if len(p) < 3:
            continue
        rows[name] = {
            "last": p.iloc[-1],
            "change_%": 100 * (p.iloc[-1] / p.iloc[0] - 1),
            "ann_vol_%": 100 * float(annualised_volatility(p, periods_per_year)),
            "max_drawdown_%": 100 * max_drawdown(p),
        }
    return pd.DataFrame(rows).T


def hhi(production) -> float:
    """Herfindahl-Hirschman index of supply concentration, 0-10,000.

    Accepts any quantities (tonnes or shares); they are normalised first.
    Above 2,500 is conventionally called highly concentrated.
    """
    q = np.asarray(list(production.values()) if isinstance(production, dict) else production,
                   dtype=float)
    if (q < 0).any() or q.sum() <= 0:
        raise ValueError("production figures must be non-negative with a positive total")
    share = 100 * q / q.sum()
    return float((share ** 2).sum())
