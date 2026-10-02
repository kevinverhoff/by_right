"""The metric catalog: what the CSV reports, plus the ratios that make rows comparable.

Raw dollars mostly track population, so a $2M township looks alarming until you
notice it has 40,000 residents. The per-resident and share-of-spending metrics
are what actually separate townships of different sizes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Anchored to this file, not the working directory: Streamlit Community Cloud
# runs from the repo root while the app lives in this subdirectory.
CSV = Path(__file__).with_name("indiana_townships_2025.csv")
GEOJSON = Path(__file__).with_name("data") / "indiana_townships.geojson"


class M:
    """One selectable measure."""

    def __init__(self, col, label, fmt, note="", positive=True, log_ok=True):
        self.col = col
        self.label = label
        self.fmt = fmt  # "usd" | "usd0" | "int" | "pct" | "ratio"
        self.note = note
        self.positive = positive  # safe to encode as bubble area
        self.log_ok = log_ok


CATALOG = [
    # --- as reported ---
    M("population_2025_est", "Population (2025 est.)", "int"),
    M("population_2020_census", "Population (2020 census)", "int"),
    M("trustee_2025_compensation", "Trustee compensation", "usd0",
      "From the 2025 Form 100R. Where a township reported more than one trustee, "
      "this is the highest-paid one; the rest are listed in the notes column."),
    M("total_receipts", "Total receipts", "usd0"),
    M("total_disbursements", "Total disbursements", "usd0"),
    M("operating_costs", "Operating costs", "usd0"),
    M("township_assistance_paid", "Township assistance paid", "usd0",
      "Direct relief to residents — the function townships exist to perform."),
    M("capital_outlays", "Capital outlays", "usd0"),
    M("beginning_balance", "Beginning cash balance", "usd0"),
    M("ending_cash_and_investments", "Ending cash and investments", "usd0"),
    # --- derived: scale-free ---
    M("disb_per_resident", "Disbursements per resident", "usd"),
    M("operating_per_resident", "Operating cost per resident", "usd"),
    M("assistance_per_resident", "Assistance paid per resident", "usd"),
    M("comp_per_resident", "Trustee pay per resident", "usd"),
    M("cash_per_resident", "Cash held per resident", "usd"),
    M("reserve_years", "Reserve (years of spending held)", "ratio",
      "Ending cash and investments divided by one year of disbursements. "
      "Above 1.0 means the township is holding more than a full year of spending."),
    M("assistance_share", "Assistance as share of spending", "pct",
      "Township assistance paid divided by total disbursements."),
    M("comp_share", "Trustee pay as share of spending", "pct",
      "Trustee compensation divided by total disbursements."),
    M("net_surplus", "Net surplus (receipts − disbursements)", "usd0",
      "Negative where a township spent down its balance.", positive=False, log_ok=False),
    M("pop_change_pct", "Population change, 2020→2025", "pct",
      "Census 2020 to the 2025 estimate.", positive=False, log_ok=False),
]

BY_COL = {m.col: m for m in CATALOG}
LABEL_TO_COL = {m.label: m.col for m in CATALOG}


def _safe_div(a, b):
    """Divide, returning NaN wherever the denominator is zero or missing."""
    b = b.replace(0, np.nan)
    return a / b


def load(path=CSV) -> pd.DataFrame:
    df = pd.read_csv(path)

    pop = df["population_2025_est"]
    df["disb_per_resident"] = _safe_div(df["total_disbursements"], pop)
    df["operating_per_resident"] = _safe_div(df["operating_costs"], pop)
    df["assistance_per_resident"] = _safe_div(df["township_assistance_paid"], pop)
    df["comp_per_resident"] = _safe_div(df["trustee_2025_compensation"], pop)
    df["cash_per_resident"] = _safe_div(df["ending_cash_and_investments"], pop)
    df["reserve_years"] = _safe_div(df["ending_cash_and_investments"], df["total_disbursements"])
    df["assistance_share"] = _safe_div(df["township_assistance_paid"], df["total_disbursements"])
    df["comp_share"] = _safe_div(df["trustee_2025_compensation"], df["total_disbursements"])
    df["net_surplus"] = df["total_receipts"] - df["total_disbursements"]
    df["pop_change_pct"] = _safe_div(
        df["population_2025_est"] - df["population_2020_census"], df["population_2020_census"]
    )

    df["name"] = df["township"] + ", " + df["county"] + " Co."
    df["geo_key"] = (
        df["county"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        + "|"
        + df["township"].str.replace(r"(?i)\s+township$", "", regex=True)
        .str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    )
    df["pop_tier"] = pd.cut(
        pop, bins=[-1, 2500, 10000, np.inf],
        labels=["Under 2,500", "2,500–10,000", "Over 10,000"],
    )
    return df


def fmt(value, kind) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if kind == "usd":
        return f"${value:,.0f}" if abs(value) >= 1000 else f"${value:,.2f}"
    if kind == "usd0":
        return f"${value:,.0f}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "pct":
        return f"{value * 100:,.1f}%"
    if kind == "ratio":
        return f"{value:,.2f}×"
    return f"{value:,.2f}"


def tickformat(kind, log: bool = False) -> str:
    """d3 format for axis ticks and hover values.

    A log axis crosses orders of magnitude, so whole-dollar rounding would label
    two different gridlines "$0". Significant-digit formats keep the small ticks
    readable and trim the trailing zeros off the large ones.
    """
    if log:
        return {"usd": "$,.3~r", "usd0": "$,.3~r", "int": ",.0f",
                "pct": ".2~%", "ratio": ",.3~r"}[kind]
    return {"usd": "$,.0f", "usd0": "$,.0f", "int": ",.0f", "pct": ".0%", "ratio": ",.1f"}[kind]
