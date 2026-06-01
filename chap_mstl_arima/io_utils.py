from __future__ import annotations

import pandas as pd


def detect_frequency(df: pd.DataFrame) -> str:
    """Return 'MS' for monthly chap data or 'W-MON' for weekly.

    Chap uses ``YYYY-MM`` for monthly and ``YYYY-Wnn`` (or
    ``YYYY-MM-DD/YYYY-MM-DD``) for weekly in ``time_period``.
    """
    sample = str(df["time_period"].iloc[0])
    lower = sample.lower()
    if "w" in lower:
        return "W-MON"
    if "/" in sample:
        return "W-MON"
    parts = sample.split("-")
    if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 12:
        return "W-MON"
    return "MS"


def period_to_timestamp(period: str, freq: str) -> pd.Timestamp:
    s = str(period)
    if freq == "MS":
        return pd.to_datetime(s + "-01")
    if "/" in s:
        return pd.to_datetime(s.split("/")[0])
    if "w" in s.lower():
        year, week = s.lower().split("-w")
        return pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%G-W%V-%u")
    return pd.to_datetime(s)


def to_long_panel(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Return a StatsForecast-style panel: columns ``unique_id, ds, y``."""
    out = pd.DataFrame(
        {
            "unique_id": df["location"].astype(str),
            "ds": df["time_period"].apply(lambda p: period_to_timestamp(p, freq)),
            "y": pd.to_numeric(df["disease_cases"], errors="coerce"),
        }
    )
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
