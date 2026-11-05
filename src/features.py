"""Feature engineering shared by training and serving (single source of truth)."""

import pandas as pd

from src.config import DATETIME_COL


def add_time_features(df: pd.DataFrame, datetime_col: str = DATETIME_COL) -> pd.DataFrame:
    """Return a copy of ``df`` with hour, dayofweek, month and is_weekend columns."""
    out = df.copy()
    ts = pd.to_datetime(out[datetime_col])
    out["hour"] = ts.dt.hour.astype("int64")
    out["dayofweek"] = ts.dt.dayofweek.astype("int64")  # Monday = 0
    out["month"] = ts.dt.month.astype("int64")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int64")
    return out
