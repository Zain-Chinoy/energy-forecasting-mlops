import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.prepare_data import load_raw, prepare, split_chronologically, validate_and_clean

N = 240


def make_raw(n: int = N, start: str = "2019-06-25") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            config.DATETIME_COL: pd.date_range(start, periods=n, freq="h"),
            config.TARGET: 1000 + rng.normal(0, 50, n),
        }
    )
    for col in config.WEATHER_COLS:
        lo, hi = config.range_for(col)
        df[col] = rng.uniform(lo + 0.3 * (hi - lo), lo + 0.6 * (hi - lo), n)
    df["Holiday_ID"] = 0
    df["holiday"] = 0
    df["school"] = 1
    return df


def test_clean_valid_data_keeps_all_rows():
    clean, report = validate_and_clean(make_raw())
    assert report["rows_clean"] == N
    assert "Holiday_ID" not in clean.columns
    for col in config.DERIVED_TIME_COLS:
        assert col in clean.columns


def test_duplicate_timestamps_are_removed():
    raw = make_raw()
    dirty = pd.concat([raw, raw.head(5)], ignore_index=True)
    clean, report = validate_and_clean(dirty)
    assert report["rows_duplicate_timestamp_dropped"] == 5
    assert len(clean) == N
    assert clean[config.DATETIME_COL].is_monotonic_increasing


def test_out_of_range_value_is_flagged_and_repaired():
    raw = make_raw()
    raw.loc[10, "T2M_toc"] = 999.0
    clean, report = validate_and_clean(raw)
    assert report["out_of_range_values"] == {"T2M_toc": 1}
    lo, hi = config.range_for("T2M_toc")
    assert clean["T2M_toc"].between(lo, hi).all()
    assert len(clean) == N


def test_rows_without_target_are_dropped():
    raw = make_raw()
    raw.loc[[3, 4, 5], config.TARGET] = np.nan
    clean, report = validate_and_clean(raw)
    assert report["rows_missing_target_dropped"] == 3
    assert len(clean) == N - 3


def test_long_gaps_are_not_invented():
    raw = make_raw()
    raw.loc[20:25, "W2M_san"] = np.nan  # 6-hour gap, only 3 hours may be interpolated
    clean, report = validate_and_clean(raw)
    assert report["rows_dropped_unfixable"] == 3
    assert len(clean) == N - 3
    assert clean["W2M_san"].notna().all()


def test_invalid_binary_flag_is_repaired():
    raw = make_raw()
    raw.loc[7, "holiday"] = 2
    clean, report = validate_and_clean(raw)
    assert report["out_of_range_values"] == {"holiday": 1}
    assert set(clean["holiday"].unique()) <= {0, 1}


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "raw.csv"
    make_raw().drop(columns=["T2M_toc"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="T2M_toc"):
        load_raw(path)


def test_split_is_chronological_and_complete():
    clean, _ = validate_and_clean(make_raw())
    train, test, shift = split_chronologically(clean, "2019-06-30", "2019-07-03")
    assert len(train) + len(test) + len(shift) == len(clean)
    assert train[config.DATETIME_COL].max() < test[config.DATETIME_COL].min()
    assert test[config.DATETIME_COL].max() < shift[config.DATETIME_COL].min()


def test_prepare_end_to_end(tmp_path):
    raw_path = tmp_path / "raw.csv"
    make_raw().to_csv(raw_path, index=False)
    out = tmp_path / "processed"
    report = prepare(raw_path, out, test_start="2019-06-30", shift_start="2019-07-03")
    for name in ("train.csv", "test.csv", "shift_2020.csv", "validation_report.json"):
        assert (out / name).exists()
    assert json.loads((out / "validation_report.json").read_text())["rows_clean"] == N
    assert report["split"]["train"]["rows"] > 0
