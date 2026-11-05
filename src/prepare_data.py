"""Standalone data preparation: clean, validate, derive features and split.

Usage:
    python -m src.prepare_data

No statistics are fitted here (no scaling or imputation of learned values), so the
chronological split cannot leak information between train and test. Any scaling is
done later inside sklearn Pipelines that are fitted on training data only.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src import config
from src.features import add_time_features

logger = logging.getLogger(__name__)

MAX_INTERPOLATION_GAP = 3  # hours


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV and check that all required columns are present."""
    df = pd.read_csv(path)
    missing = [c for c in config.REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL], errors="coerce")
    return df


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw frame and return it with a validation report.

    Steps: drop unparsable/duplicate timestamps, sort, drop rows without a target,
    mark out-of-range values as missing, interpolate short gaps in features, drop
    rows that still have missing values, then add time features.
    """
    report: dict = {"rows_raw": int(len(df))}
    df = df.copy()

    bad_ts = int(df[config.DATETIME_COL].isna().sum())
    df = df.dropna(subset=[config.DATETIME_COL])
    report["rows_bad_timestamp_dropped"] = bad_ts

    dups = int(df[config.DATETIME_COL].duplicated().sum())
    df = df.drop_duplicates(subset=[config.DATETIME_COL], keep="first")
    report["rows_duplicate_timestamp_dropped"] = dups

    df = df.sort_values(config.DATETIME_COL).reset_index(drop=True)

    expected = pd.date_range(df[config.DATETIME_COL].min(), df[config.DATETIME_COL].max(), freq="h")
    report["missing_hourly_timestamps"] = int(len(expected) - len(df))

    no_target = int(df[config.TARGET].isna().sum())
    df = df.dropna(subset=[config.TARGET])
    report["rows_missing_target_dropped"] = no_target

    # Range validation: out-of-range values are treated as missing.
    out_of_range: dict[str, int] = {}
    for col in [config.TARGET, *config.WEATHER_COLS]:
        lo, hi = config.range_for(col)
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.any():
            out_of_range[col] = int(bad.sum())
            df.loc[bad, col] = float("nan")
    for col in config.RAW_CALENDAR_COLS:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            out_of_range[col] = int(bad.sum())
            df.loc[bad, col] = float("nan")
    report["out_of_range_values"] = out_of_range

    feature_cols = [*config.WEATHER_COLS, *config.RAW_CALENDAR_COLS]
    n_missing_before = int(df[feature_cols].isna().sum().sum())
    indexed = df.set_index(config.DATETIME_COL)
    indexed[config.WEATHER_COLS] = indexed[config.WEATHER_COLS].interpolate(
        method="time", limit=MAX_INTERPOLATION_GAP
    )
    indexed[config.RAW_CALENDAR_COLS] = indexed[config.RAW_CALENDAR_COLS].ffill(limit=1)
    df = indexed.reset_index()
    report["missing_feature_values_before"] = n_missing_before
    report["missing_feature_values_after_interpolation"] = int(df[feature_cols].isna().sum().sum())

    still_missing = int(df[[config.TARGET, *feature_cols]].isna().any(axis=1).sum())
    df = df.dropna(subset=[config.TARGET, *feature_cols]).reset_index(drop=True)
    report["rows_dropped_unfixable"] = still_missing

    for col in config.RAW_CALENDAR_COLS:
        df[col] = df[col].astype("int64")
    df = df.drop(columns=["Holiday_ID"])  # identifier, redundant with `holiday`
    df = add_time_features(df)

    # Unusually low demand values (e.g. the Jan 2019 outage) are real events: kept, reported.
    report["low_demand_rows_kept(<400MWh)"] = int((df[config.TARGET] < 400).sum())
    report["rows_clean"] = int(len(df))
    return df, report


def split_chronologically(
    df: pd.DataFrame, test_start: str, shift_start: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: train < test_start <= test < shift_start <= shift."""
    ts = df[config.DATETIME_COL]
    train = df[ts < pd.Timestamp(test_start)].reset_index(drop=True)
    test = df[(ts >= pd.Timestamp(test_start)) & (ts < pd.Timestamp(shift_start))]
    shift = df[ts >= pd.Timestamp(shift_start)]
    return train, test.reset_index(drop=True), shift.reset_index(drop=True)


def prepare(
    input_path: Path = config.RAW_CSV,
    output_dir: Path = config.PROCESSED_DIR,
    test_start: str = config.TEST_START,
    shift_start: str = config.SHIFT_START,
) -> dict:
    """Run the full preparation pipeline and write train/test/shift CSVs + report."""
    df, report = validate_and_clean(load_raw(input_path))
    train, test, shift = split_chronologically(df, test_start, shift_start)
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    if train[config.DATETIME_COL].max() >= test[config.DATETIME_COL].min():
        raise ValueError("Train and test periods overlap.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    shift.to_csv(output_dir / "shift_2020.csv", index=False)

    def _span(part: pd.DataFrame) -> dict:
        has_rows = len(part) > 0
        return {
            "rows": len(part),
            "start": str(part[config.DATETIME_COL].min()) if has_rows else None,
            "end": str(part[config.DATETIME_COL].max()) if has_rows else None,
        }

    report["split"] = {"train": _span(train), "test": _span(test), "shift_2020": _span(shift)}
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Prepared data written to %s", output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.RAW_CSV)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--shift-start", default=config.SHIFT_START)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = prepare(args.input, args.output_dir, args.test_start, args.shift_start)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
