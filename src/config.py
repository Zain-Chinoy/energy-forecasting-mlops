"""Central configuration shared by data prep, training and the API."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "continuous_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"

DATASET_NAME = "panama_electricity_load"
DATETIME_COL = "datetime"
TARGET = "nat_demand"
RANDOM_STATE = 42

# Chronological split. Train < TEST_START <= test < SHIFT_START <= shift set.
# From March 2020 demand dropped ~15% (COVID-19 lockdown), so 2020 is kept out of
# train/test and used only as a distribution-shift check.
TEST_START = "2019-07-01"
SHIFT_START = "2020-01-01"

CITIES = ("toc", "san", "dav")
TEMP_COLS = [f"T2M_{c}" for c in CITIES]
HUMIDITY_COLS = [f"QV2M_{c}" for c in CITIES]
PRECIP_COLS = [f"TQL_{c}" for c in CITIES]
WIND_COLS = [f"W2M_{c}" for c in CITIES]
WEATHER_COLS = TEMP_COLS + HUMIDITY_COLS + PRECIP_COLS + WIND_COLS

RAW_CALENDAR_COLS = ["holiday", "school"]
DERIVED_TIME_COLS = ["hour", "dayofweek", "month", "is_weekend"]
CALENDAR_COLS = DERIVED_TIME_COLS + RAW_CALENDAR_COLS

# Columns required in the raw CSV (Holiday_ID is read but dropped: it is an ID).
REQUIRED_RAW_COLS = [DATETIME_COL, TARGET, *WEATHER_COLS, "Holiday_ID", *RAW_CALENDAR_COLS]

# Plausible physical ranges, used for validation in prepare_data.py and for the
# Pydantic request schema in app.py. Observed data lies well inside these.
# NOTE: the dataset metadata calls QV2M "relative humidity (%)", but the values
# (~0.01-0.02) are specific humidity (kg/kg), so the range is set accordingly.
RANGES = {
    "T2M": (10.0, 50.0),
    "QV2M": (0.0, 0.05),
    "TQL": (0.0, 5.0),
    "W2M": (0.0, 60.0),
    TARGET: (0.0, 3000.0),
}


def range_for(column: str) -> tuple[float, float]:
    """Return the (min, max) validation range for a data column."""
    if column == TARGET:
        return RANGES[TARGET]
    return RANGES[column.split("_")[0]]


# Feature sets compared in MLflow (the "feature combination" experiments).
FEATURE_SETS: dict[str, list[str]] = {
    "calendar_only": CALENDAR_COLS,
    "weather_only": WEATHER_COLS,
    "calendar_temp": CALENDAR_COLS + TEMP_COLS,
    "calendar_temp_humidity": CALENDAR_COLS + TEMP_COLS + HUMIDITY_COLS,
    "calendar_tocumen": CALENDAR_COLS + [f"{v}_toc" for v in ("T2M", "QV2M", "TQL", "W2M")],
    "all_minus_wind": CALENDAR_COLS + TEMP_COLS + HUMIDITY_COLS + PRECIP_COLS,
    "all_minus_precip": CALENDAR_COLS + TEMP_COLS + HUMIDITY_COLS + WIND_COLS,
    "all_features": CALENDAR_COLS + WEATHER_COLS,
}

MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()
MLFLOW_EXPERIMENT = "energy-demand-forecasting"
REGISTERED_MODEL_NAME = "energy-demand-regressor"
