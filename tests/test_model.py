import math

import pytest

from src import config
from src.model import load_bundle, predict_one


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_bundle_loads_with_expected_metadata(bundle):
    assert hasattr(bundle.model, "predict")
    for key in ("model_type", "feature_set", "features", "mlflow_run_id", "test"):
        assert key in bundle.meta


def test_model_features_are_known_columns(bundle):
    allowed = set(config.CALENDAR_COLS) | set(config.WEATHER_COLS)
    assert set(bundle.features) <= allowed


def test_prediction_is_finite_and_plausible(bundle, payload):
    value = predict_one(bundle, payload)
    assert math.isfinite(value)
    assert 0 < value < config.RANGES[config.TARGET][1]


def test_model_beats_the_mean_baseline(bundle):
    assert bundle.meta["test"]["rmse_improvement_vs_baseline_pct"] > 0


def test_weekday_afternoon_demand_exceeds_night(bundle, payload):
    afternoon = predict_one(bundle, {**payload, "timestamp": "2019-10-15T14:00:00"})
    night = predict_one(bundle, {**payload, "timestamp": "2019-10-15T03:00:00"})
    assert afternoon > night


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.joblib", tmp_path / "nope.json")
