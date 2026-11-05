import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import add_time_features
from src.train import MODEL_SPECS, TRAINABLE_MODELS, build_model, regression_metrics, run_training


def synthetic(n: int, start: str) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({config.DATETIME_COL: pd.date_range(start, periods=n, freq="h")})
    for col in config.WEATHER_COLS:
        lo, hi = config.range_for(col)
        df[col] = rng.uniform(lo + 0.3 * (hi - lo), lo + 0.6 * (hi - lo), n)
    df["holiday"] = 0
    df["school"] = 1
    df = add_time_features(df)
    df[config.TARGET] = 800 + 20 * df["hour"] + 5 * df["T2M_toc"] + rng.normal(0, 10, n)
    return df


def test_regression_metrics_perfect_prediction():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "x_")
    assert m["x_rmse"] == 0
    assert m["x_r2"] == 1


@pytest.mark.parametrize("model_type", TRAINABLE_MODELS)
def test_every_model_type_fits_and_predicts(model_type):
    df = synthetic(300, "2019-01-01")
    features = config.FEATURE_SETS["all_features"]
    model = build_model(model_type, features).fit(df[features], df[config.TARGET])
    assert model.predict(df[features].head(5)).shape == (5,)


def test_feature_sets_are_valid():
    allowed = set(config.CALENDAR_COLS) | set(config.WEATHER_COLS)
    assert len(config.FEATURE_SETS) >= 6
    for name, cols in config.FEATURE_SETS.items():
        assert set(cols) <= allowed, name
        assert len(cols) == len(set(cols)), name
    assert len(config.FEATURE_SETS["all_features"]) == 18


def test_search_spaces_only_use_valid_parameters():
    for name in TRAINABLE_MODELS:
        spec = MODEL_SPECS[name]
        assert set(spec["search_space"]) <= set(spec["defaults"])


def test_training_pipeline_end_to_end_quick(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    synthetic(24 * 40, "2019-01-01").to_csv(processed / "train.csv", index=False)
    synthetic(24 * 10, "2019-02-10").to_csv(processed / "test.csv", index=False)
    meta = run_training(
        processed_dir=processed,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_root=(tmp_path / "artifacts").as_uri(),
        quick=True,
        register=False,
    )
    assert (tmp_path / "models" / "model.joblib").exists()
    saved = json.loads((tmp_path / "models" / "model_meta.json").read_text())
    assert saved["mlflow_run_id"] == meta["mlflow_run_id"]
    assert saved["test"]["rmse_improvement_vs_baseline_pct"] > 0
    assert (tmp_path / "reports" / "results.md").exists()
