import numpy as np
import pandas as pd
import pytest

from src import config
from src.explain import (
    MAX_EXACT_FEATURES,
    build_background,
    explain_prediction,
    load_background,
    save_background,
)


class LinearModel:
    """Stand-in model with known weights, so the correct Shapley values are known exactly."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy(dtype=float) @ self.weights


def make_background(n_features: int, n_rows: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_rows, n_features)), columns=[f"f{i}" for i in range(n_features)]
    )


def test_exact_shapley_matches_closed_form_for_a_linear_model():
    weights = [2.0, -1.0, 0.5, 3.0]
    background = make_background(4)
    features = list(background.columns)
    row = dict(zip(features, [1.0, 2.0, -3.0, 0.5], strict=True))
    out = explain_prediction(LinearModel(weights), features, row, background)
    assert out["method"] == "shapley_exact"
    got = {c["feature"]: c["contribution_mwh"] for c in out["contributions"]}
    for w, f in zip(weights, features, strict=True):
        assert got[f] == pytest.approx(w * (row[f] - background[f].mean()))


def test_contributions_add_up_to_prediction_minus_baseline():
    background = make_background(6)
    features = list(background.columns)
    row = {f: 0.7 * i - 1.0 for i, f in enumerate(features)}
    out = explain_prediction(LinearModel([1, 2, 3, 4, 5, 6]), features, row, background)
    total = sum(c["contribution_mwh"] for c in out["contributions"])
    assert out["baseline_mwh"] + total == pytest.approx(out["prediction_mwh"])


def test_feature_the_model_ignores_gets_zero_contribution():
    background = make_background(3)
    features = list(background.columns)
    out = explain_prediction(
        LinearModel([1.0, 0.0, 2.0]), features, dict.fromkeys(features, 5.0), background
    )
    ignored = next(c for c in out["contributions"] if c["feature"] == "f1")
    assert ignored["contribution_mwh"] == pytest.approx(0.0, abs=1e-9)


def test_interaction_effect_is_shared_equally_between_symmetric_features():
    class Product:
        def predict(self, frame):
            return frame["a"].to_numpy() * frame["b"].to_numpy()

    background = pd.DataFrame({"a": [0.0], "b": [0.0]})
    out = explain_prediction(Product(), ["a", "b"], {"a": 3.0, "b": 3.0}, background)
    values = [c["contribution_mwh"] for c in out["contributions"]]
    assert values == pytest.approx([4.5, 4.5])  # 3 * 3 = 9, split evenly


def test_contributions_are_sorted_by_absolute_effect():
    background = make_background(4)
    features = list(background.columns)
    out = explain_prediction(
        LinearModel([0.1, -5.0, 2.0, 0.0]), features, dict.fromkeys(features, 3.0), background
    )
    sizes = [abs(c["contribution_mwh"]) for c in out["contributions"]]
    assert sizes == sorted(sizes, reverse=True)


def test_large_models_fall_back_to_occlusion_and_stay_correct_for_linear_models():
    n = MAX_EXACT_FEATURES + 1
    weights = np.arange(1, n + 1, dtype=float)
    background = make_background(n)
    features = list(background.columns)
    row = {f: 1.5 for f in features}
    out = explain_prediction(LinearModel(weights), features, row, background)
    assert out["method"] == "occlusion"
    got = {c["feature"]: c["contribution_mwh"] for c in out["contributions"]}
    for w, f in zip(weights, features, strict=True):
        assert got[f] == pytest.approx(w * (1.5 - background[f].mean()))


def test_background_sample_is_reproducible_and_has_the_right_columns():
    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.normal(size=(500, 5)), columns=list("abcde"))
    first = build_background(train, ["a", "c"], n_rows=30, seed=7)
    second = build_background(train, ["a", "c"], n_rows=30, seed=7)
    assert first.equals(second)
    assert list(first.columns) == ["a", "c"]
    assert len(first) == 30


def test_background_round_trip_and_missing_file(tmp_path):
    background = make_background(3, n_rows=5)
    path = tmp_path / "background.json"
    assert load_background(path) is None
    save_background(background, path)
    loaded = load_background(path)
    assert list(loaded.columns) == list(background.columns)
    np.testing.assert_allclose(loaded.to_numpy(), background.to_numpy(), atol=1e-5)


def test_committed_background_matches_the_committed_model():
    import json

    meta = json.loads(config.MODEL_META_PATH.read_text())
    background = load_background(config.BACKGROUND_PATH)
    assert background is not None
    assert list(background.columns) == meta["features"]
    assert len(background) == config.BACKGROUND_ROWS
