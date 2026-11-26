"""Load the trained model bundle and make single-row predictions."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.explain import explain_prediction, load_background
from src.features import add_time_features


@dataclass
class ModelBundle:
    """A fitted model plus the metadata saved next to it by ``src.train``."""

    model: object
    meta: dict
    background: pd.DataFrame | None = None  # reference rows for /explain (optional)

    @property
    def features(self) -> list[str]:
        return list(self.meta["features"])


def load_bundle(
    model_path: Path = config.MODEL_PATH,
    meta_path: Path = config.MODEL_META_PATH,
    background_path: Path = config.BACKGROUND_PATH,
) -> ModelBundle:
    """Load the model, its metadata and (if present) the /explain background sample.

    Raises ``FileNotFoundError`` if the model or metadata is missing, and ``ValueError`` if
    the background file was built for a different feature list (stale after a retrain).
    """
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found ({model_path}, {meta_path}). "
            "Run `python -m src.prepare_data` then `python -m src.train`."
        )
    bundle = ModelBundle(model=joblib.load(model_path), meta=json.loads(meta_path.read_text()))
    background = load_background(background_path)
    if background is not None and list(background.columns) != bundle.features:
        raise ValueError(
            f"{background_path} was built for different features than the model. "
            "Run `python -m src.explain` to regenerate it."
        )
    bundle.background = background
    return bundle


def _feature_frame(bundle: ModelBundle, payload: dict) -> pd.DataFrame:
    """Raw request fields -> one-row frame with the features the model was trained on."""
    frame = add_time_features(pd.DataFrame([payload]), datetime_col="timestamp")
    return frame[bundle.features]


def predict_one(bundle: ModelBundle, payload: dict) -> float:
    """Predict national demand (MWh) for one request payload.

    ``payload`` holds the raw API fields (``timestamp``, weather columns, ``holiday``,
    ``school``). Time features are derived with the same code used in training.
    """
    prediction = float(bundle.model.predict(_feature_frame(bundle, payload))[0])
    return max(prediction, 0.0)  # demand cannot be negative


def explain_one(bundle: ModelBundle, payload: dict) -> dict:
    """Explain one prediction: baseline plus a contribution per feature (all in MWh).

    Raises ``RuntimeError`` if the bundle has no background sample.
    """
    if bundle.background is None:
        raise RuntimeError("No background sample loaded. Run `python -m src.explain`.")
    row = _feature_frame(bundle, payload).iloc[0].to_dict()
    result = explain_prediction(bundle.model, bundle.features, row, bundle.background)
    result["prediction_mwh"] = max(result["prediction_mwh"], 0.0)
    return result
