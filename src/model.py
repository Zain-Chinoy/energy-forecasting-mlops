"""Load the trained model bundle and make single-row predictions."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.features import add_time_features


@dataclass
class ModelBundle:
    """A fitted model plus the metadata saved next to it by ``src.train``."""

    model: object
    meta: dict

    @property
    def features(self) -> list[str]:
        return list(self.meta["features"])


def load_bundle(
    model_path: Path = config.MODEL_PATH, meta_path: Path = config.MODEL_META_PATH
) -> ModelBundle:
    """Load ``model.joblib`` and ``model_meta.json``; raise if either is missing."""
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found ({model_path}, {meta_path}). "
            "Run `python -m src.prepare_data` then `python -m src.train`."
        )
    return ModelBundle(model=joblib.load(model_path), meta=json.loads(meta_path.read_text()))


def predict_one(bundle: ModelBundle, payload: dict) -> float:
    """Predict national demand (MWh) for one request payload.

    ``payload`` holds the raw API fields (``timestamp``, weather columns, ``holiday``,
    ``school``). Time features are derived with the same code used in training.
    """
    frame = add_time_features(pd.DataFrame([payload]), datetime_col="timestamp")
    prediction = float(bundle.model.predict(frame[bundle.features])[0])
    return max(prediction, 0.0)  # demand cannot be negative
