"""Explain single predictions with exact Shapley values.

For one request we show how much each feature moved the prediction away from the
*typical* prediction: ``prediction = baseline + sum(contributions)``.

Method (interventional Shapley values, computed exactly):
    * ``background`` is a small random sample of training rows (``models/background.json``).
    * ``v(S)`` = average model prediction when the features in subset S take the requested
      values and all other features are taken from a background row.
    * The contribution of a feature is its Shapley value over all subsets of the features.
      With 9 features this needs 2**9 = 512 subsets x 50 background rows = 25,600 model
      evaluations, which is a single vectorised ``predict`` call.

No extra library (such as ``shap``) is needed, so the Docker image stays small. Models with
more than ``MAX_EXACT_FEATURES`` features fall back to one-feature-at-a-time occlusion, which
is cheaper but not exactly additive.

Regenerate the background file with ``python -m src.explain`` (``src.train`` does it too).
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

MAX_EXACT_FEATURES = 12


def build_background(
    train: pd.DataFrame, features: list[str], n_rows: int, seed: int
) -> pd.DataFrame:
    """Random sample (fixed seed) of training rows, restricted to the model's features."""
    n = min(n_rows, len(train))
    idx = np.sort(np.random.default_rng(seed).choice(len(train), size=n, replace=False))
    return train.iloc[idx][features].reset_index(drop=True)


def save_background(background: pd.DataFrame, path: Path = config.BACKGROUND_PATH) -> None:
    payload = {
        "features": list(background.columns),
        "n_rows": len(background),
        "note": "Random sample of training rows used as the reference for /explain.",
        "rows": background.round(6).values.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_background(path: Path = config.BACKGROUND_PATH) -> pd.DataFrame | None:
    """Load the background sample, or ``None`` if the file does not exist."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return pd.DataFrame(data["rows"], columns=data["features"])


def explain_prediction(model, features: list[str], row: dict, background: pd.DataFrame) -> dict:
    """Return the baseline, the prediction and one contribution per feature (all in MWh).

    ``row`` maps every name in ``features`` to its value for the requested hour.
    """
    n = len(features)
    x = np.array([row[f] for f in features], dtype=float)
    bg = background[features].to_numpy(dtype=float)  # (B, n)

    if n <= MAX_EXACT_FEATURES:
        method = "shapley_exact"
        baseline, prediction, phi = _exact_shapley(model, features, x, bg)
    else:
        method = "occlusion"
        baseline, prediction, phi = _occlusion(model, features, x, bg)

    contributions = [
        {"feature": f, "value": float(v), "contribution_mwh": float(c)}
        for f, v, c in zip(features, x, phi, strict=True)
    ]
    contributions.sort(key=lambda item: abs(item["contribution_mwh"]), reverse=True)
    return {
        "method": method,
        "baseline_mwh": float(baseline),
        "prediction_mwh": float(prediction),
        "contributions": contributions,
    }


def _predict(model, features: list[str], data: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(pd.DataFrame(data, columns=features)), dtype=float)


def _exact_shapley(model, features, x, bg):
    n, n_bg = len(features), len(bg)
    n_subsets = 1 << n
    masks = ((np.arange(n_subsets)[:, None] >> np.arange(n)) & 1).astype(bool)  # (2^n, n)

    # Every subset S x every background row: keep x where the feature is in S, else the background.
    data = np.where(masks[:, None, :], x[None, None, :], bg[None, :, :])  # (2^n, B, n)
    v = _predict(model, features, data.reshape(-1, n)).reshape(n_subsets, n_bg).mean(axis=1)

    size_weight = np.array(
        [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]
    )
    sizes = masks.sum(axis=1)
    subsets = np.arange(n_subsets)
    phi = np.zeros(n)
    for i in range(n):
        without = subsets[~masks[:, i]]
        phi[i] = np.sum(size_weight[sizes[without]] * (v[without | (1 << i)] - v[without]))
    return v[0], v[-1], phi


def _occlusion(model, features, x, bg):
    n, n_bg = len(features), len(bg)
    full = _predict(model, features, x[None, :])[0]
    baseline = _predict(model, features, bg).mean()
    phi = np.zeros(n)
    for i in range(n):
        data = np.tile(x, (n_bg, 1))
        data[:, i] = bg[:, i]
        phi[i] = full - _predict(model, features, data).mean()
    return baseline, full, phi


def main() -> None:
    """Create ``models/background.json`` from the prepared training data."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = json.loads(config.MODEL_META_PATH.read_text())
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    background = build_background(
        train, meta["features"], config.BACKGROUND_ROWS, config.RANDOM_STATE
    )
    save_background(background)
    logger.info("Wrote %d background rows to %s", len(background), config.BACKGROUND_PATH)


if __name__ == "__main__":
    main()
