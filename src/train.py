"""Train, compare and select the energy-demand model, tracking everything in MLflow.

Usage:
    python -m src.prepare_data      # once, creates data/processed/*.csv
    python -m src.train             # full experiment plan (several minutes)
    python -m src.train --quick     # tiny smoke run

Experiment plan (all tracked in one MLflow experiment):
    1. exploration           baseline + 3 model types x 8 feature sets, each scored with
                             time-series cross-validation on the TRAINING period only
    2. hyperparameter_search random search for the best (model, feature set); one child
                             run per candidate
    3. final_model           refit the winner on all training data, evaluate ONCE on the
                             held-out test period, log plots, signature and the model

Model/feature selection uses cross-validation only; the test set is not used for any
choice. The winning model is saved to models/ for the API.
"""

import argparse
import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src import config  # noqa: E402

logger = logging.getLogger("train")

CATEGORICAL_COLS = ["hour", "dayofweek", "month"]  # one-hot encoded for linear models
BASELINE = "baseline_mean"

MODEL_SPECS: dict[str, dict] = {
    BASELINE: {"defaults": {}, "search_space": {}},
    "linear_regression": {"defaults": {}, "search_space": {}},
    "random_forest": {
        "defaults": {
            "n_estimators": 50,
            "max_depth": None,
            "min_samples_leaf": 10,
            "max_features": 0.5,
            "max_samples": 0.5,  # sub-sampling keeps training fast on small machines
        },
        "search_space": {
            "n_estimators": [50, 100, 150],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [5, 10, 20],
            "max_features": [0.3, 0.5, 0.8, 1.0],
        },
    },
    "hist_gradient_boosting": {
        "defaults": {
            "learning_rate": 0.1,
            "max_iter": 200,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        "search_space": {
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_iter": [200, 400, 600],
            "max_depth": [None, 6, 10],
            "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
    },
}
TRAINABLE_MODELS = ["linear_regression", "random_forest", "hist_gradient_boosting"]


# ----------------------------------------------------------------------------- models
def build_model(model_type: str, features: list[str], params: dict | None = None):
    """Create an unfitted estimator for ``model_type`` using the given hyperparameters."""
    p = {**MODEL_SPECS[model_type]["defaults"], **(params or {})}
    if model_type == BASELINE:
        return DummyRegressor(strategy="mean")
    if model_type == "linear_regression":
        cat = [c for c in CATEGORICAL_COLS if c in features]
        num = [c for c in features if c not in cat]
        prep = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                ("num", StandardScaler(), num),
            ]
        )
        return Pipeline([("prep", prep), ("model", LinearRegression())])
    if model_type == "random_forest":
        return RandomForestRegressor(**p, random_state=config.RANDOM_STATE, n_jobs=-1)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**p, random_state=config.RANDOM_STATE)
    raise ValueError(f"Unknown model type: {model_type}")


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    """RMSE (primary), MAE, R2 and MAPE (%)."""
    return {
        f"{prefix}rmse": float(root_mean_squared_error(y_true, y_pred)),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def cv_scores(model, X: pd.DataFrame, y: pd.Series, cv) -> dict[str, float]:
    """Time-series cross-validation scores (expanding window, training period only)."""
    res = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=1,
    )
    return {
        "cv_rmse_mean": float(-res["test_rmse"].mean()),
        "cv_rmse_std": float(res["test_rmse"].std()),
        "cv_mae_mean": float(-res["test_mae"].mean()),
        "cv_r2_mean": float(res["test_r2"].mean()),
        "fit_time_s": float(res["fit_time"].mean()),
    }


# ------------------------------------------------------------------------- mlflow runs
def setup_mlflow(tracking_uri: str, artifact_root: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=artifact_root)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def run_config(
    stage: str,
    run_name: str,
    model_type: str,
    fs_name: str,
    features: list[str],
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    common_tags: dict,
) -> dict:
    """Cross-validate one (model, feature set, params) combination as a child MLflow run."""
    full_params = {**MODEL_SPECS[model_type]["defaults"], **params}
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tags(
            {**common_tags, "stage": stage, "model_type": model_type, "feature_set": fs_name}
        )
        mlflow.log_params(
            {
                "model_type": model_type,
                "feature_set": fs_name,
                "n_features": len(features),
                "features": ",".join(features),
                "cv_splits": cv.get_n_splits(),
                **full_params,
            }
        )
        scores = cv_scores(build_model(model_type, features, params), X[features], y, cv)
        mlflow.log_metrics(scores)
    return {
        "run_id": run.info.run_id,
        "model_type": model_type,
        "feature_set": fs_name,
        "n_features": len(features),
        "params": full_params,
        **scores,
    }


def explore(X, y, cv, common_tags, feature_sets, models) -> pd.DataFrame:
    """Stage 1: baseline + every model type on every feature set."""
    rows = []
    with mlflow.start_run(run_name="exploration"):
        mlflow.set_tags({**common_tags, "stage": "exploration_parent"})
        mlflow.log_params({"n_models": len(models), "n_feature_sets": len(feature_sets)})
        all_feats = config.FEATURE_SETS["all_features"]
        rows.append(
            run_config(
                "exploration",
                f"{BASELINE}__none",
                BASELINE,
                "none",
                all_feats,
                {},
                X,
                y,
                cv,
                common_tags,
            )
        )
        rows[-1]["n_features"] = 0
        for fs_name, feats in feature_sets.items():
            for model_type in models:
                t0 = time.time()
                rows.append(
                    run_config(
                        "exploration",
                        f"{model_type}__{fs_name}",
                        model_type,
                        fs_name,
                        feats,
                        {},
                        X,
                        y,
                        cv,
                        common_tags,
                    )
                )
                logger.info(
                    "  %-24s %-24s cv_rmse=%8.2f  (%.0fs)",
                    model_type,
                    fs_name,
                    rows[-1]["cv_rmse_mean"],
                    time.time() - t0,
                )
        results = pd.DataFrame(rows)
        best = results[results.model_type != BASELINE].nsmallest(1, "cv_rmse_mean").iloc[0]
        mlflow.log_metrics(
            {
                "best_cv_rmse": float(best.cv_rmse_mean),
                "baseline_cv_rmse": float(
                    results[results.model_type == BASELINE].cv_rmse_mean.iloc[0]
                ),
            }
        )
        mlflow.set_tags({"best_model_type": best.model_type, "best_feature_set": best.feature_set})
    return results


def hyperparameter_search(best, features, X, y, cv, common_tags, n_iter) -> pd.DataFrame:
    """Stage 2: random search for the winning (model, feature set); child run per candidate."""
    space = MODEL_SPECS[best.model_type]["search_space"]
    rows = []
    with mlflow.start_run(run_name="hyperparameter_search"):
        mlflow.set_tags(
            {
                **common_tags,
                "stage": "search_parent",
                "model_type": best.model_type,
                "feature_set": best.feature_set,
            }
        )
        mlflow.log_params({"n_iter": n_iter, "search_type": "random"})
        if space and n_iter > 0:
            for i, params in enumerate(
                ParameterSampler(space, n_iter=n_iter, random_state=config.RANDOM_STATE)
            ):
                rows.append(
                    run_config(
                        "search",
                        f"search_{i:02d}",
                        best.model_type,
                        best.feature_set,
                        features,
                        params,
                        X,
                        y,
                        cv,
                        common_tags,
                    )
                )
                logger.info("  search_%02d cv_rmse=%8.2f %s", i, rows[-1]["cv_rmse_mean"], params)
        results = pd.DataFrame(rows)
        if len(results):
            mlflow.log_metric("best_search_cv_rmse", float(results.cv_rmse_mean.min()))
    return results


# --------------------------------------------------------------------------------- plots
def save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(results: pd.DataFrame, path: Path) -> Path:
    data = results[results.model_type != BASELINE]
    pivot = data.pivot(index="feature_set", columns="model_type", values="cv_rmse_mean")
    pivot = pivot.sort_values(pivot.columns[-1])
    base = float(results[results.model_type == BASELINE].cv_rmse_mean.iloc[0])
    ax = pivot.plot.barh(figsize=(9, 5.5), width=0.8)
    ax.axvline(base, color="grey", linestyle="--", label=f"mean baseline ({base:.0f})")
    ax.set_xlabel("Cross-validated RMSE (MWh), lower is better")
    ax.set_ylabel("")
    ax.set_title("Model x feature-set comparison (time-series CV, training period)")
    ax.legend(fontsize=8)
    return save_fig(ax.figure, path)


def plot_final(model, features, test, pred, path_dir: Path) -> list[Path]:
    y = test[config.TARGET].to_numpy()
    paths = []

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, pred, s=4, alpha=0.4)
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlabel("Actual demand (MWh)")
    ax.set_ylabel("Predicted demand (MWh)")
    ax.set_title("Predicted vs actual (test period)")
    paths.append(save_fig(fig, path_dir / "pred_vs_actual.png"))

    start = len(test) // 2
    sl = slice(start, start + 24 * 14)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test[config.DATETIME_COL].iloc[sl], y[sl], label="actual")
    ax.plot(test[config.DATETIME_COL].iloc[sl], pred[sl], label="predicted")
    ax.set_ylabel("Demand (MWh)")
    ax.set_title("Two weeks of the test period")
    ax.legend()
    fig.autofmt_xdate()
    paths.append(save_fig(fig, path_dir / "test_timeseries.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y - pred, bins=60)
    ax.set_xlabel("Residual: actual - predicted (MWh)")
    ax.set_title("Residuals (test period)")
    paths.append(save_fig(fig, path_dir / "residuals.png"))

    imp = permutation_importance(
        model,
        test[features],
        test[config.TARGET],
        n_repeats=3,
        random_state=config.RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(features) + 1.5))
    ax.barh(np.array(features)[order], imp.importances_mean[order])
    ax.set_xlabel("Increase in RMSE when feature is shuffled (MWh)")
    ax.set_title("Permutation feature importance (test period)")
    paths.append(save_fig(fig, path_dir / "feature_importance.png"))
    return paths


# ------------------------------------------------------------------------------ reports
def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- pipeline
def run_training(
    processed_dir: Path = config.PROCESSED_DIR,
    models_dir: Path = config.MODELS_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    artifact_root: str = config.MLFLOW_ARTIFACT_ROOT,
    n_iter: int = 8,
    cv_splits: int = 5,
    cv_test_days: int = 90,
    quick: bool = False,
    register: bool = True,
) -> dict:
    """Run the whole experiment plan and save the best model. Returns the metadata dict."""
    train_path, test_path = processed_dir / "train.csv", processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed data not found. Run `python -m src.prepare_data` first.")
    parse = [config.DATETIME_COL]
    train = pd.read_csv(train_path, parse_dates=parse)
    test = pd.read_csv(test_path, parse_dates=parse)
    shift_path = processed_dir / "shift_2020.csv"
    shift = pd.read_csv(shift_path, parse_dates=parse) if shift_path.exists() else None

    feature_sets = dict(config.FEATURE_SETS)
    models = list(TRAINABLE_MODELS)
    if quick:
        feature_sets = {k: feature_sets[k] for k in ("calendar_only", "all_features")}
        models = ["linear_regression", "hist_gradient_boosting"]
        n_iter, cv_splits, cv_test_days = min(n_iter, 3), 3, 7

    setup_mlflow(tracking_uri, artifact_root)
    X_train, y_train = train, train[config.TARGET]
    cv = TimeSeriesSplit(n_splits=cv_splits, test_size=24 * cv_test_days, gap=24)
    train_start = train[config.DATETIME_COL].min()
    train_end = train[config.DATETIME_COL].max()
    train_period = f"{train_start:%Y-%m-%d}..{train_end:%Y-%m-%d}"
    common_tags = {
        "dataset": config.DATASET_NAME,
        "train_rows": str(len(train)),
        "test_rows": str(len(test)),
        "train_period": train_period,
        "author": "mlops-project-team",
    }
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Stage 1/3: exploration (%d feature sets x %d models)", len(feature_sets), len(models)
    )
    results = explore(X_train, y_train, cv, common_tags, feature_sets, models)
    results.drop(columns=["params"]).to_csv(reports_dir / "exploration_results.csv", index=False)
    comparison_png = plot_model_comparison(results, figures_dir / "model_comparison.png")

    best = results[results.model_type != BASELINE].nsmallest(1, "cv_rmse_mean").iloc[0]
    best_features = feature_sets[best.feature_set]
    logger.info(
        "Best so far: %s on %s (cv_rmse=%.2f)", best.model_type, best.feature_set, best.cv_rmse_mean
    )

    logger.info("Stage 2/3: hyperparameter search for %s", best.model_type)
    search = hyperparameter_search(best, best_features, X_train, y_train, cv, common_tags, n_iter)
    chosen_params, chosen_cv = dict(best.params), best.to_dict()
    search_view = pd.DataFrame()
    if len(search):
        search_view = search.assign(params=search["params"].map(json.dumps)).drop(
            columns=["run_id", "fit_time_s", "model_type", "feature_set", "n_features"]
        )
        search_view = search_view[["params", *[c for c in search_view if c != "params"]]]
        search_view.to_csv(reports_dir / "search_results.csv", index=False)
        top = search.nsmallest(1, "cv_rmse_mean").iloc[0]
        if top.cv_rmse_mean < best.cv_rmse_mean:
            chosen_params, chosen_cv = dict(top.params), top.to_dict()

    logger.info("Stage 3/3: final fit and one-time evaluation on the test period")
    with mlflow.start_run(run_name="final_model") as run:
        mlflow.set_tags(
            {
                **common_tags,
                "stage": "final",
                "model_type": best.model_type,
                "feature_set": best.feature_set,
            }
        )
        mlflow.log_params(
            {
                "model_type": best.model_type,
                "feature_set": best.feature_set,
                "n_features": len(best_features),
                "features": ",".join(best_features),
                **{k: v for k, v in chosen_params.items()},
            }
        )
        model = build_model(best.model_type, best_features, chosen_params)
        model.fit(train[best_features], y_train)
        pred = model.predict(test[best_features])
        metrics = regression_metrics(test[config.TARGET], pred, "test_")
        base = build_model(BASELINE, best_features).fit(train[best_features], y_train)
        metrics["baseline_test_rmse"] = float(
            root_mean_squared_error(test[config.TARGET], base.predict(test[best_features]))
        )
        metrics["rmse_improvement_vs_baseline_pct"] = float(
            100 * (1 - metrics["test_rmse"] / metrics["baseline_test_rmse"])
        )
        metrics["cv_rmse_mean"] = float(chosen_cv["cv_rmse_mean"])
        metrics["cv_rmse_std"] = float(chosen_cv["cv_rmse_std"])
        shift_metrics = {}
        if shift is not None and len(shift):
            shift_metrics = regression_metrics(
                shift[config.TARGET], model.predict(shift[best_features]), "shift2020_"
            )
        mlflow.log_metrics({**metrics, **shift_metrics})

        plots = [comparison_png, *plot_final(model, best_features, test, pred, figures_dir)]
        for p in plots:
            mlflow.log_artifact(str(p), artifact_path="plots")
        results.drop(columns=["params"]).to_csv(
            reports_dir / "exploration_results.csv", index=False
        )
        mlflow.log_artifact(str(reports_dir / "exploration_results.csv"), artifact_path="reports")
        mlflow.log_dict(
            {"features": best_features, "feature_set": best.feature_set}, "features.json"
        )
        validation_report = processed_dir / "validation_report.json"
        if validation_report.exists():
            mlflow.log_artifact(str(validation_report), artifact_path="data")

        sig_input = train[best_features].astype("float64")
        signature = infer_signature(
            sig_input.head(200), model.predict(train[best_features].head(200))
        )
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=sig_input.head(3),
            # skops (the default) refuses HistGradientBoosting trees as "untrusted";
            # the model is produced by this script, so cloudpickle is fine here.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=config.REGISTERED_MODEL_NAME if register else None,
        )

        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / "model.joblib", compress=3)
        meta = {
            "model_type": best.model_type,
            "feature_set": best.feature_set,
            "features": best_features,
            "params": {k: (None if v is None else v) for k, v in chosen_params.items()},
            "cv": {
                k: chosen_cv[k]
                for k in ("cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_r2_mean")
            },
            "test": {
                k: v for k, v in metrics.items() if k != "cv_rmse_mean" and k != "cv_rmse_std"
            },
            "shift_2020": shift_metrics,
            "dataset": config.DATASET_NAME,
            "train_rows": len(train),
            "test_rows": len(test),
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment": config.MLFLOW_EXPERIMENT,
            "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        }
        (models_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(models_dir / "model_meta.json"), artifact_path="reports")

    # Human-readable summary used in the README.
    show = results.drop(columns=["params", "run_id", "fit_time_s"]).sort_values("cv_rmse_mean")
    summary = [
        "# Experiment results\n",
        f"Final model: **{best.model_type}** on feature set **{best.feature_set}** "
        f"({len(best_features)} features).\n",
        "## Final test-period metrics (evaluated once)\n",
        md_table(pd.DataFrame([{k: v for k, v in metrics.items()}])),
        "\n## Distribution-shift check (Jan-Jun 2020, not used for any decision)\n",
        md_table(pd.DataFrame([shift_metrics])) if shift_metrics else "n/a",
        "\n## Exploration: cross-validated results (training period)\n",
        md_table(show),
        "\n## Hyperparameter search\n",
        md_table(search_view.sort_values("cv_rmse_mean")) if len(search_view) else "n/a",
        f"\nBest parameters: `{json.dumps(chosen_params)}`\n",
    ]
    (reports_dir / "results.md").write_text("\n".join(summary))
    logger.info(
        "Done. Test RMSE=%.2f MWh (baseline %.2f).",
        metrics["test_rmse"],
        metrics["baseline_test_rmse"],
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--n-iter", type=int, default=8, help="random-search candidates")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="tiny smoke run")
    parser.add_argument("--no-register", action="store_true", help="skip the MLflow model registry")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    run_training(
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        quick=args.quick,
        register=not args.no_register,
    )


if __name__ == "__main__":
    main()
