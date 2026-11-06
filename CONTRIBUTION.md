# Contributions: Milestone 1 (Model & API)

## Team

| Member | ERP | Workstream |
|---|---|---|
| Muhammad Zain Chinoy | 29059 | Data pipeline |
| Areeba Salman | 29075 | Experiments and MLflow |
| Abdullah Feroz | 29219 | API, testing and repository setup |

Milestone 1 was split into three workstreams of roughly equal size. Each member owns the code, tests and documentation for one workstream, and every pull request is reviewed by a different member.

## Work split

| Workstream | Owner | Main files | Tests | Branch | Reviewer |
|---|---|---|---|---|---|
| Data pipeline | Muhammad Zain Chinoy | `src/prepare_data.py`, `src/features.py`, `src/config.py` (paths, split dates, valid ranges), `data/raw/README.md` | `tests/test_prepare_data.py` (9), `tests/test_features.py` (3) | `feat/data-prep` | Areeba Salman |
| Experiments and MLflow | Areeba Salman | `src/train.py`, `src/config.py` (feature sets), `models/`, `reports/` | `tests/test_train.py` (7) | `feat/mlflow-training` | Abdullah Feroz |
| API, testing and repository setup | Abdullah Feroz | `src/model.py`, `src/app.py`, `requirements*.txt`, `pyproject.toml`, `.gitignore`, `LICENSE` | `tests/test_api.py` (17), `tests/test_model.py` (6), `tests/conftest.py` | `feat/api` | Muhammad Zain Chinoy |
| Contribution file and final docs | All three | `CONTRIBUTION.md`, final `README.md` edits | n/a | `feat/docs` | Abdullah Feroz and Areeba Salman |

## Detailed tasks

### Muhammad Zain Chinoy: data pipeline
- Explored the Panama electricity dataset and documented its quirks: humidity labelled "%" but stored as specific humidity, the Jan 2019 outage, the COVID-era demand drop from March 2020, and the redundant `Holiday_ID` column.
- Wrote `prepare_data.py`: column checks, duplicate and timestamp handling, range validation, interpolation of gaps of at most 3 hours, a validation report, and the chronological train / test / 2020-shift split (no data leakage).
- Wrote `features.py` so training and the API derive hour, day of week, month and weekend flag with the same code.
- Defined the paths, split dates and valid physical ranges in `config.py`.
- Wrote the data-preparation and feature tests, and the Dataset and data-preparation sections of the README.

### Areeba Salman: experiments and MLflow
- Designed the experiment plan: baseline, three model types (Linear Regression, Random Forest, Gradient Boosting) and eight feature-set combinations.
- Implemented time-series cross-validation, the random hyperparameter search with one child run per candidate, and the final one-time evaluation on the test period.
- Set up MLflow tracking: parameters, metrics (RMSE, MAE, R2, MAPE), tags, plots, model signature and input example, and the model registry.
- Selected and saved the final model (`models/model.joblib`, `model_meta.json`) and generated `reports/` (results tables and figures).
- Wrote the training tests, and the MLflow design, results and findings sections of the README.

### Abdullah Feroz: API, testing and repository setup
- Built the FastAPI service: `/predict` with strict Pydantic validation (types, ranges, no unknown fields), `/health`, `/model-info` and the auto-generated `/docs`.
- Wrote `model.py` (model loading and single-row prediction) and handled the case where the model file is missing (HTTP 503).
- Wrote the API and model tests: valid and invalid requests, deterministic output, and a check that the model beats the baseline.
- Set up the repository: pinned `requirements.txt` and `requirements-dev.txt`, `pyproject.toml` (ruff and pytest settings), `.gitignore` and `LICENSE`; kept `ruff` and `pytest` passing.
- Wrote the architecture diagram, Quick start and API usage sections of the README.

### Shared work
- Project proposal and choice of dataset.
- Milestone 1 presentation: each member prepares the slides for their own workstream.
- Reviewing each other's pull requests, as listed in the table above.

## Review rotation

- Pull request for `feat/data-prep` (Zain) is reviewed by Areeba.
- Pull request for `feat/mlflow-training` (Areeba) is reviewed by Abdullah.
- Pull request for `feat/api` (Abdullah) is reviewed by Zain.
- Direct pushes to `main` are not allowed. Branches follow the `feat/` and `fix/` naming convention.

## Individual reflections

What each member built, what stood out, and what they learned.

### Muhammad Zain Chinoy
Working on the data pipeline showed me how much of an ML project happens before any model is trained. I found that the humidity column is documented as a percentage but actually holds specific humidity, and that demand drops sharply from March 2020, so I kept 2020 out of training and testing. The biggest lesson was splitting the data by date instead of randomly, so that the test results are honest and nothing leaks from the future into training.

### Areeba Salman
I designed and ran the MLflow experiments, comparing three model types across eight feature sets and logging every run so the final choice is backed by evidence. The most surprising result was that adding humidity, rainfall and wind did not improve the model, and calendar features such as hour and day of week did most of the work. I learned that choosing the best model with cross-validation, and touching the test set only once, matters more than trying many options.

### Abdullah Feroz
I built the FastAPI service and the test suite around the model. Strict Pydantic validation was the key part, because it rejects impossible inputs such as a temperature of 100 degrees before they reach the model. I also learned why we pin exact package versions and keep training and serving feature code in one place, so a saved model behaves the same everywhere it runs.
