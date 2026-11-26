# Contributions: Milestones 1 and 2

## Team

| Member | ERP | Milestone 1 workstream | Milestone 2 workstream |
|---|---|---|---|
| Muhammad Zain Chinoy | 29059 | Data pipeline and repository setup | Explainability (`/explain`), Render deployment and release |
| Areeba Salman | 29075 | Experiments and MLflow | Docker image and `/stats` monitoring |
| Abdullah Feroz | 29219 | API and testing | CI pipeline, runbook and documentation |

Each milestone was split into three workstreams. Each member owns the code, tests and documentation for their workstream, and every pull request is reviewed by a different member. Links: [repository](https://github.com/Zain-Chinoy/energy-forecasting-mlops), live service: https://energy-forecasting-mlops-n29o.onrender.com/docs.

---

# Milestone 1 (Model and API)

## Work split

| Workstream | Owner | Main files | Tests | Branch | Reviewer |
|---|---|---|---|---|---|
| Data pipeline and repository setup | Muhammad Zain Chinoy | `src/prepare_data.py`, `src/features.py`, `src/config.py` (paths, split dates, valid ranges), `data/raw/README.md`, `requirements*.txt`, `.gitignore`, `LICENSE`, GitHub repository | `tests/test_prepare_data.py` (9), `tests/test_features.py` (3) | `feat/data-prep` | Areeba Salman |
| Experiments and MLflow | Areeba Salman | `src/train.py`, `src/config.py` (feature sets), `models/`, `reports/` | `tests/test_train.py` (7) | `feat/mlflow-training` | Abdullah Feroz |
| API and testing | Abdullah Feroz | `src/model.py`, `src/app.py`, `pyproject.toml` (ruff and pytest settings) | `tests/test_api.py`, `tests/test_model.py`, `tests/conftest.py` | `feat/api` | Muhammad Zain Chinoy |
| Contribution file and final docs | All three | `CONTRIBUTION.md`, final `README.md` edits | n/a | `feat/docs` | Abdullah Feroz and Areeba Salman |

## Detailed tasks

### Muhammad Zain Chinoy: data pipeline and repository setup
- Explored the Panama electricity dataset and documented its quirks: humidity labelled "%" but stored as specific humidity, the Jan 2019 outage, the COVID-era demand drop from March 2020, and the redundant `Holiday_ID` column.
- Wrote `prepare_data.py`: column checks, duplicate and timestamp handling, range validation, interpolation of gaps of at most 3 hours, a validation report, and the chronological train / test / 2020-shift split (no data leakage).
- Wrote `features.py` so training and the API derive hour, day of week, month and weekend flag with the same code.
- Defined the paths, split dates and valid physical ranges in `config.py`.
- Wrote the data-preparation and feature tests, and the Dataset and data-preparation sections of the README.
- Set up the GitHub repository: created it, added the team as collaborators, protected `main` so changes go through reviewed pull requests, and pushed the initial project scaffold.
- Added the pinned `requirements.txt` and `requirements-dev.txt`, the `.gitignore` and the `LICENSE`, and tagged the milestone release (`v1.0-milestone1`).

### Areeba Salman: experiments and MLflow
- Designed the experiment plan: baseline, three model types (Linear Regression, Random Forest, Gradient Boosting) and eight feature-set combinations.
- Implemented time-series cross-validation, the random hyperparameter search with one child run per candidate, and the final one-time evaluation on the test period.
- Set up MLflow tracking: parameters, metrics (RMSE, MAE, R2, MAPE), tags, plots, model signature and input example, and the model registry.
- Selected and saved the final model (`models/model.joblib`, `model_meta.json`) and generated `reports/` (results tables and figures).
- Wrote the training tests, and the MLflow design, results and findings sections of the README.

### Abdullah Feroz: API and testing
- Built the FastAPI service: `/predict` with strict Pydantic validation (types, ranges, no unknown fields), `/health`, `/model-info` and the auto-generated `/docs`.
- Wrote `model.py` (model loading and single-row prediction) and handled the case where the model file is missing (HTTP 503).
- Wrote the API and model tests: valid and invalid requests, deterministic output, and a check that the model beats the baseline.
- Configured linting and testing in `pyproject.toml` (ruff and pytest settings) and kept `ruff` and `pytest` passing.
- Wrote the architecture diagram, Quick start and API usage sections of the README.

---

# Milestone 2 (Docker, CI/CD, Deployment and Monitoring)

The Milestone 2 work was divided by difficulty. The hardest part, the explainability engine behind `/explain` together with the cloud deployment, went to Zain. Areeba took the second part, containerisation and request monitoring. Abdullah took the third part, the CI pipeline, the runbook and the documentation.

## Work split

| Workstream | Owner | Main files | Tests | Branch | Reviewer |
|---|---|---|---|---|---|
| Explainability, Render deployment and release | Muhammad Zain Chinoy | `src/explain.py`, `explain_one` in `src/model.py`, `/explain` in `src/app.py`, `models/background.json`, background step in `src/train.py`, Render service, README Project Focus and Deployment sections, tag `v2.0-milestone2` | `tests/test_explain.py` (9), `/explain` cases in `tests/test_api.py` and `tests/test_model.py` | `feat/explain-and-stats` (explain part), `docs/live-url` | Areeba Salman |
| Docker image and request monitoring | Areeba Salman | `Dockerfile`, `.dockerignore`, `render.yaml`, `requirements.txt` pin fix, `src/monitoring.py`, `/stats` and the request middleware in `src/app.py`, README Docker and Monitoring sections | `tests/test_monitoring.py` (4), `/stats` cases in `tests/test_api.py` | `infra/docker`, `feat/explain-and-stats` (stats part) | Abdullah Feroz |
| CI pipeline, runbook and documentation | Abdullah Feroz | `.github/workflows/ci.yml`, `RUNBOOK.md`, `AI_USAGE.md`, `CONTRIBUTION.md`, README CI section and credits, `LICENSE` and `data/raw/README.md` updates | n/a (pipeline runs the full suite) | `infra/ci`, `docs/reflections` | Muhammad Zain Chinoy |
| Demo video | All three | 4 to 6 minute video covering intro, clean start, Docker, monitoring, CI, deployment and next steps | n/a | n/a | n/a |

The `/explain` and `/stats` code share `src/app.py` and `src/model.py`, so both were delivered in one pull request (`feat/explain-and-stats`) after a joint review.

## Detailed tasks

### Muhammad Zain Chinoy: explainability, Render deployment and release (most complex part)
- Designed and built `POST /explain`, our Project Focus. It returns the prediction, a baseline and one contribution per feature in MWh, so that `prediction = baseline + sum of contributions`.
- Implemented exact interventional Shapley values from scratch in `src/explain.py`, without the `shap` package. For the 9 features of the selected model it evaluates 512 feature subsets against 50 background rows (25,600 model evaluations) in one vectorised call, using the Shapley weights `s!(n-s-1)!/n!`. Above 12 features it falls back to a cheaper occlusion method.
- Created the fixed background sample (`models/background.json`), made `train.py` regenerate it, and made the model loader reject a background file whose columns do not match the model (so a stale file cannot give silent wrong answers).
- Wrote the explanation tests: a linear model whose Shapley values are known in closed form, the efficiency property (contributions add up), symmetry, the occlusion fallback, and a check that the committed background matches the committed model.
- Deployed the service on Render: connected the GitHub repository, chose the Docker runtime and the free instance, set the `/health` health check path and automatic deploys, and verified the live `/health`, `/explain` and `/stats` responses.
- Wrote the Project Focus and Deployment sections of the README (live URL and cURL examples), measured the `/explain` latency on three machines and corrected the README claim.
- Handled the release: merged the reviewed pull requests, and created and moved the `v2.0-milestone2` tag so that it points at a commit with a passing CI run.

### Areeba Salman: Docker image and request monitoring
- Wrote the `Dockerfile`: slim Python 3.11 base image, dependencies installed before the source so that the layer is cached, only `src/` and `models/` copied (no data, no MLflow), a non-root user, `$PORT` support for Render, and a `HEALTHCHECK` on `/health` so that `docker ps` shows `(healthy)`.
- Added `.dockerignore` and the optional `render.yaml` blueprint, and found and fixed a dependency problem: `numpy==2.5.4` is not installable on Python 3.11, so the pin was changed to 2.4.6.
- Verified the container locally: build, run, `(healthy)` status, and `/health`, `/predict`, `/explain` and `/stats` against the running container.
- Built the monitoring layer: `src/monitoring.py` (a thread-safe request counter with total, error, average and maximum latency per endpoint), the request middleware in `src/app.py`, and `GET /stats`. Health checks, `/stats` and `/docs` are deliberately not counted, so platform pings do not hide real traffic.
- Wrote the monitoring tests (counts, latency, errors, thread safety) and the `/stats` API tests, and the Docker and Monitoring sections of the README.

### Abdullah Feroz: CI pipeline, runbook and documentation
- Wrote the GitHub Actions workflow `.github/workflows/ci.yml`. On every push and pull request, and on `v*` tags, it runs three jobs: `lint` (ruff check and format check), `test` (pytest on Python 3.11) and `build`. The `build` job only runs after the other two pass. It builds the Docker image, waits until the container is healthy, and smoke-tests `/health`, `/predict`, `/explain` and `/stats` on the running container.
- Wrote `RUNBOOK.md`: how to run the service locally and in Docker, first deployment on Render, redeploying, rolling back (dashboard rollback, `git revert`, image tags), monitoring, troubleshooting and shutting the service down after grading.
- Wrote `AI_USAGE.md`, recording which AI tools were used and how the output was checked.
- Updated `CONTRIBUTION.md`, the test count and CI section of the README, the credits and open-source section, and the `LICENSE` and data-notes updates (for example, the corrected 12% COVID demand drop).
- Coordinated the pull requests and reviews, and checked every pull request for green CI before merging.

## Shared work
- Project proposal and choice of dataset.
- Reviewing each other's pull requests, as listed below.
- The Milestone 2 demo video: each member presents the part they built (intro and deployment, Docker and monitoring, CI and next steps).

## Review rotation

- Milestone 1: `feat/data-prep` (Zain) is reviewed by Areeba, `feat/mlflow-training` (Areeba) by Abdullah, `feat/api` (Abdullah) by Zain.
- Milestone 2: `feat/explain-and-stats` (Zain and Areeba) is reviewed by Abdullah, `infra/docker` (Areeba) by Zain, and `infra/ci` and `docs/reflections` (Abdullah) by Areeba.
- Direct pushes to `main` are not allowed. Branches follow the `feat/`, `infra/`, `docs/` and `fix/` naming convention. CI must pass on a pull request before it is merged.

---

## Individual reflections

What each member built, what stood out, and what they learned.

### Muhammad Zain Chinoy

**Milestone 1.** Working on the data pipeline showed me how much of an ML project happens before any model is trained. I found that the humidity column is documented as a percentage but actually holds specific humidity, and that demand drops sharply from March 2020, so I kept 2020 out of training and testing. The biggest lesson was splitting the data by date instead of randomly, so that the test results are honest and nothing leaks from the future into training. I also set up the GitHub repository and protected `main`, so the whole team works through reviewed pull requests.

**Milestone 2.** I built the `/explain` endpoint, which computes exact Shapley values so every prediction comes with a per-feature breakdown that adds up to the prediction. The most useful check was a test against a linear model, where the correct answer is known in closed form. I also learned that explaining a prediction is much more expensive than making one, because it evaluates the model about 25,000 times, and that the same service was about ten times slower on Render's free instance than on a fast laptop. Deploying on Render taught me that the free tier sleeps after 15 minutes and that the health check path has to match the app.

### Areeba Salman

**Milestone 1.** I designed and ran the MLflow experiments, comparing three model types across eight feature sets and logging every run so the final choice is backed by evidence. The most surprising result was that adding humidity, rainfall and wind did not improve the model, and calendar features such as hour and day of week did most of the work. I learned that choosing the best model with cross-validation, and touching the test set only once, matters more than trying many options.

**Milestone 2.** I wrote the Dockerfile and the `/stats` monitoring. The Dockerfile copies the requirements first so that the slow dependency install is cached, and it runs as a non-root user with a healthcheck, so `docker ps` shows `(healthy)`. For monitoring, I learned to exclude health checks and `/stats` itself from the counts, because platform pings would otherwise hide the real traffic. I also found that one pinned package did not install on Python 3.11 and had to be repinned, which showed me why the build must be tested in the same Python version as production.

### Abdullah Feroz

**Milestone 1.** I built the FastAPI service and the test suite around the model. Strict Pydantic validation was the key part, because it rejects impossible inputs such as a temperature of 100 degrees before they reach the model. I also learned why we pin exact package versions and keep training and serving feature code in one place, so a saved model behaves the same everywhere it runs.

**Milestone 2.** I wrote the GitHub Actions pipeline, the runbook and the documentation. The pipeline runs lint, the tests and a Docker build that starts the container and calls every endpoint, so each push is checked end to end rather than only by unit tests. I learned that CI is most useful when it tests the same thing that gets deployed, and that a runbook with a clear rollback step is what makes a deployment safe to change. Keeping the README, the test counts and the contribution file in step with the code was also a job of its own.