# Runbook: Energy Demand Forecasting API

How to run, deploy, redeploy and roll back the service. The deployment steps describe Render (the platform we chose). Button names were checked against the Render docs on 2026-11-25, but the dashboard may be labelled slightly differently.

## Service facts

| Item | Value |
|---|---|
| Image | Built from `Dockerfile` (Python 3.11, FastAPI, uvicorn) |
| Port | Reads `$PORT` (Render sets 10000), falls back to 8000 locally |
| Health check | `GET /health` returns 200 |
| Monitoring | `GET /stats` (request count, average and maximum latency) and one log line per request |
| Model | `models/model.joblib` is baked into the image. The data and MLflow are not needed at runtime. |

## 1. Run the container locally

```bash
docker build -t energy-api .
docker run -d --name energy-api -p 8000:8000 energy-api
docker ps                      # STATUS shows "(healthy)" after about 20 seconds
curl http://localhost:8000/health
# interactive docs: http://localhost:8000/docs
docker logs energy-api         # one line per request
docker stop energy-api && docker rm energy-api
```

## 2. First deployment on Render

1. Push the repository to GitHub (the `main` branch must contain the `Dockerfile` and `models/`).
2. In Render choose **New > Web Service** and connect the GitHub repository.
3. Settings:
   - **Language / Runtime:** Docker (Render finds the `Dockerfile` in the repo root)
   - **Branch:** `main`
   - **Instance type:** Free
   - **Advanced > Health Check Path:** `/health`
   - **Auto-Deploy:** On commit
   - Do not set `PORT` yourself. Render provides it and the container reads it.
4. Click **Create Web Service** and wait for the status **Live** (the first build takes a few minutes).
5. Verify (replace the URL with yours):
   ```bash
   export URL=https://<your-service>.onrender.com
   curl $URL/health
   curl $URL/model-info
   ```
6. Optional: `render.yaml` in the repo root describes the same service, for **New > Blueprint**.

## 3. Redeploy

- **Normal way:** merge a reviewed pull request into `main`. CI runs lint, tests and the Docker build. Render builds and deploys the new commit automatically. A new version only goes live if its `/health` check passes, otherwise Render keeps serving the previous version.
- **Manual:** open the service in Render and choose **Manual Deploy > Deploy latest commit**. Use **Clear build cache & deploy** if a dependency change was not picked up.
- **After redeploying, check:** `curl $URL/health` and `curl $URL/model-info` (compare `mlflow_run_id` and `trained_at_utc` with what you expect).

## 4. Roll back when a deployment breaks

**Option A: Render dashboard (fastest)**

1. Open the service, then the **Deploys** page.
2. Find the last deploy that worked and click **Rollback to this deploy**, then confirm.
3. Render redeploys that earlier build. Check `curl $URL/health` and `curl $URL/model-info`.
4. Turn off Auto-Deploy until the bad commit is fixed, otherwise the next push replaces the rollback.

**Option B: fix through Git (keeps history clean)**

```bash
git revert <bad-commit-sha>      # creates a new commit that undoes the change
git push origin main             # CI runs, then Render redeploys
```

**Option C: roll back to a previous Docker image tag (local machine or a VM)**

Tag every image with the commit it was built from, and keep the previous one.

```bash
docker build -t energy-api:$(git rev-parse --short HEAD) .     # build and tag the new version
docker images energy-api                                       # list tags, find the last good one
docker stop energy-api && docker rm energy-api
docker run -d --name energy-api -p 8000:8000 energy-api:<previous-tag>
docker ps                                                      # confirm "(healthy)"
```

CI builds the image as `energy-api:<commit sha>` on every push, so the tag for any commit is its SHA.

## 5. Monitoring during operation

```bash
curl $URL/stats
```

Returns the request count, error count, average and maximum latency (in milliseconds), and a breakdown per endpoint. Health checks, `/stats` and `/docs` are not counted, so Render's own health pings do not distort the numbers. The numbers are kept in memory and **reset whenever the service restarts or spins down**. Request lines are also in the Render **Logs** tab (or `docker logs` locally).

## 6. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Render deploy never goes **Live** | The app must listen on `$PORT` on `0.0.0.0` (the Dockerfile does this). Check the Health Check Path is `/health`. Read the deploy log. |
| First request takes a minute | Free instances spin down after 15 minutes without traffic and take about a minute to wake. Wake the service before a demo or recording, then wait for `/health` to answer. |
| `/stats` shows zeros | The service restarted or woke from sleep. Send some requests and check again. |
| `GET /health` shows `"model_loaded": false` or `/predict` returns 503 | `models/model.joblib` or `models/model_meta.json` is missing from the image. Check that `models/` is committed and not listed in `.dockerignore`. |
| `/explain` returns 503 | `models/background.json` is missing. Regenerate it with `python -m src.explain` (needs `data/processed/train.csv` from `python -m src.prepare_data`) and commit it. |
| `/explain` is slower than `/predict` | Expected. It evaluates the model on 25,600 rows for exact Shapley values (about 0.2 s locally, longer on a free instance). |
| CI `build` job fails but tests pass | Reproduce with `docker build -t energy-api .` locally and read the failing step. |

## 7. After grading

Free instances are not meant to run forever. Suspend the service in Render (**Settings > Suspend Service**) or delete it, and note in the README that the live endpoint was shut down.

