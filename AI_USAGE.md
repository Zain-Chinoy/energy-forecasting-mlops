# AI Usage

This file lists the AI tools we used, what for, and how we checked the results. **Group members must read it and correct anything that does not match what they actually did before submitting.**

## Tools used

| Tool | Used for |
|---|---|
| Claude (Anthropic) | Reading the course document and proposal; choosing the dataset; drafting the code for data preparation, training with MLflow, the FastAPI service, `/explain`, `/stats`, the tests, the Dockerfile, the CI workflow and `RUNBOOK.md`; drafting this documentation, the slides and the presentation scripts; explaining concepts in simple words |
| Gemini agent in Antigravity | Running the git commands to push the project to our GitHub repository |

## How we checked the output

- Every pull request was reviewed by a teammate other than the author (see `CONTRIBUTION.md`).
- `pytest` (67 tests) and `ruff` run locally and in GitHub Actions on every push.
- Numbers quoted in the README (metrics, the 2020 demand drop of about 12%) were recomputed from the data, not taken from the AI's text. One earlier AI-written figure of 15% was wrong and was corrected.
- The `/explain` code is tested against a model whose Shapley values are known exactly, and contributions are tested to add up to the prediction.
- The AI environment could not build Docker images, so the first real Docker build was done by our CI and on our own machines.

## What we changed or decided ourselves

- Choosing the Panama dataset, the feature sets to compare, the Project Focus (MLflow experimentation and explainability) and Render as the platform.
- Running and deploying the service, recording the demo and writing our reflections.
- _TODO (each member): add specific examples where you changed, rejected or fixed AI output._

## Limits

AI-written code can look correct and still be wrong. We treated it like code from a classmate: read it, run it, test it, and only then merge it.
