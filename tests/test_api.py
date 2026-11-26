import math

import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_request(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert math.isfinite(body["predicted_demand_mwh"])
    assert body["predicted_demand_mwh"] > 0
    assert body["unit"] == "MWh"
    assert {"model_type", "feature_set", "mlflow_run_id"} <= body.keys()


def test_predict_is_deterministic(client, payload):
    first = client.post("/predict", json=payload).json()
    second = client.post("/predict", json=payload).json()
    assert first["predicted_demand_mwh"] == second["predicted_demand_mwh"]


def test_predict_accepts_timezone_suffix(client, payload):
    response = client.post("/predict", json={**payload, "timestamp": "2019-10-15T14:00:00-05:00"})
    assert response.status_code == 200


def test_missing_field_is_rejected(client, payload):
    del payload["T2M_toc"]
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("T2M_toc", 100.0),  # impossible temperature
        ("QV2M_san", -0.1),  # negative humidity
        ("TQL_dav", 50.0),  # absurd rainfall
        ("W2M_toc", -3.0),  # negative wind
        ("holiday", 2),  # not a binary flag
        ("school", -1),
        ("T2M_dav", "hot"),  # wrong type
        ("timestamp", "not-a-date"),
    ],
)
def test_invalid_values_are_rejected(client, payload, field, value):
    payload[field] = value
    assert client.post("/predict", json=payload).status_code == 422


def test_unexpected_field_is_rejected(client, payload):
    payload["surprise"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["features"]
    assert "mlflow_run_id" in body


def test_predict_returns_503_when_model_missing(client, payload):
    saved = client.app.state.bundle
    client.app.state.bundle = None
    try:
        assert client.post("/predict", json=payload).status_code == 503
        assert client.get("/health").json()["model_loaded"] is False
    finally:
        client.app.state.bundle = saved


def test_docs_and_openapi_are_served(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]


# ---------------------------------------------------------------- monitoring (/stats)
def get_stats(client) -> dict:
    return client.get("/stats").json()


def endpoint_count(stats: dict, endpoint: str) -> int:
    return stats["by_endpoint"].get(endpoint, {}).get("count", 0)


def test_stats_count_rises_with_predict_requests(client, payload):
    before = get_stats(client)
    for _ in range(3):
        assert client.post("/predict", json=payload).status_code == 200
    after = get_stats(client)
    assert endpoint_count(after, "/predict") == endpoint_count(before, "/predict") + 3
    assert after["total_requests"] == before["total_requests"] + 3
    assert after["avg_latency_ms"] > 0
    assert after["max_latency_ms"] >= after["avg_latency_ms"]


def test_health_checks_and_stats_calls_are_not_counted(client):
    before = get_stats(client)
    for _ in range(5):
        client.get("/health")
        client.get("/stats")
    assert get_stats(client)["total_requests"] == before["total_requests"]


def test_invalid_requests_are_counted_as_errors(client, payload):
    before = get_stats(client)
    payload["T2M_toc"] = 100.0
    assert client.post("/predict", json=payload).status_code == 422
    after = get_stats(client)
    assert after["error_requests"] == before["error_requests"] + 1


def test_unknown_paths_are_grouped_as_unmatched(client):
    before = endpoint_count(get_stats(client), "unmatched")
    assert client.get("/no-such-page").status_code == 404
    assert endpoint_count(get_stats(client), "unmatched") == before + 1


# ------------------------------------------------------------------ explanation (/explain)
def test_explain_returns_contributions_that_add_up(client, payload):
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "shapley_exact"
    total = sum(c["contribution_mwh"] for c in body["contributions"])
    assert body["baseline_mwh"] + total == pytest.approx(body["predicted_demand_mwh"], abs=0.1)


def test_explain_prediction_matches_predict(client, payload):
    predicted = client.post("/predict", json=payload).json()["predicted_demand_mwh"]
    explained = client.post("/explain", json=payload).json()["predicted_demand_mwh"]
    assert explained == pytest.approx(predicted, abs=0.01)


def test_explain_covers_exactly_the_model_features(client, payload):
    features = client.get("/model-info").json()["features"]
    contributions = client.post("/explain", json=payload).json()["contributions"]
    assert sorted(c["feature"] for c in contributions) == sorted(features)


def test_hour_of_day_explains_why_afternoon_is_high_and_night_is_low(client, payload):
    def hour_effect(timestamp: str) -> float:
        body = client.post("/explain", json={**payload, "timestamp": timestamp}).json()
        return next(c["contribution_mwh"] for c in body["contributions"] if c["feature"] == "hour")

    assert hour_effect("2019-10-15T14:00:00") > 0
    assert hour_effect("2019-10-15T03:00:00") < 0


def test_explain_rejects_invalid_input(client, payload):
    payload["W2M_toc"] = -1.0
    assert client.post("/explain", json=payload).status_code == 422


def test_explain_returns_503_without_model_or_background(client, payload):
    bundle = client.app.state.bundle
    saved_background = bundle.background
    try:
        bundle.background = None
        assert client.post("/explain", json=payload).status_code == 503
        client.app.state.bundle = None
        assert client.post("/explain", json=payload).status_code == 503
    finally:
        bundle.background = saved_background
        client.app.state.bundle = bundle
