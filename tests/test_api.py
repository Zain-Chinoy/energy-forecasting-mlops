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
