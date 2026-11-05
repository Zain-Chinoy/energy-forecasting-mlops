import pytest
from fastapi.testclient import TestClient

from src.app import EXAMPLE_REQUEST, app


@pytest.fixture
def payload() -> dict:
    return dict(EXAMPLE_REQUEST)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # runs the lifespan, so the model is loaded
        yield test_client
