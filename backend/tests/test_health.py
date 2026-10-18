from fastapi.testclient import TestClient

from policylens.main import app


def test_health_endpoint_returns_success() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "PolicyLens AI",
    }