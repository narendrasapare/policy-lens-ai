import json

import pytest
from fastapi.testclient import TestClient

from policylens.api import dependencies
from policylens.api.routers import workspace
from policylens.main import app


class FakeRepository:
    def get_document_status(self, document_id):
        return True, 2

    def get_vector_count(self):
        return 2


@pytest.fixture
def workspace_client(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "DATA_DIRECTORY", tmp_path)
    monkeypatch.setattr(dependencies, "get_vector_repository", lambda: FakeRepository())
    with TestClient(app) as client:
        yield client, tmp_path


def test_empty_workspace(workspace_client):
    client, _ = workspace_client
    response = client.get("/api/v1/workspace/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": [], "total_vectors": 2}


def test_document_catalog_exposes_summary_and_index_state(workspace_client):
    client, directory = workspace_client
    processed = directory / "processed"
    processed.mkdir()
    document_id = "a" * 32
    summary = {"document_id": document_id, "filename": "policy.pdf", "page_count": 1,
               "character_count": 100, "chunk_count": 2}
    (processed / f"{document_id}.json").write_text(json.dumps({"document": summary}))
    response = client.get("/api/v1/workspace/documents")
    assert response.status_code == 200
    assert response.json()["documents"] == [{**summary, "indexed": True, "preparation_status": "ready", "preparation_error": None}]


def test_invalid_metadata_is_actionable(workspace_client):
    client, directory = workspace_client
    processed = directory / "processed"
    processed.mkdir()
    (processed / f"{'b' * 32}.json").write_text("invalid json")
    response = client.get("/api/v1/workspace/documents")
    assert response.status_code == 409
    assert "metadata" in response.json()["detail"]


def test_cors_rejects_unknown_origin(workspace_client):
    client, _ = workspace_client
    response = client.options("/api/v1/chat", headers={"Origin": "https://untrusted.example",
                                                      "Access-Control-Request-Method": "POST"})
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_health_route_remains_available(workspace_client):
    client, _ = workspace_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
