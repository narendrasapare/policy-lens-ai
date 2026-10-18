from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from policylens.api.routers.documents import (
    get_document_indexing_service,
    get_document_ingestion_service,
)
from policylens.api.routers.search import get_semantic_search_service
from policylens.main import app
from policylens.repositories.faiss_vector_repository import FaissVectorRepository
from policylens.services.document_indexing import DocumentIndexingService
from policylens.services.document_ingestion import DocumentIngestionService
from policylens.services.embedding_service import EmbeddingService
from policylens.services.semantic_search import SemanticSearchService


class FakeEmbeddingService(EmbeddingService):
    def __init__(self) -> None:
        super().__init__(model_name="fake-keyword-embeddings")

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lower = text.lower()
        if "vacation" in lower or "leave" in lower:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "expense" in lower or "reimbursement" in lower:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return self._vector(query)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    embedding_service = FakeEmbeddingService()
    repository = FaissVectorRepository(tmp_path / "indexes")

    ingestion_service = DocumentIngestionService(data_directory=tmp_path)
    indexing_service = DocumentIndexingService(
        processed_directory=tmp_path / "processed",
        embedding_service=embedding_service,
        vector_repository=repository,
    )
    search_service = SemanticSearchService(
        embedding_service=embedding_service,
        vector_repository=repository,
    )

    app.dependency_overrides[get_document_ingestion_service] = (
        lambda: ingestion_service
    )
    app.dependency_overrides[get_document_indexing_service] = (
        lambda: indexing_service
    )
    app.dependency_overrides[get_semantic_search_service] = (
        lambda: search_service
    )

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def create_pdf(text: str) -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return document.tobytes()


def upload_and_index(client: TestClient, filename: str, text: str) -> None:
    upload_response = client.post(
        "/api/v1/documents",
        files={"file": (filename, create_pdf(text), "application/pdf")},
    )
    assert upload_response.status_code == 201

    document_id = upload_response.json()["document"]["document_id"]
    indexing_response = client.post(f"/api/v1/documents/{document_id}/index")
    assert indexing_response.status_code == 200


def test_search_requires_an_index(client: TestClient) -> None:
    response = client.post(
        "/api/v1/search",
        json={"query": "annual vacation", "top_k": 5},
    )

    assert response.status_code == 409


def test_semantic_search_returns_ranked_source_passages(
    client: TestClient,
) -> None:
    upload_and_index(
        client,
        "leave_policy.pdf",
        "Employees receive paid vacation under the annual leave policy.",
    )
    upload_and_index(
        client,
        "expenses.pdf",
        "Business expense reimbursement requires prior approval.",
    )

    response = client.post(
        "/api/v1/search",
        json={"query": "  annual vacation  ", "top_k": 5},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "annual vacation"
    assert body["total_indexed_chunks"] == 2
    assert len(body["results"]) == 2
    assert body["results"][0]["source_filename"] == "leave_policy.pdf"
    assert body["results"][0]["page_number"] == 1
    assert "vacation" in body["results"][0]["text"].lower()
    assert body["results"][0]["similarity_score"] == pytest.approx(1.0)
    assert body["results"][0]["similarity_score"] >= body["results"][1]["similarity_score"]


def test_top_k_limits_results(client: TestClient) -> None:
    upload_and_index(client, "leave.pdf", "Employees receive annual leave.")
    upload_and_index(client, "expense.pdf", "Expense reimbursement is approved.")

    response = client.post(
        "/api/v1/search",
        json={"query": "annual leave", "top_k": 1},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


@pytest.mark.parametrize(
    "request_body",
    [
        {"query": "   ", "top_k": 5},
        {"query": "vacation", "top_k": 0},
        {"query": "vacation", "top_k": 21},
    ],
)
def test_search_rejects_invalid_requests(
    client: TestClient,
    request_body: dict[str, str | int],
) -> None:
    response = client.post("/api/v1/search", json=request_body)
    assert response.status_code == 422