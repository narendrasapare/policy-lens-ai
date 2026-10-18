import json
from collections.abc import Iterator
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from policylens.api.routers.documents import (
    get_document_indexing_service,
    get_document_ingestion_service,
)
from policylens.main import app
from policylens.repositories.faiss_vector_repository import (
    FaissVectorRepository,
)
from policylens.services.document_indexing import (
    DocumentIndexingService,
)
from policylens.services.document_ingestion import (
    DocumentIngestionService,
)
from policylens.services.embedding_service import EmbeddingService


class FakeEmbeddingService(EmbeddingService):
    def __init__(self) -> None:
        super().__init__(model_name="test-embedding-model")
        self.calls = 0

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.calls += 1

        return np.asarray(
            [
                [
                    float(index + 1),
                    float(len(text) % 23 + 1),
                    1.0,
                ]
                for index, text in enumerate(texts)
            ],
            dtype=np.float32,
        )


@pytest.fixture
def indexing_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, FakeEmbeddingService, Path]]:
    embedding_service = FakeEmbeddingService()

    indexing_service = DocumentIndexingService(
        processed_directory=tmp_path / "processed",
        embedding_service=embedding_service,
        vector_repository=FaissVectorRepository(
            index_directory=tmp_path / "indexes",
        ),
    )

    app.dependency_overrides[get_document_ingestion_service] = (
        lambda: DocumentIngestionService(data_directory=tmp_path)
    )

    app.dependency_overrides[get_document_indexing_service] = (
        lambda: indexing_service
    )

    try:
        with TestClient(app) as client:
            yield client, embedding_service, tmp_path
    finally:
        app.dependency_overrides.clear()


def create_test_pdf(text: str) -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), text)

        return document.tobytes()


def upload_test_document(
    client: TestClient,
    filename: str,
    text: str,
) -> tuple[str, int]:
    response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                filename,
                create_test_pdf(text),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    document = response.json()["document"]

    return document["document_id"], document["chunk_count"]


def test_index_document_creates_persistent_faiss_files(
    indexing_client: tuple[
        TestClient,
        FakeEmbeddingService,
        Path,
    ],
) -> None:
    client, embedding_service, data_directory = indexing_client

    document_id, chunk_count = upload_test_document(
        client,
        "hr_policy.pdf",
        "Employees must follow the annual leave policy.",
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/index"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "indexed"
    assert result["document_id"] == document_id
    assert result["chunks_added"] == chunk_count
    assert result["total_vectors"] == chunk_count
    assert embedding_service.calls == 1

    index_path = data_directory / "indexes" / "policy_vectors.faiss"

    metadata_path = (
        data_directory / "indexes" / "chunk_metadata.json"
    )

    assert index_path.exists()
    assert metadata_path.exists()

    index = faiss.read_index(str(index_path))

    assert index.ntotal == chunk_count
    assert index.d == 3

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["embedding_model"] == "test-embedding-model"
    assert len(metadata["chunks"]) == chunk_count
    assert metadata["chunks"][0]["document_id"] == document_id
    assert metadata["chunks"][0]["source_filename"] == "hr_policy.pdf"
    assert metadata["chunks"][0]["page_number"] == 1


def test_indexing_same_document_twice_does_not_duplicate_vectors(
    indexing_client: tuple[
        TestClient,
        FakeEmbeddingService,
        Path,
    ],
) -> None:
    client, embedding_service, _ = indexing_client

    document_id, chunk_count = upload_test_document(
        client,
        "employee_manual.pdf",
        "Employees must follow workplace safety requirements.",
    )

    first_response = client.post(
        f"/api/v1/documents/{document_id}/index"
    )

    second_response = client.post(
        f"/api/v1/documents/{document_id}/index"
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    result = second_response.json()

    assert result["status"] == "already_indexed"
    assert result["chunks_added"] == 0
    assert result["total_vectors"] == chunk_count
    assert embedding_service.calls == 1


def test_indexing_multiple_documents_uses_shared_index(
    indexing_client: tuple[
        TestClient,
        FakeEmbeddingService,
        Path,
    ],
) -> None:
    client, _, _ = indexing_client

    first_id, first_chunk_count = upload_test_document(
        client,
        "leave_policy.pdf",
        "Employees can apply for annual leave.",
    )

    second_id, second_chunk_count = upload_test_document(
        client,
        "travel_policy.pdf",
        "Employees must obtain approval for business travel.",
    )

    first_response = client.post(
        f"/api/v1/documents/{first_id}/index"
    )

    second_response = client.post(
        f"/api/v1/documents/{second_id}/index"
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert second_response.json()["total_vectors"] == (
        first_chunk_count + second_chunk_count
    )


def test_indexing_unknown_document_returns_not_found(
    indexing_client: tuple[
        TestClient,
        FakeEmbeddingService,
        Path,
    ],
) -> None:
    client, _, _ = indexing_client

    response = client.post(
        f"/api/v1/documents/{'0' * 32}/index"
    )

    assert response.status_code == 404


def test_indexing_invalid_document_id_is_rejected(
    indexing_client: tuple[
        TestClient,
        FakeEmbeddingService,
        Path,
    ],
) -> None:
    client, _, _ = indexing_client

    response = client.post(
        "/api/v1/documents/invalid-id/index"
    )

    assert response.status_code == 422