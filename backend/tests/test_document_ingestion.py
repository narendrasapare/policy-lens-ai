import json
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from policylens.api.routers.documents import (
    get_document_ingestion_service,
)
from policylens.main import app
from policylens.services.document_ingestion import (
    DocumentIngestionService,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app.dependency_overrides[get_document_ingestion_service] = (
        lambda: DocumentIngestionService(data_directory=tmp_path)
    )

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def create_test_pdf(page_texts: list[str]) -> bytes:
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()

            if text:
                page.insert_text((72, 72), text)

        return document.tobytes()


def test_upload_pdf_preserves_page_metadata(
    client: TestClient,
    tmp_path: Path,
) -> None:
    pdf_content = create_test_pdf(
        [
            "Employees receive annual leave benefits.",
            "Employees must follow workplace safety policies.",
        ]
    )

    response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "hr_manual.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    result = response.json()

    assert result["status"] == "processed"
    assert result["document"]["filename"] == "hr_manual.pdf"
    assert result["document"]["page_count"] == 2
    assert result["document"]["chunk_count"] >= 2

    document_id = result["document"]["document_id"]

    pdf_path = tmp_path / "uploads" / f"{document_id}.pdf"
    metadata_path = (
        tmp_path / "processed" / f"{document_id}.json"
    )

    assert pdf_path.exists()
    assert pdf_path.read_bytes() == pdf_content
    assert metadata_path.exists()

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["document"]["document_id"] == document_id

    page_numbers = {
        chunk["page_number"] for chunk in metadata["chunks"]
    }

    assert page_numbers == {1, 2}


def test_upload_rejects_invalid_pdf(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "invalid.pdf",
                b"This is not a PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422


def test_upload_rejects_pdf_without_text(
    client: TestClient,
) -> None:
    pdf_content = create_test_pdf([""])

    response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "empty.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422


def test_upload_rejects_non_pdf_extension(
    client: TestClient,
) -> None:
    pdf_content = create_test_pdf(["Company policy"])

    response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "document.txt",
                pdf_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 422

def test_duplicate_upload_reuses_existing_document(
    client: TestClient,
    tmp_path: Path,
) -> None:
    pdf_content = create_test_pdf(
        ["Employees must follow workplace safety policies."]
    )

    first_response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "hr_manual.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    second_response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "hr_manual.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    renamed_response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "renamed_manual.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert first_response.status_code == 201
    assert first_response.json()["status"] == "processed"

    assert second_response.status_code == 200
    assert second_response.json()["status"] == "already_exists"

    assert renamed_response.status_code == 200
    assert renamed_response.json()["status"] == "already_exists"

    first_document = first_response.json()["document"]

    assert second_response.json()["document"] == first_document
    assert renamed_response.json()["document"] == first_document

    assert len(list((tmp_path / "uploads").glob("*.pdf"))) == 1
    assert len(list((tmp_path / "processed").glob("*.json"))) == 1