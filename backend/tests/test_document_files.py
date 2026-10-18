from pathlib import Path

from fastapi.testclient import TestClient

from policylens.api.routers import document_files
from policylens.main import app


DOCUMENT_ID = "a" * 32


def test_document_file_opens_inline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(document_files, "DATA_DIRECTORY", tmp_path)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    pdf_bytes = b"%PDF-1.4\nexample local PDF for routing test\n%%EOF"
    (uploads / f"{DOCUMENT_ID}.pdf").write_bytes(pdf_bytes)

    with TestClient(app) as client:
        response = client.get(f"/api/v1/documents/{DOCUMENT_ID}/file")

    assert response.status_code == 200
    assert response.content == pdf_bytes
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")


def test_document_file_not_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(document_files, "DATA_DIRECTORY", tmp_path)
    with TestClient(app) as client:
        response = client.get(f"/api/v1/documents/{DOCUMENT_ID}/file")
    assert response.status_code == 404


def test_document_file_rejects_invalid_id() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/documents/not-a-document-id/file")
    assert response.status_code == 422
