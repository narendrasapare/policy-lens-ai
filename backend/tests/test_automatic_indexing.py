from threading import Event

import pymupdf
from fastapi.testclient import TestClient

from policylens.api.dependencies import get_rag_service
from policylens.api.routers import documents, workspace
from policylens.main import app
from policylens.services.document_ingestion import DocumentIngestionService
from policylens.services.indexing_jobs import IndexingJobs


def pdf_bytes():
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Employees receive twenty days of annual leave.")
        return pdf.tobytes()


def test_upload_queues_without_waiting_and_blocks_chat_until_ready(tmp_path, monkeypatch):
    started, release = Event(), Event()
    indexed = set()

    class Indexer:
        def index_document(self, document_id):
            started.set()
            assert release.wait(5)
            indexed.add(document_id)

    class Repository:
        def get_document_status(self, document_id):
            return document_id in indexed, len(indexed)

        def get_vector_count(self):
            return len(indexed)

    from policylens.api import dependencies
    monkeypatch.setattr(workspace, "DATA_DIRECTORY", tmp_path)
    monkeypatch.setattr(dependencies, "get_vector_repository", lambda: Repository())
    app.dependency_overrides[documents.get_document_ingestion_service] = lambda: DocumentIngestionService(tmp_path)
    app.dependency_overrides[get_rag_service] = lambda: object()
    try:
        with TestClient(app) as client:
            app.state.indexing_jobs.shutdown()
            jobs = IndexingJobs(lambda: Indexer())
            app.state.indexing_jobs = jobs
            try:
                response = client.post("/api/v1/documents", files={"file": ("leave.pdf", pdf_bytes(), "application/pdf")})
                assert response.status_code == 201
                assert started.wait(2)
                document_id = response.json()["document"]["document_id"]
                catalog = client.get("/api/v1/workspace/documents").json()
                assert catalog["documents"][0]["preparation_status"] == "indexing"
                response = client.post("/api/v1/chat", json={"question": "How much leave?", "top_k": 5})
                assert response.status_code == 409
                assert "prepared" in response.json()["detail"]
            finally:
                release.set()
                jobs.shutdown()
            assert client.get("/api/v1/workspace/documents").json()["documents"][0]["indexed"]
            assert document_id in indexed
    finally:
        app.dependency_overrides.clear()


def test_failed_job_can_be_retried_without_duplicate_queued_work():
    started, release = Event(), Event()
    calls = []

    class Indexer:
        def index_document(self, document_id):
            calls.append(document_id)
            started.set()
            assert release.wait(5)
            if len(calls) == 1:
                raise RuntimeError("Embedding model unavailable")

    jobs = IndexingJobs(lambda: Indexer())
    jobs.enqueue("a" * 32)
    assert started.wait(2)
    jobs.enqueue("a" * 32)
    release.set()
    # A barrier on the single worker waits for the first task without timing assumptions.
    jobs._executor.submit(lambda: None).result(timeout=5)
    assert jobs.status("a" * 32)[0] == "failed"
    jobs.enqueue("a" * 32)
    jobs.shutdown()
    assert jobs.status("a" * 32) == ("ready", None)
    assert len(calls) == 2


def test_startup_recovers_unfinished_documents(tmp_path, monkeypatch):
    import policylens.main as application

    pending_id, ready_id = "b" * 32, "c" * 32
    processed = tmp_path / "processed"
    processed.mkdir()
    for document_id in (pending_id, ready_id):
        (processed / f"{document_id}.json").write_text("{}")
    recovered = Event()
    calls = []

    class Repository:
        def get_document_status(self, document_id):
            return document_id == ready_id, 1

    class Indexer:
        def index_document(self, document_id):
            calls.append(document_id)
            recovered.set()

    monkeypatch.setattr(application, "DATA_DIRECTORY", tmp_path)
    monkeypatch.setattr(application, "get_vector_repository", lambda: Repository())
    monkeypatch.setattr(application, "get_document_indexing_service", lambda: Indexer())
    with TestClient(app):
        assert recovered.wait(2)
    assert calls == [pending_id]
