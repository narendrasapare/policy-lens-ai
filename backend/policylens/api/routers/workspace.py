import json
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from policylens.api.dependencies import DATA_DIRECTORY
from policylens.repositories.faiss_vector_repository import VectorStoreError
from policylens.schemas.document import DocumentSummary

router = APIRouter(prefix="/api/v1/workspace", tags=["Workspace"])


class WorkspaceDocument(DocumentSummary):
    indexed: bool
    preparation_status: str = "ready"
    preparation_error: str | None = None


class WorkspaceResponse(BaseModel):
    documents: list[WorkspaceDocument]
    total_vectors: int


def list_workspace_documents(request: Request) -> WorkspaceResponse:
    from policylens.api.dependencies import get_vector_repository

    repository = get_vector_repository()
    documents = []
    try:
        for path in sorted((DATA_DIRECTORY / "processed").glob("*.json")):
            if not re.fullmatch(r"[0-9a-f]{32}", path.stem):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = DocumentSummary.model_validate(payload["document"])
            if summary.document_id != path.stem:
                raise ValueError("Document ID mismatch")
            indexed, _ = repository.get_document_status(summary.document_id)
            jobs = getattr(request.app.state, "indexing_jobs", None)
            state, error = ("ready", None) if indexed else (jobs.status(summary.document_id) if jobs else ("queued", None))
            documents.append(WorkspaceDocument(
                **summary.model_dump(), indexed=indexed,
                preparation_status=state, preparation_error=error,
            ))
        return WorkspaceResponse(documents=documents, total_vectors=repository.get_vector_count())
    except VectorStoreError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=409, detail="Stored document metadata is unreadable or inconsistent.") from error


router.add_api_route("/documents", list_workspace_documents, methods=["GET"], response_model=WorkspaceResponse)


@router.post("/documents/{document_id}/retry", status_code=202)
def retry_preparation(document_id: str, request: Request):
    if not re.fullmatch(r"[0-9a-f]{32}", document_id) or not (DATA_DIRECTORY / "processed" / f"{document_id}.json").is_file():
        raise HTTPException(status_code=404, detail="Document not found.")
    request.app.state.indexing_jobs.enqueue(document_id)
    return {"status": "queued"}
