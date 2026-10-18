from typing import Annotated

from fastapi import (
    APIRouter,
    Request,
    Depends,
    File,
    HTTPException,
    Path as ApiPath,
    Response,
    UploadFile,
    status,
)

from policylens.repositories.faiss_vector_repository import (
    VectorStoreError,
)
from policylens.schemas.document import DocumentUploadResponse
from policylens.schemas.indexing import DocumentIndexResponse
from policylens.services.document_ingestion import (
    DocumentIngestionError,
    DocumentIngestionService,
)
from policylens.services.document_indexing import (
    DocumentIndexingService,
    InvalidProcessedDocumentError,
    ProcessedDocumentNotFoundError,
)


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


# Use the same repository instance for indexing and search so their writes and reads share a lock.
from policylens.api.dependencies import (
    get_document_ingestion_service,
    get_document_indexing_service,
)


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    response: Response,
    request: Request,
    file: Annotated[UploadFile, File()],
    ingestion_service: Annotated[
        DocumentIngestionService,
        Depends(get_document_ingestion_service),
    ],
) -> DocumentUploadResponse:
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF size cannot exceed 10 MB.",
        )

    try:
        result = ingestion_service.ingest(
            filename=file.filename or "",
            content=content,
        )

    except DocumentIngestionError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    if result.already_exists:
        response.status_code = status.HTTP_200_OK

    jobs = getattr(request.app.state, "indexing_jobs", None)
    if jobs is not None:
        jobs.enqueue(result.document.document_id)

    return DocumentUploadResponse(
        status=(
            "already_exists"
            if result.already_exists
            else "processed"
        ),
        document=result.document,
    )


@router.post(
    "/{document_id}/index",
    response_model=DocumentIndexResponse,
    status_code=status.HTTP_200_OK,
)
def index_document(
    document_id: Annotated[
        str,
        ApiPath(pattern=r"^[0-9a-f]{32}$"),
    ],
    indexing_service: Annotated[
        DocumentIndexingService,
        Depends(get_document_indexing_service),
    ],
) -> DocumentIndexResponse:
    try:
        return indexing_service.index_document(document_id)

    except ProcessedDocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    except InvalidProcessedDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error

    except VectorStoreError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error