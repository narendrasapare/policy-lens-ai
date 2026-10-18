from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import FileResponse

from policylens.api.dependencies import DATA_DIRECTORY

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.get("/{document_id}/file", response_class=FileResponse)
def view_document(
    document_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
) -> FileResponse:
    pdf_path = DATA_DIRECTORY / "uploads" / f"{document_id}.pdf"

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document PDF not found.",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{document_id}.pdf",
        content_disposition_type="inline",
    )
