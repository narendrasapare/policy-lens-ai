from typing import Literal

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)
    text: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    page_count: int
    character_count: int
    chunk_count: int


class DocumentUploadResponse(BaseModel):
    status: Literal["processed", "already_exists"]
    document: DocumentSummary