from typing import Literal

from pydantic import BaseModel, Field


class DocumentIndexResponse(BaseModel):
    status: Literal["indexed", "already_indexed"]
    document_id: str
    chunks_added: int = Field(ge=0)
    total_vectors: int = Field(ge=0)
    embedding_model: str