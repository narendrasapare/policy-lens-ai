from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question cannot be empty.")
        return stripped


class ChatSource(BaseModel):
    reference: int = Field(ge=1)
    document_id: str
    chunk_id: str
    filename: str
    page_number: int = Field(ge=1)
    excerpt: str
    similarity_score: float
    rerank_score: float | None = None
    retrieved_for: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    retrieved_chunks: int = Field(ge=0)
    query_type: Literal["factual", "summary", "multi_part"] = "factual"
    sub_questions: list[str] = Field(default_factory=list)
    reranking_applied: bool = False