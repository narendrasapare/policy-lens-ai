from pydantic import BaseModel, Field, field_validator


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("Search query cannot be empty.")
        return query


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int
    text: str
    similarity_score: float


class SemanticSearchResponse(BaseModel):
    query: str
    total_indexed_chunks: int
    results: list[SearchResult]