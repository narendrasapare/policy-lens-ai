from policylens.repositories.faiss_vector_repository import FaissVectorRepository
from policylens.schemas.search import SemanticSearchResponse, SearchResult
from policylens.services.embedding_service import EmbeddingService


class NoIndexedDocumentsError(Exception):
    pass


class SemanticSearchService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: FaissVectorRepository,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_repository = vector_repository

    def summary_context(self) -> list[SearchResult]:
        chunks = self._vector_repository.get_chunks()
        if not chunks:
            raise NoIndexedDocumentsError("Upload a PDF and wait for preparation before asking questions.")
        # Zero is a placeholder: these passages were read in document order, not ranked.
        return [SearchResult(**chunk.model_dump(), similarity_score=0.0) for chunk in chunks]

    def search(self, query: str, top_k: int) -> SemanticSearchResponse:
        if self._vector_repository.get_vector_count() == 0:
            raise NoIndexedDocumentsError(
                "No indexed documents found. Index a PDF before searching."
            )

        query_vector = self._embedding_service.encode_query(query)
        search_result = self._vector_repository.search(query_vector, top_k)

        return SemanticSearchResponse(
            query=query,
            total_indexed_chunks=search_result.total_vectors,
            results=[
                SearchResult(
                    chunk_id=hit.chunk.chunk_id,
                    document_id=hit.chunk.document_id,
                    source_filename=hit.chunk.source_filename,
                    page_number=hit.chunk.page_number,
                    text=hit.chunk.text,
                    similarity_score=hit.similarity_score,
                )
                for hit in search_result.hits
            ],
        )
