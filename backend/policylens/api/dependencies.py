from functools import lru_cache
from pathlib import Path

from policylens.core.config import get_settings
from policylens.repositories.faiss_vector_repository import FaissVectorRepository
from policylens.services.answer_generation import LangChainAnswerGenerationService
from policylens.services.document_ingestion import DocumentIngestionService
from policylens.services.document_indexing import DocumentIndexingService
from policylens.services.embedding_service import EmbeddingService
from policylens.services.llm_factory import create_chat_model
from policylens.services.rag_service import RagService
from policylens.services.reranking import CrossEncoderReranker
from policylens.services.reranking_evidence import RerankingEvidenceRecorder
from policylens.services.semantic_search import SemanticSearchService


PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_vector_repository() -> FaissVectorRepository:
    return FaissVectorRepository(index_directory=DATA_DIRECTORY / "indexes")


@lru_cache(maxsize=1)
def get_document_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService(data_directory=DATA_DIRECTORY)


@lru_cache(maxsize=1)
def get_document_indexing_service() -> DocumentIndexingService:
    return DocumentIndexingService(
        processed_directory=DATA_DIRECTORY / "processed",
        embedding_service=get_embedding_service(),
        vector_repository=get_vector_repository(),
    )


@lru_cache(maxsize=1)
def get_semantic_search_service() -> SemanticSearchService:
    return SemanticSearchService(
        embedding_service=get_embedding_service(),
        vector_repository=get_vector_repository(),
    )


@lru_cache(maxsize=1)
def get_answer_generation_service() -> LangChainAnswerGenerationService:
    settings = get_settings()
    return LangChainAnswerGenerationService(
        model_factory=lambda: create_chat_model(settings),
    )


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=get_settings().reranker_model)


@lru_cache(maxsize=1)
def get_reranking_evidence_recorder() -> RerankingEvidenceRecorder:
    return RerankingEvidenceRecorder(
        output_file=PROJECT_DIRECTORY / "evaluations" / "logs" / "rerank_comparisons.jsonl",
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(
        search_service=get_semantic_search_service(),
        answer_generation_service=get_answer_generation_service(),
        reranker=get_reranker() if settings.reranking_enabled else None,
        evidence_recorder=(
            get_reranking_evidence_recorder()
            if settings.reranking_enabled and settings.rerank_log_enabled
            else None
        ),
        candidate_count=settings.rerank_candidate_count,
    )