import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import SecretStr

from policylens.core.config import Settings
from policylens.schemas.chat import ChatSource
from policylens.schemas.search import SearchResult, SemanticSearchResponse
from policylens.services.answer_generation import (
    GeminiConfigurationError,
    LangChainAnswerGenerationService,
)
from policylens.services.llm_factory import create_chat_model
from policylens.services.rag_service import RagService


class FakeSearchService:
    def search(self, query: str, top_k: int) -> SemanticSearchResponse:
        return SemanticSearchResponse(
            query=query,
            total_indexed_chunks=33,
            results=[
                SearchResult(
                    chunk_id="document1-p8-c2",
                    document_id="document1",
                    source_filename="opengov_hr_manual.pdf",
                    page_number=8,
                    text="Employees should take 10 business days of paid vacation.",
                    similarity_score=0.88,
                )
            ][:top_k],
        )


def test_lcel_generates_answer_with_local_fake_model() -> None:
    fake_model = FakeListChatModel(
        responses=["The minimum is 10 business days. [1]"]
    )
    generator = LangChainAnswerGenerationService(
        model_factory=lambda: fake_model
    )
    source = ChatSource(
        reference=1,
        document_id="document1",
        chunk_id="document1-p8-c2",
        filename="opengov_hr_manual.pdf",
        page_number=8,
        excerpt="Employees should take 10 business days of paid vacation.",
        similarity_score=0.88,
    )

    answer = generator.generate_answer(
        "How many vacation days?", [source]
    )

    assert answer == "The minimum is 10 business days. [1]"


def test_lcel_pipeline_returns_answer_and_source_metadata() -> None:
    fake_model = FakeListChatModel(responses=["10 business days. [1]"])
    generator = LangChainAnswerGenerationService(
        model_factory=lambda: fake_model
    )
    service = RagService(
        search_service=FakeSearchService(),
        answer_generation_service=generator,
    )

    response = service.answer_question(
        "How many paid vacation days?", top_k=5
    )

    assert response.answer == "10 business days. [1]"
    assert response.retrieved_chunks == 1
    assert response.sources[0].filename == "opengov_hr_manual.pdf"
    assert response.sources[0].page_number == 8


def test_cloud_requests_are_blocked_without_explicit_opt_in() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        allow_cloud_llm=False,
        google_api_key=SecretStr("test-key-not-sent"),
    )

    with pytest.raises(GeminiConfigurationError, match="disabled"):
        create_chat_model(settings)