from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from policylens.api.dependencies import get_rag_service
from policylens.main import app
from policylens.schemas.chat import ChatResponse
from policylens.schemas.search import SearchResult, SemanticSearchResponse
from policylens.services.answer_generation import (
    AnswerGenerationError,
    GeminiConfigurationError,
)
from policylens.services.rag_service import RagService
from policylens.services.semantic_search import NoIndexedDocumentsError


class FakeSearchService:
    def search(
        self,
        query: str,
        top_k: int,
    ) -> SemanticSearchResponse:
        return SemanticSearchResponse(
            query=query,
            total_indexed_chunks=33,
            results=[
                SearchResult(
                    chunk_id="doc123-p8-c2",
                    document_id="doc123",
                    source_filename="opengov_hr_manual.pdf",
                    page_number=8,
                    text="Employees should take 10 business days of paid vacation.",
                    similarity_score=0.91,
                )
            ][:top_k],
        )


class FakeAnswerGenerationService:
    def generate_answer(
        self,
        question: str,
        sources: list,
    ) -> str:
        assert question == "How much paid vacation?"
        assert sources[0].page_number == 8

        return "Employees should take 10 business days of paid vacation. [1]"


@pytest.fixture
def client() -> Iterator[TestClient]:
    rag_service = RagService(
        search_service=FakeSearchService(),
        answer_generation_service=FakeAnswerGenerationService(),
    )

    app.dependency_overrides[get_rag_service] = lambda: rag_service

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_chat_returns_grounded_answer_and_sources(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "  How much paid vacation?  ",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"].endswith("[1]")
    assert body["retrieved_chunks"] == 1
    assert body["sources"][0]["filename"] == "opengov_hr_manual.pdf"
    assert body["sources"][0]["page_number"] == 8
    assert body["sources"][0]["reference"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "Question", "top_k": 0},
        {"question": "Question", "top_k": 11},
    ],
)
def test_chat_rejects_invalid_requests(
    client: TestClient,
    payload: dict,
) -> None:
    response = client.post(
        "/api/v1/chat",
        json=payload,
    )

    assert response.status_code == 422


def test_chat_requires_indexed_documents(
    client: TestClient,
) -> None:
    class NoIndexService:
        def answer_question(
            self,
            question: str,
            top_k: int,
        ) -> ChatResponse:
            raise NoIndexedDocumentsError(
                "No indexed documents found."
            )

    app.dependency_overrides[get_rag_service] = (
        lambda: NoIndexService()
    )

    response = client.post(
        "/api/v1/chat",
        json={"question": "Where is the policy?"},
    )

    assert response.status_code == 409


def test_chat_handles_missing_api_key(
    client: TestClient,
) -> None:
    class UnconfiguredService:
        def answer_question(
            self,
            question: str,
            top_k: int,
        ) -> ChatResponse:
            raise GeminiConfigurationError(
                "Google Gemini API key is not configured."
            )

    app.dependency_overrides[get_rag_service] = (
        lambda: UnconfiguredService()
    )

    response = client.post(
        "/api/v1/chat",
        json={"question": "Where is the policy?"},
    )

    assert response.status_code == 503


def test_chat_handles_llm_failure(
    client: TestClient,
) -> None:
    class UnavailableService:
        def answer_question(
            self,
            question: str,
            top_k: int,
        ) -> ChatResponse:
            raise AnswerGenerationError(
                "Gemini could not generate an answer right now."
            )

    app.dependency_overrides[get_rag_service] = (
        lambda: UnavailableService()
    )

    response = client.post(
        "/api/v1/chat",
        json={"question": "Where is the policy?"},
    )

    assert response.status_code == 502