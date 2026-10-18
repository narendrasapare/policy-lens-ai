from policylens.schemas.search import SearchResult, SemanticSearchResponse
from policylens.services.query_routing import QueryRouter, QueryType
from policylens.services.rag_service import RagService


class FakeSearchService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int) -> SemanticSearchResponse:
        self.queries.append(query)

        if "vacation" in query.lower():
            filename = "vacation.pdf"
            chunk_id = "vacation-p1-c1"
            text = "Vacation allowance is 10 business days."
        else:
            filename = "insurance.pdf"
            chunk_id = "insurance-p2-c1"
            text = "Employer pays 80 percent of the insurance premium."

        return SemanticSearchResponse(
            query=query,
            total_indexed_chunks=2,
            results=[
                SearchResult(
                    chunk_id=chunk_id,
                    document_id="document-1",
                    source_filename=filename,
                    page_number=1 if "vacation" in query.lower() else 2,
                    text=text,
                    similarity_score=0.85,
                )
            ][:top_k],
        )


class FakeAnswerGenerator:
    def generate_answer(self, question: str, sources: list) -> str:
        return "Vacation is 10 business days [1]. Insurance coverage is 80% [2]."


def test_factual_query_is_not_over_split() -> None:
    result = QueryRouter().route("What are benefits and compensation?")

    assert result.query_type == QueryType.FACTUAL
    assert result.sub_questions == ("What are benefits and compensation?",)


def test_summary_question_is_detected() -> None:
    result = QueryRouter().route("Summarize the vacation policy.")

    assert result.query_type == QueryType.SUMMARY
    assert len(result.sub_questions) == 1


def test_explicit_multi_part_query_is_decomposed() -> None:
    question = (
        "How many vacation days do employees get, and "
        "what percentage of health insurance does the company pay?"
    )

    result = QueryRouter().route(question)

    assert result.query_type == QueryType.MULTI_PART
    assert len(result.sub_questions) == 2
    assert "vacation" in result.sub_questions[0]
    assert "insurance" in result.sub_questions[1]


def test_comparison_question_is_decomposed() -> None:
    result = QueryRouter().route(
        "Compare vacation policy with insurance policy"
    )

    assert result.query_type == QueryType.MULTI_PART
    assert len(result.sub_questions) == 2


def test_rag_searches_both_questions_and_retains_evidence() -> None:
    search_service = FakeSearchService()

    service = RagService(
        search_service=search_service,
        answer_generation_service=FakeAnswerGenerator(),
    )

    response = service.answer_question(
        question=(
            "How many vacation days do employees get, and "
            "what percentage of health insurance does the company pay?"
        ),
        top_k=5,
    )

    assert response.query_type == "multi_part"
    assert len(search_service.queries) == 2
    assert len(response.sub_questions) == 2
    assert response.retrieved_chunks == 2
    assert [source.reference for source in response.sources] == [1, 2]
    assert {source.filename for source in response.sources} == {
        "vacation.pdf",
        "insurance.pdf",
    }
    assert response.sources[0].retrieved_for is not None