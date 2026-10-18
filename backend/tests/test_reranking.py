import json
from pathlib import Path

import numpy as np
import pytest

from policylens.schemas.search import SearchResult, SemanticSearchResponse
from policylens.services.rag_service import RagService
from policylens.services.reranking import CrossEncoderReranker, RerankingError
from policylens.services.reranking_evidence import RerankingEvidenceRecorder


def make_result(chunk_id: str, text: str, similarity: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="document1",
        source_filename="opengov_hr_manual.pdf",
        page_number=8,
        text=text,
        similarity_score=similarity,
    )


class FakePairScorer:
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        assert all(question for question, _ in pairs)
        return np.asarray(
            [8.0 if "correct evidence" in text else -3.0 for _, text in pairs],
            dtype=np.float32,
        )


def make_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_name="fake-cross-encoder",
        model_factory=FakePairScorer,
    )


def test_reranker_changes_order_using_pair_scores() -> None:
    candidates = [
        make_result("irrelevant", "Unrelated policy text", 0.92),
        make_result("relevant", "This is correct evidence about vacation", 0.72),
    ]

    outcome = make_reranker().rerank("What is vacation?", candidates, top_k=1)

    assert outcome.before[0].chunk_id == "irrelevant"
    assert outcome.after[0].chunk_id == "relevant"
    assert outcome.ranked_results[0].result.chunk_id == "relevant"
    assert outcome.ranked_results[0].original_rank == 2
    assert outcome.top_result_changed is True
    assert outcome.after[0].rerank_score > outcome.after[1].rerank_score


def test_rerank_evidence_is_valid_jsonl(tmp_path: Path) -> None:
    candidates = [
        make_result("irrelevant", "Other policy", 0.9),
        make_result("relevant", "correct evidence about annual leave", 0.7),
    ]
    outcome = make_reranker().rerank("Annual leave?", candidates, top_k=2)
    evidence_file = tmp_path / "logs" / "rerank_comparisons.jsonl"
    recorder = RerankingEvidenceRecorder(output_file=evidence_file)

    recorder.record(outcome)
    recorder.record(outcome)

    records = [
        json.loads(line)
        for line in evidence_file.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 2
    assert records[0]["top_result_changed"] is True
    assert records[0]["before"][0]["chunk_id"] == "irrelevant"
    assert records[0]["after"][0]["chunk_id"] == "relevant"
    assert records[0]["candidate_count"] == 2
    assert records[0]["timestamp_utc"]


def test_empty_results_do_not_load_cross_encoder() -> None:
    def unexpected_model_load() -> FakePairScorer:
        raise AssertionError("Model should not load for empty candidates")

    reranker = CrossEncoderReranker(model_factory=unexpected_model_load)
    outcome = reranker.rerank("Any query?", [], top_k=5)

    assert outcome.ranked_results == []
    assert outcome.top_result_changed is False


def test_invalid_model_scores_are_rejected() -> None:
    class BadScorer:
        def predict(
            self,
            pairs: list[tuple[str, str]],
            **kwargs: object,
        ) -> np.ndarray:
            return np.asarray([float("nan")], dtype=np.float32)

    reranker = CrossEncoderReranker(model_factory=BadScorer)

    with pytest.raises(RerankingError, match="invalid scores"):
        reranker.rerank(
            "Vacation?",
            [make_result("a", "Text", 0.5)],
            top_k=5,
        )


class FakeSearchService:
    def search(self, query: str, top_k: int) -> SemanticSearchResponse:
        topic = "vacation" if "vacation" in query.lower() else "insurance"
        results = [
            make_result(
                f"{topic}-unrelated",
                f"Less relevant information about {topic}",
                0.9,
            ),
            make_result(
                f"{topic}-relevant",
                f"This is correct evidence about {topic}",
                0.7,
            ),
        ]
        return SemanticSearchResponse(
            query=query,
            total_indexed_chunks=4,
            results=results[:top_k],
        )


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.received_sources = []

    def generate_answer(self, question: str, sources: list) -> str:
        self.received_sources = sources
        return "Answer supported by the supplied source passages. [1]"


def test_rag_uses_reranked_passages_before_generation(
    tmp_path: Path,
) -> None:
    generator = FakeAnswerGenerator()
    recorder = RerankingEvidenceRecorder(tmp_path / "evidence.jsonl")
    service = RagService(
        search_service=FakeSearchService(),
        answer_generation_service=generator,
        reranker=make_reranker(),
        evidence_recorder=recorder,
        candidate_count=15,
    )

    response = service.answer_question("How much vacation?", top_k=1)

    assert response.reranking_applied is True
    assert response.retrieved_chunks == 1
    assert generator.received_sources[0].chunk_id == "vacation-relevant"
    assert response.sources[0].rerank_score == pytest.approx(8.0)
    assert len(
        recorder.output_file.read_text(encoding="utf-8").splitlines()
    ) == 1


def test_multi_part_queries_rerank_each_part(tmp_path: Path) -> None:
    generator = FakeAnswerGenerator()
    recorder = RerankingEvidenceRecorder(tmp_path / "comparison.jsonl")
    service = RagService(
        search_service=FakeSearchService(),
        answer_generation_service=generator,
        reranker=make_reranker(),
        evidence_recorder=recorder,
    )

    response = service.answer_question(
        "How much vacation do employees get, and what is the insurance policy?",
        top_k=2,
    )

    assert response.query_type == "multi_part"
    assert response.reranking_applied is True
    assert response.retrieved_chunks == 2
    assert {source.chunk_id for source in response.sources} == {
        "vacation-relevant",
        "insurance-relevant",
    }
    assert len(
        recorder.output_file.read_text(encoding="utf-8").splitlines()
    ) == 2