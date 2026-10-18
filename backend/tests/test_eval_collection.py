import json
from pathlib import Path

import pytest

from policylens.schemas.chat import ChatResponse, ChatSource
from scripts.collect_eval_samples import EvaluationCase, collect_samples, load_cases


class FakeRagService:
    def __init__(self) -> None:
        self.calls = 0

    def answer_question(self, question: str, top_k: int) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            answer="Employees should take 10 business days. [1]",
            sources=[
                ChatSource(
                    reference=1,
                    document_id="d1",
                    chunk_id="d1-p8-c2",
                    filename="policy.pdf",
                    page_number=8,
                    excerpt="Employees should take 10 business days.",
                    similarity_score=0.5,
                    rerank_score=3.4,
                )
            ],
            retrieved_chunks=1,
            reranking_applied=True,
        )


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    case = {
        "id": "case-1",
        "category": "factual",
        "question": "Vacation?",
        "reference": "10 days",
        "expected_pages": [8],
    }
    path.write_text(json.dumps([case, case]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


def test_collection_is_idempotent_and_keeps_real_sources(tmp_path: Path) -> None:
    cases = [
        EvaluationCase(
            id="case-1",
            category="factual",
            question="Vacation?",
            reference="10 days",
            expected_pages=[8],
        )
    ]
    service = FakeRagService()
    output = tmp_path / "rag_samples.jsonl"

    assert collect_samples(cases, service, output, "fake-local") == 1
    assert collect_samples(cases, service, output, "fake-local") == 0
    assert service.calls == 1

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["retrieved_contexts"] == ["Employees should take 10 business days."]
    assert records[0]["expected_pages_present"] is True
    assert records[0]["reranking_applied"] is True
    assert records[0]["generator_model"] == "fake-local"
