import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.score_ragas import (
    read_jsonl,
    run,
    sample_digest,
    score_one,
    validate_local_settings,
)


class FixedMetric:
    def __init__(self, score: float) -> None:
        self.score_value = score
        self.calls = 0

    async def ascore(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        assert kwargs
        return SimpleNamespace(value=self.score_value)


class BrokenMetric:
    async def ascore(self, **kwargs: object) -> object:
        raise ValueError("Invalid structured evaluation output")


@pytest.fixture
def sample() -> dict:
    return {
        "case_id": "hr-001",
        "question": "How much annual leave?",
        "answer": "Ten business days. [1]",
        "reference": "Ten business days",
        "retrieved_contexts": ["Ten business days of paid annual leave."],
        "retrieved_pages": [8],
        "generator_model": "local-test",
    }


def test_sample_digest_changes_when_context_changes(sample: dict) -> None:
    first = sample_digest(sample)
    sample["retrieved_contexts"].append("Different text")
    assert sample_digest(sample) != first


def test_score_one_records_actual_metric_results(sample: dict) -> None:
    metric = FixedMetric(0.75)
    result = asyncio.run(
        score_one(sample, {"faithfulness": metric}, None, "local-model", "0.4.2")
    )
    assert result["metrics"]["faithfulness"] == 0.75
    assert result["errors"] == {}
    assert metric.calls == 1


def test_score_one_preserves_scoring_failures(sample: dict) -> None:
    result = asyncio.run(
        score_one(sample, {"faithfulness": BrokenMetric()}, None, "local-model", "0.4.2")
    )
    assert result["metrics"]["faithfulness"] is None
    assert "ValueError" in result["errors"]["faithfulness"]


def test_repeated_run_does_not_rescore_successful_metrics(
    sample: dict, tmp_path: Path
) -> None:
    samples_file = tmp_path / "samples.jsonl"
    scores_file = tmp_path / "scores.jsonl"
    samples_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    metric = FixedMetric(0.5)
    options = (samples_file, scores_file, {"faithfulness": metric}, "test", "0.4.2", 1, False)
    asyncio.run(run(*options))
    asyncio.run(run(*options))
    assert metric.calls == 1
    assert len(read_jsonl(scores_file)) == 1


def test_local_endpoint_rejects_remote_urls() -> None:
    settings = SimpleNamespace(
        llm_provider="ollama", allow_cloud_llm=False,
        ollama_base_url="https://remote.example.org"
    )
    with pytest.raises(RuntimeError, match="local HTTP"):
        validate_local_settings(settings)
