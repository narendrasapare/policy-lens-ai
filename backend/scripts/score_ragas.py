import argparse
import asyncio
import hashlib
import json
import math
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit
from uuid import uuid4


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
SAMPLES_PATH = PROJECT_DIRECTORY / "evaluations" / "results" / "rag_samples.jsonl"
SCORES_PATH = PROJECT_DIRECTORY / "evaluations" / "results" / "ragas_scores.jsonl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METRIC_INPUTS = {
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
}


class MetricScorer(Protocol):
    async def ascore(self, **kwargs: Any) -> Any: ...


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation file: {path}")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Missing case_id at line {line_number}")
        if case_id in seen_ids:
            raise ValueError(f"Repeated case_id in {path.name}: {case_id}")
        seen_ids.add(case_id)
        rows.append(row)
    return rows


def validate_sample(row: dict[str, Any]) -> None:
    for field in ("question", "answer", "reference"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{row['case_id']}: missing {field}")
    contexts = row.get("retrieved_contexts")
    if not isinstance(contexts, list) or not all(
        isinstance(text, str) for text in contexts
    ):
        raise ValueError(f"{row['case_id']}: invalid retrieved_contexts")


def sample_digest(row: dict[str, Any]) -> str:
    values = {
        "question": row["question"],
        "answer": row["answer"],
        "reference": row["reference"],
        "retrieved_contexts": row["retrieved_contexts"],
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def save_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def validate_local_settings(settings: Any) -> str:
    if settings.llm_provider != "ollama" or settings.allow_cloud_llm:
        raise RuntimeError("RAGAS scoring requires local-only Ollama settings.")

    parsed = urlsplit(settings.ollama_base_url)
    if parsed.scheme != "http" or parsed.hostname not in (
        "localhost", "127.0.0.1", "::1"
    ):
        raise RuntimeError("Ollama evaluator must use a local HTTP endpoint.")
    return settings.ollama_base_url.rstrip("/") + "/v1"


def create_metrics(settings: Any) -> tuple[Any, dict[str, MetricScorer]]:
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

    base_url = validate_local_settings(settings)
    client = AsyncOpenAI(
        api_key="ollama", base_url=base_url, timeout=180, max_retries=0
    )
    evaluator = llm_factory(settings.ollama_model, provider="openai", client=client)
    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL, use_api=False, device="cpu"
    )
    return client, {
        "faithfulness": Faithfulness(llm=evaluator),
        "answer_relevancy": AnswerRelevancy(llm=evaluator, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=evaluator),
    }


async def score_one(
    sample: dict[str, Any],
    metrics: dict[str, MetricScorer],
    previous: dict[str, Any] | None,
    model_name: str,
    ragas_version: str,
) -> dict[str, Any]:
    validate_sample(sample)
    digest = sample_digest(sample)

    if previous is not None and previous.get("sample_sha256") != digest:
        raise ValueError(
            f"{sample['case_id']}: saved sample changed. Run with --overwrite "
            "to intentionally start a new evaluation."
        )

    inputs = {
        "user_input": sample["question"],
        "response": sample["answer"],
        "reference": sample["reference"],
        "retrieved_contexts": sample["retrieved_contexts"],
    }
    scores = dict(previous.get("metrics", {})) if previous else {}
    errors = dict(previous.get("errors", {})) if previous else {}

    for name, scorer in metrics.items():
        if scores.get(name) is not None:
            continue
        arguments = {field: inputs[field] for field in METRIC_INPUTS[name]}
        try:
            result = await scorer.ascore(**arguments)
            value = float(result.value)
            if not math.isfinite(value):
                raise ValueError("Metric returned a non-finite value.")
            scores[name] = value
            errors.pop(name, None)
        except Exception as error:
            scores[name] = None
            errors[name] = f"{type(error).__name__}: {str(error)[:240]}"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": sample["case_id"],
        "question": sample["question"],
        "answer_preview": sample["answer"][:220],
        "source_pages": sample.get("retrieved_pages", []),
        "generator_model": sample.get("generator_model"),
        "evaluator_model": model_name,
        "embedding_model": EMBEDDING_MODEL,
        "ragas_version": ragas_version,
        "sample_sha256": digest,
        "metrics": scores,
        "errors": errors,
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    for name in METRIC_INPUTS:
        values = [
            row["metrics"].get(name)
            for row in rows
            if row["metrics"].get(name) is not None
        ]
        if values:
            print(f"{name}: mean={sum(values) / len(values):.4f} ({len(values)}/{len(rows)} scored)")
        else:
            print(f"{name}: not scored (0/{len(rows)})")


async def run(
    samples_path: Path,
    scores_path: Path,
    metrics: dict[str, MetricScorer],
    model_name: str,
    ragas_version: str,
    limit: int | None,
    overwrite: bool,
) -> None:
    samples = read_jsonl(samples_path)
    if not samples:
        raise ValueError("No collected RAG samples exist yet.")
    selected = samples[:limit] if limit else samples
    previous_rows = [] if overwrite or not scores_path.exists() else read_jsonl(scores_path)
    existing = {row["case_id"]: row for row in previous_rows}

    for sample in selected:
        earlier = existing.get(sample["case_id"])
        if earlier is not None and earlier.get("sample_sha256") != sample_digest(sample):
            raise ValueError(
                f"Sample changed for {sample['case_id']}. Run with --overwrite."
            )
        if earlier is not None and all(
            earlier.get("metrics", {}).get(name) is not None for name in metrics
        ):
            print(f"Skipping {sample['case_id']}: all metrics already scored")
            continue

        record = await score_one(sample, metrics, earlier, model_name, ragas_version)
        existing[sample["case_id"]] = record
        ordered = [existing[row["case_id"]] for row in samples if row["case_id"] in existing]
        save_scores(scores_path, ordered)
        metrics_out = ", ".join(
            f"{key}={value if value is not None else 'FAILED'}"
            for key, value in record["metrics"].items()
        )
        print(f"{sample['case_id']}: {metrics_out}")

    ordered = [existing[row["case_id"]] for row in samples if row["case_id"] in existing]
    print_summary(ordered)
    print(f"Results saved: {scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS scoring using local Ollama")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N samples")
    parser.add_argument("--overwrite", action="store_true", help="Replace earlier score results")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    from policylens.core.config import get_settings

    settings = get_settings()
    validate_local_settings(settings)
    print(f"Evaluator: local Ollama / {settings.ollama_model}")
    print("Gemini/cloud inference: disabled")
    client, metrics = create_metrics(settings)

    async def evaluate_and_close() -> None:
        try:
            await run(
                SAMPLES_PATH,
                SCORES_PATH,
                metrics,
                settings.ollama_model,
                version("ragas"),
                args.limit,
                args.overwrite,
            )
        finally:
            await client.close()

    asyncio.run(evaluate_and_close())


if __name__ == "__main__":
    main()
