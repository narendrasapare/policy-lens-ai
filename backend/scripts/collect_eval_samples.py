import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from policylens.schemas.chat import ChatResponse


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIRECTORY / "evaluations" / "datasets" / "hr_policy_questions.json"
OUTPUT_PATH = PROJECT_DIRECTORY / "evaluations" / "results" / "rag_samples.jsonl"


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    expected_pages: list[int] = Field(default_factory=list)


class RagAnswerService(Protocol):
    def answer_question(self, question: str, top_k: int) -> ChatResponse: ...


def load_cases(dataset_path: Path) -> list[EvaluationCase]:
    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Evaluation dataset must be a non-empty JSON array.")
    cases = [EvaluationCase.model_validate(case) for case in raw_cases]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique.")
    return cases


def read_existing_ids(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    ids: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            ids.add(str(record["case_id"]))
    return ids


def collect_samples(
    cases: list[EvaluationCase],
    service: RagAnswerService,
    output_path: Path,
    model_name: str,
    *,
    top_k: int = 5,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = read_existing_ids(output_path)
    collected = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        for case in cases:
            if case.id in completed_ids:
                print(f"Skipping {case.id}: already collected")
                continue

            response = service.answer_question(case.question, top_k=top_k)
            source_pages = sorted({source.page_number for source in response.sources})
            expected_pages_present = all(
                page in source_pages for page in case.expected_pages
            )

            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "case_id": case.id,
                "category": case.category,
                "question": case.question,
                "reference": case.reference,
                "expected_pages": case.expected_pages,
                "retrieved_pages": source_pages,
                "expected_pages_present": expected_pages_present,
                "answer": response.answer,
                "retrieved_contexts": [source.excerpt for source in response.sources],
                "source_chunks": [source.model_dump() for source in response.sources],
                "query_type": response.query_type,
                "sub_questions": response.sub_questions,
                "reranking_applied": response.reranking_applied,
                "generator_model": model_name,
                "top_k": top_k,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            completed_ids.add(case.id)
            collected += 1
            print(
                f"Collected {case.id}: {len(response.sources)} sources, "
                f"pages={source_pages}, expected_pages_present={expected_pages_present}"
            )

    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real local RAG evaluation samples")
    parser.add_argument("--limit", type=int, default=1, help="Number of dataset rows (default: 1)")
    parser.add_argument("--all", action="store_true", help="Collect every dataset row")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace prior collected samples (old output must no longer be needed)",
    )
    args = parser.parse_args()

    if args.limit < 1 or not 1 <= args.top_k <= 10:
        parser.error("--limit must be >= 1 and --top-k must be between 1 and 10")

    from policylens.core.config import get_settings

    settings = get_settings()
    if settings.llm_provider != "ollama" or settings.allow_cloud_llm:
        raise RuntimeError("Evaluation collection requires local-only Ollama settings.")

    from policylens.api.dependencies import get_rag_service

    if args.overwrite and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    cases = load_cases(DATASET_PATH)
    selected_cases = cases if args.all else cases[: args.limit]
    added = collect_samples(
        selected_cases,
        get_rag_service(),
        OUTPUT_PATH,
        settings.ollama_model,
        top_k=args.top_k,
    )
    print(f"New samples: {added} | Total stored: {len(read_existing_ids(OUTPUT_PATH))}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
