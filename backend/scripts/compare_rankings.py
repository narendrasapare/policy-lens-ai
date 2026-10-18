from policylens.api.dependencies import (
    get_reranker,
    get_reranking_evidence_recorder,
    get_semantic_search_service,
)
from policylens.services.semantic_search import NoIndexedDocumentsError


QUESTIONS = [
    "How many paid vacation days do employees get?",
    "What happens if payday falls on a weekend?",
    "Who approves business expenses over 1000 dollars?",
    "How do employees report workplace harassment?",
    "Does the organization allow remote work?",
    "What percentage of medical insurance does the employer pay?",
    "Which expenses require receipts?",
    "What is the process for professional development approval?",
    "What happens after a report of policy violations?",
    "How do employees request time off?",
    "Who handles media inquiries?",
    "How long are daily standup meetings?",
    "What kinds of professional development are supported?",
    "How does the organization handle conflicts of interest?",
]


def main() -> None:
    search_service = get_semantic_search_service()
    reranker = get_reranker()
    recorder = get_reranking_evidence_recorder()
    changed_count = 0

    for question in QUESTIONS:
        try:
            candidates = search_service.search(
                query=question, top_k=15
            ).results
        except NoIndexedDocumentsError:
            print("Index a PDF before running the reranking comparison.")
            return

        outcome = reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=5,
        )
        recorder.record(outcome)
        changed_count += int(outcome.top_result_changed)

        before = outcome.before[0].chunk_id if outcome.before else "none"
        after = outcome.after[0].chunk_id if outcome.after else "none"

        print(f"Question: {question}")
        print(f"  FAISS top chunk:    {before}")
        print(f"  Reranked top chunk: {after}")
        print(f"  Top changed:        {outcome.top_result_changed}")

    print(f"\nTop-result changes: {changed_count}/{len(QUESTIONS)}")
    print(f"Evidence saved to: {recorder.output_file}")

    if changed_count == 0:
        print(
            "No top-result change was observed for these questions. "
            "Try additional genuine questions; do not invent a changed ranking."
        )


if __name__ == "__main__":
    main()