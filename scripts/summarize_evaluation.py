"""Summarize saved RAGAS results without invoking a model."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, help="Write a Markdown results table")
    args = parser.parse_args()
    path = ROOT / "evaluations/results/ragas_scores.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Saved evaluation cases: {len(rows)}")
    means = {}
    for metric in METRICS:
        values = [row.get("metrics", {}).get(metric) for row in rows]
        values = [value for value in values if isinstance(value, (int, float))]
        means[metric] = f"{sum(values)/len(values):.4f} ({len(values)}/{len(rows)})" if values else "not scored"
        print(f"{metric}: {means[metric]}")
    if args.markdown:
        lines = ["# Evaluation results", "",
                 "These are RAGAS judgments on the saved responses in `evaluations/results/rag_samples.jsonl`. "
                 "Scoring uses local Ollama and sentence-transformer embeddings. Generating this report does not run inference.", "",
                 "| Case | Faithfulness | Answer relevancy | Context precision |",
                 "| --- | ---: | ---: | ---: |"]
        for row in rows:
            cells = [f"{row.get('metrics', {}).get(metric):.4f}" if isinstance(row.get('metrics', {}).get(metric), (int, float)) else "FAILED" for metric in METRICS]
            lines.append("| " + " | ".join([row["case_id"], *cells]) + " |")
        lines += ["", "Mean (successful cases / recorded cases):", ""]
        lines += [f"- {metric}: {means[metric]}" for metric in METRICS]
        models = sorted({str(row.get("evaluator_model")) for row in rows})
        lines += ["", "Evaluator: " + ", ".join(models) + ".",
                  "RAGAS versions: " + ", ".join(sorted({str(row.get("ragas_version")) for row in rows})) + ".", "",
                  "## How to interpret this", "",
                  "The dataset has 12 questions over one public handbook: factual lookups, one multipart question and one unsupported question. "
                  "It is a small project check, not a general benchmark. Generator and evaluator use the same local model family, so these scores are not an independent human assessment. "
                  "Low scores remain visible and need manual inspection; no scores are replaced or adjusted to improve the table.", "",
                  "The JSONL records retain question, answer preview, timestamps, model names, source pages, sample hash, metric values and any scoring errors. "
                  "Existing completed metrics are reused; their original record timestamps are preserved. This run scores previously saved responses, rather than regenerating every answer after each code change.", "",
                  "## Reproduce", "", "From `backend` with Ollama running:", "", "```powershell",
                  ".\\.venv\\Scripts\\python.exe -m scripts.collect_eval_samples --all",
                  ".\\.venv\\Scripts\\python.exe -m scripts.score_ragas", "```", "",
                  "From the project root:", "", "```powershell",
                  ".\\backend\\.venv\\Scripts\\python.exe scripts/summarize_evaluation.py --markdown docs/EVALUATION.md", "```", "",
                  "Collection skips saved cases. To deliberately refresh responses, archive both JSONL files first, "
                  "then collect with `--all --overwrite` and score with `--overwrite`."]
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Report: {args.markdown}")


if __name__ == "__main__":
    main()
