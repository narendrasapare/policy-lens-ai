# Evaluation results

These are RAGAS judgments on the saved responses in `evaluations/results/rag_samples.jsonl`. Scoring uses local Ollama and sentence-transformer embeddings. Generating this report does not run inference.

| Case | Faithfulness | Answer relevancy | Context precision |
| --- | ---: | ---: | ---: |
| hr-001 | 1.0000 | 1.0000 | 1.0000 |
| hr-002 | 1.0000 | 0.9070 | 1.0000 |
| hr-003 | 1.0000 | 0.9116 | 1.0000 |
| hr-004 | 0.5000 | 0.7308 | 1.0000 |
| hr-005 | 1.0000 | 1.0000 | 1.0000 |
| hr-006 | 1.0000 | 0.9576 | 1.0000 |
| hr-007 | 1.0000 | 0.8951 | 1.0000 |
| hr-008 | 1.0000 | 0.6511 | 1.0000 |
| hr-009 | 1.0000 | 0.8984 | 1.0000 |
| hr-010 | 1.0000 | 1.0000 | 1.0000 |
| hr-011 | 1.0000 | 0.8430 | 0.0000 |
| hr-012 | 0.0000 | 1.0000 | 1.0000 |

Mean (successful cases / recorded cases):

- faithfulness: 0.8750 (12/12)
- answer_relevancy: 0.8996 (12/12)
- context_precision: 0.9167 (12/12)

Evaluator: qwen3:4b-instruct.
RAGAS versions: 0.4.3.

## How to interpret this

The dataset has 12 questions over one public handbook: factual lookups, one multipart question and one unsupported question. It is a small project check, not a general benchmark. Generator and evaluator use the same local model family, so these scores are not an independent human assessment. Low scores remain visible and need manual inspection; no scores are replaced or adjusted to improve the table.

The JSONL records retain question, answer preview, timestamps, model names, source pages, sample hash, metric values and any scoring errors. Existing completed metrics are reused; their original record timestamps are preserved. This run scores previously saved responses, rather than regenerating every answer after each code change.

## Reproduce

From `backend` with Ollama running:

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_eval_samples --all
.\.venv\Scripts\python.exe -m scripts.score_ragas
```

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe scripts/summarize_evaluation.py --markdown docs/EVALUATION.md
```

Collection skips saved cases. To deliberately refresh responses, archive both JSONL files first, then collect with `--all --overwrite` and score with `--overwrite`.
