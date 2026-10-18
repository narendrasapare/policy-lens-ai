import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from policylens.services.reranking import RerankOutcome


class RerankingEvidenceError(Exception):
    pass


class RerankingEvidenceRecorder:
    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        self._lock = Lock()

    def record(self, outcome: RerankOutcome) -> None:
        evidence = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": outcome.query,
            "reranker_model": outcome.model_name,
            "candidate_count": len(outcome.before),
            "top_result_changed": outcome.top_result_changed,
            "before": [asdict(item) for item in outcome.before],
            "after": [asdict(item) for item in outcome.after],
        }

        try:
            with self._lock:
                self.output_file.parent.mkdir(parents=True, exist_ok=True)
                with self.output_file.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(evidence, ensure_ascii=False) + "\n")
        except OSError as error:
            raise RerankingEvidenceError(
                "Unable to save reranking evidence."
            ) from error