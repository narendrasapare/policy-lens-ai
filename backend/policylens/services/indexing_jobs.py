"""Single-worker indexing queue. Unfinished documents are recovered at startup."""

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

logger = logging.getLogger(__name__)


class IndexingJobs:
    def __init__(self, service_factory):
        self._service_factory = service_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexing")
        self._lock = Lock()
        self._states: dict[str, tuple[str, str | None]] = {}

    def status(self, document_id: str) -> tuple[str, str | None]:
        with self._lock:
            return self._states.get(document_id, ("queued", None))

    def enqueue(self, document_id: str) -> None:
        with self._lock:
            if self._states.get(document_id, (None,))[0] in {"queued", "indexing"}:
                return
            self._states[document_id] = ("queued", None)
            self._executor.submit(self._run, document_id)

    def _run(self, document_id: str) -> None:
        with self._lock:
            self._states[document_id] = ("indexing", None)
        try:
            self._service_factory().index_document(document_id)
        except Exception:
            logger.exception("Indexing failed for %s", document_id)
            with self._lock:
                self._states[document_id] = (
                    "failed", "Could not prepare this PDF. Try preparing it again."
                )
        else:
            with self._lock:
                self._states[document_id] = ("ready", None)

    def has_pending(self) -> bool:
        with self._lock:
            return any(state in {"queued", "indexing"} for state, _ in self._states.values())

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
