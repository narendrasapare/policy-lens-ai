from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Protocol

import numpy as np

from policylens.schemas.search import SearchResult


class RerankingError(Exception):
    pass


class PairScoringModel(Protocol):
    def predict(
        self,
        pairs: list[tuple[str, str]],
        **kwargs: object,
    ) -> object: ...


@dataclass(frozen=True)
class RankedSearchResult:
    result: SearchResult
    rerank_score: float
    original_rank: int


@dataclass(frozen=True)
class RankingEntry:
    rank: int
    chunk_id: str
    filename: str
    page_number: int
    faiss_score: float
    rerank_score: float


@dataclass(frozen=True)
class RerankOutcome:
    query: str
    model_name: str
    ranked_results: list[RankedSearchResult]
    before: list[RankingEntry]
    after: list[RankingEntry]
    top_result_changed: bool


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> RerankOutcome: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        model_factory: Callable[[], PairScoringModel] | None = None,
    ) -> None:
        self.model_name = model_name
        self._model_factory = model_factory
        self._model: PairScoringModel | None = None
        self._lock = Lock()

    def _load_model(self) -> PairScoringModel:
        if self._model_factory is not None:
            return self._model_factory()

        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> RerankOutcome:
        if top_k < 1:
            raise ValueError("top_k must be at least one.")
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if not candidates:
            return RerankOutcome(
                query=query,
                model_name=self.model_name,
                ranked_results=[],
                before=[],
                after=[],
                top_result_changed=False,
            )

        pairs = [(query, candidate.text) for candidate in candidates]

        try:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model()

                raw_scores = self._model.predict(
                    pairs,
                    batch_size=16,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as error:
            raise RerankingError(
                "Local cross-encoder inference failed. Check the model download."
            ) from error

        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise RerankingError("Cross-encoder returned invalid scores.")

        order = sorted(
            range(len(candidates)),
            key=lambda position: (-float(scores[position]), position),
        )

        def entry(position: int, rank: int) -> RankingEntry:
            item = candidates[position]
            return RankingEntry(
                rank=rank,
                chunk_id=item.chunk_id,
                filename=item.source_filename,
                page_number=item.page_number,
                faiss_score=float(item.similarity_score),
                rerank_score=float(scores[position]),
            )

        before = [
            entry(position, rank)
            for rank, position in enumerate(range(len(candidates)), start=1)
        ]
        after = [
            entry(position, rank)
            for rank, position in enumerate(order, start=1)
        ]
        ranked_results = [
            RankedSearchResult(
                result=candidates[position],
                rerank_score=float(scores[position]),
                original_rank=position + 1,
            )
            for position in order[:top_k]
        ]

        return RerankOutcome(
            query=query,
            model_name=self.model_name,
            ranked_results=ranked_results,
            before=before,
            after=after,
            top_result_changed=order[0] != 0,
        )