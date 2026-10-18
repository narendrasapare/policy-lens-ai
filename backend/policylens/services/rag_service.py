from typing import TypedDict

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from policylens.schemas.chat import ChatResponse, ChatSource
from policylens.schemas.search import SearchResult
from policylens.services.answer_generation import AnswerGenerator
from policylens.services.query_routing import QueryRouter, QueryType
from policylens.services.reranking import Reranker
from policylens.services.reranking_evidence import RerankingEvidenceRecorder
from policylens.services.semantic_search import SemanticSearchService


class _RagInput(TypedDict):
    question: str
    top_k: int


class _RetrievalState(TypedDict):
    question: str
    sources: list[ChatSource]
    retrieved_chunks: int
    query_type: str
    sub_questions: list[str]
    reranking_applied: bool


class RagService:
    def __init__(
        self,
        search_service: SemanticSearchService,
        answer_generation_service: AnswerGenerator,
        query_router: QueryRouter | None = None,
        reranker: Reranker | None = None,
        evidence_recorder: RerankingEvidenceRecorder | None = None,
        candidate_count: int = 15,
    ) -> None:
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive.")

        self._search_service = search_service
        self._answer_generation_service = answer_generation_service
        self._query_router = query_router or QueryRouter()
        self._reranker = reranker
        self._evidence_recorder = evidence_recorder
        self._candidate_count = candidate_count
        self._chain = (
            RunnableLambda(self._retrieve)
            | RunnablePassthrough.assign(answer=RunnableLambda(self._generate))
        )

    def _retrieve(self, request: _RagInput) -> _RetrievalState:
        plan = self._query_router.route(request["question"])
        sub_questions = list(plan.sub_questions)
        requested_top_k = request["top_k"]
        if plan.query_type == QueryType.SUMMARY:
            results = self._search_service.summary_context()
            return {
                "question": request["question"],
                "sources": [ChatSource(
                    reference=position, document_id=result.document_id,
                    chunk_id=result.chunk_id, filename=result.source_filename,
                    page_number=result.page_number, excerpt=result.text,
                    similarity_score=0.0,
                ) for position, result in enumerate(results, start=1)],
                "retrieved_chunks": len(results), "query_type": "summary",
                "sub_questions": sub_questions, "reranking_applied": False,
            }
        candidate_count = (
            max(requested_top_k, self._candidate_count)
            if self._reranker is not None
            else requested_top_k
        )

        groups: list[list[tuple[SearchResult, float | None]]] = []

        for sub_question in sub_questions:
            response = self._search_service.search(
                query=sub_question,
                top_k=candidate_count,
            )

            if self._reranker is None:
                groups.append([(result, None) for result in response.results])
                continue

            outcome = self._reranker.rerank(
                query=sub_question,
                candidates=response.results,
                top_k=requested_top_k,
            )

            if self._evidence_recorder is not None:
                self._evidence_recorder.record(outcome)

            groups.append([
                (item.result, item.rerank_score)
                for item in outcome.ranked_results
            ])

        seen_chunk_ids: set[str] = set()
        selected: list[tuple[SearchResult, str, float | None]] = []
        max_group_size = max((len(group) for group in groups), default=0)
        max_results = max(requested_top_k, len(sub_questions))

        for rank in range(max_group_size):
            for question, group in zip(sub_questions, groups):
                if rank >= len(group):
                    continue

                result, rerank_score = group[rank]
                if result.chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(result.chunk_id)
                selected.append((result, question, rerank_score))

                if len(selected) >= max_results:
                    break

            if len(selected) >= max_results:
                break

        sources = [
            ChatSource(
                reference=position,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                filename=result.source_filename,
                page_number=result.page_number,
                excerpt=result.text,
                similarity_score=result.similarity_score,
                rerank_score=rerank_score,
                retrieved_for=sub_question,
            )
            for position, (result, sub_question, rerank_score) in enumerate(
                selected, start=1
            )
        ]

        return {
            "question": request["question"],
            "sources": sources,
            "retrieved_chunks": len(sources),
            "query_type": plan.query_type.value,
            "sub_questions": sub_questions,
            "reranking_applied": self._reranker is not None,
        }

    def _generate(self, state: _RetrievalState) -> str:
        if not state["sources"]:
            return "I could not find relevant information in the indexed documents."

        if state["query_type"] == "summary":
            return self._answer_generation_service.generate_summary(
                question=state["question"], sources=state["sources"],
            )
        return self._answer_generation_service.generate_answer(
            question=state["question"],
            sources=state["sources"],
        )

    def answer_question(self, question: str, top_k: int) -> ChatResponse:
        result = self._chain.invoke({"question": question, "top_k": top_k})

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieved_chunks=result["retrieved_chunks"],
            query_type=result["query_type"],
            sub_questions=result["sub_questions"],
            reranking_applied=result["reranking_applied"],
        )
