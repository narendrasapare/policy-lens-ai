import numpy as np
import pytest

from policylens.repositories.faiss_vector_repository import FaissVectorRepository
from policylens.schemas.chat import ChatSource
from policylens.schemas.document import DocumentChunk
from policylens.services.answer_generation import AnswerGenerationError, LangChainAnswerGenerationService
from policylens.services.rag_service import RagService
from policylens.services.semantic_search import NoIndexedDocumentsError, SemanticSearchService


def test_summary_reads_later_pages_without_embedding_or_top_k_limit(tmp_path):
    repository = FaissVectorRepository(tmp_path)
    chunks = [DocumentChunk(
        chunk_id=f"document-p{page}-c1", document_id="document", source_filename="handbook.pdf",
        page_number=page, chunk_index=1, text=f"Policy on page {page}.",
    ) for page in range(1, 11)]
    repository.add_document("document", chunks, np.ones((10, 3), dtype=np.float32), "test")
    search = SemanticSearchService(embedding_service=None, vector_repository=repository)

    class Generator:
        def generate_summary(self, question, sources):
            assert [source.page_number for source in sources] == list(range(1, 11))
            return "The last page has a policy too. [10]"

    response = RagService(search, Generator()).answer_question("Summarize the handbook.", top_k=1)
    assert response.retrieved_chunks == 10
    assert response.sources[-1].reference == 10
    assert not response.reranking_applied


def test_summary_has_a_clear_empty_workspace_error(tmp_path):
    search = SemanticSearchService(None, FaissVectorRepository(tmp_path))
    with pytest.raises(NoIndexedDocumentsError):
        search.summary_context()


def test_summary_batches_preserve_all_passages_and_original_references():
    sources = [ChatSource(
        reference=i, document_id="document", chunk_id=f"chunk-{i}", filename="handbook.pdf",
        page_number=i, excerpt=f"Unique policy {i}. " + "text " * 180, similarity_score=0,
    ) for i in range(1, 21)]
    calls = []

    class Generator(LangChainAnswerGenerationService):
        def _invoke(self, question, context):
            calls.append((question, context))
            if question.startswith("User request:"):
                import re
                return "Cited notes: " + " ".join(re.findall(r"\[\d+\]", context))
            return "The final page is included. [20]"

    answer = Generator(lambda: None).generate_summary("Summarize the policies.", sources)
    assert answer.endswith("[20]")
    map_contexts = [context for question, context in calls if question.startswith("User request:")]
    assert len(map_contexts) > 1
    assert all(len(context) <= 4020 for context in map_contexts)
    for source in sources:
        assert sum(source.excerpt in context for context in map_contexts) == 1
    assert "[20]" in calls[-1][1]


def test_summary_rejects_nonexistent_citations():
    class Generator(LangChainAnswerGenerationService):
        def _invoke(self, question, context):
            return "A rule with an invented reference. [999]"

    source = ChatSource(reference=1, document_id="doc", chunk_id="chunk", filename="policy.pdf",
                        page_number=1, excerpt="A rule.", similarity_score=0)
    with pytest.raises(AnswerGenerationError, match="invalid source"):
        Generator(lambda: None).generate_summary("Summarize.", [source])
