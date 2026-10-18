import json
import re
from pathlib import Path

from policylens.repositories.faiss_vector_repository import (
    FaissVectorRepository,
)
from policylens.schemas.document import DocumentChunk, DocumentSummary
from policylens.schemas.indexing import DocumentIndexResponse
from policylens.services.embedding_service import EmbeddingService


class ProcessedDocumentNotFoundError(Exception):
    pass


class InvalidProcessedDocumentError(Exception):
    pass


class DocumentIndexingService:
    def __init__(
        self,
        processed_directory: Path,
        embedding_service: EmbeddingService,
        vector_repository: FaissVectorRepository,
    ) -> None:
        self._processed_directory = processed_directory
        self._embedding_service = embedding_service
        self._vector_repository = vector_repository

    def index_document(
        self,
        document_id: str,
    ) -> DocumentIndexResponse:
        if not re.fullmatch(r"[0-9a-f]{32}", document_id):
            raise InvalidProcessedDocumentError(
                "Invalid document identifier."
            )

        chunks = self._load_document_chunks(document_id)

        already_indexed, total_vectors = (
            self._vector_repository.get_document_status(document_id)
        )

        if already_indexed:
            return DocumentIndexResponse(
                status="already_indexed",
                document_id=document_id,
                chunks_added=0,
                total_vectors=total_vectors,
                embedding_model=self._embedding_service.model_name,
            )

        embeddings = self._embedding_service.encode_documents(
            [chunk.text for chunk in chunks]
        )

        result = self._vector_repository.add_document(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=self._embedding_service.model_name,
        )

        return DocumentIndexResponse(
            status=(
                "indexed"
                if result.chunks_added > 0
                else "already_indexed"
            ),
            document_id=document_id,
            chunks_added=result.chunks_added,
            total_vectors=result.total_vectors,
            embedding_model=self._embedding_service.model_name,
        )

    def _load_document_chunks(
        self,
        document_id: str,
    ) -> list[DocumentChunk]:
        document_path = (
            self._processed_directory / f"{document_id}.json"
        )

        if not document_path.is_file():
            raise ProcessedDocumentNotFoundError(
                "The document has not been processed or does not exist."
            )

        try:
            payload = json.loads(
                document_path.read_text(encoding="utf-8")
            )

            summary = DocumentSummary.model_validate(
                payload["document"]
            )

            chunks = [
                DocumentChunk.model_validate(chunk)
                for chunk in payload["chunks"]
            ]

        except (OSError, KeyError, TypeError, ValueError) as error:
            raise InvalidProcessedDocumentError(
                "The processed document metadata is invalid."
            ) from error

        chunk_ids = {chunk.chunk_id for chunk in chunks}

        if (
            not chunks
            or summary.document_id != document_id
            or summary.chunk_count != len(chunks)
            or len(chunk_ids) != len(chunks)
            or any(
                chunk.document_id != document_id
                or chunk.source_filename != summary.filename
                or not chunk.text.strip()
                for chunk in chunks
            )
        ):
            raise InvalidProcessedDocumentError(
                "The processed document contains inconsistent chunks."
            )

        return chunks