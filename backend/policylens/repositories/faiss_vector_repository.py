from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

import faiss
import numpy as np
from pydantic import BaseModel, Field

from policylens.schemas.document import DocumentChunk


class VectorStoreError(Exception):
    pass


class StoredIndexMetadata(BaseModel):
    embedding_model: str
    vector_dimension: int = Field(gt=0)
    chunks: list[DocumentChunk]


@dataclass(frozen=True)
class IndexWriteResult:
    chunks_added: int
    total_vectors: int


@dataclass(frozen=True)
class VectorSearchHit:
    chunk: DocumentChunk
    similarity_score: float


@dataclass(frozen=True)
class VectorSearchResult:
    hits: list[VectorSearchHit]
    total_vectors: int


class FaissVectorRepository:
    def __init__(self, index_directory: Path) -> None:
        self._index_directory = index_directory
        self._index_path = index_directory / "policy_vectors.faiss"
        self._metadata_path = index_directory / "chunk_metadata.json"
        self._lock = Lock()

    def get_document_status(
        self,
        document_id: str,
    ) -> tuple[bool, int]:
        with self._lock:
            index, metadata = self._load()

            total_vectors = int(index.ntotal) if index is not None else 0

            already_indexed = (
                metadata is not None
                and any(
                    chunk.document_id == document_id
                    for chunk in metadata.chunks
                )
            )

            return already_indexed, total_vectors

    def get_vector_count(self) -> int:
        with self._lock:
            index, _ = self._load()
            return int(index.ntotal) if index is not None else 0

    def get_chunks(self) -> list[DocumentChunk]:
        """Return document order for summaries, rather than similarity order."""
        with self._lock:
            _, metadata = self._load()
            if metadata is None:
                return []
            return sorted(metadata.chunks, key=lambda chunk: (
                chunk.document_id, chunk.page_number, chunk.chunk_index
            ))

    def add_document(
        self,
        document_id: str,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        embedding_model: str,
    ) -> IndexWriteResult:
        vectors = np.array(
            embeddings,
            dtype=np.float32,
            order="C",
            copy=True,
        )

        if (
            not chunks
            or vectors.ndim != 2
            or vectors.shape[0] != len(chunks)
            or vectors.shape[1] == 0
        ):
            raise ValueError("Chunk and embedding counts must match.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Embeddings cannot contain zero vectors.")

        faiss.normalize_L2(vectors)

        with self._lock:
            index, existing_metadata = self._load()

            if existing_metadata is not None:
                if any(
                    chunk.document_id == document_id
                    for chunk in existing_metadata.chunks
                ):
                    return IndexWriteResult(
                        chunks_added=0,
                        total_vectors=int(index.ntotal),
                    )

                if existing_metadata.embedding_model != embedding_model:
                    raise VectorStoreError(
                        "The FAISS index uses a different embedding model."
                    )

            vector_dimension = int(vectors.shape[1])

            if index is None:
                index = faiss.IndexFlatIP(vector_dimension)

            elif int(index.d) != vector_dimension:
                raise VectorStoreError(
                    "Embedding dimensions do not match the existing index."
                )

            index.add(vectors)

            previous_chunks = (
                existing_metadata.chunks
                if existing_metadata is not None
                else []
            )

            updated_metadata = StoredIndexMetadata(
                embedding_model=embedding_model,
                vector_dimension=vector_dimension,
                chunks=[*previous_chunks, *chunks],
            )

            self._save(index, updated_metadata)

            return IndexWriteResult(
                chunks_added=len(chunks),
                total_vectors=int(index.ntotal),
            )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> VectorSearchResult:
        if top_k < 1:
            raise ValueError("top_k must be at least one.")

        vector = np.asarray(query_vector, dtype=np.float32)

        if (
            vector.ndim != 1
            or vector.size == 0
            or not np.isfinite(vector).all()
            or np.linalg.norm(vector) == 0
        ):
            raise ValueError("Query embedding is invalid.")

        normalized_query = np.ascontiguousarray(vector.reshape(1, -1))
        faiss.normalize_L2(normalized_query)

        with self._lock:
            index, metadata = self._load()

            if index is None or metadata is None:
                return VectorSearchResult(hits=[], total_vectors=0)

            if int(index.d) != normalized_query.shape[1]:
                raise VectorStoreError(
                    "Query embedding dimensions do not match the FAISS index."
                )

            scores, positions = index.search(
                normalized_query,
                min(top_k, int(index.ntotal)),
            )

            hits = [
                VectorSearchHit(
                    chunk=metadata.chunks[int(position)],
                    similarity_score=float(score),
                )
                for score, position in zip(scores[0], positions[0])
                if position >= 0
            ]

            return VectorSearchResult(
                hits=hits,
                total_vectors=int(index.ntotal),
            )

    def _load(
        self,
    ) -> tuple[faiss.Index | None, StoredIndexMetadata | None]:
        index_exists = self._index_path.is_file()
        metadata_exists = self._metadata_path.is_file()

        if not index_exists and not metadata_exists:
            return None, None

        if index_exists != metadata_exists:
            raise VectorStoreError(
                "FAISS index and metadata files are inconsistent."
            )

        try:
            index = faiss.read_index(str(self._index_path))

            metadata = StoredIndexMetadata.model_validate_json(
                self._metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise VectorStoreError(
                "Unable to load the stored FAISS index."
            ) from error

        if (
            int(index.ntotal) != len(metadata.chunks)
            or int(index.d) != metadata.vector_dimension
            or index.metric_type != faiss.METRIC_INNER_PRODUCT
        ):
            raise VectorStoreError(
                "Stored vectors and chunk metadata do not match."
            )

        return index, metadata

    def _save(
        self,
        index: faiss.Index,
        metadata: StoredIndexMetadata,
    ) -> None:
        self._index_directory.mkdir(parents=True, exist_ok=True)

        temporary_id = uuid4().hex

        temporary_index = (
            self._index_directory / f".policy_vectors_{temporary_id}.tmp"
        )
        temporary_metadata = (
            self._index_directory / f".chunk_metadata_{temporary_id}.tmp"
        )

        try:
            faiss.write_index(index, str(temporary_index))

            temporary_metadata.write_text(
                metadata.model_dump_json(indent=2),
                encoding="utf-8",
            )

            temporary_index.replace(self._index_path)
            temporary_metadata.replace(self._metadata_path)

        except (OSError, RuntimeError) as error:
            raise VectorStoreError(
                "Unable to persist the FAISS index."
            ) from error

        finally:
            temporary_index.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
