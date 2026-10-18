import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from policylens.schemas.document import (
    DocumentChunk,
    DocumentSummary,
)


class DocumentIngestionError(Exception):
    pass


@dataclass(frozen=True)
class DocumentIngestionResult:
    document: DocumentSummary
    already_exists: bool


class DocumentIngestionService:
    def __init__(
        self,
        data_directory: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        max_pages: int = 100,
    ) -> None:
        self._uploads_directory = data_directory / "uploads"
        self._processed_directory = data_directory / "processed"
        self._max_pages = max_pages
        self._ingestion_lock = Lock()

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def ingest(
        self,
        filename: str,
        content: bytes,
    ) -> DocumentIngestionResult:
        source_filename = filename.replace("\\", "/").split("/")[-1].strip()

        if not source_filename.lower().endswith(".pdf"):
            raise DocumentIngestionError(
                "Only PDF documents are supported."
            )

        if not content.startswith(b"%PDF-"):
            raise DocumentIngestionError(
                "The uploaded file is not a valid PDF."
            )

        content_hash = hashlib.sha256(content).hexdigest()

        with self._ingestion_lock:
            existing_document = self._find_existing_document(
                content_hash
            )

            if existing_document is not None:
                return DocumentIngestionResult(
                    document=existing_document,
                    already_exists=True,
                )

            pages = self._extract_pages(content)

            document_id = uuid4().hex
            chunks: list[DocumentChunk] = []

            for page_number, page_text in pages:
                text_chunks = self._text_splitter.split_text(page_text)

                for chunk_index, chunk_text in enumerate(
                    text_chunks,
                    start=1,
                ):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=(
                                f"{document_id}-p{page_number}-c{chunk_index}"
                            ),
                            document_id=document_id,
                            source_filename=source_filename,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            text=chunk_text,
                        )
                    )

            if not chunks:
                raise DocumentIngestionError(
                    "No extractable text found in the PDF."
                )

            document_summary = DocumentSummary(
                document_id=document_id,
                filename=source_filename,
                page_count=len(pages),
                character_count=sum(
                    len(page_text) for _, page_text in pages
                ),
                chunk_count=len(chunks),
            )

            self._save_document(
                document_id=document_id,
                original_content=content,
                content_hash=content_hash,
                document_summary=document_summary,
                chunks=chunks,
            )

            return DocumentIngestionResult(
                document=document_summary,
                already_exists=False,
            )

    def _find_existing_document(
        self,
        content_hash: str,
    ) -> DocumentSummary | None:
        if not self._processed_directory.is_dir():
            return None

        metadata_files = sorted(
            self._processed_directory.glob("*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )

        for metadata_path in metadata_files:
            stored_pdf = (
                self._uploads_directory / f"{metadata_path.stem}.pdf"
            )

            if not stored_pdf.is_file():
                continue

            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )

                stored_hash = metadata.get("content_sha256")

                if not isinstance(stored_hash, str):
                    stored_hash = self._hash_file(stored_pdf)

                if stored_hash != content_hash:
                    continue

                document = DocumentSummary.model_validate(
                    metadata["document"]
                )

                if document.document_id != metadata_path.stem:
                    raise ValueError("Document ID mismatch.")

                return document

            except (OSError, ValueError, TypeError, KeyError) as error:
                raise DocumentIngestionError(
                    "Unable to validate existing document metadata."
                ) from error

        return None

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        digest = hashlib.sha256()

        with file_path.open("rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(block)

        return digest.hexdigest()

    def _extract_pages(
        self,
        content: bytes,
    ) -> list[tuple[int, str]]:
        try:
            with pymupdf.open(
                stream=content,
                filetype="pdf",
            ) as document:
                if document.needs_pass:
                    raise DocumentIngestionError(
                        "Password-protected PDFs are not supported."
                    )

                if document.page_count > self._max_pages:
                    raise DocumentIngestionError(
                        f"PDF exceeds the {self._max_pages}-page limit."
                    )

                pages: list[tuple[int, str]] = []

                for page_index in range(document.page_count):
                    page = document.load_page(page_index)
                    text = page.get_text(
                        "text",
                        sort=True,
                    ).strip()

                    pages.append(
                        (page_index + 1, text)
                    )

                return pages

        except (
            pymupdf.FileDataError,
            RuntimeError,
            ValueError,
        ) as error:
            raise DocumentIngestionError(
                "Unable to read the uploaded PDF."
            ) from error

    def _save_document(
        self,
        document_id: str,
        original_content: bytes,
        content_hash: str,
        document_summary: DocumentSummary,
        chunks: list[DocumentChunk],
    ) -> None:
        self._uploads_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._processed_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path = self._uploads_directory / f"{document_id}.pdf"

        metadata_path = (
            self._processed_directory / f"{document_id}.json"
        )

        metadata = {
            "content_sha256": content_hash,
            "document": document_summary.model_dump(),
            "chunks": [
                chunk.model_dump()
                for chunk in chunks
            ],
        }

        try:
            pdf_path.write_bytes(original_content)

            metadata_path.write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        except OSError:
            pdf_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise