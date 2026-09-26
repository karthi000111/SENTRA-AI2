"""Query-driven retrieval over one isolated uploaded-document corpus."""
from __future__ import annotations

from .chunker import PageChunker
from .models import Chunk, ExtractionReport, RetrievalResult
from .vector_store import FaissVectorStore


class Retriever:
    def __init__(self, embedder: object, vector_store: FaissVectorStore) -> None:
        self._embedder, self._vector_store = embedder, vector_store

    @classmethod
    def from_extraction_report(cls, report: ExtractionReport, embedder: object,
                               chunker: PageChunker | None = None) -> "Retriever":
        if not report.has_text:
            details = "; ".join(problem.message for problem in report.problems) or "No PDF pages were supplied."
            raise ValueError(f"Cannot build a RAG corpus: {details}")
        chunks = (chunker or PageChunker()).chunk_pages(report.pages)
        return cls.from_chunks(chunks, embedder)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], embedder: object) -> "Retriever":
        if not chunks:
            raise ValueError("Cannot build a RAG corpus: no chunks were provided.")
        return cls(embedder, FaissVectorStore(chunks, embedder.encode([chunk.text for chunk in chunks])))

    @property
    def size(self) -> int:
        return self._vector_store.size

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Return matching evidence ordered by ascending squared-L2 distance."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        matches = self._vector_store.search(self._embedder.encode([query]), top_k)
        return [RetrievalResult(chunk.document, chunk.page, chunk.chunk_id, chunk.text, distance,
                                chunk.session_id, chunk.document_id, chunk.is_reference_section).as_dict()
                for chunk, distance in matches]
