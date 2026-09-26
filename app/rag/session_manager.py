"""Lifecycle management for isolated, temporary uploaded-PDF RAG sessions."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import UUID, uuid4

from app.config.rag import RAGSettings, get_rag_settings

from .chunker import PageChunker
from .embeddings import SentenceTransformerEmbedder
from .models import Chunk, ExtractionProblem, ExtractionReport
from .pdf_processor import PDFProcessor
from .retriever import Retriever
from .vector_store import FaissVectorStore


@dataclass(frozen=True)
class SessionIngestionResult:
    session_id: str
    chunk_count: int
    problems: list[ExtractionProblem]


class SessionRAGManager:
    """Create, query, and delete temporary FAISS corpora isolated by session ID.

    The manager never combines indexes or metadata between session directories.
    A process may reopen a persisted temporary session safely via its own index
    and metadata files.
    """

    def __init__(self, *, settings: RAGSettings | None = None, embedder: object | None = None,
                 processor: PDFProcessor | None = None) -> None:
        self.settings = settings or get_rag_settings()
        self.settings.session_root.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder or SentenceTransformerEmbedder(self.settings.embedding_model)
        self._processor = processor or PDFProcessor()
        self._retrievers: dict[str, Retriever] = {}

    def cleanup_orphaned_sessions(self) -> None:
        """Remove sessions older than session_ttl_hours to prevent state drift."""
        if not self.settings.session_root.exists():
            return
        now = time.time()
        ttl_seconds = self.settings.session_ttl_hours * 3600
        for session_dir in self.settings.session_root.iterdir():
            if not session_dir.is_dir():
                continue
            try:
                UUID(session_dir.name)
            except ValueError:
                continue
            
            # Use the directory modification time to check age
            if now - session_dir.stat().st_mtime > ttl_seconds:
                try:
                    self.delete_session(session_dir.name)
                except Exception:
                    pass

    def create_session(self) -> str:
        """Create and return a new opaque session ID and storage directory."""
        self.cleanup_orphaned_sessions()
        session_id = str(uuid4())
        session_dir = self._session_dir(session_id)
        (session_dir / "uploads").mkdir(parents=True)
        return session_id

    def add_documents(self, session_id: str, files: list[tuple[str, bytes]]) -> SessionIngestionResult:
        """Ingest one or more uploaded PDF byte streams into one session only."""
        session_dir = self._session_dir(session_id)
        uploads_dir = session_dir / "uploads"
        if not uploads_dir.is_dir():
            raise KeyError(f"Unknown RAG session: {session_id}")
        if not files:
            raise ValueError("At least one uploaded file is required.")

        report = ExtractionReport()
        new_chunks: list[Chunk] = []
        chunker = PageChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        for filename, content in files:
            display_name = Path(filename).name or "upload.pdf"
            document_id = str(uuid4())
            # Retain the original upload only inside its temporary session directory.
            (uploads_dir / f"{document_id}_{display_name}").write_bytes(content)
            document_report = self._processor.extract_bytes([(display_name, content)])
            report.problems.extend(document_report.problems)
            pages = [replace(page, document_id=document_id) for page in document_report.pages]
            report.pages.extend(pages)
            new_chunks.extend(
                replace(chunk, session_id=session_id, document_id=document_id,
                        chunk_id=f"{document_id}_{chunk.chunk_id}")
                for chunk in chunker.chunk_pages(pages)
            )

        existing_chunks = self._current_chunks(session_id)
        all_chunks = [*existing_chunks, *new_chunks]
        if not all_chunks:
            messages = "; ".join(problem.message for problem in report.problems) or "No extractable PDF text found."
            raise ValueError(f"No session index was created: {messages}")

        retriever = Retriever.from_chunks(all_chunks, self._embedder)
        self._retrievers[session_id] = retriever
        self._persist(session_id, retriever)
        return SessionIngestionResult(session_id, len(all_chunks), report.problems)

    def retrieve(self, session_id: str, query: str, top_k: int | None = None, include_references: bool = False, bias_first_pages: bool = False) -> list[dict[str, object]]:
        """Search only the FAISS index and metadata associated with ``session_id``."""
        retriever = self._retrievers.get(session_id) or self._load_retriever(session_id)
        # Fetch extra results to allow for filtering
        results = retriever.retrieve(query, (top_k or self.settings.default_top_k) * 3)
        
        filtered = []
        for result in results:
            if result["session_id"] != session_id or result["score"] < self.settings.relevance_threshold:
                continue
            if not include_references and result.get("is_reference_section", False):
                continue
            
            if bias_first_pages:
                page_num = result.get("page", 1)
                if page_num <= 3:
                    result["score"] *= 1.2
            filtered.append(result)
            
        if bias_first_pages:
            filtered.sort(key=lambda x: x["score"], reverse=True)
            
        return filtered[:(top_k or self.settings.default_top_k)]

    def delete_session(self, session_id: str) -> None:
        """Delete one validated session's uploads, FAISS index, and metadata."""
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            raise KeyError(f"Unknown RAG session: {session_id}")
        self._retrievers.pop(session_id, None)
        shutil.rmtree(session_dir)

    def session_path(self, session_id: str) -> Path:
        """Return a validated session path for diagnostics and tests."""
        return self._session_dir(session_id)

    def session_summary(self, session_id: str) -> dict[str, object]:
        """Return safe, session-local status information for application views."""
        session_dir = self._session_dir(session_id)
        if not session_dir.is_dir():
            raise KeyError(f"Unknown RAG session: {session_id}")
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.is_file():
            return {"session_id": session_id, "documents": [], "chunk_count": 0, "ready": False}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in metadata.get("chunks", [])]
        if metadata.get("session_id") != session_id or any(chunk.session_id != session_id for chunk in chunks):
            raise ValueError("Session metadata does not match the requested session.")
        return {
            "session_id": session_id,
            "documents": sorted({chunk.document for chunk in chunks}),
            "chunk_count": len(chunks),
            "ready": bool(chunks),
        }

    def _current_chunks(self, session_id: str) -> list[Chunk]:
        if session_id in self._retrievers:
            return list(self._retrievers[session_id]._vector_store.chunks)
        index_path = self._session_dir(session_id) / "faiss.index"
        return list(self._load_retriever(session_id)._vector_store.chunks) if index_path.exists() else []

    def _persist(self, session_id: str, retriever: Retriever) -> None:
        session_dir = self._session_dir(session_id)
        vector_store = retriever._vector_store
        vector_store.save(session_dir / "faiss.index")
        metadata = {"session_id": session_id, "chunks": [asdict(chunk) for chunk in vector_store.chunks]}
        (session_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_retriever(self, session_id: str) -> Retriever:
        session_dir = self._session_dir(session_id)
        if not session_dir.is_dir():
            raise KeyError(f"Unknown RAG session: {session_id}")
        metadata_path, index_path = session_dir / "metadata.json", session_dir / "faiss.index"
        if not metadata_path.is_file() or not index_path.is_file():
            raise KeyError(f"Session has no searchable index: {session_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("session_id") != session_id:
            raise ValueError("Session metadata does not match the requested session.")
        chunks = [Chunk(**item) for item in metadata.get("chunks", [])]
        if any(chunk.session_id != session_id for chunk in chunks):
            raise ValueError("Session metadata contains chunks from another session.")
        retriever = Retriever(self._embedder, FaissVectorStore.load(index_path, chunks))
        self._retrievers[session_id] = retriever
        return retriever

    def _session_dir(self, session_id: str) -> Path:
        try:
            normalized = str(UUID(session_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("session_id must be a valid UUID.") from exc
        if normalized != session_id:
            raise ValueError("session_id must use canonical UUID form.")
        return self.settings.session_root / normalized
