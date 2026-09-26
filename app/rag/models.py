"""Data structures used by the RAG evidence layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExtractedPage:
    document: str
    page: int
    text: str
    document_id: str = ""
    is_reference_section: bool = False


@dataclass(frozen=True)
class Chunk:
    document: str
    page: int
    chunk_id: str
    text: str
    session_id: str = ""
    document_id: str = ""
    is_reference_section: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    """FAISS IndexFlatL2 result; lower distance means closer evidence."""
    document: str
    page: int
    chunk_id: str
    text: str
    distance: float
    session_id: str = ""
    document_id: str = ""
    is_reference_section: bool = False

    def as_dict(self) -> dict[str, object]:
        score = 1.0 / (1.0 + self.distance)
        return {"document": self.document, "filename": self.document, "source": self.document, "page": self.page,
                "chunk_id": self.chunk_id, "text": self.text, "distance": self.distance,
                "score": score,
                "session_id": self.session_id, "document_id": self.document_id,
                "is_reference_section": self.is_reference_section}


@dataclass(frozen=True)
class ExtractionProblem:
    source: str
    message: str


@dataclass
class ExtractionReport:
    pages: list[ExtractedPage] = field(default_factory=list)
    problems: list[ExtractionProblem] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return bool(self.pages)


PDFInput = str | Path
