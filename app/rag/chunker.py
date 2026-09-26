"""Page-aware overlapping word chunking."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from .models import Chunk, ExtractedPage


class PageChunker:
    """Create 500-word chunks with 50-word overlap, retaining page boundaries."""
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size.")
        self.chunk_size, self.overlap = chunk_size, overlap

    def chunk_pages(self, pages: Iterable[ExtractedPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            words = page.text.split()
            stem = Path(page.document).stem.replace(" ", "_")
            for number, start in enumerate(range(0, len(words), self.chunk_size - self.overlap), start=1):
                text = " ".join(words[start:start + self.chunk_size])
                if text:
                    chunks.append(Chunk(page.document, page.page, f"{stem}_p{page.page}_c{number:02d}", text,
                                        document_id=page.document_id,
                                        is_reference_section=page.is_reference_section))
        return chunks
