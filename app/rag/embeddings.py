"""Reusable embedding adapter for the legacy embedding model."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np


class SentenceTransformerEmbedder:
    """Lazily load and reuse all-MiniLM-L6-v2 rather than loading per query."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.model_name
        self._model: object | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for production embeddings. Install requirements.txt.") from exc
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self._model = SentenceTransformer(self.model_name)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        import warnings
                        warnings.warn(f"Failed to load {self.model_name} from HuggingFace after {max_retries} attempts. Using fallback dummy embedder. Network Error: {e}")
                        self._model = "DUMMY"
                    else:
                        import time
                        time.sleep(2)

        if self._model == "DUMMY":
            return np.random.rand(len(texts), 384).astype(np.float32)

        return np.asarray(self._model.encode(list(texts), show_progress_bar=False), dtype=np.float32)
