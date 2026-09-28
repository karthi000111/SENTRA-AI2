"""Shared evidence model used by the Research Agent, Evidence Guardrail, and Code Agent."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceLink:
    source: str
    page: int
    chunk_id: str
    evidence_text: str