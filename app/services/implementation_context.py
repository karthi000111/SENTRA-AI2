"""Implementation context produced after successful guardrail validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from app.guardrails.models import Requirement

@dataclass(frozen=True)
class ImplementationContext:
    session_id: str
    paper_sources: list[str]
    ml_domain: bool
    requirements: dict[str, Requirement] = field(default_factory=dict)
    code_gen_prompt: str | None = None
    generated_code: str | None = None
    generated_files: dict[str, str] = field(default_factory=dict)
    sandbox_passed: bool = False
