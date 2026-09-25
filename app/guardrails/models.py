"""Data models for Guardrail components."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from app.agents.models import EvidenceLink

class AIParadigm(str, Enum):
    DEEP_LEARNING = "DEEP_LEARNING"
    SYMBOLIC_PROBABILISTIC = "SYMBOLIC_PROBABILISTIC"
    CLASSICAL_ML = "CLASSICAL_ML"
    CONTROL_INDUSTRIAL = "CONTROL_INDUSTRIAL"

class RequirementClassification(str, Enum):
    PAPER_SUPPORTED = "PAPER_SUPPORTED"
    IMPLEMENTATION_CHOICE = "IMPLEMENTATION_CHOICE"

class EvidenceState(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_FOUND = "NOT_FOUND"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTING = "CONFLICTING"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    NOT_APPLICABLE_TO_PARADIGM = "NOT_APPLICABLE_TO_PARADIGM"

class GuardrailTerminalState(str, Enum):
    PASS = "PASS"
    DOMAIN_UNSUPPORTED = "DOMAIN_UNSUPPORTED"
    IMPLEMENTATION_UNSUPPORTED = "IMPLEMENTATION_UNSUPPORTED"
    UNRESOLVED = "UNRESOLVED"

class PaperRelevance(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"

class ImplementationSupport(str, Enum):
    SUPPORTED = "SUPPORTED"
    IMPLEMENTATION_UNSUPPORTED = "IMPLEMENTATION_UNSUPPORTED"
    APPLICABLE = "APPLICABLE"

@dataclass(frozen=True)
class ContextInfo:
    document_id: str | None = None
    dataset: str | None = None
    task: str | None = None
    model: str | None = None
    experiment: str | None = None
    section: str | None = None

@dataclass(frozen=True)
class Requirement:
    name: str
    value: str | None = None
    classification: RequirementClassification = RequirementClassification.PAPER_SUPPORTED
    state: EvidenceState = EvidenceState.NOT_FOUND
    evidence: EvidenceLink | None = None
    reason: str | None = None
    context: ContextInfo | None = None

@dataclass(frozen=True)
class GuardrailResult:
    terminal_state: GuardrailTerminalState
    attempt_count: int
    requirements: dict[str, Requirement] = field(default_factory=dict)
    reason: str | None = None
    session_id: str | None = None
    code_generation_allowed: bool = False
    task_description: str | None = None
    paper_relevance: PaperRelevance = PaperRelevance.SUPPORTED
    implementation_support: ImplementationSupport = ImplementationSupport.SUPPORTED
    detected_paradigm: AIParadigm | None = None
    generated_code: str | None = None
    sandbox_passed: bool = False
    
    @property
    def completeness_score(self) -> float:
        if self.total_required_fields == 0:
            return 0.0
        return self.supported_required_fields / self.total_required_fields
        
    @property
    def supported_required_fields(self) -> int:
        reqs = [r for r in self.requirements.values() if r.classification == RequirementClassification.PAPER_SUPPORTED]
        return sum(1 for r in reqs if r.state in (EvidenceState.SUPPORTED, EvidenceState.NOT_APPLICABLE_TO_PARADIGM))
        
    @property
    def total_required_fields(self) -> int:
        reqs = [r for r in self.requirements.values() if r.classification == RequirementClassification.PAPER_SUPPORTED]
        return len(reqs)
        
    @property
    def paper_supported_count(self) -> int:
        return sum(1 for r in self.requirements.values() if r.classification == RequirementClassification.PAPER_SUPPORTED)
        
    @property
    def implementation_choice_count(self) -> int:
        return sum(1 for r in self.requirements.values() if r.classification == RequirementClassification.IMPLEMENTATION_CHOICE)

    @property
    def verified_context(self) -> dict[str, Any]:
        from typing import Any
        specs = {}
        for k, v in self.requirements.items():
            status = "NOT_APPLICABLE" if v.state == EvidenceState.NOT_APPLICABLE_TO_PARADIGM else v.state.value
            specs[k] = {
                "value": v.value,
                "status": status,
                "citation": v.evidence.page if v.evidence else None
            }
        return specs

    @property
    def grounded_specs(self) -> dict[str, Any]:
        """Return only active, non-empty, paradigm-relevant specifications."""
        from typing import Any
        if not self.verified_context:
            return {}

        filtered = {}

        for key, value in self.verified_context.items():
            # Support both raw strings and dicts for robustness
            val_to_check = value.get("value") if isinstance(value, dict) else value
            
            # Ignore empty/placeholders
            if val_to_check in [
                None,
                "",
                "None",
                "..........",
                "NOT_FOUND",
                "N/A",
            ]:
                continue

            # Ignore explicitly non-applicable specifications
            if isinstance(value, dict) and value.get("status") == "NOT_APPLICABLE":
                continue

            filtered[key] = value

        return filtered

