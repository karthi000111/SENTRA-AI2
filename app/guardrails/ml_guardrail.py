"""
Simpler, ML-only guardrail for Sentra AI.
This module verifies that an uploaded paper contains concrete, evidence-backed
detail to be implemented as an ML model or algorithmic system.
"""
from __future__ import annotations

import re
import logging
from typing import Literal
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class FieldEvidence(BaseModel):
    field: str
    status: Literal["SUPPORTED", "INFERRED", "MISSING"]
    value: str | None = None
    source_page: int | None = None
    reasoning: str | None = None  # required if status == INFERRED

class GuardrailResult(BaseModel):
    fields: list[FieldEvidence]
    code_generation_allowed: bool
    blocking_reason: str | None = None
    warnings: list[str] = []
    audit_summary: str = ""

def retrieve(session_id: str, query: str) -> list[str]:
    """
    RAG retrieval function using the actual SessionRAGManager.
    """
    from app.rag.session_manager import SessionRAGManager
    manager = SessionRAGManager()
    try:
        results = manager.retrieve(session_id, query, top_k=5)
        return [str(res.get("text", "")) for res in results]
    except Exception:
        return []

CORE_SIGNALS = [
    "equations_or_formulas",
    "algorithm_or_architecture_description",
    "parameters_or_hyperparameters",
    "training_or_optimization_procedure",
    "dataset_or_example_input"
]

def _evaluate_field(field: str, text: str) -> FieldEvidence | None:
    text_lower = text.lower()
    
    if field == "equations_or_formulas":
        if re.search(r"\b(equation|formula|math|activation|softmax|relu|theorem|chain rule|probability)\b", text_lower):
            return FieldEvidence(field=field, status="SUPPORTED", value="Found equations or formulas")
            
    elif field == "algorithm_or_architecture_description":
        if re.search(r"\b(encoder|decoder|attention|conv layer|convolutional|transformer|resnet|lstm|architecture|step|procedure|pseudocode|algorithm|loop|propagate|propagation)\b", text_lower):
            return FieldEvidence(field=field, status="SUPPORTED", value="Found algorithm or architecture description")
            
    elif field == "parameters_or_hyperparameters":
        if re.search(r"\b(dimension|depth|width|heads|dropout|learning rate|batch size|parameter|threshold|alpha|beta|weight|probability|constraint)\b", text_lower):
            return FieldEvidence(field=field, status="SUPPORTED", value="Found parameters or hyperparameters")
            
    elif field == "training_or_optimization_procedure":
        if re.search(r"\b(loss|objective function|cross-entropy|mse|mae|maximize|minimize|optimization|training|gradient)\b", text_lower):
            return FieldEvidence(field=field, status="SUPPORTED", value="Found training or optimization procedure")
            
    elif field == "dataset_or_example_input":
        if re.search(r"\b(dataset|input format|corpus|cifar|imagenet|wmt|data|example|table \d+|figure \d+|case study|demonstration|results)\b", text_lower):
            return FieldEvidence(field=field, status="SUPPORTED", value="Found dataset or example input")

    return None

def verify_paper_evidence(retrieved_chunks: list[str], session_id: str) -> GuardrailResult:
    from app.rag.session_manager import SessionRAGManager
    from app.agents.research_agent import ResearchAgent
    
    agent = ResearchAgent(SessionRAGManager())
    stage1_q = "Is this paper's main contribution a computational method, algorithm, model, or system that could be implemented as code? Answer only YES or NO, with a one-sentence reason."
    res = agent.run(session_id=session_id, query=stage1_q)
    
    ans_upper = res.answer.upper().strip()
    passed_stage1 = res.sufficient_evidence and not ans_upper.startswith("NO")
    
    if not passed_stage1:
        return GuardrailResult(
            fields=[],
            code_generation_allowed=False,
            blocking_reason="Not an ML/AI paper",
            warnings=[],
            audit_summary=f"Stage 1 Classification: {res.answer}"
        )
        
    combined_text = " ".join(retrieved_chunks)
    
    fields_evidence: list[FieldEvidence] = []
    
    for field in CORE_SIGNALS:
        evidence = _evaluate_field(field, combined_text)
        if evidence:
            fields_evidence.append(evidence)
        else:
            fields_evidence.append(FieldEvidence(field=field, status="MISSING", value="Not present in paper — will use a documented default/TODO in generated code."))

    # Calculate score
    found_signals = [ev.field for ev in fields_evidence if ev.status in ["SUPPORTED", "INFERRED"]]
    relevant_found = len(found_signals)
    total_required = len(CORE_SIGNALS)
    
    # 2 out of 5 required
    passed = relevant_found >= 2
    
    # Debug log as requested by verification
    print(f"DEBUG GUARDRAIL: Found {relevant_found}/{total_required} signals. Matched signals: {found_signals}")
    
    status_str = "PASS" if passed else "FAIL"
    audit_summary = f"[VERSION: v2-FLAT] ML evidence: {relevant_found}/{total_required} signals found | Status: {status_str}"

    warnings = []
    
    for ev in fields_evidence:
        if ev.status == "MISSING":
            warnings.append(
                f"Field '{ev.field}' was MISSING. The generated code MUST include a clearly "
                f"marked TODO/comment stating this was not found and substitute a reasonable default."
            )

    blocking_reason = None
    if not passed:
        blocking_reason = "insufficient technical content for code generation"

    return GuardrailResult(
        fields=fields_evidence,
        code_generation_allowed=passed,
        blocking_reason=blocking_reason,
        warnings=warnings,
        audit_summary=audit_summary
    )
