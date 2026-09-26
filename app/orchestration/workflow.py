"""Bounded corrective retrieval and reconsideration workflow."""
from __future__ import annotations

from app.agents.research_agent import ResearchAgent
from app.guardrails.domain_guardrail import DomainGuardrail
from app.guardrails.evidence_guardrail import EvidenceGuardrail
from app.guardrails.models import GuardrailResult, GuardrailTerminalState
from app.agents.code_gen_agent import CodeGenAgent
from app.services.implementation_context import ImplementationContext

DEFAULT_PAPER_TASK = (
    "Extract all core architectural specifications, mathematical equations,"
    " constraint rules, and dataset definitions required for a complete"
    " reproduction of this paper, and generate a functional Python"
    " implementation."
)

def run_research_to_spec_workflow(
    session_id: str,
    task: str = None,
    research_agent = None,
    domain_guardrail = None,
    evidence_guardrail = None,
    code_gen_agent = None,
    task_description: str = None
):
    from app.guardrails.ml_guardrail import verify_paper_evidence
    from app.rag.session_manager import SessionRAGManager

    if task is None and task_description is not None:
        task = task_description
    if not task:
        task = DEFAULT_PAPER_TASK

    manager = SessionRAGManager()
    
    try:
        # Retrieve an initial broad set of chunks for the guardrail to evaluate
        initial_chunks = manager.retrieve(
            session_id, 
            "architecture components model math equations layers training loss optimizer dataset format", 
            top_k=15
        )
        retrieved_texts = [str(c.get("text", "")) for c in initial_chunks]
    except Exception:
        retrieved_texts = []

    guardrail_result = verify_paper_evidence(retrieved_texts, session_id)
    
    return None, guardrail_result

