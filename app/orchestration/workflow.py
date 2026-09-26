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
    if research_agent is None:
        from app.rag.session_manager import SessionRAGManager
        research_agent = ResearchAgent(SessionRAGManager())
        
    if evidence_guardrail is None:
        evidence_guardrail = EvidenceGuardrail(max_attempts=3)
        
    if code_gen_agent is None:
        code_gen_agent = CodeGenAgent()
        
    guardrail_result = evidence_guardrail.validate(session_id, task, research_agent)
    
    context = None
    if guardrail_result.code_generation_allowed:
        code, passed = code_gen_agent.generate_code(guardrail_result)
        # Update guardrail result with code
        guardrail_result = GuardrailResult(
            terminal_state=guardrail_result.terminal_state,
            attempt_count=guardrail_result.attempt_count,
            requirements=guardrail_result.requirements,
            reason=guardrail_result.reason,
            session_id=guardrail_result.session_id,
            code_generation_allowed=guardrail_result.code_generation_allowed,
            task_description=guardrail_result.task_description,
            paper_relevance=guardrail_result.paper_relevance,
            implementation_support=guardrail_result.implementation_support,
            detected_paradigm=guardrail_result.detected_paradigm,
            generated_code=code,
            sandbox_passed=passed
        )
        context = ImplementationContext(
            session_id=session_id,
            paper_sources=[],
            ml_domain=True,
            requirements=guardrail_result.requirements,
            code_gen_prompt=code_gen_agent.generate_prompt_template(guardrail_result),
            generated_code=code,
            sandbox_passed=passed
        )
    else:
        context = ImplementationContext(
            session_id=session_id,
            paper_sources=[],
            ml_domain=False,
            requirements=guardrail_result.requirements
        )
    
    return context, guardrail_result

