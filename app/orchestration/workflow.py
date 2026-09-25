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
    research_agent: ResearchAgent = None,
    domain_guardrail: DomainGuardrail = None,
    evidence_guardrail: EvidenceGuardrail = None,
    code_gen_agent: CodeGenAgent = None,
    task_description: str = None
) -> tuple[ImplementationContext | None, GuardrailResult]:
    # Support backward compatibility for test_workflow.py
    if task is None and task_description is not None:
        task = task_description
    
    if not task:
        task = DEFAULT_PAPER_TASK

    if code_gen_agent is None:
        code_gen_agent = CodeGenAgent()
    """
    Executes the Domain and Evidence Guardrail verification.
    """
    paper_relevance, impl_support, reason = domain_guardrail.validate(session_id, task)
    
    if impl_support == "IMPLEMENTATION_UNSUPPORTED":
        # If paper isn't supported at all, terminal state can be DOMAIN_UNSUPPORTED
        term_state = GuardrailTerminalState.DOMAIN_UNSUPPORTED if paper_relevance == "UNSUPPORTED" else GuardrailTerminalState.IMPLEMENTATION_UNSUPPORTED
        
        return None, GuardrailResult(
            terminal_state=term_state,
            attempt_count=0,
            reason=reason,
            session_id=session_id,
            code_generation_allowed=False,
            task_description=task,
            paper_relevance=paper_relevance,
            implementation_support=impl_support
        )
    
    guardrail_result = evidence_guardrail.validate(
        session_id=session_id, 
        task=task,
        research_agent=research_agent
    )

    from dataclasses import replace
    if guardrail_result.terminal_state == GuardrailTerminalState.PASS:
        source_docs = list({
            req.evidence.source
            for req in guardrail_result.requirements.values()
            if req.evidence is not None
        })
        
        code_gen_prompt = code_gen_agent.generate_prompt_template(guardrail_result)
        code, sandbox_passed = code_gen_agent.generate_code(guardrail_result)
        guardrail_result = replace(guardrail_result, generated_code=code, sandbox_passed=sandbox_passed)
        
        context = ImplementationContext(
            session_id=session_id,
            paper_sources=source_docs,
            ml_domain=True,
            requirements=guardrail_result.requirements,
            code_gen_prompt=code_gen_prompt
        )
        return context, guardrail_result

    return None, guardrail_result

