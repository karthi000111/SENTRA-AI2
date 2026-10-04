"""Bounded corrective retrieval and reconsideration workflow."""
from __future__ import annotations

from app.agents.research_agent import ResearchAgent
from app.agents.llm_backend import call_hf_inference
from app.guardrails.domain_guardrail import DomainGuardrail
from app.guardrails.evidence_guardrail import EvidenceGuardrail
from app.guardrails.models import GuardrailResult, GuardrailTerminalState
from app.agents.code_agent import CodeAgent, CodeGenResult
from app.services.implementation_context import ImplementationContext

DEFAULT_PAPER_TASK = (
    "Extract all core architectural specifications, mathematical equations,"
    " constraint rules, and dataset definitions required for a complete"
    " reproduction of this paper, and generate a functional Python"
    " implementation."
)

# ------------------------------------------------------------------
# Phase 1: guardrail only (fast — no LLM code-gen call)
# ------------------------------------------------------------------
def run_research_to_spec_workflow(
    session_id: str,
    task: str = None,
    research_agent=None,
    domain_guardrail=None,
    evidence_guardrail=None,
    code_gen_agent=None,       # kept for backwards compat, not used here
    task_description: str = None,
):
    """Run the evidence guardrail and return validated specs.

    Code generation is NOT triggered here — call
    :func:`run_code_generation` separately when the user clicks
    "Implement Paper".
    """
    if research_agent is None:
        from app.rag.session_manager import SessionRAGManager
        research_agent = ResearchAgent(SessionRAGManager(), llm_callable=call_hf_inference)

    if evidence_guardrail is None:
        evidence_guardrail = EvidenceGuardrail(max_attempts=3)

    guardrail_result = evidence_guardrail.validate(session_id, task, research_agent)

    context = ImplementationContext(
        session_id=session_id,
        paper_sources=[],
        ml_domain=guardrail_result.code_generation_allowed,
        requirements=guardrail_result.requirements,
    )

    return context, guardrail_result


# ------------------------------------------------------------------
# Phase 2: code generation (calls real LLM — runs on demand)
# ------------------------------------------------------------------
def run_code_generation(
    session_id: str,
    guardrail_result: GuardrailResult,
    code_gen_agent: CodeAgent | None = None,
) -> tuple[ImplementationContext, GuardrailResult, CodeGenResult]:
    """Generate a paper implementation from a passing GuardrailResult.

    Returns the updated ``(ImplementationContext, GuardrailResult,
    CodeGenResult)`` triple.
    """
    if code_gen_agent is None:
        code_gen_agent = CodeAgent()   # uses call_llm → HF Inference

    gen_result = code_gen_agent.generate_implementation(guardrail_result)

    # Rebuild the guardrail result with the generated code attached
    updated_gr = GuardrailResult(
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
        generated_code=gen_result.generated_code,
        sandbox_passed=gen_result.success,
    )

    files_dict = {f.filename: f.code for f in gen_result.files} if gen_result.files else {}

    context = ImplementationContext(
        session_id=session_id,
        paper_sources=[],
        ml_domain=True,
        requirements=guardrail_result.requirements,
        code_gen_prompt=code_gen_agent.build_prompt(guardrail_result),
        generated_code=gen_result.generated_code,
        generated_files=files_dict,
        sandbox_passed=gen_result.success,
    )

    return context, updated_gr, gen_result

