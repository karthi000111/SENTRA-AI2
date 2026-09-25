import pathlib

p = pathlib.Path('app/orchestration/workflow.py')
t = p.read_text()

if "CodeGenAgent" not in t:
    t = t.replace(
        "from app.guardrails.models import GuardrailResult, GuardrailTerminalState",
        "from app.guardrails.models import GuardrailResult, GuardrailTerminalState\nfrom app.agents.code_gen_agent import CodeGenAgent"
    )

    t = t.replace(
        """def run_research_to_spec_workflow(
    session_id: str,
    task_description: str,
    research_agent: ResearchAgent,
    domain_guardrail: DomainGuardrail,
    evidence_guardrail: EvidenceGuardrail
) -> tuple[ImplementationContext | None, GuardrailResult]:""",
        """def run_research_to_spec_workflow(
    session_id: str,
    task_description: str,
    research_agent: ResearchAgent,
    domain_guardrail: DomainGuardrail,
    evidence_guardrail: EvidenceGuardrail,
    code_gen_agent: CodeGenAgent = None
) -> tuple[ImplementationContext | None, GuardrailResult]:
    if code_gen_agent is None:
        code_gen_agent = CodeGenAgent()"""
    )
    
    t = t.replace(
        """        context = ImplementationContext(
            session_id=session_id,
            paper_sources=source_docs,
            ml_domain=True,
            requirements=guardrail_result.requirements
        )""",
        """        code_gen_prompt = code_gen_agent.generate_prompt_template(guardrail_result)
        
        context = ImplementationContext(
            session_id=session_id,
            paper_sources=source_docs,
            ml_domain=True,
            requirements=guardrail_result.requirements,
            code_gen_prompt=code_gen_prompt
        )"""
    )

p.write_text(t)
