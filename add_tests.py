import pathlib

p = pathlib.Path('tests/test_workflow.py')
t = p.read_text()

new_tests = """
def test_valid_paper_unsupported_implementation(evidence_guardrail, session_id):
    \"\"\"Test 2: Valid paper but incompatible neural-network request\"\"\"
    responses = {}
    
    # 1. Paper Relevance succeeds
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "Bayesian logic-based belief maintenance", [create_evidence("bayesian logic")], True
    )
    
    # 2. Implementation Support fails
    task = "functional PyTorch neural-network model with layers, optimizer, learning rate and loss"
    impl_query = f"Find methodology, architecture, or algorithm details for: {task}"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "Not found", [], False
    )
    
    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description=task,
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    from app.guardrails.models import PaperRelevance, ImplementationSupport
    assert gr.paper_relevance == PaperRelevance.SUPPORTED
    assert gr.implementation_support == ImplementationSupport.IMPLEMENTATION_UNSUPPORTED
    assert gr.terminal_state == GuardrailTerminalState.IMPLEMENTATION_UNSUPPORTED
    assert gr.code_generation_allowed is False


def test_no_keyword_rejection(evidence_guardrail, session_id):
    \"\"\"Test 5: No keyword-based rejection. Research document without typical ML keywords is supported.\"\"\"
    responses = {}
    
    # 1. Paper Relevance succeeds without ML keywords
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "A novel heuristic search algorithm", [create_evidence("heuristic search")], True
    )
    
    # 2. Implementation Support succeeds
    task = "heuristic search algorithm implementation"
    impl_query = f"Find methodology, architecture, or algorithm details for: {task}"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "heuristic search methodology", [create_evidence("heuristic search")], True
    )
    
    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    # Let EvidenceGuardrail fail on architecture, dataset etc. since it's just a test of DomainGuardrail
    # We will get UNRESOLVED, but crucially NOT DOMAIN_UNSUPPORTED or IMPLEMENTATION_UNSUPPORTED
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description=task,
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    from app.guardrails.models import PaperRelevance, ImplementationSupport
    assert gr.paper_relevance == PaperRelevance.SUPPORTED
    assert gr.implementation_support == ImplementationSupport.APPLICABLE
    assert gr.terminal_state == GuardrailTerminalState.UNRESOLVED

"""

p.write_text(t + new_tests)
