import pathlib

p = pathlib.Path('tests/test_workflow.py')
t = p.read_text()

if "AIParadigm" not in t:
    t = t.replace(
        "from app.guardrails.models import GuardrailResult, GuardrailTerminalState, EvidenceState, RequirementClassification, Requirement",
        "from app.guardrails.models import GuardrailResult, GuardrailTerminalState, EvidenceState, RequirementClassification, Requirement, AIParadigm"
    )

t = t.replace(
    "expected_reqs = evidence_guardrail._get_required_fields_for_task(\"Task\")",
    "expected_reqs = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING) + evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)"
)

t = t.replace(
    "expected_reqs = evidence_guardrail._get_required_fields_for_task(task)",
    "expected_reqs = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING) + evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)"
)

new_mock = """def create_mock_research_agent(responses: dict):
    agent = Mock(spec=ResearchAgent)
    
    def side_effect(session_id: str, query: str, top_k: int = 5):
        if "determine the AI paradigm" in query:
            return ResearchResult(session_id, query, "neural network deep learning", [create_evidence("neural network machine learning model")], True)
        # Allow exact match, or fallback to default
        if query in responses:
            return responses[query]
        # Return fallback missing
        return ResearchResult(
            session_id=session_id,
            query=query,
            answer="No sufficient evidence.",
            evidence=[],
            sufficient_evidence=False
        )"""

t = t.replace("""def create_mock_research_agent(responses: dict):
    agent = Mock(spec=ResearchAgent)
    
    def side_effect(session_id: str, query: str, top_k: int = 5):
        # Allow exact match, or fallback to default
        if query in responses:
            return responses[query]
        # Return fallback missing
        return ResearchResult(
            session_id=session_id,
            query=query,
            answer="No sufficient evidence.",
            evidence=[],
            sufficient_evidence=False
        )""", new_mock)

p.write_text(t)
