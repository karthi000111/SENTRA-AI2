import pytest
from app.guardrails.evidence_guardrail import EvidenceGuardrail
from app.guardrails.models import EvidenceState, GuardrailTerminalState, RequirementClassification, AIParadigm
from tests.test_workflow import create_mock_research_agent, create_evidence
from app.agents.research_agent import ResearchResult

def test_pass_first_logic(tmp_path):
    # Setup
    evidence_guardrail = EvidenceGuardrail(max_attempts=3)
    session_id = "test-session-pass-first"
    task_description = "A deep learning test task"

    # We mock the research agent so that core fields (network_architecture, layers) 
    # are found immediately. We do NOT provide any evidence for optional fields.
    responses = {}
    
    # We only need to provide evidence for critical fields
    critical_fields = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING)
    for req in critical_fields:
        query = evidence_guardrail._get_query_for_attempt(req, 1)
        ev_text = f"{req.replace('_', ' ')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"
        responses[query] = ResearchResult(
            session_id, query, ev_text, [create_evidence(ev_text)], True
        )

    # For any other queries, we return not found
    class PassFirstMockAgent:
        def __init__(self, responses):
            self.responses = responses
            self.call_count = 0
            
        def run(self, session_id, query, top_k=3):
            self.call_count += 1
            if query in self.responses:
                return self.responses[query]
            return ResearchResult(session_id, query, "Not found", [], False)
            
        def _keywords(self, q): return set()
        def _best_sentence(self, t, k): return t

    agent = PassFirstMockAgent(responses)

    # Execute validate
    result = evidence_guardrail.validate(session_id, task_description, agent)

    # Verify Pass 1 auto-passed without targeted queries for optional fields
    assert result.terminal_state == GuardrailTerminalState.PASS
    assert result.attempt_count == 1
    
    # Verify optional fields automatically defaulted
    opt_fields = evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)
    for opt in opt_fields:
        assert result.requirements[opt].state == EvidenceState.SUPPORTED
        assert result.requirements[opt].classification == RequirementClassification.IMPLEMENTATION_CHOICE

def test_targeted_doubts(tmp_path):
    # Setup
    evidence_guardrail = EvidenceGuardrail(max_attempts=3)
    session_id = "test-session-targeted"
    task_description = "A deep learning test task"

    responses = {}
    
    critical_fields = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING)
    target_field = critical_fields[0] # Let's say network_architecture is missing on attempt 1
    
    for req in critical_fields:
        query1 = evidence_guardrail._get_query_for_attempt(req, 1)
        query2 = evidence_guardrail._get_query_for_attempt(req, 2)
        
        if req == target_field:
            # Missing in attempt 1, found in attempt 2
            responses[query1] = ResearchResult(session_id, query1, "Not found", [], False)
            ev_text = f"{req.replace('_', ' ')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule in attempt 2"
            responses[query2] = ResearchResult(session_id, query2, ev_text, [create_evidence(ev_text)], True)
        else:
            # Found immediately in attempt 1
            ev_text = f"{req.replace('_', ' ')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"
            responses[query1] = ResearchResult(session_id, query1, ev_text, [create_evidence(ev_text)], True)

    class TargetedMockAgent:
        def __init__(self, responses):
            self.responses = responses
            
        def run(self, session_id, query, top_k=3):
            if query in self.responses:
                return self.responses[query]
            return ResearchResult(session_id, query, "Not found", [], False)
            
        def _keywords(self, q): return set()
        def _best_sentence(self, t, k): return t

    agent = TargetedMockAgent(responses)

    # Execute validate
    result = evidence_guardrail.validate(session_id, task_description, agent)

    # Verify Pass 2 resolved it
    assert result.terminal_state == GuardrailTerminalState.PASS
    assert result.attempt_count == 2
    assert result.requirements[target_field].state == EvidenceState.SUPPORTED
