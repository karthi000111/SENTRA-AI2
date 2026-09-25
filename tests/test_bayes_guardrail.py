from __future__ import annotations
import pytest
from app.guardrails.evidence_guardrail import EvidenceGuardrail
from app.guardrails.models import (
    GuardrailTerminalState, EvidenceState, AIParadigm
)
from app.agents.research_agent import ResearchResult
from app.agents.code_gen_agent import CodeGenAgent

class MockResearchAgent:
    def __init__(self, responses):
        self.responses = responses
        
    def run(self, session_id, query, top_k=5):
        for key, result in self.responses.items():
            if key in query.lower():
                return result
        return ResearchResult(session_id, query, "Not found", [], False)
        
    def _keywords(self, text):
        return ["mock"]
        
    def _best_sentence(self, text, kw):
        return text

def create_evidence(text: str) -> dict:
    return {
        "source": "Belief Maintenance in Bayesian Networks.pdf",
        "page": 1,
        "chunk_id": "chunk_1",
        "text": text,
        "score": 0.95
    }

def test_bayes_guardrail_paradigm():
    session_id = "test_bayes"
    task = "Implement a logic-based belief maintenance system"
    
    # We need to mock the paradigm detection query and the critical fields
    responses = {
        "paradigm": ResearchResult(
            session_id, "paradigm", "bayesian belief maintenance", 
            [create_evidence("This paper describes a logic-based belief maintenance system in Bayesian networks.")], True
        ),
        "proposition representation": ResearchResult(
            session_id, "proposition", "proposition logic", 
            [create_evidence("proposition representation using logical clauses")], True
        ),
        "probability intervals": ResearchResult(
            session_id, "intervals", "intervals", 
            [create_evidence("probability intervals are bounded by lower and upper bounds")], True
        ),
        "constraint rules": ResearchResult(
            session_id, "rules", "constraint rules", 
            [create_evidence("constraint rules enforce consistency")], True
        ),
        "propagation engine": ResearchResult(
            session_id, "engine", "propagation engine", 
            [create_evidence("propagation engine updates the intervals")], True
        )
    }
    
    agent = MockResearchAgent(responses)
    guardrail = EvidenceGuardrail(max_attempts=1)
    
    result = guardrail.validate(session_id, task, agent)
    
    # 1. Correctly classifies the paper as SYMBOLIC_PROBABILISTIC
    assert result.detected_paradigm == AIParadigm.SYMBOLIC_PROBABILISTIC
    
    # 2. Marks loss_function, optimizer, and learning_rate as NOT_APPLICABLE_TO_PARADIGM
    assert result.requirements["loss_function"].state == EvidenceState.NOT_APPLICABLE_TO_PARADIGM
    assert result.requirements["optimizer"].state == EvidenceState.NOT_APPLICABLE_TO_PARADIGM
    assert result.requirements["learning_rate"].state == EvidenceState.NOT_APPLICABLE_TO_PARADIGM
    
    # 3. Achieves a PASS status instead of UNRESOLVED
    assert result.terminal_state == GuardrailTerminalState.PASS
    assert result.code_generation_allowed is True
    
    # 4. Successfully generates functional Python code template/prompt for IBN/LBMS
    codegen = CodeGenAgent()
    prompt = codegen.generate_prompt_template(result)
    assert "Symbolic AI" in prompt
    assert "logic proposition managers" in prompt
    assert "Do NOT force PyTorch nn.Module skeletons" in prompt
