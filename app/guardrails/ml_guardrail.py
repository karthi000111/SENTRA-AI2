"""
Simplified ML guardrail for Sentra AI.
This module verifies that an uploaded paper contains concrete, evidence-backed
detail to be implemented as an ML model or algorithmic system.
"""
from __future__ import annotations

import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class GuardrailResult(BaseModel):
    code_generation_allowed: bool
    blocking_reason: str | None = None
    audit_summary: str = ""

def verify_paper_evidence(retrieved_chunks: list[str], session_id: str) -> GuardrailResult:
    from app.rag.session_manager import SessionRAGManager
    from app.agents.research_agent import ResearchAgent
    
    agent = ResearchAgent(SessionRAGManager())
    
    # Check A (Relevance)
    stage1_q = "Is this paper's main contribution a computational method, algorithm, model, or system that could be implemented as code? Answer only YES or NO."
    res1 = agent.run(session_id=session_id, query=stage1_q)
    ans1_upper = res1.answer.upper().strip()
    passed_stage1 = res1.sufficient_evidence and ans1_upper.startswith("YES")
    
    if not passed_stage1:
        return GuardrailResult(
            code_generation_allowed=False,
            blocking_reason="Not an ML/AI or computational method paper.",
            audit_summary=f"Relevance Check: {res1.answer}"
        )
        
    # Check B (Method Content)
    stage2_q = "Does this paper contain enough concrete technical details (like equations, algorithms, architecture diagrams, or pseudo-code) to write a basic software implementation? Answer only YES or NO."
    res2 = agent.run(session_id=session_id, query=stage2_q)
    ans2_upper = res2.answer.upper().strip()
    passed_stage2 = res2.sufficient_evidence and ans2_upper.startswith("YES")
    
    if not passed_stage2:
        return GuardrailResult(
            code_generation_allowed=False,
            blocking_reason="Paper lacks sufficient technical details (equations/algorithms) for implementation.",
            audit_summary=f"Content Check: {res2.answer}"
        )

    return GuardrailResult(
        code_generation_allowed=True,
        blocking_reason=None,
        audit_summary="Paper verified: Contains implementable computational methods and sufficient technical details."
    )
