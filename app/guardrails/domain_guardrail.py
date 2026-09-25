"""Domain Guardrail to verify if a paper is relevant to Machine Learning."""
from __future__ import annotations

from app.agents.research_agent import ResearchAgent

class DomainGuardrail:
    """Verifies that the document actually pertains to machine learning research."""
    
    def __init__(self, research_agent: ResearchAgent):
        self.research_agent = research_agent
        
    def validate(self, session_id: str, task: str) -> tuple['PaperRelevance', 'ImplementationSupport', str]:
        """Determines if the document is a valid research paper and supports the requested task."""
        from app.guardrails.models import PaperRelevance, ImplementationSupport
        
        # 1. Check if it's a valid research document
        relevance_queries = [
            "What is the core methodology, model, or algorithm proposed in this research paper?",
            "What research approach or scientific methodology is discussed in this paper?"
        ]
        
        is_relevant = False
        for q in relevance_queries:
            result = self.research_agent.run(session_id=session_id, query=q)
            if result.sufficient_evidence:
                is_relevant = True
                break
                
        if not is_relevant:
            return PaperRelevance.UNSUPPORTED, ImplementationSupport.IMPLEMENTATION_UNSUPPORTED, "The document does not appear to be a supported research paper."
            
        # 2. Check if the requested implementation is supported by the paper
        impl_query = f"Find methodology, architecture, or algorithm details for: {task}"
        impl_result = self.research_agent.run(session_id=session_id, query=impl_query)
        
        if not impl_result.sufficient_evidence:
            return PaperRelevance.SUPPORTED, ImplementationSupport.IMPLEMENTATION_UNSUPPORTED, f"The requested implementation is not supported by the paper."
            
        return PaperRelevance.SUPPORTED, ImplementationSupport.APPLICABLE, "The paper supports the requested implementation."
