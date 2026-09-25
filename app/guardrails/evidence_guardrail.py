"""Evidence Guardrail to ensure sufficient implementation details."""
from __future__ import annotations

import re
from app.agents.research_agent import ResearchAgent
from app.agents.models import EvidenceLink
from app.guardrails.models import (
    Requirement, RequirementClassification, GuardrailResult, 
    EvidenceState, GuardrailTerminalState, ContextInfo, AIParadigm
)

class EvidenceGuardrail:
    """Validates that a paper contains sufficient evidence to implement a task."""
    
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def _detect_paradigm(self, session_id: str, research_agent: ResearchAgent) -> AIParadigm:
        query = "Extract keywords, abstract summary, and core methodology to determine the AI paradigm."
        result = research_agent.run(session_id=session_id, query=query, top_k=5)
        text = " ".join(str(ev.get("text", "")).lower() for ev in (result.evidence or []))
        
        symbolic_terms = ["bayesian", "belief maintenance", "clause", "probability interval", "boolean constraint propagation", "truth maintenance"]
        if any(term in text for term in symbolic_terms):
            return AIParadigm.SYMBOLIC_PROBABILISTIC
            
        classical_terms = ["clustering", "decision tree", "svm", "support vector", "random forest"]
        if any(term in text for term in classical_terms):
            return AIParadigm.CLASSICAL_ML
            
        control_terms = ["iec 61131", "state machine", "automation logic", "plc"]
        if any(term in text for term in control_terms):
            return AIParadigm.CONTROL_INDUSTRIAL
            
        return AIParadigm.DEEP_LEARNING

    def _get_critical_fields(self, paradigm: AIParadigm) -> list[str]:
        if paradigm == AIParadigm.DEEP_LEARNING:
            return ["architecture_layers", "loss_function", "model_dimensions"]
        elif paradigm == AIParadigm.SYMBOLIC_PROBABILISTIC:
            return ["proposition_representation", "probability_intervals", "constraint_rules", "propagation_engine"]
        elif paradigm == AIParadigm.CLASSICAL_ML:
            return ["algorithm_type", "feature_dimensions", "objective_or_distance_metric"]
        return ["architecture", "dataset", "target", "loss_function"]

    def _get_optional_fields(self, paradigm: AIParadigm) -> list[str]:
        if paradigm == AIParadigm.DEEP_LEARNING:
            return ["optimizer", "learning_rate", "batch_size"]
        # Symbolic, Classical, Control do not typically use these exact NN hyperparameters
        # in the same way, but they could have others. We'll leave them empty for now.
        return []
        
    def _get_all_possible_fields(self) -> list[str]:
        return [
            "architecture_layers", "loss_function", "model_dimensions",
            "proposition_representation", "probability_intervals", "constraint_rules", "propagation_engine",
            "algorithm_type", "feature_dimensions", "objective_or_distance_metric",
            "architecture", "dataset", "target",
            "optimizer", "learning_rate", "batch_size"
        ]

    def _get_query_for_attempt(self, req_name: str, attempt: int) -> str:
        clean_name = req_name.replace('_', ' ')
        if attempt == 1:
            return f"What {clean_name} is explicitly reported or specified in the paper?"
        elif attempt == 2:
            return f"Find specific implementation details or methodology about the {clean_name} used in the paper."
        else:
            return f"Search for exact hyperparameters, setup, or {clean_name} configuration in the text."

    def _normalize_value(self, req_name: str, text: str) -> str | None:
        text_lower = text.lower()
        if req_name == "optimizer":
            if "sgd" in text_lower or "stochastic gradient descent" in text_lower:
                return "SGD"
            if "adam" in text_lower:
                return "Adam"
            if "rmsprop" in text_lower:
                return "RMSprop"
            return None
        return text

    def _explicitly_supports(self, req_name: str, text: str) -> bool:
        """Requirement-level validation to ensure the chunk actually supports the field."""
        text = text.lower()
        if req_name in ["architecture", "architecture_layers", "algorithm", "algorithm_type"]:
            return bool(re.search(r"\b(architecture|layer|layers|convolutional|conv|residual|network|transformer|block|hidden|algorithm|method)\b", text))
        elif req_name in ["dataset", "features", "target", "feature_dimensions"]:
            return bool(re.search(r"\b(dataset|datasets|corpus|cifar|cifar-10|imagenet|mnist|coco|data|training set|test set|features|target|labels|dimension|dimensions)\b", text))
        elif req_name == "loss_function":
            if "f(x) + x" in text and not re.search(r"\b(cross.?entropy|mse|negative log likelihood|nll|loss)\b", text):
                return False
            return bool(re.search(r"\b(cross.?entropy|mse|negative log likelihood|nll|loss|objective)\b", text))
        elif req_name == "optimizer":
            return bool(re.search(r"\b(adam|sgd|stochastic gradient descent|rmsprop|adagrad)\b", text))
        elif req_name == "learning_rate":
            return bool(re.search(r"\b(learning.?rate|lr)\b", text))
        elif req_name == "batch_size":
            return bool(re.search(r"\b(batch.?size|mini.?batch)\b", text))
        elif req_name == "num_clusters":
            return bool(re.search(r"\b(clusters|k\s*=)\b", text))
        elif req_name in ["distance_metric", "objective_or_distance_metric"]:
            return bool(re.search(r"\b(distance|euclidean|manhattan|cosine|objective|metric)\b", text))
        elif req_name in ["proposition_representation", "probability_intervals", "constraint_rules", "propagation_engine", "model_dimensions"]:
            # Broader match for symbolic topics since it varies wildly
            return bool(re.search(r"\b(proposition|probability|interval|constraint|rule|propagation|engine|dimension|bayesian|logic)\b", text))
        return False

    def _attempt_resolve(self, req_name: str, requirements: dict, session_id: str, task: str, research_agent: ResearchAgent, attempt: int):
        from app.guardrails.models import Requirement, RequirementClassification, EvidenceState, ContextInfo
        from app.agents.models import EvidenceLink
        
        query = self._get_query_for_attempt(req_name, attempt)
        result = research_agent.run(session_id=session_id, query=query, top_k=5)
        
        if result.sufficient_evidence and result.evidence:
            supporting_evidences = []
            for ev in result.evidence:
                if self._explicitly_supports(req_name, str(ev.get("text", ""))):
                    supporting_evidences.append(ev)
            
            if supporting_evidences:
                best_ev = supporting_evidences[0]
                
                if best_ev:
                    ev_link = EvidenceLink(
                        source=str(best_ev.get("source", best_ev.get("filename", ""))),
                        page=int(best_ev.get("page", 0)),
                        chunk_id=str(best_ev.get("chunk_id", "")),
                        evidence_text=str(best_ev.get("text", ""))
                    )
                    
                    val = self._normalize_value(req_name, str(best_ev.get("text", "")))
                    if val is None:
                        val = str(best_ev.get("text", ""))
                        
                    keywords = research_agent._keywords(query)
                    excerpt = research_agent._best_sentence(str(best_ev["text"]), keywords)
                    answer = f"{val} (Based on {best_ev['source']}, page {best_ev['page']}: {excerpt})"
                    
                    ctx = ContextInfo(
                        document_id=ev_link.source,
                        experiment=task
                    )
                    
                    requirements[req_name] = Requirement(
                        name=req_name,
                        value=answer,
                        classification=RequirementClassification.PAPER_SUPPORTED,
                        state=EvidenceState.SUPPORTED,
                        evidence=ev_link,
                        context=ctx
                    )
                else:
                    if len(supporting_evidences) > 1:
                        requirements[req_name] = Requirement(
                            name=req_name,
                            state=EvidenceState.CONTEXT_MISMATCH,
                            reason="Evidence found but did not match target context."
                        )
                    else:
                        requirements[req_name] = Requirement(
                            name=req_name,
                            state=EvidenceState.NOT_FOUND,
                            reason="Explicit evidence not found."
                        )

    def validate(
        self, 
        session_id: str, 
        task: str,
        research_agent: ResearchAgent
    ) -> GuardrailResult:
        from app.guardrails.models import Requirement, RequirementClassification, EvidenceState, GuardrailTerminalState, PaperRelevance, ImplementationSupport
        
        paradigm = self._detect_paradigm(session_id, research_agent)
        
        critical_fields = self._get_critical_fields(paradigm)
        optional_fields = self._get_optional_fields(paradigm)
        all_possible_fields = self._get_all_possible_fields()
        
        requirements = {}
        for req in all_possible_fields:
            if req in optional_fields:
                requirements[req] = Requirement(
                    name=req, 
                    state=EvidenceState.SUPPORTED,
                    classification=RequirementClassification.IMPLEMENTATION_CHOICE,
                    value="Assigned to default implementation choice.",
                    reason="Optional engineering parameter mapped to default."
                )
            elif req not in critical_fields:
                requirements[req] = Requirement(
                    name=req, 
                    state=EvidenceState.NOT_APPLICABLE_TO_PARADIGM,
                    reason=f"Not applicable to {paradigm.value}"
                )
            else:
                requirements[req] = Requirement(name=req, state=EvidenceState.NOT_FOUND)
        
        # Pass 1: Automatic Spec Review
        for req_name in critical_fields:
            if requirements[req_name].state != EvidenceState.SUPPORTED:
                self._attempt_resolve(req_name, requirements, session_id, task, research_agent, attempt=1)
                
        all_critical_supported = all(requirements[req].state == EvidenceState.SUPPORTED for req in critical_fields)
        if all_critical_supported:
            return GuardrailResult(
                terminal_state=GuardrailTerminalState.PASS,
                attempt_count=1,
                requirements=requirements,
                session_id=session_id,
                code_generation_allowed=True,
                task_description=task,
                paper_relevance=PaperRelevance.SUPPORTED,
                implementation_support=ImplementationSupport.SUPPORTED,
                detected_paradigm=paradigm
            )
            
        attempt = 2
        while attempt <= self.max_attempts:
            missing_core_fields = [f for f in critical_fields if requirements[f].state != EvidenceState.SUPPORTED]
            for req_name in missing_core_fields:
                self._attempt_resolve(req_name, requirements, session_id, task, research_agent, attempt=attempt)
                
            all_critical_supported = all(requirements[req].state == EvidenceState.SUPPORTED for req in critical_fields)
            if all_critical_supported:
                return GuardrailResult(
                    terminal_state=GuardrailTerminalState.PASS,
                    attempt_count=attempt,
                    requirements=requirements,
                    session_id=session_id,
                    code_generation_allowed=True,
                    task_description=task,
                    paper_relevance=PaperRelevance.SUPPORTED,
                    implementation_support=ImplementationSupport.SUPPORTED,
                    detected_paradigm=paradigm
                )
            attempt += 1

        return GuardrailResult(
            terminal_state=GuardrailTerminalState.UNRESOLVED,
            attempt_count=self.max_attempts,
            requirements=requirements,
            reason="Could not retrieve sufficient evidence for all critical required fields after maximum attempts.",
            session_id=session_id,
            code_generation_allowed=False,
            task_description=task,
            paper_relevance=PaperRelevance.SUPPORTED,
            implementation_support=ImplementationSupport.APPLICABLE,
            detected_paradigm=paradigm
        )
