"""Evidence Guardrail to ensure sufficient implementation details using method-type-aware schema."""
from __future__ import annotations

import re
import numpy as np
from app.agents.research_agent import ResearchAgent
from app.agents.models import EvidenceLink
from app.guardrails.models import (
    Requirement, RequirementClassification, GuardrailResult, 
    EvidenceState, GuardrailTerminalState, ContextInfo, AIParadigm,
    PaperRelevance, ImplementationSupport
)

class EvidenceGuardrail:
    """Validates that a paper contains sufficient evidence to implement a task."""
    
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.ml_references = [
            "Deep learning neural networks with transformer architectures, convolutional layers, and self-attention mechanisms.",
            "Classical machine learning including random forests, support vector machines, decision trees, and k-means clustering.",
            "Symbolic and probabilistic artificial intelligence, bayesian networks, belief maintenance, constraint satisfaction, and logic systems.",
            "Reinforcement learning, agentic systems, Q-learning, and markov decision processes.",
            "Algorithmic optimization, statistical learning theory, and computational complexity procedures."
        ]
        self.non_ml_references = [
            "Theoretical physics, quantum mechanics, string theory, and mathematical physics equations.",
            "Astronomy, astrophysics, molecular clouds, stellar formation, and sub-mm telescope observations.",
            "Biological sciences, genomics, genetics, protein folding, and molecular biology.",
            "Organic chemistry, synthesis pathways, chemical reactions, and materials science.",
            "Pure social sciences, qualitative studies, anthropological research, and sociological analysis."
        ]
        self.ml_embeddings = None
        self.non_ml_embeddings = None

    def _detect_method_type(self, session_id: str, research_agent: ResearchAgent) -> str:
        query = "Does this paper propose a method that requires a training or optimization loop on a dataset (like deep learning or gradient descent), or is it a deterministic/symbolic algorithm (like clustering, constraint satisfaction, boolean propagation, tree search)? Answer 'TRAINABLE' or 'DETERMINISTIC'."
        result = research_agent.run(session_id=session_id, query=query, top_k=5)
        text = str(result.answer).upper()
        if "TRAINABLE" in text:
            return "TRAINABLE"
        elif "DETERMINISTIC" in text:
            return "DETERMINISTIC"
        
        # fallback based on evidence text
        evidence_text = " ".join(str(ev.get("text", "")).lower() for ev in (result.evidence or []))
        if re.search(r"\b(train|learning|gradient|loss|optimization|epoch|batch)\b", evidence_text):
            return "TRAINABLE"
        return "DETERMINISTIC"

    def _get_critical_fields(self, method_type: str) -> list[str]:
        fields = [
            "core_method_description",
            "core_equations_or_formal_rules"
        ]
        if method_type == "TRAINABLE":
            fields.extend(["key_parameters", "training_or_optimization_procedure", "dataset_or_example_input"])
        return fields

    def _get_all_possible_fields(self) -> list[str]:
        return [
            "core_method_description",
            "core_equations_or_formal_rules",
            "key_parameters",
            "training_or_optimization_procedure",
            "dataset_or_example_input"
        ]

    def _get_query_for_attempt(self, req_name: str, attempt: int) -> str:
        if req_name == "core_method_description":
            return "method algorithm architecture network transformer block layer procedure logic"
        elif req_name == "core_equations_or_formal_rules":
            return "equation equations function loss rule rules probability metric formula formal constraint"
        elif req_name == "key_parameters":
            return "parameter parameters hyperparameter hyperparameters rate size dimension threshold beta"
        elif req_name == "training_or_optimization_procedure":
            return "train training optimization epoch batch gradient descent"
        elif req_name == "dataset_or_example_input":
            return "dataset datasets corpus data set test example input"
        return req_name

    def _explicitly_supports(self, req_name: str, text: str) -> bool:
        """Requirement-level validation to ensure the chunk actually supports the field."""
        text = text.lower()
        if req_name == "core_method_description":
            return bool(re.search(r"\b(architecture|layer|layers|convolutional|residual|network|transformer|block|hidden|algorithm|method|procedure|steps)\b", text))
        elif req_name == "core_equations_or_formal_rules":
            if "f(x) + x" in text and not re.search(r"\b(equation|equations|loss|function|functions|constraint|constraints|probability|probabilities|rule|rules)\b", text):
                return False
            return bool(re.search(r"\b(equation|equations|loss|function|functions|constraint|constraints|probability|probabilities|rule|rules|objective|metric|metrics|formula|formulas)\b", text))
        elif req_name == "key_parameters":
            return bool(re.search(r"\b(parameter|parameters|hyperparameter|hyperparameters|rate|size|dimension|dimensions|clusters|k\s*=)\b", text))
        elif req_name == "training_or_optimization_procedure":
            return bool(re.search(r"\b(train|training|optimize|optimization|epoch|epochs|batch|batches|gradient|descent|adam|sgd|learning)\b", text))
        elif req_name == "dataset_or_example_input":
            return bool(re.search(r"\b(dataset|datasets|corpus|data|training set|test set|features|target|labels|example|input|inputs)\b", text))
        return False

    def _attempt_resolve(self, req_name: str, requirements: dict, session_id: str, task: str, research_agent: ResearchAgent, attempt: int):
        query = self._get_query_for_attempt(req_name, attempt)
        result = research_agent.run(session_id=session_id, query=query, top_k=5)
        
        if result.sufficient_evidence and result.evidence:
            supporting_evidences = []
            for ev in result.evidence:
                if self._explicitly_supports(req_name, str(ev.get("text", ""))):
                    supporting_evidences.append(ev)
            
            if supporting_evidences:
                best_ev = supporting_evidences[0]
                
                ev_link = EvidenceLink(
                    source=str(best_ev.get("source", best_ev.get("filename", ""))),
                    page=int(best_ev.get("page", 0)),
                    chunk_id=str(best_ev.get("chunk_id", "")),
                    evidence_text=str(best_ev.get("text", ""))
                )
                
                val = str(best_ev.get("text", ""))
                keywords = research_agent._keywords(query)
                excerpt = research_agent._best_sentence(val, keywords)
                answer = f"Evidence found in {best_ev['source']}, page {best_ev['page']}: {excerpt}"
                
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
                if len(result.evidence) > 0:
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
        
        # Pre-check Domain Relevance (Embedding-based)
        chunks = research_agent._rag_manager._current_chunks(session_id)
        if not chunks:
            return GuardrailResult(
                terminal_state=GuardrailTerminalState.DOMAIN_UNSUPPORTED,
                attempt_count=1,
                reason="Not an ML/AI paper. Reason: No substantive text chunks found.",
                session_id=session_id,
                code_generation_allowed=False,
                task_description=task,
                paper_relevance=PaperRelevance.UNSUPPORTED,
                implementation_support=ImplementationSupport.IMPLEMENTATION_UNSUPPORTED
            )

        embedder = research_agent._rag_manager._embedder

        if self.ml_embeddings is None:
            self.ml_embeddings = embedder.encode(self.ml_references)
            self.non_ml_embeddings = embedder.encode(self.non_ml_references)

        first_chunk = chunks[0].text
        chunk_embedding = embedder.encode([first_chunk])[0]
        
        chunk_norm = np.linalg.norm(chunk_embedding)
        if chunk_norm > 0:
            chunk_embedding = chunk_embedding / chunk_norm
            
        ml_norms = np.linalg.norm(self.ml_embeddings, axis=1, keepdims=True)
        ml_embeddings_norm = np.divide(self.ml_embeddings, ml_norms, out=np.zeros_like(self.ml_embeddings), where=ml_norms!=0)
        
        non_ml_norms = np.linalg.norm(self.non_ml_embeddings, axis=1, keepdims=True)
        non_ml_embeddings_norm = np.divide(self.non_ml_embeddings, non_ml_norms, out=np.zeros_like(self.non_ml_embeddings), where=non_ml_norms!=0)

        ml_sims = np.dot(ml_embeddings_norm, chunk_embedding)
        non_ml_sims = np.dot(non_ml_embeddings_norm, chunk_embedding)
        
        avg_ml_sim = float(np.mean(ml_sims))
        avg_non_ml_sim = float(np.mean(non_ml_sims))

        if avg_non_ml_sim >= avg_ml_sim:
            return GuardrailResult(
                terminal_state=GuardrailTerminalState.DOMAIN_UNSUPPORTED,
                attempt_count=1,
                reason="Not an ML/AI paper.",
                session_id=session_id,
                code_generation_allowed=False,
                task_description=task,
                paper_relevance=PaperRelevance.UNSUPPORTED,
                implementation_support=ImplementationSupport.IMPLEMENTATION_UNSUPPORTED
            )

        method_type = self._detect_method_type(session_id, research_agent)
        
        critical_fields = self._get_critical_fields(method_type)
        all_possible_fields = self._get_all_possible_fields()
        
        requirements = {}
        for req in all_possible_fields:
            if req not in critical_fields:
                requirements[req] = Requirement(
                    name=req, 
                    state=EvidenceState.NOT_APPLICABLE_TO_PARADIGM,
                    reason=f"Not applicable to {method_type} paradigm"
                )
            else:
                requirements[req] = Requirement(name=req, state=EvidenceState.NOT_FOUND)
        
        # Pass 1: Automatic Spec Review
        for req_name in critical_fields:
            self._attempt_resolve(req_name, requirements, session_id, task, research_agent, attempt=1)
                
        def is_substantive_algorithmic_context_present() -> bool:
            return (requirements.get("core_method_description").state == EvidenceState.SUPPORTED or 
                    requirements.get("core_equations_or_formal_rules").state == EvidenceState.SUPPORTED)

        if method_type == "DETERMINISTIC":
            if is_substantive_algorithmic_context_present():
                for req in critical_fields:
                    if requirements[req].state != EvidenceState.SUPPORTED:
                        requirements[req] = Requirement(
                            name=req,
                            state=EvidenceState.NOT_APPLICABLE_TO_PARADIGM,
                            reason="Not required due to presence of other substantive algorithmic context."
                        )
                return GuardrailResult(
                    terminal_state=GuardrailTerminalState.PASS,
                    attempt_count=1,
                    requirements=requirements,
                    session_id=session_id,
                    code_generation_allowed=True,
                    task_description=task,
                    paper_relevance=PaperRelevance.SUPPORTED,
                    implementation_support=ImplementationSupport.SUPPORTED,
                    detected_paradigm=AIParadigm.CLASSICAL_ML
                )
        else:
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
                    detected_paradigm=AIParadigm.DEEP_LEARNING
                )
            
        attempt = 2
        while attempt <= self.max_attempts:
            missing_core_fields = [f for f in critical_fields if requirements[f].state != EvidenceState.SUPPORTED]
            for req_name in missing_core_fields:
                self._attempt_resolve(req_name, requirements, session_id, task, research_agent, attempt=attempt)
                
            if method_type == "DETERMINISTIC":
                if is_substantive_algorithmic_context_present():
                    for req in critical_fields:
                        if requirements[req].state != EvidenceState.SUPPORTED:
                            requirements[req] = Requirement(
                                name=req,
                                state=EvidenceState.NOT_APPLICABLE_TO_PARADIGM,
                                reason="Not required due to presence of other substantive algorithmic context."
                            )
                    return GuardrailResult(
                        terminal_state=GuardrailTerminalState.PASS,
                        attempt_count=attempt,
                        requirements=requirements,
                        session_id=session_id,
                        code_generation_allowed=True,
                        task_description=task,
                        paper_relevance=PaperRelevance.SUPPORTED,
                        implementation_support=ImplementationSupport.SUPPORTED,
                        detected_paradigm=AIParadigm.CLASSICAL_ML
                    )
            else:
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
                        detected_paradigm=AIParadigm.DEEP_LEARNING
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
            detected_paradigm=AIParadigm.DEEP_LEARNING if method_type == "TRAINABLE" else AIParadigm.CLASSICAL_ML
        )
