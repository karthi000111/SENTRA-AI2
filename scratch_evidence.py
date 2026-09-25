import pathlib
import re

p = pathlib.Path('app/guardrails/evidence_guardrail.py')
t = p.read_text()

# 1. Update imports
t = t.replace(
    "from app.guardrails.models import (",
    "from app.guardrails.models import (\n    AIParadigm,"
)

# 2. Add paradigm detection and dynamic fields
new_methods = """
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

    def _get_required_fields_for_paradigm(self, paradigm: AIParadigm) -> list[str]:
        if paradigm == AIParadigm.DEEP_LEARNING:
            return ["architecture_layers", "loss_function", "model_dimensions", "optimizer", "learning_rate", "batch_size"]
        elif paradigm == AIParadigm.SYMBOLIC_PROBABILISTIC:
            return ["proposition_representation", "probability_intervals", "constraint_rules", "propagation_engine", "optimizer", "learning_rate", "batch_size"]
        elif paradigm == AIParadigm.CLASSICAL_ML:
            return ["algorithm_type", "feature_dimensions", "objective_or_distance_metric", "optimizer", "learning_rate", "batch_size"]
        return ["architecture", "dataset", "target", "loss_function", "optimizer", "learning_rate", "batch_size"]

    def _get_critical_fields_for_paradigm(self, paradigm: AIParadigm) -> list[str]:
        if paradigm == AIParadigm.DEEP_LEARNING:
            return ["architecture_layers", "loss_function", "model_dimensions"]
        elif paradigm == AIParadigm.SYMBOLIC_PROBABILISTIC:
            return ["proposition_representation", "probability_intervals", "constraint_rules", "propagation_engine"]
        elif paradigm == AIParadigm.CLASSICAL_ML:
            return ["algorithm_type", "feature_dimensions", "objective_or_distance_metric"]
        return ["architecture", "dataset", "target", "loss_function"]

    def _get_applicable_fields(self, paradigm: AIParadigm) -> list[str]:
        if paradigm == AIParadigm.SYMBOLIC_PROBABILISTIC:
            return ["proposition_representation", "probability_intervals", "constraint_rules", "propagation_engine"]
        elif paradigm == AIParadigm.CLASSICAL_ML:
            return ["algorithm_type", "feature_dimensions", "objective_or_distance_metric"]
        elif paradigm == AIParadigm.DEEP_LEARNING:
            return ["architecture_layers", "loss_function", "model_dimensions", "optimizer", "learning_rate", "batch_size"]
        return ["architecture", "dataset", "target", "loss_function", "optimizer", "learning_rate", "batch_size"]

"""

# Let's replace _get_required_fields_for_task and _get_query_for_attempt 
# and _explicitly_supports with more generic versions that handle all fields.

# Actually, I'll rewrite the class methods completely using regex and string replacement on the file.
