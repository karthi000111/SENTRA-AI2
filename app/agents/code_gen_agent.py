from __future__ import annotations
from app.guardrails.models import GuardrailResult, AIParadigm

class CodeGenAgent:
    def __init__(self):
        pass
        
    def generate_prompt_template(self, guardrail_result: GuardrailResult) -> str:
        paradigm = guardrail_result.detected_paradigm
        
        base_prompt = "You are an AI code generation agent.\n"
        base_prompt += f"Task: {guardrail_result.task_description}\n\n"
        
        if paradigm == AIParadigm.SYMBOLIC_PROBABILISTIC:
            base_prompt += (
                "Context: The uploaded document belongs to the Symbolic AI / Probabilistic Graphical Models paradigm.\n"
                "Instructions:\n"
                "- Synthesize Python class structures implementing logic proposition managers, probability interval propagators, or constraint solvers (e.g., Boolean Constraint Propagation).\n"
                "- Do NOT force PyTorch nn.Module skeletons or neural network logic.\n"
                "- Focus on the core methodology established in the paper.\n"
            )
        elif paradigm == AIParadigm.DEEP_LEARNING:
            base_prompt += (
                "Context: The uploaded document belongs to the Deep Learning paradigm.\n"
                "Instructions:\n"
                "- Synthesize PyTorch nn.Module skeletons.\n"
                "- Implement the neural network architecture, forward pass, and specify the loss function and optimizer.\n"
            )
        elif paradigm == AIParadigm.CLASSICAL_ML:
            base_prompt += (
                "Context: The uploaded document belongs to the Classical ML paradigm.\n"
                "Instructions:\n"
                "- Synthesize scikit-learn compatible class structures or custom logic for clustering, decision trees, SVMs, etc.\n"
            )
        elif paradigm == AIParadigm.CONTROL_INDUSTRIAL:
            base_prompt += (
                "Context: The uploaded document belongs to the Control / Industrial paradigm.\n"
                "Instructions:\n"
                "- Synthesize logic for state machines, automation logic, or control theory systems.\n"
            )
            
        base_prompt += "\nRequired Fields from Evidence Guardrail:\n"
        for req in guardrail_result.requirements.values():
            if req.state == "SUPPORTED":
                base_prompt += f"- {req.name}: {req.value}\n"
                
        return base_prompt

    def generate_code(self, guardrail_result: GuardrailResult) -> tuple[str, bool]:
        """Generates code and validates it via a simulated sandbox."""
        prompt = self.generate_prompt_template(guardrail_result)
        # Mock code generation for now
        paradigm = guardrail_result.detected_paradigm
        
        if paradigm == AIParadigm.SYMBOLIC_PROBABILISTIC:
            code = (
                "class LogicPropositionManager:\n"
                "    def __init__(self):\n"
                "        self.intervals = []\n\n"
                "    def propagate(self):\n"
                "        pass\n"
            )
        elif paradigm == AIParadigm.DEEP_LEARNING:
            code = (
                "import torch.nn as nn\n\n"
                "class DeepModel(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n\n"
                "    def forward(self, x):\n"
                "        return x\n"
            )
        else:
            code = (
                "class SynthesizedModel:\n"
                "    def fit(self, X, y):\n"
                "        pass\n"
            )
            
        return code, True
