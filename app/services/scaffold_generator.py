from __future__ import annotations
from app.guardrails.models import GuardrailResult, GuardrailTerminalState, EvidenceState

def generate_scaffold(result: GuardrailResult) -> str:
    """Generates an Evidence-Bounded Implementation Scaffold for UNRESOLVED tasks."""
    if result.terminal_state != GuardrailTerminalState.UNRESOLVED:
        return ""
        
    lines = []
    lines.append("# Evidence-Bounded Implementation Scaffold")
    lines.append("")
    
    missing_fields = [req for req in result.requirements.values() if req.state != EvidenceState.SUPPORTED]
    
    for req in missing_fields:
        lines.append("# ================================================================")
        lines.append("# TODO — MISSING PAPER EVIDENCE")
        lines.append("#")
        lines.append(f"# The source document does not establish the {req.name.replace('_', ' ')}")
        lines.append(f"# after the maximum bounded recovery attempts ({result.attempt_count}).")
        lines.append("#")
        lines.append("# Sentra AI intentionally does NOT select a default value.")
        lines.append("# Researcher action is required.")
        lines.append("# ================================================================")
        lines.append("")
        var_name = req.name.upper()
        lines.append(f"{var_name} = None")
        lines.append("")
        lines.append(f"if {var_name} is None:")
        lines.append(f"    raise NotImplementedError(")
        lines.append(f"        \"{req.name.replace('_', ' ').capitalize()} is not established by the source document.\"")
        lines.append(f"    )")
        lines.append("")
        
    return "\n".join(lines)
