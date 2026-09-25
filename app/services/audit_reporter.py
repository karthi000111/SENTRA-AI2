from __future__ import annotations
from app.guardrails.models import GuardrailResult, GuardrailTerminalState, EvidenceState

def generate_audit_report(result: GuardrailResult) -> str:
    """Generates a structured audit report for the guardrail evaluation."""
    lines = []
    
    lines.append("## Verification Summary")
    lines.append(f"* status: {result.terminal_state.value}")
    lines.append(f"* paper_relevance: {result.paper_relevance.value if hasattr(result.paper_relevance, 'value') else result.paper_relevance}")
    lines.append(f"* implementation_support: {result.implementation_support.value if hasattr(result.implementation_support, 'value') else result.implementation_support}")
    lines.append(f"* reason: {result.reason}")
    lines.append(f"* recovery attempts: {result.attempt_count}")
    lines.append(f"* requested task: {result.task_description}")
    lines.append("")
    
    lines.append("## Evidence Inventory")
    verified_fields = [req for req in result.requirements.values() if req.state == EvidenceState.SUPPORTED]
    for req in verified_fields:
        lines.append(f"* field: {req.name}")
        lines.append(f"  value: {req.value}")
        lines.append(f"  evidence status: {req.state.value}")
        if req.evidence:
            lines.append(f"  source: {req.evidence.source}")
            lines.append(f"  page: {req.evidence.page}")
            if req.evidence.chunk_id:
                lines.append(f"  section/chunk: {req.evidence.chunk_id}")
        if req.context:
            lines.append(f"  context: {req.context.experiment}")
    lines.append("")
    
    lines.append("## Missing/Problematic Evidence")
    missing_fields = [req for req in result.requirements.values() if req.state != EvidenceState.SUPPORTED]
    for req in missing_fields:
        lines.append(f"* field: {req.name}")
        lines.append(f"  status: {req.state.value}")
        lines.append(f"  reason: {req.reason}")
        # Note: In a complete implementation we might log exact queries.
        lines.append(f"  recovery queries attempted: {result.attempt_count}")
    lines.append("")
    
    lines.append("## Completeness")
    lines.append(f"* supported required fields: {result.supported_required_fields}")
    lines.append(f"* total required fields: {result.total_required_fields}")
    lines.append(f"* completeness ratio: {result.completeness_score:.2f}")
    lines.append("")
    
    lines.append("## Downstream Decision")
    lines.append(f"* code_generation_allowed: {result.code_generation_allowed}")
    if not result.code_generation_allowed:
        lines.append(f"* blocking_reason: {result.reason}")
        
    return "\n".join(lines)
