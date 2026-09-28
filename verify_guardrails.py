import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services import ResearchWorkspace

def format_paradigm(raw_paradigm: str) -> str:
    val = str(raw_paradigm).upper()
    if val.startswith("AIPARADIGM."):
        val = val.split(".")[-1]
    
    PARADIGM_DISPLAY_MAP = {
        "SYMBOLIC_PROBABILISTIC": "Symbolic Logic / Probabilistic",
        "DEEP_LEARNING": "Deep Learning",
        "CLASSICAL_ML": "Classical Machine Learning",
        "INDUSTRIAL_CONTROL": "Industrial Control System",
    }
    return PARADIGM_DISPLAY_MAP.get(val, val.replace("_", " ").title())

def test_paper(file_path: str):
    print(f"\n============================================================")
    print(f"Testing Paper: {file_path}")
    print(f"============================================================")
    
    service = ResearchWorkspace()
    session_id = service.create_session()
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    filename = Path(file_path).name
    ingestion = service.ingest(session_id, [(filename, content)])
    print(f"Knowledge Base READY — {ingestion.chunk_count} chunks indexed.")
    
    context, gr = service.generate_ml_specification(session_id)
    
    print("\n#### Guardrail Execution Summary: " + gr.terminal_state.value)
    if gr.reason:
        print(f"Reason: {gr.reason}")
    if gr.detected_paradigm:
        print(f"**Detected Paradigm:** {format_paradigm(gr.detected_paradigm)}")
    
    if getattr(gr, "requirements", None):
        print("### Evidence Claims")
        for name, req in gr.requirements.items():
            if req.state.value == "NOT_APPLICABLE_TO_PARADIGM":
                continue
            if req.state.value != "SUPPORTED" and gr.attempt_count < 3:
                continue
                
            status_symbol = "[OK]" if req.state.value == "SUPPORTED" else "[FAIL]"
            print(f"\n{status_symbol} **{name.replace('_', ' ').title()}** - {req.state.value}")
            if req.value:
                print(f"> {req.value}")
            if req.reason and req.state.value != "SUPPORTED":
                print(f"*Reason:* {req.reason}")
    if context and context.generated_code:
        print("\n### Generated Code")
        print(context.generated_code)
    print("------------------------------------------------------------\n")

if __name__ == "__main__":
    test_paper("D:\\SENTRA AI\\Attention.pdf")
    test_paper("D:\\SENTRA AI\\Belief.pdf")
    test_paper("D:\\SENTRA AI\\Orion.pdf")
