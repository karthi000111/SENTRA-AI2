import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from app.guardrails.ml_guardrail import verify_paper_evidence
from app.rag.session_manager import SessionRAGManager

rag = SessionRAGManager()

pdfs = [
    r"D:\SENTRA AI\data\sessions\c77116a9-8510-444d-adb6-406d292ac5d2\uploads\1f9ff8d0-0213-45d8-b191-245b5a6e720e_attention.pdf",
    r"D:\SENTRA AI\data\sessions\37d20f0d-1caf-4f3f-883e-09929e9d1ef8\uploads\78d15ba7-2b53-4809-8e15-a829238b0c16_Belief Maintenance in Bayesian Networks.pdf",
    r"D:\SENTRA AI\data\sessions\489eb6fb-194c-4812-8578-8d23ecc654c5\uploads\886d664f-a3f5-4dd0-a196-373e188fd21a_DETECTION OF IRREGULAR, SUB-MM OPAQUE STRUCTURES IN THE ORION MOLECULAR CLOUDS.pdf"
]

def render_ui(gr):
    if not gr.code_generation_allowed:
        print(f"✗ Code generation blocked: {gr.blocking_reason}")
    else:
        print(f"**{gr.audit_summary}**")
        print("✓ Code generation allowed.")
    
    if gr.audit_summary and not gr.code_generation_allowed:
        print(f"**{gr.audit_summary}**")
        
    for field in gr.fields:
        if field.status == "MISSING":
            print(f"- **{field.field}**: ⚠️ MISSING — {field.value}")
        else:
            print(f"- **{field.field}**: ✓ {field.status} — {field.value}")

for pdf in pdfs:
    s = rag.create_session()
    with open(pdf, 'rb') as f:
        rag.add_documents(s, [(os.path.basename(pdf), f.read())])
    
    # Get retrieved chunks for context
    try:
        results = rag.retrieve(s, "machine learning deep learning algorithm equations methodology", top_k=5)
        chunks = [str(res.get("text", "")) for res in results]
    except:
        chunks = []
        
    gr = verify_paper_evidence(chunks, s)
    print(f"\n==================================================")
    print(f"UI RENDERED OUTPUT FOR: {os.path.basename(pdf)}")
    print(f"==================================================")
    render_ui(gr)
