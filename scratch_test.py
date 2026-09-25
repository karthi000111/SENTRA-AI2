import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')
from app.services.research_workspace import ResearchWorkspace

async def test_resnet():
    workspace = ResearchWorkspace()
    session_id = workspace.create_session()
    
    with open(r"D:\SENTRA AI\data\sessions\3e15d0f6-881e-4904-b6f6-f335acde2b75\uploads\04ac308a-04eb-4127-a728-c30d20e07a0a_Deep Residual Learning for Image Recognition.pdf", "rb") as f:
        pdf_bytes = f.read()
        
    workspace.ingest(session_id, [("Deep Residual Learning for Image Recognition.pdf", pdf_bytes)])
    
    context, gr = workspace.generate_ml_specification(session_id, "Implement ResNet for image classification")
    
    print("\n==================================================")
    print("GUARDRAIL VERIFICATION RESULT")
    print(f"Action: {gr.action.name}")
    print("==================================================")
    
    for req_name in [
        "architecture", "dataset", "optimizer", "learning_rate", 
        "batch_size", "training_procedure", "activation", 
        "loss_function", "framework"
    ]:
        req = gr.requirements.get(req_name)
        if not req:
            continue
            
        print(f"\n1. Field name: {req_name.upper()}")
        print(f"2. Status: {req.classification.name}")
        
        if req.classification.name == "PAPER_SUPPORTED" and req.evidence:
            print(f"3. Extracted value: {req.value}")
            print(f"4. Source page: {req.evidence.page}")
            print(f"5. Chunk ID: {req.evidence.chunk_id}")
            print(f"6. Exact supporting evidence: {req.evidence.evidence_text.strip()}")
            print(f"7. One-line explanation: Passed strict regex validation checking explicitly for {req_name} terminology in the extracted chunk.")
        else:
            print(f"3. Extracted value: N/A")
            print(f"4. Source page: N/A")
            print(f"5. Chunk ID: N/A")
            print(f"6. Exact supporting evidence: N/A")
            print(f"7. One-line explanation: No explicit chunk in the paper satisfied the strict requirement-level regex for {req_name}.")

if __name__ == "__main__":
    asyncio.run(test_resnet())
