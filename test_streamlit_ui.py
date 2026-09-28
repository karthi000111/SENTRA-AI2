from streamlit.testing.v1 import AppTest

def get_button_by_label(at, label):
    for idx, b in enumerate(at.button):
        if b.label == label:
            return b
    return None

def test_ui(pdf_path):
    print(f"\n==================================================")
    print(f"UI E2E TEST: {pdf_path}")
    print(f"==================================================")
    
    at = AppTest.from_file("ui/app.py").run()
    
    # 1. Create Research Session
    btn = get_button_by_label(at, "Create Research Session")
    if btn:
        btn.click().run()
    
    # 2. Upload file
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    at.file_uploader[0].set_value([(pdf_path, file_bytes, "application/pdf")]).run()
    
    # 3. Index uploaded documents
    btn = get_button_by_label(at, "Index uploaded documents")
    if btn:
        btn.click().run()
        
    # 4. Run Guardrail
    btn = get_button_by_label(at, "Run Guardrail")
    if btn:
        btn.click().run()

    print("\n[EXTRACTED AUDIT PANEL TEXT FROM UI RENDERING]")
    # We want to extract specifically the output from the guardrail audit
    # which is mostly in markdown, success, warning, error.
    # Let's print them in order they appear if possible. 
    # For AppTest we can just dump the specific ones that match our output.
    
    for m in at.markdown:
        if m.value and ("ML evidence:" in m.value or "Verified Specifications" in m.value):
            print(f"{m.value}")
    
    for s in at.success:
        if s.value and ("Code generation allowed" in s.value or "**" in s.value):
            print(f"[SUCCESS RENDERING]: {s.value}")
            
    for w in at.warning:
        if w.value and ("MISSING" in w.value or "MUST include" in w.value):
            print(f"[WARNING RENDERING]: {w.value}")
            
    for e in at.error:
        if e.value and "Code generation blocked" in e.value:
            print(f"[ERROR RENDERING]: {e.value}")

test_ui("Attention.pdf")
test_ui("Belief.pdf")
