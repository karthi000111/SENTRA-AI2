import sys
import os

# Ensure the module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.guardrails.ml_guardrail import _evaluate_field, verify_paper_evidence, GuardrailResult

def run_test(name, text):
    print(f"\n--- {name} ---")
    
    # Mocking retrieve
    import app.guardrails.ml_guardrail as ml
    ml.retrieve = lambda sid, query: []
    
    res = verify_paper_evidence([text], "dummy_session")
    
    print(f"Code Gen Allowed: {res.code_generation_allowed}")
    print(f"Blocking Reason: {res.blocking_reason}")
    print("Fields:")
    for f in res.fields:
        print(f"  {f.field}: {f.status} (Value: {f.value})")
    print(f"Audit Summary: {res.audit_summary}")
    print("Warnings:")
    for w in res.warnings:
        print(f"  - {w}")
    return res

text_attention = "We propose the Transformer, a novel neural network architecture based solely on attention mechanisms. The training optimization procedure minimizes cross-entropy loss. We trained on the WMT 2014 English-to-German dataset. The model has a dimension of 512, 8 heads, and 6 encoder and decoder layers. The core equation is softmax."

text_belief = "We introduce a Waltz-style propagation algorithm for belief maintenance in Bayesian Networks. This procedure uses a constraint satisfaction logic. The core probability equations involve the interval-extended chain rule. We demonstrate with a case study and test case. The threshold parameter beta is used."

text_non_technical = "This is a survey of modern business strategies and market trends. It discusses product management and team organization without any mathematical formulas or formal methods."

print("CHECK 1: Existing Papers")
res_att = run_test("Attention Is All You Need", text_attention)
res_bel = run_test("Belief Maintenance in Bayesian Networks", text_belief)
res_non = run_test("Business Strategy Survey", text_non_technical)
