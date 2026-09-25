import pathlib

p = pathlib.Path('tests/test_workflow.py')
t = p.read_text()

# Fix test_valid_ml_paper_missing_optimizer
t = t.replace('def test_valid_ml_paper_missing_optimizer(evidence_guardrail, session_id):',
              'def test_valid_ml_paper_missing_loss(evidence_guardrail, session_id):')
t = t.replace('"""Test 3: Valid ML paper with missing optimizer -> UNRESOLVED, optimizer = NOT_FOUND"""',
              '"""Test 3: Valid ML paper with missing loss function -> UNRESOLVED"""')
t = t.replace('if req == "optimizer":\n                # Missing in all attempts',
              'if req == "loss_function":\n                # Missing in all attempts')

# Fix test_targeted_retrieval_succeeds
t = t.replace('assert gr.requirements["architecture"].state == EvidenceState.SUPPORTED',
              'assert gr.requirements["architecture_layers"].state == EvidenceState.SUPPORTED')

# Fix test_provenance
t = t.replace('assert arch.evidence.chunk_id == "chunk_architecture"',
              'assert arch.evidence.chunk_id == "chunk_architecture_layers"')

p.write_text(t)
