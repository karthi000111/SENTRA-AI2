import pathlib

p = pathlib.Path('tests/test_workflow.py')
t = p.read_text()

# Fix test_false_attribution_prevention
t = t.replace('def test_false_attribution_prevention(evidence_guardrail, session_id):',
              'def test_false_attribution_prevention_loss(evidence_guardrail, session_id):')
t = t.replace('if req == "optimizer":\n                # Missing in all attempts',
              'if req == "loss_function":\n                # Missing in all attempts')
t = t.replace('assert gr.requirements["optimizer"].state == EvidenceState.NOT_FOUND',
              'assert gr.requirements["loss_function"].state == EvidenceState.NOT_FOUND')

p.write_text(t)
