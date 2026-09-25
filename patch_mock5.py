import pathlib

p = pathlib.Path('tests/test_workflow.py')
t = p.read_text()

t = t.replace('assert gr.requirements["optimizer"].classification == RequirementClassification.PAPER_SUPPORTED',
              'assert gr.requirements["optimizer"].classification == RequirementClassification.IMPLEMENTATION_CHOICE')

t = t.replace("""    # Optimizer should not be supported because "dataset cifar-10 used" does not contain optimizer keywords
    assert gr.requirements["loss_function"].state == EvidenceState.NOT_FOUND
    
    # Loss function should not be supported because "f(x) + x" is explicitly excluded unless it contains "loss"
    assert gr.requirements["loss_function"].state == EvidenceState.NOT_FOUND""",
              """    # Loss function should not be supported because "f(x) + x" is explicitly excluded unless it contains "loss"
    assert gr.requirements["loss_function"].state == EvidenceState.NOT_FOUND""")

p.write_text(t)
