import pathlib

p = pathlib.Path('tests/test_workflow.py')
t = p.read_text()

# Fix mock evidence strings in test_workflow.py for targeted retrieval and provenance
t = t.replace('ev = f"{req} found"',
              'ev = f"{req.replace(\'_\', \' \')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"')

t = t.replace('ev_text = f"{req} found"',
              'ev_text = f"{req.replace(\'_\', \' \')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"')

t = t.replace('arch = gr.requirements["architecture"]',
              'arch = gr.requirements["architecture_layers"]')

p.write_text(t)
