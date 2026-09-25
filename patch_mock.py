import pathlib

p = pathlib.Path('tests/test_workflow.py')
t = p.read_text()

# Fix mock evidence strings in test_workflow.py to include standalone words matching the regex
t = t.replace('evidence_str = f"{req} explicitly found"',
              'evidence_str = f"{req.replace(\'_\', \' \')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"')

p.write_text(t)
