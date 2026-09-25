import pathlib

p = pathlib.Path('tests/test_guardrail_pass_first.py')
t = p.read_text()

t = t.replace('ev_text = f"explicitly found {req} layers architecture"',
              'ev_text = f"{req.replace(\'_\', \' \')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"')

t = t.replace('ev_text = f"explicitly found {req} layers architecture in attempt 2"',
              'ev_text = f"{req.replace(\'_\', \' \')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule in attempt 2"')

p.write_text(t)
