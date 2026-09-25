import pathlib

p = pathlib.Path('tests/test_workflow.py')
t = p.read_text()

target = """    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )"""

replacement = """    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = "Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )"""

t = t.replace(target, replacement)
p.write_text(t)
