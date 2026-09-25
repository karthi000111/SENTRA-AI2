"""Tests for the Domain and Evidence Guardrails."""
import pytest
from unittest.mock import Mock

from app.agents.research_agent import ResearchAgent, ResearchResult
from app.guardrails.models import GuardrailResult, GuardrailTerminalState, EvidenceState, RequirementClassification, Requirement, AIParadigm
from app.guardrails.domain_guardrail import DomainGuardrail
from app.guardrails.evidence_guardrail import EvidenceGuardrail
from app.orchestration.workflow import run_research_to_spec_workflow
from app.services.implementation_context import ImplementationContext


@pytest.fixture
def domain_guardrail(research_agent):
    return DomainGuardrail(research_agent=research_agent)

@pytest.fixture
def evidence_guardrail():
    return EvidenceGuardrail(max_attempts=3)

@pytest.fixture
def session_id():
    return "test-session-123"

def create_evidence(text: str, filename="test.pdf", chunk_id="chunk1", page=1):
    return {
        "text": text,
        "source": filename,
        "filename": filename,
        "page": page,
        "chunk_id": chunk_id,
        "distance": 0.1
    }

def create_mock_research_agent(responses: dict):
    agent = Mock(spec=ResearchAgent)
    
    def side_effect(session_id: str, query: str, top_k: int = 5):
        if "determine the AI paradigm" in query:
            return ResearchResult(session_id, query, "neural network deep learning", [create_evidence("neural network machine learning model")], True)
        # Allow exact match, or fallback to default
        if query in responses:
            return responses[query]
        # Return fallback missing
        return ResearchResult(
            session_id=session_id,
            query=query,
            answer="No sufficient evidence.",
            evidence=[],
            sufficient_evidence=False
        )
        
    agent.run.side_effect = side_effect
    return agent


def test_valid_ml_paper(evidence_guardrail, session_id):
    """Test 1: Valid ML paper -> PASS"""
    # Mock finding evidence for all fields
    responses = {}
    
    # Domain check
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = "Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = f"Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )

    expected_reqs = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING) + evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)
    # Evidence checks
    for req in expected_reqs:
        query = evidence_guardrail._get_query_for_attempt(req, 1)
        if req == "optimizer":
            evidence_str = "SGD"
        elif req == "loss_function":
            evidence_str = "cross-entropy loss"
        else:
            evidence_str = f"{req.replace('_', ' ')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"
        responses[query] = ResearchResult(
            session_id, query, f"{req} found", [create_evidence(evidence_str)], True
        )
        
    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    assert gr.terminal_state == GuardrailTerminalState.PASS
    assert context is not None


def test_non_ml_paper(evidence_guardrail, session_id):
    """Test 2: Clearly non-ML/random PDF -> DOMAIN_UNSUPPORTED"""
    agent = create_mock_research_agent({})
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    assert gr.terminal_state == GuardrailTerminalState.DOMAIN_UNSUPPORTED
    assert context is None
    assert gr.code_generation_allowed is False


def test_valid_ml_paper_missing_loss(evidence_guardrail, session_id):
    """Test 3: Valid ML paper with missing loss function -> UNRESOLVED"""
    responses = {}
    
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = "Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )

    expected_reqs = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING) + evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)
    for req in expected_reqs:
        query1 = evidence_guardrail._get_query_for_attempt(req, 1)
        query2 = evidence_guardrail._get_query_for_attempt(req, 2)
        query3 = evidence_guardrail._get_query_for_attempt(req, 3)
        if req == "loss_function":
            # Missing in all attempts
            responses[query1] = ResearchResult(session_id, query1, "Not found", [], False)
            responses[query2] = ResearchResult(session_id, query2, "Not found", [], False)
            responses[query3] = ResearchResult(session_id, query3, "Not found", [], False)
        else:
            if req == "optimizer":
                ev = "SGD"
            else:
                ev = f"{req.replace('_', ' ')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"
            responses[query1] = ResearchResult(session_id, query1, ev, [create_evidence(ev)], True)

    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    assert gr.terminal_state == GuardrailTerminalState.UNRESOLVED
    assert gr.requirements["loss_function"].state == EvidenceState.NOT_FOUND


def test_targeted_retrieval_succeeds(evidence_guardrail, session_id):
    """Test 4: Initial retrieval insufficient but targeted retrieval succeeds"""
    responses = {}
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = "Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )

    expected_reqs = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING) + evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)
    for req in expected_reqs:
        query = evidence_guardrail._get_query_for_attempt(req, 1)
        responses[query] = ResearchResult(session_id, query, "Not found", [], False)
        
        # Succeed on attempt 2 for all except we mock specific ones below
        query2 = evidence_guardrail._get_query_for_attempt(req, 2)
        if req == "optimizer":
            ev = "SGD"
        elif req == "loss_function":
            ev = "cross-entropy loss"
        else:
            ev = f"{req.replace('_', ' ')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"
        responses[query2] = ResearchResult(session_id, query2, ev, [create_evidence(ev)], True)

    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    assert gr.terminal_state == GuardrailTerminalState.PASS
    assert gr.attempt_count == 2
    assert gr.requirements["architecture_layers"].state == EvidenceState.SUPPORTED
    assert gr.requirements["loss_function"].state == EvidenceState.SUPPORTED


def test_evidence_remains_insufficient(evidence_guardrail, session_id):
    """Test 5 & 6: Evidence remains insufficient, max attempts bounded to 3"""
    responses = {}
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = "Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    
    # Do not add any responses for requirements, so all will fail on all 3 attempts
    
    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    assert gr.terminal_state == GuardrailTerminalState.UNRESOLVED
    assert gr.attempt_count == 3
    assert context is None


def test_false_attribution_prevention_loss(evidence_guardrail, session_id):
    """Test 7: False attribution prevention - not specified is NOT_FOUND"""
    responses = {}
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = "Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )

    expected_reqs = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING) + evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)
    for req in expected_reqs:
        query1 = evidence_guardrail._get_query_for_attempt(req, 1)
        query2 = evidence_guardrail._get_query_for_attempt(req, 2)
        query3 = evidence_guardrail._get_query_for_attempt(req, 3)
        if req == "loss_function":
            # Missing in all attempts
            responses[query1] = ResearchResult(session_id, query1, "Not found", [], False)
            responses[query2] = ResearchResult(session_id, query2, "Not found", [], False)
            responses[query3] = ResearchResult(session_id, query3, "Not found", [], False)
        else:
            ev = f"{req.replace('_', ' ')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"
            responses[query1] = ResearchResult(session_id, query1, ev, [create_evidence(ev)], True)
            
    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    assert gr.requirements["loss_function"].state == EvidenceState.NOT_FOUND
    assert gr.requirements["optimizer"].classification == RequirementClassification.IMPLEMENTATION_CHOICE


def test_provenance(evidence_guardrail, session_id):
    """Test 8: Provenance must retain source, page, chunk_id, session_id"""
    responses = {}
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = "Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )

    expected_reqs = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING) + evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)
    for req in expected_reqs:
        query = evidence_guardrail._get_query_for_attempt(req, 1)
        if req == "optimizer":
            ev_text = "SGD"
        elif req == "loss_function":
            ev_text = "cross-entropy loss"
        else:
            ev_text = f"{req.replace('_', ' ')} explicitly found dimension layers architecture algorithm dataset metric clusters propagation rule"
        responses[query] = ResearchResult(
            session_id, query, f"{req} found", 
            [create_evidence(ev_text, filename="special.pdf", chunk_id=f"chunk_{req}", page=5)], 
            True
        )

    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    arch = gr.requirements["architecture_layers"]
    assert arch.evidence.source == "special.pdf"
    assert arch.evidence.page == 5
    assert arch.evidence.chunk_id == "chunk_architecture_layers"


def test_session_isolation(evidence_guardrail):
    """Test 9: Session isolation"""
    agent = Mock(spec=ResearchAgent)
    # Mock domain success
    def side_effect(session_id: str, query: str, top_k: int = 5):
        return ResearchResult(
            session_id=session_id,
            query=query,
            answer="machine learning",
            evidence=[create_evidence("neural network machine learning model")],
            sufficient_evidence=True
        )
    agent.run.side_effect = side_effect
    domain_guardrail = DomainGuardrail(agent)
    
    run_research_to_spec_workflow("sess-1", "Task", agent, domain_guardrail, evidence_guardrail)
    
    for call_args in agent.run.call_args_list:
        assert call_args.kwargs["session_id"] == "sess-1"


def test_requirement_level_validation(evidence_guardrail, session_id):
    """Test 10: Ensure irrelevant candidate evidence is rejected."""
    responses = {}
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )
    impl_query = "Find methodology, architecture, or algorithm details for: Task"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "machine learning methodology", [create_evidence("neural network machine learning model")], True
    )

    expected_reqs = evidence_guardrail._get_critical_fields(AIParadigm.DEEP_LEARNING) + evidence_guardrail._get_optional_fields(AIParadigm.DEEP_LEARNING)
    for req in expected_reqs:
        query1 = evidence_guardrail._get_query_for_attempt(req, 1)
        query2 = evidence_guardrail._get_query_for_attempt(req, 2)
        query3 = evidence_guardrail._get_query_for_attempt(req, 3)
        
        if req == "optimizer":
            # Provide explicitly wrong evidence for optimizer
            ev = "dataset cifar-10 used"
            r = ResearchResult(session_id, query1, ev, [create_evidence(ev)], True)
            responses[query1] = responses[query2] = responses[query3] = r
        elif req == "loss_function":
            # Provide F(x) + x for loss function (which should be rejected by regex)
            ev = "the residual function f(x) + x is added"
            r = ResearchResult(session_id, query1, ev, [create_evidence(ev)], True)
            responses[query1] = responses[query2] = responses[query3] = r
        else:
            ev = f"{req} explicitly found"
            responses[query1] = ResearchResult(session_id, query1, ev, [create_evidence(ev)], True)

    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description="Task",
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    # Loss function should not be supported because "f(x) + x" is explicitly excluded unless it contains "loss"
    assert gr.requirements["loss_function"].state == EvidenceState.NOT_FOUND


def test_scaffold_generation():
    from app.services.scaffold_generator import generate_scaffold
    result = GuardrailResult(
        terminal_state=GuardrailTerminalState.UNRESOLVED,
        attempt_count=3,
        requirements={
            "missing_hyperparameter": Requirement(
                name="missing_hyperparameter",
                state=EvidenceState.NOT_FOUND
            )
        }
    )
    scaffold = generate_scaffold(result)
    assert "MISSING PAPER EVIDENCE" in scaffold
    assert "MISSING_HYPERPARAMETER = None" in scaffold
    assert "raise NotImplementedError" in scaffold

def test_scaffold_generation_skipped_on_pass():
    from app.services.scaffold_generator import generate_scaffold
    result = GuardrailResult(
        terminal_state=GuardrailTerminalState.PASS,
        attempt_count=1,
        requirements={}
    )
    scaffold = generate_scaffold(result)
    assert scaffold == ""

def test_audit_report_generation():
    from app.services.audit_reporter import generate_audit_report
    result = GuardrailResult(
        terminal_state=GuardrailTerminalState.UNRESOLVED,
        attempt_count=3,
        reason="Missing fields",
        requirements={
            "learning_rate": Requirement(
                name="learning_rate",
                value="0.01",
                state=EvidenceState.SUPPORTED,
                evidence=None
            ),
            "batch_size": Requirement(
                name="batch_size",
                state=EvidenceState.NOT_FOUND,
                reason="Not in text"
            )
        }
    )
    report = generate_audit_report(result)
    assert "status: UNRESOLVED" in report
    assert "recovery attempts: 3" in report
    assert "batch_size" in report
    assert "status: NOT_FOUND" in report

def test_valid_paper_unsupported_implementation(evidence_guardrail, session_id):
    """Test 2: Valid paper but incompatible neural-network request"""
    responses = {}
    
    # 1. Paper Relevance succeeds
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "Bayesian logic-based belief maintenance", [create_evidence("bayesian logic")], True
    )
    
    # 2. Implementation Support fails
    task = "functional PyTorch neural-network model with layers, optimizer, learning rate and loss"
    impl_query = f"Find methodology, architecture, or algorithm details for: {task}"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "Not found", [], False
    )
    
    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description=task,
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    from app.guardrails.models import PaperRelevance, ImplementationSupport
    assert gr.paper_relevance == PaperRelevance.SUPPORTED
    assert gr.implementation_support == ImplementationSupport.IMPLEMENTATION_UNSUPPORTED
    assert gr.terminal_state == GuardrailTerminalState.IMPLEMENTATION_UNSUPPORTED
    assert gr.code_generation_allowed is False


def test_no_keyword_rejection(evidence_guardrail, session_id):
    """Test 5: No keyword-based rejection. Research document without typical ML keywords is supported."""
    responses = {}
    
    # 1. Paper Relevance succeeds without ML keywords
    domain_query = "What is the core methodology, model, or algorithm proposed in this research paper?"
    responses[domain_query] = ResearchResult(
        session_id, domain_query, "A novel heuristic search algorithm", [create_evidence("heuristic search")], True
    )
    
    # 2. Implementation Support succeeds
    task = "heuristic search algorithm implementation"
    impl_query = f"Find methodology, architecture, or algorithm details for: {task}"
    responses[impl_query] = ResearchResult(
        session_id, impl_query, "heuristic search methodology", [create_evidence("heuristic search")], True
    )
    
    agent = create_mock_research_agent(responses)
    domain_guardrail = DomainGuardrail(agent)
    
    # Let EvidenceGuardrail fail on architecture, dataset etc. since it's just a test of DomainGuardrail
    # We will get UNRESOLVED, but crucially NOT DOMAIN_UNSUPPORTED or IMPLEMENTATION_UNSUPPORTED
    
    context, gr = run_research_to_spec_workflow(
        session_id=session_id,
        task_description=task,
        research_agent=agent,
        domain_guardrail=domain_guardrail,
        evidence_guardrail=evidence_guardrail
    )
    
    from app.guardrails.models import PaperRelevance, ImplementationSupport
    assert gr.paper_relevance == PaperRelevance.SUPPORTED
    assert gr.implementation_support == ImplementationSupport.APPLICABLE
    assert gr.terminal_state == GuardrailTerminalState.UNRESOLVED

