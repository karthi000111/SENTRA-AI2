from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
import numpy as np
import pytest

from app.agents.research_agent import ResearchAgent
from app.config.rag import RAGSettings
from app.rag.session_manager import SessionRAGManager


class KeywordEmbedder:
    vocabulary = ("alpha", "beta", "dataset", "monitoring", "methodology")
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[text.lower().split().count(word) for word in self.vocabulary] for text in texts], dtype=np.float32)


def pdf_bytes(pages: list[str]) -> bytes:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    result = BytesIO()
    document.save(result)
    document.close()
    return result.getvalue()


def make_agent(tmp_path: Path, llm_callable=None) -> tuple[SessionRAGManager, ResearchAgent]:
    rag = SessionRAGManager(settings=RAGSettings(session_root=tmp_path / "sessions", relevance_threshold=0.20), embedder=KeywordEmbedder())
    return rag, ResearchAgent(rag, llm_callable=llm_callable)


# ---------------------------------------------------------------------------
# Original extractive tests (backward-compatible, no LLM)
# ---------------------------------------------------------------------------

def test_agent_returns_evidence_grounded_answer_with_source_and_page(tmp_path: Path) -> None:
    rag, agent = make_agent(tmp_path)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("study.pdf", pdf_bytes(["Introductory material.", "The alpha methodology uses a curated dataset for monitoring."]))])
    result = agent.run(session_id=session_id, query="What methodology uses the dataset?")
    assert result.sufficient_evidence
    assert "study.pdf, page 2" in result.answer
    assert result.evidence[0]["source"] == "study.pdf"
    assert result.evidence[0]["page"] == 2
    assert result.evidence[0]["score"] > 0
    assert result.as_dict()["session_id"] == session_id
    assert result.faithfulness_score == 1.0  # extractive = trivially faithful


def test_agent_handles_multiple_documents_and_keeps_source_traceability(tmp_path: Path) -> None:
    rag, agent = make_agent(tmp_path)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("methods.pdf", pdf_bytes(["The alpha methodology is evaluated carefully."])), ("data.pdf", pdf_bytes(["The beta dataset contains monitoring records."]))])
    result = agent.run(session_id=session_id, query="What beta dataset was used?")
    assert result.sufficient_evidence
    assert result.evidence[0]["source"] == "data.pdf"


def test_agent_refuses_to_guess_when_session_has_no_supporting_evidence(tmp_path: Path) -> None:
    rag, agent = make_agent(tmp_path)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("alpha.pdf", pdf_bytes(["Alpha methodology is documented."]))])
    result = agent.run(session_id=session_id, query="What beta dataset was used?")
    assert not result.sufficient_evidence
    assert result.evidence == []
    assert "sufficient evidence" in result.answer.lower()
    assert result.faithfulness_score == 1.0  # no answer = no hallucination


def test_agent_enforces_session_isolation_and_deleted_sessions_fail(tmp_path: Path) -> None:
    rag, agent = make_agent(tmp_path)
    session_a, session_b = rag.create_session(), rag.create_session()
    rag.add_documents(session_a, [("alpha.pdf", pdf_bytes(["Alpha methodology belongs only to A."]))])
    rag.add_documents(session_b, [("beta.pdf", pdf_bytes(["Beta dataset belongs only to B."]))])
    result_a = agent.run(session_id=session_a, query="What alpha methodology is described?")
    leaked = agent.run(session_id=session_a, query="What beta dataset is described?")
    result_b = agent.run(session_id=session_b, query="What beta dataset is described?")
    assert result_a.evidence[0]["source"] == "alpha.pdf"
    assert not leaked.sufficient_evidence and leaked.evidence == []
    assert result_b.evidence[0]["source"] == "beta.pdf"
    rag.delete_session(session_a)
    with pytest.raises(KeyError, match="Unknown RAG session"):
        agent.run(session_id=session_a, query="alpha methodology")
    assert agent.run(session_id=session_b, query="beta dataset").sufficient_evidence


def test_agent_validates_empty_query_and_session_without_index(tmp_path: Path) -> None:
    rag, agent = make_agent(tmp_path)
    session_id = rag.create_session()
    with pytest.raises(ValueError, match="non-empty"):
        agent.run(session_id=session_id, query=" ")
    with pytest.raises(KeyError, match="no searchable index"):
        agent.run(session_id=session_id, query="alpha")


# ---------------------------------------------------------------------------
# LLM-grounded answer tests
# ---------------------------------------------------------------------------

def _mock_llm_faithful(prompt: str) -> str:
    """Mock LLM that returns an answer grounded in the evidence context."""
    return (
        "Based on the evidence from study.pdf, page 2, the alpha methodology "
        "uses a curated dataset for monitoring purposes."
    )


def _mock_llm_hallucinating(prompt: str) -> str:
    """Mock LLM that hallucinates information not in the evidence."""
    return (
        "The quantum entanglement method uses a photonic crystal array "
        "for teleportation-based communication across galaxies."
    )


def _mock_llm_partial(prompt: str) -> str:
    """Mock LLM that is partially faithful and partially hallucinating."""
    return (
        "The alpha methodology uses a dataset for monitoring. "
        "Additionally, it leverages quantum computing for speed optimization."
    )


def _mock_llm_failing(prompt: str) -> str:
    """Mock LLM that raises an error."""
    raise ConnectionError("API unavailable")


def test_llm_grounded_answer_has_high_faithfulness(tmp_path: Path) -> None:
    """When the LLM sticks to the evidence, faithfulness should be 1.0."""
    rag, agent = make_agent(tmp_path, llm_callable=_mock_llm_faithful)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("study.pdf", pdf_bytes(["Intro.", "The alpha methodology uses a curated dataset for monitoring."]))])
    result = agent.run(session_id=session_id, query="What methodology uses the dataset?")
    assert result.sufficient_evidence
    assert result.faithfulness_score >= 0.8  # high faithfulness
    assert "faithfulness_score" in result.as_dict()


def test_hallucinating_llm_has_low_faithfulness(tmp_path: Path) -> None:
    """When the LLM makes up information, faithfulness should be low."""
    rag, agent = make_agent(tmp_path, llm_callable=_mock_llm_hallucinating)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("study.pdf", pdf_bytes(["Intro.", "The alpha methodology uses a curated dataset for monitoring."]))])
    result = agent.run(session_id=session_id, query="What methodology uses the dataset?")
    assert result.sufficient_evidence
    assert result.faithfulness_score < 0.5  # low faithfulness = hallucination detected


def test_partially_faithful_llm_has_moderate_score(tmp_path: Path) -> None:
    """A mix of faithful and hallucinated content yields a mid-range score."""
    rag, agent = make_agent(tmp_path, llm_callable=_mock_llm_partial)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("study.pdf", pdf_bytes(["Intro.", "The alpha methodology uses a dataset for monitoring."]))])
    result = agent.run(session_id=session_id, query="What methodology uses the dataset?")
    assert result.sufficient_evidence
    assert 0.3 <= result.faithfulness_score <= 0.8  # somewhere in between


def test_llm_failure_falls_back_to_extractive(tmp_path: Path) -> None:
    """When the LLM fails, the agent gracefully falls back to extractive answers."""
    rag, agent = make_agent(tmp_path, llm_callable=_mock_llm_failing)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("study.pdf", pdf_bytes(["Intro.", "The alpha methodology uses a curated dataset for monitoring."]))])
    result = agent.run(session_id=session_id, query="What methodology uses the dataset?")
    assert result.sufficient_evidence
    assert result.faithfulness_score == 1.0  # extractive fallback = trivially faithful
    assert "study.pdf, page 2" in result.answer


def test_faithfulness_score_in_as_dict(tmp_path: Path) -> None:
    """Ensure faithfulness_score is included in the serialised dict."""
    rag, agent = make_agent(tmp_path, llm_callable=_mock_llm_faithful)
    session_id = rag.create_session()
    rag.add_documents(session_id, [("study.pdf", pdf_bytes(["The alpha methodology uses monitoring."]))])
    result = agent.run(session_id=session_id, query="What alpha methodology?")
    data = result.as_dict()
    assert "faithfulness_score" in data
    assert 0.0 <= data["faithfulness_score"] <= 1.0


def test_compute_faithfulness_all_grounded() -> None:
    """Direct test of the faithfulness computation function."""
    evidence = [{"text": "The alpha methodology uses a dataset for monitoring."}]
    answer = "The alpha methodology uses a dataset for monitoring."
    score = ResearchAgent._compute_faithfulness(answer, evidence)
    assert score == 1.0


def test_compute_faithfulness_completely_hallucinated() -> None:
    """When the answer shares no keywords with evidence, faithfulness ≈ 0."""
    evidence = [{"text": "The alpha methodology uses a dataset for monitoring."}]
    answer = "Quantum entanglement enables teleportation across galaxies."
    score = ResearchAgent._compute_faithfulness(answer, evidence)
    assert score < 0.5
