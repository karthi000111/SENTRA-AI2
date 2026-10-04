"""Thin application-facing integration layer for the Research Agent UI."""
from __future__ import annotations

from dataclasses import dataclass

from app.agents import ResearchAgent, ResearchResult
from app.agents.llm_backend import call_hf_inference
from app.guardrails.models import GuardrailResult
from app.guardrails.domain_guardrail import DomainGuardrail
from app.guardrails.evidence_guardrail import EvidenceGuardrail
from app.services.implementation_context import ImplementationContext
from app.rag import SessionIngestionResult, SessionRAGManager


@dataclass(frozen=True)
class WorkspaceStatus:
    session_id: str
    documents: list[str]
    chunk_count: int
    ready: bool


class ResearchWorkspace:
    """Coordinates UI actions without duplicating RAG or agent behavior."""

    def __init__(self, manager: SessionRAGManager | None = None) -> None:
        self._manager = manager or SessionRAGManager()
        self._agent = ResearchAgent(self._manager, llm_callable=call_hf_inference)
        self._domain_guardrail = DomainGuardrail(self._agent)
        self._evidence_guardrail = EvidenceGuardrail(max_attempts=3)

    def create_session(self) -> str:
        return self._manager.create_session()

    def status(self, session_id: str) -> WorkspaceStatus:
        summary = self._manager.session_summary(session_id)
        return WorkspaceStatus(
            session_id=session_id,
            documents=list(summary["documents"]),
            chunk_count=int(summary["chunk_count"]),
            ready=bool(summary["ready"]),
        )

    def ingest(self, session_id: str, files: list[tuple[str, bytes]]) -> SessionIngestionResult:
        if not files:
            raise ValueError("Please upload at least one PDF.")
        existing = set(self.status(session_id).documents)
        unique_files = [(name, content) for name, content in files if name not in existing]
        if not unique_files:
            raise ValueError("These documents are already indexed in this session.")
        return self._manager.add_documents(session_id, unique_files)

    def ask(self, session_id: str, query: str) -> ResearchResult:
        return self._agent.run(session_id=session_id, query=query)

    def delete_session(self, session_id: str) -> None:
        self._manager.delete_session(session_id)

    def generate_ml_specification(
        self,
        session_id: str,
        task: str = None
    ) -> tuple[ImplementationContext | None, GuardrailResult]:
        """Runs the bounded research-to-specification workflow."""
        from app.orchestration.workflow import run_research_to_spec_workflow
        return run_research_to_spec_workflow(
            session_id=session_id,
            task=task,
            research_agent=self._agent,
            domain_guardrail=self._domain_guardrail,
            evidence_guardrail=self._evidence_guardrail
        )

    def implement_paper(
        self,
        session_id: str,
        guardrail_result: GuardrailResult,
    ) -> tuple[ImplementationContext, GuardrailResult, "CodeGenResult"]:
        """Generate code from a passing guardrail result (calls real LLM).

        Should only be called after a PASS guardrail result.
        """
        import importlib
        import app.orchestration.workflow as wf
        if not hasattr(wf, "run_code_generation"):
            importlib.reload(wf)
        return wf.run_code_generation(
            session_id=session_id,
            guardrail_result=guardrail_result,
        )

    def test_implementation(
        self,
        code_or_files: Any,
        specs: dict | None = None,
    ) -> "TestSuiteResult":
        """Runs the 10 dynamic ML domain test cases against generated code."""
        from app.testing.ml_testing_agent import MLTestingAgent
        agent = MLTestingAgent()
        return agent.run_test_suite(code_or_files, specs=specs)

