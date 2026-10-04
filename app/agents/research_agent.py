"""LLM-grounded research agent backed by one uploaded-document session.

The agent retrieves evidence from the session corpus, then uses an LLM to
synthesise an answer *strictly grounded* in that evidence.  A faithfulness
score (0.0 – 1.0) quantifies how much of the generated answer is supported
by the retrieved context, enabling direct measurement of hallucination
avoidance.

When no LLM callable is provided the agent falls back to extractive-only
answers (identical to the original behaviour) so all existing call-sites
remain compatible.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from app.rag.session_manager import SessionRAGManager

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "according", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "paper", "proposed",
    "the", "this", "to", "was", "what", "were", "which", "with",
})

# ---------------------------------------------------------------------------
# Grounded system prompt — instructs the LLM to answer ONLY from context
# ---------------------------------------------------------------------------
_RESEARCH_SYSTEM_PROMPT = (
    "You are a precise research assistant. You MUST answer the user's "
    "question using ONLY the evidence provided below.  Do NOT add any "
    "information that is not explicitly stated in the evidence context. "
    "If the evidence is insufficient, say so clearly. "
    "Cite the source document and page number for every claim you make."
)

_RESEARCH_USER_TEMPLATE = (
    "### EVIDENCE CONTEXT\n{context}\n\n"
    "### QUESTION\n{query}\n\n"
    "Answer the question based ONLY on the evidence context above. "
    "For each statement in your answer, cite the source (document name and page). "
    "If the context does not contain enough information, state that explicitly."
)


@dataclass(frozen=True)
class ResearchResult:
    """Stable hand-off shape for later agents (such as Specification Agent).

    Attributes
    ----------
    faithfulness_score : float
        Fraction of claims in the generated answer that are supported by the
        retrieved evidence (1.0 = no hallucination, 0.0 = fully hallucinated).
        Set to 1.0 for extractive-only answers (they quote verbatim evidence).
    """

    session_id: str
    query: str
    answer: str
    evidence: list[dict[str, object]]
    sufficient_evidence: bool
    faithfulness_score: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "evidence": self.evidence,
            "sufficient_evidence": self.sufficient_evidence,
            "faithfulness_score": self.faithfulness_score,
        }


class ResearchAgent:
    """Answer questions grounded in the evidence from a supplied session.

    There is intentionally no filesystem or global-corpus access here: all
    document access occurs through :class:`SessionRAGManager.retrieve`.

    When ``llm_callable`` is provided the agent synthesises a natural-language
    answer via the LLM and then computes a **faithfulness score** — the
    proportion of sentences in the answer whose key claims can be traced back
    to the retrieved evidence text.  This enables the RAGAS-style
    "faithfulness" metric for measuring hallucination avoidance.

    When ``llm_callable`` is ``None`` the agent falls back to extractive
    answers (direct quotes), which are trivially faithful.
    """

    def __init__(
        self,
        rag_manager: SessionRAGManager,
        llm_callable: Callable[[str], str] | None = None,
    ) -> None:
        self._rag_manager = rag_manager
        self._llm = llm_callable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *, session_id: str, query: str, top_k: int = 3) -> ResearchResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        evidence = self._rag_manager.retrieve(session_id, query, top_k)
        keywords = self._keywords(query)
        supported = [
            item for item in evidence
            if self._supports_query(str(item["text"]), keywords)
        ]

        if not supported:
            return ResearchResult(
                session_id=session_id,
                query=query,
                answer="The uploaded documents do not provide sufficient evidence to answer this question.",
                evidence=[],
                sufficient_evidence=False,
                faithfulness_score=1.0,  # no answer generated → no hallucination
            )

        # ----- LLM path: generate a grounded answer --------------------
        if self._llm is not None:
            return self._llm_answer(session_id, query, supported)

        # ----- Extractive fallback (original behaviour) -----------------
        best = supported[0]
        excerpt = self._best_sentence(str(best["text"]), keywords)
        answer = f"Based on {best['source']}, page {best['page']}: {excerpt}"
        return ResearchResult(session_id, query, answer, supported, True, faithfulness_score=1.0)

    # ------------------------------------------------------------------
    # LLM-grounded answer generation
    # ------------------------------------------------------------------

    def _llm_answer(
        self,
        session_id: str,
        query: str,
        evidence: list[dict[str, object]],
    ) -> ResearchResult:
        """Build a context-grounded prompt, call the LLM, and score faithfulness."""

        context_block = self._build_context_block(evidence)
        prompt = (
            f"{_RESEARCH_SYSTEM_PROMPT}\n\n"
            f"{_RESEARCH_USER_TEMPLATE.format(context=context_block, query=query)}"
        )

        try:
            raw_answer = self._llm(prompt)
            if not raw_answer or not raw_answer.strip():
                raise ValueError("LLM returned an empty response.")
        except Exception as exc:
            logger.warning("LLM call failed (%s), falling back to extractive answer.", exc)
            best = evidence[0]
            keywords = self._keywords(query)
            excerpt = self._best_sentence(str(best["text"]), keywords)
            answer = f"Based on {best['source']}, page {best['page']}: {excerpt}"
            return ResearchResult(session_id, query, answer, evidence, True, faithfulness_score=1.0)

        answer = raw_answer.strip()

        # Compute faithfulness: what fraction of answer sentences are
        # grounded in the evidence context.
        faithfulness = self._compute_faithfulness(answer, evidence)

        logger.info(
            "Research LLM answer: %d chars, faithfulness=%.2f, evidence_chunks=%d",
            len(answer), faithfulness, len(evidence),
        )

        return ResearchResult(
            session_id=session_id,
            query=query,
            answer=answer,
            evidence=evidence,
            sufficient_evidence=True,
            faithfulness_score=faithfulness,
        )

    # ------------------------------------------------------------------
    # Faithfulness scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_faithfulness(
        answer: str,
        evidence: list[dict[str, object]],
    ) -> float:
        """Compute a RAGAS-style faithfulness score.

        Algorithm
        ---------
        1. Split the answer into individual sentences (claims).
        2. For each sentence, extract non-trivial keywords.
        3. A sentence is considered *faithful* if ≥ 50% of its keywords
           appear in at least one evidence chunk.
        4. ``faithfulness = faithful_sentences / total_sentences``.

        This is a lightweight, embedding-free approximation of the RAGAS
        faithfulness metric.  It runs locally with zero API cost.
        """
        sentences = ResearchAgent._split_sentences(answer)
        if not sentences:
            return 1.0

        # Combine all evidence text into one searchable corpus
        evidence_corpus = " ".join(
            str(ev.get("text", "")).lower() for ev in evidence
        )
        evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_corpus))

        faithful_count = 0
        for sentence in sentences:
            claim_keywords = ResearchAgent._keywords(sentence)
            if not claim_keywords:
                # Trivial sentence (stop-words only) — considered faithful
                faithful_count += 1
                continue
            overlap = claim_keywords & evidence_tokens
            # A sentence is faithful if ≥50% of its keywords are in evidence
            if len(overlap) / len(claim_keywords) >= 0.5:
                faithful_count += 1

        return round(faithful_count / len(sentences), 4)

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_block(evidence: list[dict[str, object]]) -> str:
        """Format retrieved evidence chunks into a numbered context block."""
        lines: list[str] = []
        for i, ev in enumerate(evidence, 1):
            source = ev.get("source", ev.get("filename", "unknown"))
            page = ev.get("page", "?")
            text = str(ev.get("text", "")).strip()
            lines.append(f"[{i}] Source: {source}, Page {page}\n{text}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Text-processing utilities (preserved from original implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences, filtering out empty strings."""
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _keywords(query: str) -> set[str]:
        return {
            word for word in re.findall(r"[a-z0-9]+", query.lower())
            if len(word) > 2 and word not in _STOP_WORDS
        }

    @staticmethod
    def _supports_query(text: str, keywords: set[str]) -> bool:
        # A query made only of stop words should never be treated as evidence.
        if not keywords:
            return False
        terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        return bool(keywords & terms)

    @staticmethod
    def _best_sentence(text: str, keywords: set[str]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        matching = [
            sentence for sentence in sentences
            if ResearchAgent._supports_query(sentence, keywords)
        ]
        return (matching[0] if matching else text).strip()
