"""Command-line demonstration for the session-scoped Research Agent."""
from __future__ import annotations

import argparse
from pathlib import Path

from app.agents import ResearchAgent
from app.agents.llm_backend import call_hf_inference
from app.rag import SessionRAGManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SENTRA AI Research Agent against uploaded PDFs.")
    parser.add_argument("--demo", nargs="+", metavar="PDF", help="One or more local PDF files to ingest temporarily.")
    parser.add_argument("--question", help="Research question to ask of the supplied PDFs.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM and use extractive-only answers.")
    args = parser.parse_args()
    if not args.demo:
        print("Sentra AI project initialized. Use --demo paper.pdf --question 'Your question'.")
        return
    if not args.question:
        parser.error("--question is required with --demo")
    files: list[tuple[str, bytes]] = []
    for raw_path in args.demo:
        path = Path(raw_path)
        if not path.is_file():
            parser.error(f"PDF does not exist: {path}")
        files.append((path.name, path.read_bytes()))
    rag = SessionRAGManager()
    session_id = rag.create_session()
    try:
        ingestion = rag.add_documents(session_id, files)
        llm_fn = None if args.no_llm else call_hf_inference
        result = ResearchAgent(rag, llm_callable=llm_fn).run(
            session_id=session_id, query=args.question,
        )
        print("=" * 50)
        print("SENTRA AI - RESEARCH AGENT DEMO")
        print("=" * 50)
        print(f"Session: {session_id}")
        print(f"Documents: {', '.join(name for name, _ in files)}")
        print(f"Chunks indexed: {ingestion.chunk_count}")
        print(f"Question: {args.question}\n")
        print("RESEARCH AGENT")
        print(result.answer)
        print(f"\nFAITHFULNESS SCORE: {result.faithfulness_score:.1%}")
        print(f"  → {result.faithfulness_score:.1%} of the answer is grounded in evidence")
        print(f"  → {1 - result.faithfulness_score:.1%} potential hallucination")
        print("\nEVIDENCE")
        for item in result.evidence:
            print(f"Source: {item['source']} | Page: {item['page']} | Score: {item['score']:.3f}")
            print(f'"{item["text"]}"\n')
    finally:
        rag.delete_session(session_id)


if __name__ == "__main__":
    main()
