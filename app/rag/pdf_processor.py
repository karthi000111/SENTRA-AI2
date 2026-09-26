"""PDF text extraction that preserves source and page provenance."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import fitz

from .models import ExtractedPage, ExtractionProblem, ExtractionReport, PDFInput


class PDFProcessor:
    """Extract selectable text from one or more PDFs; OCR is intentionally absent."""

    def extract_files(self, pdf_files: Iterable[PDFInput]) -> ExtractionReport:
        report = ExtractionReport()
        for raw_path in pdf_files:
            path = Path(raw_path)
            if not path.is_file():
                report.problems.append(ExtractionProblem(str(path), "File does not exist."))
            else:
                self._extract(report, path.name, filename=str(path))
        return report

    def extract_bytes(self, files: Iterable[tuple[str, bytes]]) -> ExtractionReport:
        report = ExtractionReport()
        for name, content in files:
            if not content:
                report.problems.append(ExtractionProblem(name, "PDF content is empty."))
            else:
                self._extract(report, name, stream=content)
        return report

    @staticmethod
    def _extract(report: ExtractionReport, name: str, *, filename: str | None = None,
                 stream: bytes | None = None) -> None:
        try:
            document = fitz.open(filename=filename) if filename else fitz.open(stream=stream, filetype="pdf")
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            report.problems.append(ExtractionProblem(name, f"Could not open PDF: {exc}"))
            return
        import re
        try:
            before = len(report.pages)
            in_references = False
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                
                # Check for headings that look like references
                if not in_references:
                    if re.search(r"(?im)^(?:\d+\.?\s*)?(References|Bibliography|Works Cited)\s*$", text):
                        in_references = True
                        
                is_ref = in_references
                if not is_ref:
                    # Regex safeguard for citation-shaped text
                    if re.search(r"(?i)(?:,\s*\d+\(\d+\):\d+-\d+,?\s*(?:19|20)\d{2}|[A-Z][a-z]+,\s*[A-Z]\.\s*\((?:19|20)\d{2}\)\.\s*[A-Z])", text):
                        # check if a large chunk of the text looks like references
                        matches = len(re.findall(r"(?i)(?:,\s*\d+\(\d+\):\d+-\d+,?\s*(?:19|20)\d{2}|[A-Z][a-z]+,\s*[A-Z]\.\s*\((?:19|20)\d{2}\)\.\s*[A-Z]|arxiv:\d{4}\.\d{4,5})", text))
                        if matches >= 3:
                            is_ref = True
                
                report.pages.append(ExtractedPage(name, page_number, text, is_reference_section=is_ref))
            if len(report.pages) == before:
                report.problems.append(ExtractionProblem(name, "No extractable text was found; OCR is not enabled."))
        finally:
            document.close()
