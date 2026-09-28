"""Professional Streamlit workspace for the existing SENTRA Research Agent."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env so HF_TOKEN persists across restarts
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

# Purge stale app modules from sys.modules to prevent long-running Streamlit processes from caching old code in RAM
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("app.") or mod_name == "app":
        del sys.modules[mod_name]

import streamlit as st

from app.services import ResearchWorkspace, WorkspaceStatus
from ui.components.evidence_panel import render_result
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.components.upload_panel import render_upload_panel
from ui.styles.theme import inject_theme


PARADIGM_DISPLAY_MAP = {
    "SYMBOLIC_PROBABILISTIC": "Symbolic Logic / Probabilistic",
    "DEEP_LEARNING": "Deep Learning",
    "CLASSICAL_ML": "Classical Machine Learning",
    "INDUSTRIAL_CONTROL": "Industrial Control System",
}

def format_paradigm(raw_paradigm: str) -> str:
    # If raw_paradigm is an Enum, we must extract its value or str representation.
    # The Enum in models.py inherits from str, so str(raw_paradigm) works.
    val = str(raw_paradigm).upper()
    if val.startswith("AIPARADIGM."):
        val = val.split(".")[-1]
    return PARADIGM_DISPLAY_MAP.get(
        val,
        val.replace("_", " ").title(),
    )

@st.cache_resource
def workspace() -> ResearchWorkspace:
    """One application service; backend session IDs remain the isolation boundary."""
    from app.services.research_workspace import ResearchWorkspace
    inst = ResearchWorkspace()
    if not hasattr(inst, "implement_paper"):
        # Force reload module if stale class definition was loaded
        import importlib
        import app.services.research_workspace as rw_mod
        importlib.reload(rw_mod)
        inst = rw_mod.ResearchWorkspace()
    return inst


def reset_ui_state() -> None:
    st.session_state.pop("sentra_session_id", None)
    st.session_state.pop("history", None)
    st.session_state.pop("confirm_delete", None)
    st.session_state.pop("last_context", None)
    st.session_state.pop("last_guardrail", None)


def active_status(service: ResearchWorkspace) -> WorkspaceStatus | None:
    session_id = st.session_state.get("sentra_session_id")
    if not session_id:
        return None
    try:
        return service.status(session_id)
    except (KeyError, ValueError):
        reset_ui_state()
        st.warning("The previous research session is no longer available.")
        return None


def create_session(service: ResearchWorkspace) -> None:
    st.session_state.sentra_session_id = service.create_session()
    st.session_state.history = []
    st.session_state.confirm_delete = False
    st.success("New research session created. Upload PDF papers to build its knowledge base.")


def main() -> None:
    st.set_page_config(page_title="SENTRA AI · Research Workspace", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_theme(st)
    service = workspace()
    status = active_status(service)
    render_header(st, bool(status))
    new_session, request_delete = render_sidebar(st, status)
    if new_session:
        if status:
            st.info("The current session remains available until you explicitly delete it.")
        create_session(service)
        st.rerun()
    if request_delete:
        st.session_state.confirm_delete = True

    status = active_status(service)
    if st.session_state.get("confirm_delete") and status:
        st.warning("Delete this temporary session and all of its uploaded PDFs, index, and evidence history?")
        confirm, cancel = st.columns(2)
        if confirm.button("Confirm delete", type="primary"):
            try:
                service.delete_session(status.session_id)
                reset_ui_state()
                st.success("Research session deleted.")
                st.rerun()
            except (KeyError, ValueError) as exc:
                st.error(f"Could not delete this session: {exc}")
        if cancel.button("Keep session"):
            st.session_state.confirm_delete = False
            st.rerun()

    if not status:
        st.markdown("## WELCOME TO SENTRA AI")
        st.markdown('<div class="card"><h3>Transform research papers into evidence-grounded research intelligence.</h3><p class="muted">Create a temporary research session. Your papers, FAISS index, and evidence remain isolated to that session.</p></div>', unsafe_allow_html=True)
        if st.button("Create Research Session", type="primary"):
            create_session(service)
            st.rerun()
        return

    left, right = st.columns([1, 1.75], gap="large")
    with left:
        files = render_upload_panel(st, disabled=False)
        if files:
            pending = [(file.name, file.getvalue()) for file in files]
            if st.button("Index uploaded documents", type="primary", use_container_width=True):
                try:
                    with st.spinner("Extracting text and building this session's knowledge base..."):
                        ingestion = service.ingest(status.session_id, pending)
                    if ingestion.problems:
                        st.warning("Some PDF content could not be extracted: " + "; ".join(problem.message for problem in ingestion.problems))
                    st.success(f"Knowledge Base READY — {ingestion.chunk_count} chunks indexed.")
                    st.rerun()
                except (KeyError, ValueError, RuntimeError) as exc:
                    st.error(f"The uploaded documents could not be processed: {exc}")
        st.markdown("---")
        st.markdown("#### SESSION-ISOLATED KNOWLEDGE")
        st.info(f"Only documents uploaded to this research session are available to the Research Agent.\n\nCurrent knowledge space: `{status.session_id}`")

    with right:
        st.markdown("### RESEARCH WORKSPACE")
        if not status.ready:
            st.markdown('<div class="card"><div class="label">YOUR RESEARCH SPACE IS EMPTY</div><h3>Upload one or more research papers to create your temporary knowledge base.</h3><p class="muted">Documents remain isolated within this research session.</p></div>', unsafe_allow_html=True)
            return
            
        qa_tab, spec_tab, code_tab = st.tabs(["Q&A", "Implementation Guardrail", "🔬 Code Implementation"])
        
        with qa_tab:
            query = st.text_area("Ask Research Agent", placeholder="Ask something about your uploaded research papers...", height=105)
            if st.button("Ask Research Agent", type="primary"):
                if not query.strip():
                    st.warning("Please enter a research question.")
                else:
                    try:
                        with st.spinner("Research Agent is analyzing your evidence..."):
                            result = service.ask(status.session_id, query)
                        st.session_state.history = [{"query": query, "result": result}, *st.session_state.get("history", [])][:8]
                    except (KeyError, ValueError, RuntimeError) as exc:
                        st.error(f"The Research Agent could not retrieve evidence: {exc}")
            history = st.session_state.get("history", [])
            if history:
                render_result(st, history[0]["result"])
                with st.expander("RECENT QUESTIONS"):
                    for item in history:
                        st.caption(f"• {item['query']}")
                    
        with spec_tab:
            st.markdown("### Guardrail Verification")
            st.markdown("Ensure the uploaded paper provides sufficient ML evidence before code generation.")
            
            if st.button("Run Guardrail", type="primary", use_container_width=True):
                with st.spinner("Executing domain and evidence guardrails..."):
                    context, gr = service.generate_ml_specification(status.session_id)
                st.session_state.last_context = context
                st.session_state.last_guardrail = gr
            
            gr = st.session_state.get("last_guardrail")
            context = st.session_state.get("last_context")
            
            if gr:
                st.markdown("---")
                st.markdown(f"#### Guardrail Execution Summary: {gr.terminal_state.value}")
                
                if gr.reason:
                    st.error(f"Reason: {gr.reason}")
                
                if gr.detected_paradigm:
                    st.info(f"**Detected Paradigm:** {format_paradigm(gr.detected_paradigm)}")
                
                if getattr(gr, "requirements", None):
                    st.markdown("### Evidence Claims")
                    for name, req in gr.requirements.items():
                        # Hide non-applicable claims
                        if req.state.value == "NOT_APPLICABLE_TO_PARADIGM":
                            continue
                        # Only show unresolved claims if they are not supported
                        if req.state.value != "SUPPORTED" and gr.attempt_count < 3:
                            continue
                            
                        # Formatting
                        if req.state.value == "SUPPORTED":
                            st.success(f"**{name.replace('_', ' ').title()}** - {req.state.value}")
                        else:
                            st.warning(f"**{name.replace('_', ' ').title()}** - {req.state.value}")
                            
                        if req.evidence and req.evidence.evidence_text:
                            st.markdown(f"> Evidence found in {req.evidence.source}, page {req.evidence.page}: {req.evidence.evidence_text}")
                        elif req.value:
                            st.markdown(f"> {req.value}")
                        if req.reason and req.state.value != "SUPPORTED":
                            st.markdown(f"*Reason:* {req.reason}")
                            
                st.markdown("---")

        # -----------------------------------------------------------------
        # Code Implementation tab
        # -----------------------------------------------------------------
        with code_tab:
            st.markdown("### 🔬 Code Implementation")
            st.markdown(
                "Generate a runnable Python implementation of the paper, "
                "grounded in the verified specifications extracted by the guardrail."
            )

            gr = st.session_state.get("last_guardrail")

            if not gr:
                st.markdown(
                    '<div class="card">'
                    '<div class="label">NO GUARDRAIL RESULT YET</div>'
                    '<h3>Run the Implementation Guardrail first.</h3>'
                    '<p class="muted">Switch to the "Implementation Guardrail" tab '
                    'and click <strong>Run Guardrail</strong> to validate the paper.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif not gr.code_generation_allowed:
                st.warning(
                    "⛔ The guardrail did **not** pass — code generation is blocked.  \n"
                    "Reason: " + (gr.reason or "insufficient evidence")
                )
            else:
                # --- Guardrail passed → show Implement Paper button --------
                st.success(
                    f"✅ Guardrail **PASSED** "
                    f"({gr.completeness_score:.0%} spec coverage, "
                    f"paradigm: {format_paradigm(gr.detected_paradigm or 'unknown')}).  \n"
                    "Ready to generate a paper implementation."
                )

                if st.button(
                    "🚀 Implement Paper",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Implementing your paper..."):
                        try:
                            if not hasattr(service, "implement_paper"):
                                st.cache_resource.clear()
                                service = workspace()
                            ctx, updated_gr, gen_result = service.implement_paper(
                                status.session_id, gr
                            )
                            st.session_state.last_guardrail = updated_gr
                            st.session_state.last_context = ctx
                            st.session_state.last_gen_result = gen_result
                        except Exception as exc:
                            st.error(f"Code generation failed: {exc}")

                # --- Display previous / just-generated result ---------------
                gen_result = st.session_state.get("last_gen_result")
                if gen_result:
                    if gen_result.success:
                        st.markdown("---")
                        if getattr(gen_result, "fully_grounded", True):
                            st.markdown("#### ✅ Implementation Generated (Fully Grounded)")
                        else:
                            st.markdown(f"#### ⚠️ Implementation Generated ({len(gen_result.missing_specs)} Specs Uncited)")

                        # Missing Specs warning (if any)
                        missing = getattr(gen_result, "missing_specs", [])
                        if missing:
                            with st.expander(f"⚠️ Uncited Grounded Specs ({len(missing)})", expanded=True):
                                st.caption("The following grounded specifications were extracted from the paper but not explicitly tagged with `# SPEC:` in the generated code:")
                                for m_spec in missing:
                                    st.markdown(f"- `{m_spec}`")

                        # Spec citations
                        if gen_result.spec_citations_found:
                            with st.expander(f"📎 Spec Citations ({len(gen_result.spec_citations_found)})", expanded=False):
                                for cite in gen_result.spec_citations_found:
                                    st.markdown(f"- `{cite}`")

                        # Disclosed Deferrals (if any)
                        deferred_items = getattr(gen_result, "deferred_specs", [])
                        if deferred_items:
                            with st.expander(f"📌 Disclosed Deferrals ({len(deferred_items)})", expanded=True):
                                st.caption("The generator explicitly disclosed the following deferred items / out-of-scope components:")
                                for d_item in deferred_items:
                                    st.markdown(f"- `{d_item}`")

                        # Assumptions
                        if gen_result.assumptions_found:
                            with st.expander(f"⚠️ Assumptions Made ({len(gen_result.assumptions_found)})", expanded=True):
                                for assumption in gen_result.assumptions_found:
                                    st.markdown(f"- {assumption}")

                        # The code
                        st.markdown("#### Generated Code")
                        files = getattr(gen_result, "files", [])
                        if files:
                            file_tabs = st.tabs([f"📄 {f.filename}" for f in files])
                            for tab, f in zip(file_tabs, files):
                                with tab:
                                    st.code(f.code, language="python")
                                    st.download_button(
                                        label=f"💾 Download {f.filename}",
                                        data=f.code,
                                        file_name=f.filename,
                                        mime="text/x-python",
                                        key=f"download_{f.filename}",
                                    )
                            
                            # ZIP download for whole project
                            import io
                            import zipfile
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                for f in files:
                                    zf.writestr(f.filename, f.code)
                            zip_buffer.seek(0)
                            
                            st.markdown("---")
                            st.download_button(
                                label="📦 Download Complete Project (.zip)",
                                data=zip_buffer.getvalue(),
                                file_name="paper_implementation.zip",
                                mime="application/zip",
                                use_container_width=True,
                            )
                        else:
                            st.code(gen_result.generated_code, language="python")
                            st.download_button(
                                label="💾 Download as .py",
                                data=gen_result.generated_code,
                                file_name="paper_implementation.py",
                                mime="text/x-python",
                                use_container_width=True,
                            )
                    else:
                        st.markdown("---")
                        st.error(f"Code generation failed: {gen_result.error_reason}")
                        if gen_result.generated_code:
                            with st.expander("Partial / failed output"):
                                st.code(gen_result.generated_code, language="python")

if __name__ == "__main__":
    main()

