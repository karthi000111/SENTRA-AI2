from __future__ import annotations

from app.services import WorkspaceStatus


def render_sidebar(st: object, status: WorkspaceStatus | None) -> tuple[bool, bool]:
    with st.sidebar:
        st.markdown("### RESEARCH SESSION")
        if status:
            st.markdown('<div class="label">CURRENT SESSION</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="session-id">{status.session_id}</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### KNOWLEDGE BASE")
            state = "● READY" if status.ready else "● EMPTY"
            css = "ready" if status.ready else "empty"
            st.markdown(f'<div class="{css}">{state}</div>', unsafe_allow_html=True)
            one, two = st.columns(2)
            one.metric("DOCUMENTS", len(status.documents))
            two.metric("CHUNKS", status.chunk_count)
            if status.documents:
                st.markdown("#### DOCUMENTS")
                for document in status.documents:
                    st.caption(f"✓ {document}")
            st.markdown("---")
        new_session = st.button("+ New Research Session", use_container_width=True)
        delete_session = bool(status) and st.button("Delete Session", type="secondary", use_container_width=True)
        st.markdown("---")
        st.markdown("#### PIPELINE")
        st.caption("✓ Research")
        st.caption("✓ Evidence Guardrail")
        st.caption("✓ Code Generation")

        # --- HF Token input ---
        st.markdown("---")
        st.markdown("#### 🔑 HUGGING FACE")
        hf_token = st.text_input(
            "HF Token",
            type="password",
            value=st.session_state.get("hf_token", ""),
            placeholder="hf_...",
            help="Paste your [Hugging Face token](https://huggingface.co/settings/tokens) to enable code generation.",
        )
        if hf_token:
            st.session_state.hf_token = hf_token
            import os
            os.environ["HF_TOKEN"] = hf_token
            st.caption("✅ Token set")
        else:
            st.caption("Paste token to enable Implement Paper")

        return new_session, delete_session
