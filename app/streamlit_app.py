"""
FinReg Navigator — Streamlit App
Location: project_root/app/streamlit_app.py

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import uuid
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import streamlit as st
from logs.logging_config import setup_logging
from src.core.router import Router

setup_logging()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinReg Navigator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load CSS ─────────────────────────────────────────────────────────────────
_css_path = Path(__file__).parent / "style.css"
with open(_css_path, "r", encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────
if "router" not in st.session_state:
    st.session_state.router = Router()

if "history" not in st.session_state:
    st.session_state.history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ FinReg Navigator")
    st.markdown("*Pakistan Financial Regulations Intelligence*")
    st.markdown("---")

    st.markdown("**📂 Upload Documents**")
    uploaded_files = st.file_uploader(
        "PDF or image files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader"
    )

    if uploaded_files:
        for f in uploaded_files:
            size_kb = max(1, len(f.getvalue()) // 1024)
            icon = "📄" if f.name.lower().endswith(".pdf") else "🖼️"
            st.markdown(f"`{icon} {f.name}` — {size_kb} KB")

    st.markdown("---")
    verbose = st.toggle("Verbose mode", value=False)

    # Clear chat
    if st.session_state.history:
        st.markdown("---")
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.rerun()

    # History preview
    if st.session_state.history:
        st.markdown("---")
        st.markdown("**🕐 History**")
        for h in reversed(st.session_state.history[-8:]):
            preview = h["query"][:42] + "..." if len(h["query"]) > 42 else h["query"]
            st.markdown(f'<div class="history-item">↳ {preview}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;'
        'color:#444;letter-spacing:0.1em;">FINREG NAVIGATOR v1.0<br>'
        'SBP · SECP · FBR</div>',
        unsafe_allow_html=True
    )

# ─── Main header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-header">
    <div class="nav-title">FinReg Navigator</div>
    <div class="nav-sub">Pakistan · Financial Regulation Intelligence</div>
</div>
<div class="nav-rule">── SBP &nbsp;·&nbsp; SECP &nbsp;·&nbsp; FBR &nbsp;·&nbsp; EMI &nbsp;·&nbsp; NBFC &nbsp;·&nbsp; Banking ──</div>
""", unsafe_allow_html=True)

# ─── Render chat history ──────────────────────────────────────────────────────
mode_labels = {
    "regulatory_only": "⚖ Regulatory KB",
    "uploaded_only":   "📄 Uploaded Doc",
    "compare":         "⚖ ↔ 📄 Compare",
}

for h in st.session_state.history:
    # User bubble
    st.markdown(
        f'<div class="chat-user">'
        f'<span class="chat-label">You</span>{h["query"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    # Build pills
    pills_html = ""
    if h["reg_chunks"]:
        pills_html += f'<span class="ctx-pill hit">📚 {h["reg_chunks"]} KB chunk(s)</span>'
    if h["upl_chunks"]:
        pills_html += f'<span class="ctx-pill hit">📄 {h["upl_chunks"]} doc chunk(s)</span>'
    if h["web"]:
        pills_html += f'<span class="ctx-pill web">🌐 {len(h["web"])} web source(s)</span>'
    if not pills_html:
        pills_html = '<span class="ctx-pill">⚠ No sources found</span>'

    # Build web sources block
    web_html = ""
    if h["web"]:
        web_html = '<div class="web-sources-block"><div class="web-sources-label">Web Sources</div>'
        for r in h["web"]:
            web_html += (
                f'<div class="web-source">'
                f'<a href="{r["url"]}" target="_blank">{r["title"]}</a>'
                f'<br><span style="color:#999;font-size:0.68rem">{r["url"]}</span>'
                f'</div>'
            )
        web_html += '</div>'

    answer_html = h["answer"].replace("\n", "<br>")

    # Assistant bubble
    st.markdown(f"""
    <div class="chat-assistant">
        <span class="chat-label">FinReg Navigator</span>
        <div class="mode-badge">{mode_labels.get(h["mode"], h["mode"])}</div>
        <div class="pills-row">{pills_html}</div>
        <div class="answer-body">{answer_html}</div>
        {web_html}
    </div>
    """, unsafe_allow_html=True)

    if h["steps"]:
        with st.expander("Execution trace", expanded=False):
            for s in h["steps"]:
                st.markdown(f'<div class="step-row">{s}</div>', unsafe_allow_html=True)

# ─── Processing status placeholder ───────────────────────────────────────────
status_placeholder = st.empty()

# Show "thinking" bubble while processing
if st.session_state.is_processing and st.session_state.pending_query:
    st.markdown(
        f'<div class="chat-user">'
        f'<span class="chat-label">You</span>{st.session_state.pending_query}'
        f'</div>',
        unsafe_allow_html=True
    )
    status_placeholder.markdown(
        '<div class="chat-assistant thinking">'
        '<span class="chat-label">FinReg Navigator</span>'
        '<div class="thinking-dots">'
        'Searching knowledge base<span class="dots">...</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ─── Input bar ────────────────────────────────────────────────────────────────
st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])

with col_input:
    placeholder_text = (
        "Answering your query..." if st.session_state.is_processing
        else "Type your question here..."
    )
    query = st.text_area(
        "Query",
        placeholder=placeholder_text,
        height=80,
        label_visibility="collapsed",
        key="query_input",
        disabled=st.session_state.is_processing,
    )

with col_btn:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    btn_label = "⟳ Processing..." if st.session_state.is_processing else "→ Ask"
    submit = st.button(
        btn_label,
        use_container_width=True,
        disabled=st.session_state.is_processing,
    )

# ─── On submit: flip processing flag and rerun to show thinking state ─────────
if submit and query.strip() and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.pending_query = query.strip()
    st.rerun()

# ─── On rerun with processing=True: run the actual query ─────────────────────
if st.session_state.is_processing and st.session_state.pending_query:
    q = st.session_state.pending_query

    files = None
    if uploaded_files:
        files = [(f.name, f.getvalue()) for f in uploaded_files]

    try:
        result = st.session_state.router.handle_input(
            query=q,
            files=files,
            upload_id=st.session_state.session_id if files else None,
        )
        st.session_state.history.append({
            "query":      q,
            "answer":     result.get("answer", ""),
            "mode":       result.get("mode", "regulatory_only"),
            "steps":      result.get("progress", []),
            "web":        result.get("web_results", []),
            "reg_chunks": len(result.get("regulatory_text", [])),
            "upl_chunks": len(result.get("uploaded_text", [])),
        })

    except Exception as e:
        st.session_state.history.append({
            "query":      q,
            "answer":     f"⚠️ Error processing query: {e}",
            "mode":       "regulatory_only",
            "steps":      [],
            "web":        [],
            "reg_chunks": 0,
            "upl_chunks": 0,
        })

    finally:
        st.session_state.is_processing = False
        st.session_state.pending_query = ""
        st.rerun()