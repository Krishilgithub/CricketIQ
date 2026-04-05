"""
CricketIQ — Unified Streamlit Entry Point
Thin routing shell that delegates rendering to individual page modules.
"""
import sys, os, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Disable LangSmith tracing when API key is not configured ───────────────
# Prevents 403 Forbidden errors on Streamlit Cloud when LANGCHAIN_API_KEY
# is not set in the app's secrets.
if not os.environ.get("LANGCHAIN_API_KEY") and not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

from src.pages.shared import THEME_CSS
from src.logger import get_logger

log = get_logger(__name__)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricketIQ Intelligence Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's default multipage nav (auto-discovered .py files in /pages)
hide_pages_css = """
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>
"""
st.markdown(hide_pages_css, unsafe_allow_html=True)

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ── Session State Initialization ───────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state["sessions"] = {}
if "current_session_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state["current_session_id"] = new_id
    st.session_state["sessions"][new_id] = {
        "title": "New Chat",
        "timestamp": pd.Timestamp.now(),
        "messages": [],
    }
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""

# ── Sidebar Navigation ─────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='text-align:center; font-size:3.5rem; margin-bottom:0;'>🏏</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='text-align:center; font-size:1.4rem; font-weight:700; color:#38bdf8; letter-spacing:0.5px;'>CricketIQ</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='text-align:center; font-size:0.8rem; color:#64748b; margin-bottom:8px;'>T20 Analytics Platform</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

NAV_ITEMS = {
    "🤖  AI Chat Bot": "🤖 AI Chat Bot",
    "🔮  Match Prediction": "🔮 Match Prediction",
    "📊  Intelligence Hub": "📊 Intelligence Hub",
    "📈  Advanced Analytics": "📈 Advanced Analytics",
    "💬  Chat History": "💬 Chat History",
}

app_mode_label = st.sidebar.radio(
    "Navigate",
    list(NAV_ITEMS.keys()),
)
app_mode = NAV_ITEMS[app_mode_label]

# Active session info in sidebar
st.sidebar.markdown("---")
current_sessions = st.session_state.get("sessions", {})
active_session = current_sessions.get(st.session_state.get("current_session_id", ""), {})

if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
    new_id = str(uuid.uuid4())
    st.session_state["current_session_id"] = new_id
    st.session_state["sessions"][new_id] = {
        "title": "New Chat",
        "timestamp": pd.Timestamp.now(),
        "messages": [],
    }
    st.rerun()

st.sidebar.caption(f"💬 Active Chat: *{active_session.get('title', 'New Chat')}*")
msg_count = len([m for m in active_session.get("messages", []) if m["role"] == "user"])
st.sidebar.caption(f"📝 {msg_count} question(s) in this session")

# ── Page Routing ───────────────────────────────────────────────────────────
if app_mode == "🤖 AI Chat Bot":
    from src.pages import chatbot
    chatbot.render()

elif app_mode == "🔮 Match Prediction":
    from src.pages import prediction
    prediction.render()

elif app_mode == "📊 Intelligence Hub":
    from src.pages import dashboard
    dashboard.render()

elif app_mode == "📈 Advanced Analytics":
    from src.pages import analytics
    analytics.render()

elif app_mode == "💬 Chat History":
    from src.pages import history
    history.render()
