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
    "<div style='text-align:center; font-size:4rem; margin-bottom:-10px;'>🏏</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("## 🏏 CricketIQ")
st.sidebar.markdown("*T20 Analytics Platform*")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigate",
    [
        "🤖 AI Analyst",
        "🔮 Match Prediction",
        "📊 Intelligence Hub",
        "📈 Advanced Analytics",
        "💬 Chat History",
    ],
)

# Active session info in sidebar
st.sidebar.markdown("---")
current_sessions = st.session_state.get("sessions", {})
active_session = current_sessions.get(st.session_state.get("current_session_id", ""), {})
st.sidebar.caption(f"💬 Active Chat: *{active_session.get('title', 'New Chat')}*")
msg_count = len([m for m in active_session.get("messages", []) if m["role"] == "user"])
st.sidebar.caption(f"📝 {msg_count} question(s) in this session")

# ── Page Routing ───────────────────────────────────────────────────────────
if app_mode == "🤖 AI Analyst":
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
