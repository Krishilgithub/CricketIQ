"""
src/genai/chatbot_app.py
─────────────────────────
CricketIQ AI Analyst — Streamlit Chatbot

Provides a conversational cricket analyst powered by:
  1. RAG (Retrieval-Augmented Generation) via DuckDB Gold Layer
  2. OpenAI GPT-4o (or GPT-3.5 fallback) for response generation
  3. Pre-match match win probability from the champion model

Usage:
  streamlit run src/genai/chatbot_app.py

Set OPENAI_API_KEY in .env or environment before running.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.genai.rag_context import build_rag_context
from src.logger import get_logger

log = get_logger(__name__)

# ── OpenAI Client ─────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    HAS_OPENAI = False

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CricketIQ, an expert cricket analytics assistant with deep knowledge of T20 international cricket. 
You have access to a real cricket database covering 3,000+ T20I matches.

When answering:
- Use specific statistics from the context provided (marked [DATABASE CONTEXT])
- Be concise and insightful — like a professional cricket analyst
- If the context has data, cite exact numbers
- If asked about predictions, use the win probabilities provided
- Keep responses structured but conversational
- Add emojis for key stats to improve readability 🏏

If context is unavailable, say so honestly rather than making up numbers."""

# ── LLM Call ──────────────────────────────────────────────────────────────────
def ask_llm(user_query: str, rag_context: str, history: list) -> str:
    """Call OpenAI with RAG context injected into the system message."""
    if not HAS_OPENAI:
        # Graceful fallback — show only the RAG context
        return (
            f"**[AI Response — OpenAI key not configured]**\n\n"
            f"Here's what I found in the database for your query:\n\n"
            f"```\n{rag_context}\n```\n\n"
            f"To enable full AI responses, add your `OPENAI_API_KEY` to `.env`."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\n[DATABASE CONTEXT]\n{rag_context}"},
    ]
    # Include last 6 turns of history for context window efficiency
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_query})

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}\n\nDatabase context:\n```\n{rag_context}\n```"


# ── Streamlit App ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricketIQ — AI Analyst",
    page_icon="🏏",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1117; }
    .chat-msg-user { background:#1f6feb; border-radius:12px; padding:10px 16px; margin:4px 0; }
    .chat-msg-bot  { background:#161b22; border-radius:12px; padding:10px 16px; margin:4px 0; border:1px solid #30363d; }
    .stTextInput > div > div > input { background: #0d1117; color: white; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏏 CricketIQ AI Analyst")
st.sidebar.markdown("*Ask anything about T20 cricket*")
st.sidebar.markdown("---")

openai_status = "✅ Connected" if HAS_OPENAI else "⚠️ Key not set (showing DB context only)"
st.sidebar.markdown(f"**OpenAI**: {openai_status}")
st.sidebar.markdown("**Database**: DuckDB Gold Layer")
st.sidebar.markdown("**RAG**: Intent-aware context retrieval")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Sample Questions")
sample_qs = [
    "What is India's win rate in T20Is?",
    "Show India vs Pakistan head to head record",
    "Who are the top 10 run scorers?",
    "What is the average score at Eden Gardens?",
    "Who will win between India and Australia?",
    "Tell me about Virat Kohli's batting stats",
]
for q in sample_qs:
    if st.sidebar.button(q, key=f"sample_{q[:20]}"):
        st.session_state["pending_query"] = q

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state["messages"] = []
    st.session_state["pending_query"] = ""

# ── Chat State ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""

# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🏏 CricketIQ — AI Cricket Analyst")
st.markdown("*Powered by your 1.14M delivery cricket database + AI. Ask anything!*")

# Render existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# Input — either from sidebar button or chat input
user_input = st.chat_input("Ask about teams, players, venues, predictions...")
if st.session_state["pending_query"]:
    user_input = st.session_state["pending_query"]
    st.session_state["pending_query"] = ""

if user_input:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Build RAG context
    with st.spinner("🔍 Querying cricket database..."):
        context = build_rag_context(user_input)

    # Get LLM response
    with st.spinner("🤖 Analysing..."):
        reply = ask_llm(user_input, context, st.session_state["messages"])

    # Store and display reply
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(reply)

    # Show expandable RAG context
    with st.expander("📊 Database Context Used", expanded=False):
        st.code(context, language="text")
