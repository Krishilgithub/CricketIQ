"""
src/genai/chatbot_app.py
─────────────────────────
CricketIQ AI Analyst — Streamlit Chatbot (Agentic Version)

Provides a conversational cricket analyst powered by:
  1. Semantic Query Rewriting
  2. Text-to-SQL (Agentic Loop over DuckDB)
  3. Strict Verification & Fuzzy Entity Resolution
"""

import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import json
import requests
import uuid
from dotenv import load_dotenv

load_dotenv()

from src.genai.rag_context import execute_sql, get_schema_string, extract_entities, get_con
from src.logger import get_logger

log = get_logger(__name__)

# ── OpenRouter API Config ─────────────────────────────────────────────────────
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Using a fast, high-quality model
MODEL_ID = "google/gemini-2.5-flash"


# ── System Prompt (Agentic Text-to-SQL) ────────────────────────────────────────
SYSTEM_PROMPT = """You are CricketIQ, an expert AI cricket analyst with direct database access.

You have access to a DuckDB database containing detailed T20I cricket statistics. 
{SCHEMA}

AVAILABLE ENTITIES (Correctly spelled):
{ENTITIES}

INSTRUCTIONS:
1. When asked a statistical question, YOU MUST query the database first.
2. To query the database, output a SQL query wrapped EXACTLY in <SQL>...</SQL> tags.
   Example: <SQL>SELECT match_date, team_1, winner, result_margin FROM main_gold.fact_matches WHERE team_1='India' LIMIT 5</SQL>
3. The system will run the query and return the result to you in the next turn. 
4. Analyze the result and provide your final insightful response to the user.
5. PREDICTIONS AND PROBABILITIES: If asked to *predict* or calculate *chances* (e.g., "chances India wins finals"), you MUST NOT refuse using pre-trained warnings. Instead, query the database for the team's historical win rate, recent form (e.g., mart_team_form), and past performance in finals. Formulate a clear percentage chance based on these stats, and clearly present the queried statistics as your "Proof".
6. SHOW YOUR PROOF: When answering historical queries (e.g., "matches India won"), you MUST include a formatted Markdown table showing the raw rows (Match Date, Venue, Winner, Margin, etc.) from the execution result to support your answer. If there are many rows, show the top 10 most recent ones as proof.
7. If the database result is completely empty or insufficient after querying, respond EXACTLY with: "I don't have enough data in the current CricketIQ database to answer that accurately."
8. IMPORTANT SCHEMA NOTE: `fact_matches` DOES NOT have a `team_2` column. It only has `team_1` (which is the toss winner) and `winner`. To find matches between two specific teams (e.g., India vs Pakistan), you can check if `team_1` is one team and `winner` is either team, OR join `fact_matches` with `fact_innings` (which has `batting_team`) on `match_id` to reliably find both playing teams.
9. Add emojis for key stats to improve readability 🏏"""

# ── LLM Call ──────────────────────────────────────────────────────────────────
def rewrite_query_with_llm(user_query: str, history: list) -> str:
    """Rewrite follow-up questions into standalone queries using chat history."""
    if not history:
        return user_query
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8503", 
        "X-Title": "CricketIQ Local App",
        "Content-Type": "application/json"
    }
    
    sys_prompt = "You are a strict query rewriter. Given a chat history and a follow-up user question, rewrite the user question to be a singular standalone query containing all necessary entities (names, venues, teams) from the history. DO NOT answer the question. ONLY output the rewritten query."
    
    messages = [{"role": "system", "content": sys_prompt}]
    for turn in history[-4:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"]})
            
    messages.append({"role": "user", "content": f"Rewrite this query: {user_query}"})

    payload = {"model": MODEL_ID, "messages": messages, "max_tokens": 100, "temperature": 0.0}
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        rewritten = response.json()["choices"][0]["message"]["content"].strip()
        return rewritten.strip('"\'')
    except Exception as e:
        log.warning(f"Query rewrite failed: {e}")
        return user_query


def agent_loop(standalone_query: str, history: list, st_placeholder) -> str:
    """Runs the Text-to-SQL Agent loop."""
    con = get_con()
    entities = extract_entities(standalone_query, con)
    con.close()
    
    entity_str = json.dumps(entities, indent=2)
    schema_str = get_schema_string()
    
    dynamic_sys_prompt = SYSTEM_PROMPT.replace("{SCHEMA}", schema_str).replace("{ENTITIES}", entity_str)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8503", 
        "X-Title": "CricketIQ Local App",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": dynamic_sys_prompt}]
    for turn in history[-6:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": standalone_query})

    loop_limit = 4
    for step in range(loop_limit):
        payload = {"model": MODEL_ID, "messages": messages, "max_tokens": 800, "temperature": 0.2}
        
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"
            
        messages.append({"role": "assistant", "content": reply})
        
        sql_match = re.search(r'<SQL>(.*?)</SQL>', reply, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
            
            db_result = execute_sql(sql_query)
            
            # Save query and results for the persistent UI
            st.session_state["sessions"][st.session_state["current_session_id"]].append({
                "role": "tool", 
                "content": sql_query,
                "result": db_result
            })
            
            with st_placeholder.container():
                st.info(f"💾 Agent Executing SQL:\n```sql\n{sql_query}\n```")

            # Feed back to agent
            messages.append({"role": "user", "content": f"[DATABASE RESULT]\n{db_result}"})
            continue 
        else:
            return reply

    return "⚠️ Agent loop limit exceeded without a final answer."


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

# State init
if "sessions" not in st.session_state:
    st.session_state["sessions"] = {}
if "current_session_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state["current_session_id"] = new_id
    st.session_state["sessions"][new_id] = []
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""
if "sandbox_query" not in st.session_state:
    st.session_state["sandbox_query"] = "SELECT * FROM main_gold.fact_matches LIMIT 5"
if "show_sandbox" not in st.session_state:
    st.session_state["show_sandbox"] = False

def toggle_sandbox():
    st.session_state["show_sandbox"] = not st.session_state["show_sandbox"]

def load_sandbox(sql: str):
    st.session_state["sandbox_query"] = sql
    st.session_state["show_sandbox"] = True

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏏 CricketIQ AI Analyst")
st.sidebar.markdown("*Ask anything about T20 cricket*")
st.sidebar.button("🛠️ Toggle SQL Sandbox", on_click=toggle_sandbox)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Chat History")

if st.sidebar.button("➕ New Conversation"):
    new_id = str(uuid.uuid4())
    st.session_state["current_session_id"] = new_id
    st.session_state["sessions"][new_id] = []
    st.rerun()

for sess_id, msgs in st.session_state["sessions"].items():
    title = "New Chat"
    user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
    if user_msgs:
        title = user_msgs[0][:25] + "..."
        
    if st.sidebar.button(title, key=f"sess_{sess_id}"):
        st.session_state["current_session_id"] = sess_id
        st.rerun()

if st.sidebar.button("🗑 Clear Current Chat"):
    st.session_state["sessions"][st.session_state["current_session_id"]] = []
    st.session_state["pending_query"] = ""
    st.rerun()

# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🏏 CricketIQ — AI Cricket Analyst")
st.markdown("*Powered by Agentic RAG and DuckDB. Ask anything!*")

# ── Interactive Sandbox ──
if st.session_state["show_sandbox"]:
    st.markdown("### 🛠️ Interactive SQL Sandbox")
    st.info("Edit the agent's query below or write your own to explore the database.")
    
    with st.form("sandbox_form"):
        edited_sql = st.text_area("SQL Query", value=st.session_state["sandbox_query"], height=150)
        run_btn = st.form_submit_button("▶️ Run Query")
        
    if run_btn:
        st.session_state["sandbox_query"] = edited_sql
        try:
            con = get_con()
            df = con.execute(edited_sql).df()
            con.close()
            st.success(f"Query returned {len(df)} rows.")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")
            if 'con' in locals(): con.close()
    st.markdown("---")

# Render chat history
current_msgs = st.session_state["sessions"][st.session_state["current_session_id"]]

if not current_msgs:
    st.markdown("### 💡 Sample Questions")
    cols = st.columns(3)
    sample_qs = [
        "What is India's win rate in T20Is?",
        "Show India vs Pakistan record",
        "Who are the top 10 run scorers?",
        "What is the avg score at Eden Gardens?",
        "Tell me about Virat Kohli's stats",
    ]
    for i, q in enumerate(sample_qs):
        if cols[i % 3].button(q, key=f"sample_{q[:20]}"):
            st.session_state["pending_query"] = q
            st.rerun()

for i, msg in enumerate(current_msgs):
    if msg["role"] == "tool":
        with st.chat_message("tool", avatar="🛠️"):
            with st.expander("💾 Agent Executed SQL (Expand to see results)", expanded=False):
                st.code(msg["content"], language="sql")
                if "result" in msg:
                    st.markdown("**Database Result:**")
                    st.markdown(msg["result"])
                if st.button("Edit & Run in Sandbox", key=f"edit_sql_{i}"):
                    load_sandbox(msg["content"])
                    st.rerun()
    else:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

user_input = st.chat_input("Ask about teams, players, venues, predictions...")
if st.session_state["pending_query"]:
    user_input = st.session_state["pending_query"]
    st.session_state["pending_query"] = ""

if user_input:
    st.session_state["sessions"][st.session_state["current_session_id"]].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    status_placeholder = st.empty()
    
    with st.spinner("🧠 Agent is thinking..."):
        history_for_rewrite = st.session_state["sessions"][st.session_state["current_session_id"]][:-1]
        standalone_query = rewrite_query_with_llm(user_input, history_for_rewrite)
        if standalone_query != user_input and standalone_query.lower() != user_input.lower():
            status_placeholder.caption(f"*Contextualized Query: '{standalone_query}'*")

        reply = agent_loop(standalone_query, history_for_rewrite, status_placeholder)

    st.session_state["sessions"][st.session_state["current_session_id"]].append({"role": "assistant", "content": reply})
    st.rerun()

