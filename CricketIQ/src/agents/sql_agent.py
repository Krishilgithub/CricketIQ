"""
src/agents/sql_agent.py
────────────────────────
LangSmith-traced Text-to-SQL Agent loop.
"""

import requests
import re
from langsmith import traceable

from src.rag.rag_pipeline import gather_query_context
from src.rag.retriever import execute_sql

import os

# ── API Config ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "google/gemini-2.5-flash"

@traceable(run_type="llm", name="OpenRouter Query Rewriting")
def rewrite_query_with_llm(user_query: str, history: list) -> str:
    if not history:
        return user_query
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    sys_prompt = "You are a strict query rewriter. Given a chat history and a follow-up user question, rewrite the user question to be a singular standalone query containing all necessary entities (names, venues, teams) from the history. DO NOT answer the question. ONLY output the rewritten query."
    messages = [{"role": "system", "content": sys_prompt}]
    for turn in history[-4:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"Rewrite this query: {user_query}"})
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json={"model": MODEL_ID, "messages": messages, "max_tokens": 100, "temperature": 0.0}, timeout=10)
        return response.json()["choices"][0]["message"]["content"].strip().strip('"\'')
    except Exception:
        return user_query


@traceable(run_type="llm", name="OpenRouter Text-to-SQL Generation")
def call_openrouter(messages: list) -> str:
    """Wrapper to trace the LLM call independently."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8503",
        "X-Title": "CricketIQ Local App",
    }
    payload = {"model": MODEL_ID, "messages": messages, "max_tokens": 1000, "temperature": 0.2}
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=25)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@traceable(run_type="agent", name="SQL Agent Loop")
def agent_loop(standalone_query: str, history: list, st_session_cb=None) -> str:
    """
    Main Text-to-SQL agent loop with tracing.
    st_session_cb is an optional callback to save real-time state back to Streamlit without importing it here.
    """
    dynamic_sys_prompt = gather_query_context(standalone_query)
    
    messages = [{"role": "system", "content": dynamic_sys_prompt}]
    for turn in history[-6:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": standalone_query})

    for _ in range(4):
        try:
            reply = call_openrouter(messages)
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"

        messages.append({"role": "assistant", "content": reply})
        sql_match = re.search(r'<SQL>(.*?)</SQL>', reply, re.IGNORECASE | re.DOTALL)
        
        if sql_match:
            sql_query = sql_match.group(1).strip()

            # Security check
            forbidden = ["drop", "delete", "insert", "update", "truncate"]
            if any(kw in sql_query.lower() for kw in forbidden):
                return "🚫 Security: Destructive SQL operations are not permitted."

            db_result = execute_sql(sql_query)
            
            # Use callback to log the tool execution to the frontend / session
            if st_session_cb:
                st_session_cb(sql_query, db_result)

            messages.append({"role": "user", "content": f"[DATABASE RESULT]\n{db_result}"})
            continue
        else:
            return reply
            
    return "⚠️ Agent loop limit exceeded without a final answer."
