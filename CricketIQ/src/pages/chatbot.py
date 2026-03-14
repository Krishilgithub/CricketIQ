"""AI Analyst chatbot page — Text-to-SQL with context memory and SQL sandbox."""
import re, json
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from src.genai.rag_context import execute_sql, get_schema_string, extract_entities, get_con
from src.pages.shared import get_hub_con, load_model, get_h2h_rate, get_venue_avg, get_team_form

# ── API Config ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "google/gemini-2.5-flash"

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
5. PREDICTIONS AND PROBABILITIES: If asked to *predict* or calculate *chances*, query the database for historical win rates, recent form, and past performance. Formulate a clear percentage chance based on these stats, and clearly present the queried statistics as your "Proof".
6. SHOW YOUR PROOF: When answering historical queries, you MUST include a formatted Markdown table showing the raw rows from the execution result to support your answer. If there are many rows, show the top 10 most recent ones as proof.
7. If the database result is completely empty or insufficient after querying, respond EXACTLY with: "I don't have enough data in the current CricketIQ database to answer that accurately."
8. IMPORTANT SCHEMA NOTE: `fact_matches` DOES NOT have a `team_2` column. It only has `team_1` (which is the toss winner) and `winner`. To find matches between two specific teams, join `fact_matches` with `fact_innings` (which has `batting_team`) on `match_id` to reliably find both playing teams.
9. SECURITY: Never generate DROP, DELETE, INSERT, or UPDATE queries. ONLY read-only SELECT queries are allowed.
10. EDGE CASES: If a user asks about a player who did not play T20 Internationals (e.g., Don Bradman), politely clarify this database strictly tracks modern T20I matches.
11. AUTO-SUGGESTIONS: At the very end of your final response, ALWAYS provide exactly 3 highly relevant follow-up questions under a "### 🤔 Suggested Follow-ups" heading as bullet points.
12. Add emojis for key stats to improve readability 🏏"""


def route_query_intent(user_query: str, history: list) -> dict:
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    sys_prompt = """You are an intelligent query router for a Cricket AI. 
Analyze the user query and decide if it requires:
- 'ML': A predictive model for match outcomes (e.g., 'predict the winner', 'what are the chances', 'win probability between A and B').
- 'SQL': General historical statistics, past match details, scores, player records, etc.

Return EXACTLY a JSON dict with NO markdown wrapping, like this:
{
  "route": "ML" or "SQL",
  "team1": "Specific Team Name or null (e.g. 'India')",
  "team2": "Specific Team Name or null (e.g. 'Australia')",
  "venue": "Venue Name or null (e.g. 'Eden Gardens')",
  "toss_decision": "Bat" or "Field" (default to "Bat")
}"""
    messages = [{"role": "system", "content": sys_prompt}]
    for turn in history[-4:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_query})
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json={"model": MODEL_ID, "messages": messages, "max_tokens": 100, "temperature": 0.0}, timeout=10)
        content = response.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"route": "SQL"}
    except Exception:
        return {"route": "SQL"}


def ml_agent_loop(intent_data: dict, history: list, st_placeholder) -> str:
    champion = load_model()
    if not champion:
        return "⚠️ ML Model not found. Please train it first."
    
    t1 = intent_data.get("team1") or "India"
    t2 = intent_data.get("team2") or "Australia"
    venue = intent_data.get("venue") or "Eden Gardens"
    toss = intent_data.get("toss_decision") or "Bat"
    
    with st_placeholder.container():
        st.info(f"🔮 **ML Prediction Route Triggered**\n\nPredicting: **{t1}** vs **{t2}** at **{venue}** (Toss: **{toss}**)")
    
    # Calculate features
    try:
        h2h = get_h2h_rate(t1)
        v_avg = get_venue_avg(venue)
        t1_form = get_team_form(t1)
        t2_form = get_team_form(t2)
        
        feats = pd.DataFrame([{
            "toss_bat": 1 if toss == "Bat" else 0,
            "venue_avg_1st_inns_runs": v_avg,
            "team_1_h2h_win_rate": h2h,
            "team_1_form_last5": t1_form,
            "team_2_form_last5": 1 - t2_form,
        }])
        
        win_prob_t1 = float(champion["model"].predict_proba(feats)[0][1])
        win_prob_t2 = 1.0 - win_prob_t1
    except Exception as e:
        return f"⚠️ ML Execution error: {e}"

    # Ask the LLM to format the response nicely based on these ML results
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    sys_prompt = "You are CricketIQ. Structure a compelling match prediction narrative using the provided ML model probabilities, and include the feature data used (H2H, Team Form, Venue Avg). Always include suggestions for follow-ups."
    
    ml_context = f"""
    ML Prediction Results:
    Match: {t1} vs {t2}
    Venue: {venue}
    {t1} Win Probability: {win_prob_t1*100:.1f}%
    {t2} Win Probability: {win_prob_t2*100:.1f}%
    
    Features Used:
    - {t1} Head-to-Head Win Rate overall: {h2h*100:.1f}%
    - Venue ({venue}) Avg 1st Inns: {v_avg:.1f}
    - {t1} Recent Form Score: {t1_form:.2f}
    - {t2} Recent Form Score: {t2_form:.2f}
    """
    
    messages = [{"role": "system", "content": sys_prompt}]
    for turn in history[-4:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"The ML Model just returned these results:\n{ml_context}\n\nPlease generate a response to the user."})
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json={"model": MODEL_ID, "messages": messages, "max_tokens": 800, "temperature": 0.4}, timeout=20)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ LLM Error Formatting ML Result: {e}"

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


def agent_loop(standalone_query: str, history: list, st_placeholder) -> str:
    con = get_con()
    entities = extract_entities(standalone_query, con)
    con.close()

    dynamic_sys_prompt = SYSTEM_PROMPT.replace("{SCHEMA}", get_schema_string()).replace("{ENTITIES}", json.dumps(entities, indent=2))
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": dynamic_sys_prompt}]

    for turn in history[-6:]:
        if turn["role"] in ["user", "assistant"]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": standalone_query})

    hub_con = get_hub_con()
    for _ in range(4):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json={"model": MODEL_ID, "messages": messages, "max_tokens": 1000, "temperature": 0.2}, timeout=25)
            reply = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"

        messages.append({"role": "assistant", "content": reply})
        sql_match = re.search(r'<SQL>(.*?)</SQL>', reply, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()

            # Inject security check
            forbidden = ["drop", "delete", "insert", "update", "truncate"]
            if any(kw in sql_query.lower() for kw in forbidden):
                return "🚫 Security: Destructive SQL operations are not permitted."

            db_result = execute_sql(sql_query)

            # Try to render as a real DataFrame for richer display
            try:
                result_df = hub_con.execute(sql_query).df()
            except Exception:
                result_df = None

            current_session = st.session_state["sessions"][st.session_state["current_session_id"]]
            current_session["messages"].append({
                "role": "tool",
                "content": sql_query,
                "result": db_result,
                "result_df": result_df.to_dict() if result_df is not None else None,
            })

            with st_placeholder.container():
                st.info(f"💾 Agent Executing SQL:\n```sql\n{sql_query}\n```")
                if result_df is not None and not result_df.empty:
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

            messages.append({"role": "user", "content": f"[DATABASE RESULT]\n{db_result}"})
            continue
        else:
            return reply
    return "⚠️ Agent loop limit exceeded without a final answer."


def _try_auto_chart(reply: str, msgs: list):
    """If the last SQL result has numeric data, automatically render a bar chart."""
    tool_msgs = [m for m in msgs if m.get("role") == "tool" and m.get("result_df")]
    if not tool_msgs:
        return
    last_tool = tool_msgs[-1]
    try:
        df = pd.DataFrame(last_tool["result_df"])
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if numeric_cols and str_cols and len(df) > 1:
            x_col = str_cols[0]
            y_col = numeric_cols[0]
            fig = px.bar(df, x=x_col, y=y_col, title=f"📊 Auto-Chart: {y_col} by {x_col}",
                         color=y_col, color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def render():
    st.title("🤖 AI Analyst")
    st.markdown("Ask anything about T20 cricket. I'll write SQL, query the database, and return an expert analysis.")

    hub_con = get_hub_con()

    # ── SQL Sandbox ────────────────────────────────────────────────────────
    with st.expander("🛠️ Interactive SQL Sandbox", expanded=False):
        st.info("Write your own SQL query against the `main_gold` schema to explore the database directly.")
        with st.form("sandbox_form"):
            default_query = "SELECT * FROM main_gold.fact_matches LIMIT 5"
            edited_sql = st.text_area("SQL Query", value=default_query, height=120, key="sandbox_sql")
            col_run, col_export = st.columns([1, 1])
            if col_run.form_submit_button("▶️ Run Query"):
                try:
                    df = hub_con.execute(edited_sql).df()
                    st.success(f"✅ Query returned {len(df)} rows.")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    # CSV export
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download CSV", csv, "query_result.csv", "text/csv")
                except Exception as e:
                    st.error(f"SQL Error: {e}")

    st.markdown("---")

    current_session = st.session_state["sessions"][st.session_state["current_session_id"]]
    msgs = current_session["messages"]

    # ── Sample Questions (only for empty chats) ────────────────────────────
    if not msgs:
        st.markdown("<div class='section-header'>💡 Try asking...</div>", unsafe_allow_html=True)
        sample_qs = [
            "What is India's T20I win rate?",
            "Show India vs Pakistan head-to-head record",
            "Who are the top 10 run scorers?",
            "Predict India's chances in the World Cup final",
            "What is the avg score at Eden Gardens?",
            "Best bowling spells in T20 World Cup history",
        ]
        cols = st.columns(3)
        for i, q in enumerate(sample_qs):
            if cols[i % 3].button(q, key=f"sample_{i}"):
                st.session_state["pending_query"] = q
                st.rerun()

    # ── Render Message History ─────────────────────────────────────────────
    for msg in msgs:
        if msg["role"] == "tool":
            with st.chat_message("assistant", avatar="🛠️"):
                with st.expander("💾 SQL Executed", expanded=False):
                    st.code(msg["content"], language="sql")
                    if "result_df" in msg and msg["result_df"]:
                        try:
                            df_display = pd.DataFrame(msg["result_df"])
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                            csv = df_display.to_csv(index=False).encode("utf-8")
                            st.download_button("⬇️ Download Result CSV", csv, "sql_result.csv", "text/csv", key=f"dl_{id(msg)}")
                        except Exception:
                            st.markdown(msg.get("result", ""))
                    else:
                        st.markdown(msg.get("result", ""))
        else:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                # Auto-chart after AI responses
                if msg["role"] == "assistant":
                    _try_auto_chart(msg["content"], msgs)

    # ── Chat Input ─────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about teams, players, venues, predictions...")
    if st.session_state.get("pending_query"):
        user_input = st.session_state["pending_query"]
        st.session_state["pending_query"] = ""

    if user_input:
        if len(msgs) == 0:
            current_session["title"] = user_input[:35] + "..."

        current_session["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        status_placeholder = st.empty()
        with st.spinner("🧠 Agent is thinking..."):
            history_for_rewrite = msgs[:-1]
            # Route Intent
            intent = route_query_intent(user_input, history_for_rewrite)
            
            if intent.get("route") == "ML":
                reply = ml_agent_loop(intent, history_for_rewrite, status_placeholder)
            else:
                standalone_query = rewrite_query_with_llm(user_input, history_for_rewrite)
                if standalone_query.lower() != user_input.lower():
                    status_placeholder.caption(f"*Contextualized: '{standalone_query}'*")
                reply = agent_loop(standalone_query, history_for_rewrite, status_placeholder)

        current_session["messages"].append({"role": "assistant", "content": reply})
        st.rerun()
