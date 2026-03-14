"""AI Analyst chatbot page — Text-to-SQL with context memory and SQL sandbox."""
import re, json, uuid
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from langsmith import traceable

from src.agents.sql_agent import agent_loop, rewrite_query_with_llm
from src.chat.intent_classifier import classify_intent
from src.chat.entity_extractor import extract_prediction_entities
from src.ml.predictor import predict_match
from src.ui.prediction_display import render_prediction_result
from src.pages.shared import get_hub_con

@traceable(run_type="chain", name="User Chat Interaction")
def process_chat(user_input: str, history: list, st_placeholder) -> str:
    """Traced entry point for the entire agentic run. Routes based on intent."""
    
    intent = classify_intent(user_input, history)
    
    if intent == "PREDICTION":
        st_placeholder.caption("🎯 *Intent Detected: Match Prediction Model*")
        
        entities = extract_prediction_entities(user_input, history)
        teamA = entities.get("Team A")
        teamB = entities.get("Team B")
        
        if not teamA or not teamB:
            found_team = teamA or teamB
            if found_team:
                return f"I detected you want a match prediction involving **{found_team}**, but I need **two valid international cricket teams** to run the prediction model. Could you please specify who they are playing against?"
            else:
                return "I detected you want a match prediction, but I couldn't identify the teams. Could you please specify the two teams that are playing?"
            
        venue = entities.get("Venue") or "Wankhede Stadium"
        toss = entities.get("Toss") or "Bat"
        
        try:
            pred = predict_match(teamA, teamB, venue, toss)
            
            # Save the raw prediction result in the session history under a special key
            current_session = st.session_state["sessions"][st.session_state["current_session_id"]]
            current_session["messages"].append({
                "role": "tool",
                "content": f"Predicted {teamA} vs {teamB}",
                "prediction_dict": pred
            })
            
            with st_placeholder.container():
                render_prediction_result(pred)
                
            model_name = pred.get("model_loaded", "champion model")
            return f"I've run the numbers for {teamA} vs {teamB}. Based on the {model_name}, {pred['favourite']} is favored to win with a {pred['fav_prob']:.1f}% probability."
            
        except Exception as e:
            return f"⚠️ Prediction Error: {e}"
            
    # Default SQL / Knowledge Flow
    st_placeholder.caption("📊 *Intent Detected: Data Analytics Agent*")
    standalone_query = rewrite_query_with_llm(user_input, history)
    if standalone_query.lower() != user_input.lower():
        st_placeholder.caption(f"*Contextualized: '{standalone_query}'*")

    # Define a callback to log tools executed to Streamlit state
    def st_session_cb(sql_query, db_result):
        hub_con = get_hub_con()
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

    reply = agent_loop(standalone_query, history, st_session_cb=st_session_cb)
    return reply


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
    col_title, col_btn = st.columns([0.85, 0.15])
    with col_title:
        st.title("🤖 AI Analyst")
    with col_btn:
        st.write("") # Vertical padding alignment
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_id = str(uuid.uuid4())
            st.session_state["current_session_id"] = new_id
            st.session_state["sessions"][new_id] = {
                "title": "New Chat",
                "timestamp": pd.Timestamp.now(),
                "messages": [],
            }
            st.rerun()

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
                if "prediction_dict" in msg:
                    # It's a prediction visualization
                    with st.expander("🎯 Prediction Analysis", expanded=True):
                        from src.ui.prediction_display import render_prediction_result
                        render_prediction_result(msg["prediction_dict"])
                else:
                    # It's a SQL execution
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
            reply = process_chat(user_input, history_for_rewrite, status_placeholder)

        current_session["messages"].append({"role": "assistant", "content": reply})
        st.rerun()
