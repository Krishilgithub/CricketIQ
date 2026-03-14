import sys, os, re, json, requests, uuid, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

from src.config import get_config, resolve_path
from src.genai.rag_context import execute_sql, get_schema_string, extract_entities, get_con
from src.logger import get_logger

log = get_logger(__name__)

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
5. PREDICTIONS AND PROBABILITIES: If asked to *predict* or calculate *chances* (e.g., "chances India wins finals"), query the database for the team's historical win rate, recent form, and past performance. Formulate a clear percentage chance based on these stats, and clearly present the queried statistics as your "Proof".
6. SHOW YOUR PROOF: When answering historical queries, you MUST include a formatted Markdown table showing the raw rows (Match Date, Venue, Winner, Margin, etc.) from the execution result to support your answer. If there are many rows, show the top 10 most recent ones as proof.
7. If the database result is completely empty or insufficient after querying, respond EXACTLY with: "I don't have enough data in the current CricketIQ database to answer that accurately."
8. IMPORTANT SCHEMA NOTE: `fact_matches` DOES NOT have a `team_2` column. It only has `team_1` (which is the toss winner) and `winner`. To find matches between two specific teams, check if `team_1` is one team and `winner` is either team, OR join `fact_matches` with `fact_innings` (which has `batting_team`) on `match_id` to reliably find both playing teams.
9. SECURITY: Never generate DROP, DELETE, INSERT, or UPDATE queries. ONLY read-only SELECT queries are allowed.
10. EDGE CASES: If a user asks about a player who did not play T20 Internationals (e.g., Don Bradman, Vivian Richards), politely clarify that this database strictly tracks modern T20I matches. If a question is too ambiguous (e.g., "Who won?"), ask them to specify the teams, venue, or year.
11. AUTO-SUGGESTIONS: At the very end of your final response, ALWAYS provide exactly 3 highly relevant and interesting follow-up questions that the user could ask next to explore the data further. Format these under a "### 🤔 Suggested Follow-ups" heading as bullet points.
12. Add emojis for key stats to improve readability 🏏"""

# ── App Config & Styling ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CricketIQ",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0b0f19; }
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .metric-card { background: #1e293b; border-radius: 12px; padding: 20px; margin: 8px 0; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .section-header { font-size: 1.5rem; font-weight: 700; color: #38bdf8; margin-bottom: 16px; border-bottom: 2px solid #1e293b; padding-bottom: 8px; }
    .chat-msg-user { background: #0284c7; border-radius: 12px 12px 0 12px; padding: 12px 16px; margin: 4px 0; color: white; float: right; clear: both; max-width: 80%; }
    .chat-msg-bot { background: #1e293b; border-radius: 12px 12px 12px 0; padding: 12px 16px; margin: 4px 0; border: 1px solid #334155; color: #f8fafc; float: left; clear: both; max-width: 80%; }
    .stTextInput > div > div > input { background: #1e293b; color: white; border: 1px solid #334155; border-radius: 8px; }
    .st-expander { background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
    .stCodeBlock { background-color: #0b0f19 !important; border-radius: 6px !important; border: 1px solid #334155 !important; }
</style>
""", unsafe_allow_html=True)

# ── Initialization ─────────────────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state["sessions"] = {}
if "current_session_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state["current_session_id"] = new_id
    st.session_state["sessions"][new_id] = {
        "title": "New Chat",
        "timestamp": pd.Timestamp.now(),
        "messages": []
    }
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = ""

# ── Intelligence Hub Data Loaders ──────────────────────────────────────────
@st.cache_resource
def get_hub_con():
    cfg = get_config()
    return duckdb.connect(str(resolve_path(cfg["paths"]["duckdb_path"])), read_only=True)

@st.cache_resource
def load_model():
    cfg = get_config()
    db_path = str(resolve_path(cfg["paths"]["models_dir"])) + "/champion_model.pkl"
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            return pickle.load(f)
    return None

hub_con = get_hub_con()
champion = load_model()

@st.cache_data
def get_teams():
    return hub_con.execute("SELECT DISTINCT team_1 FROM main_gold.fact_matches ORDER BY team_1").df()["team_1"].tolist()

@st.cache_data
def get_venues():
    return hub_con.execute("SELECT DISTINCT venue FROM main_gold.fact_matches ORDER BY venue").df()["venue"].tolist()

@st.cache_data
def get_h2h_rate(team1):
    h2h_q = f"""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN winner = '{team1}' THEN 1 ELSE 0 END) as wins
    FROM main_gold.fact_matches
    WHERE (toss_winner = '{team1}' OR winner = '{team1}')
    """
    h2h_df = hub_con.execute(h2h_q).df()
    return float(h2h_df.iloc[0]["wins"]) / max(float(h2h_df.iloc[0]["total"]), 1)

@st.cache_data
def get_venue_avg(venue):
    venue_q = f"""
    SELECT AVG(i.total_runs) as avg_runs
    FROM main_gold.fact_innings i
    JOIN main_gold.fact_matches m ON i.match_id = m.match_id
    WHERE m.venue = '{venue}' AND i.innings_number = 1
    """
    venue_df = hub_con.execute(venue_q).df()
    return float(venue_df.iloc[0]["avg_runs"] or 150)

@st.cache_data
def get_team_form(team1):
    form_q = f"""
    SELECT ROUND(AVG(team_1_win)::FLOAT, 3) as form
    FROM (
        SELECT team_1_win FROM main_gold.fact_matches
        WHERE toss_winner = '{team1}'
        ORDER BY match_date DESC LIMIT 5
    )
    """
    form_df = hub_con.execute(form_q).df()
    return float(form_df.iloc[0]["form"] or 0.5)

@st.cache_data
def get_phase_data(venue_sel):
    phase_q = f"""
    SELECT
        CASE WHEN d.over_number <= 5 THEN 'Powerplay (0-5)'
             WHEN d.over_number <= 14 THEN 'Middle (6-14)'
             ELSE 'Death (15-19)' END as phase,
        ROUND(SUM(d.runs_batter + d.runs_extras)::FLOAT / NULLIF(SUM(d.is_legal_ball), 0) * 6, 2) as run_rate,
        ROUND(SUM(d.is_wicket)::FLOAT / NULLIF(SUM(d.is_legal_ball), 0) * 6, 4) as wicket_rate,
        COUNT(*) as balls
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    WHERE m.venue = '{venue_sel}'
    GROUP BY 1 ORDER BY 1
    """
    return hub_con.execute(phase_q).df()

@st.cache_data
def get_toss_recommendation(venue_sel):
    toss_q = f"""
    SELECT toss_decision,
           COUNT(*) total,
           SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) as toss_wins
    FROM main_gold.fact_matches
    WHERE venue = '{venue_sel}' AND result_type NOT IN ('no result', 'tie')
    GROUP BY toss_decision
    """
    return hub_con.execute(toss_q).df()

@st.cache_data
def get_global_kpis():
    q = """
    SELECT 
        (SELECT COUNT(*) FROM main_gold.fact_matches) as total_matches,
        (SELECT SUM(runs_batter + runs_extras) FROM main_gold.fact_deliveries) as total_runs,
        (SELECT COUNT(*) FROM main_gold.fact_wickets) as total_wickets,
        (SELECT COUNT(DISTINCT batter) FROM main_gold.fact_deliveries) as total_players
    """
    try:
        return hub_con.execute(q).df().iloc[0]
    except Exception:
        return pd.Series({"total_matches": 0, "total_runs": 0, "total_wickets": 0, "total_players": 0})

@st.cache_data
def get_top_batters():
    bat_q = """
    SELECT batter, SUM(runs_batter) as runs,
           SUM(is_legal_ball) as balls,
           ROUND(SUM(runs_batter)::FLOAT / NULLIF(SUM(is_legal_ball),0) * 100, 2) as sr
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    WHERE m.match_date >= (CURRENT_DATE - INTERVAL 730 DAY)
    GROUP BY batter HAVING balls > 100
    ORDER BY runs DESC LIMIT 15
    """
    return hub_con.execute(bat_q).df()

@st.cache_data
def get_top_bowlers():
    bowl_q = """
    SELECT bowler, SUM(is_wicket) as wickets,
           SUM(is_legal_ball) as balls,
           ROUND(SUM(runs_batter + runs_extras)::FLOAT / NULLIF(SUM(is_legal_ball),0) * 6, 2) as econ
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    WHERE m.match_date >= (CURRENT_DATE - INTERVAL 730 DAY)
    GROUP BY bowler HAVING balls > 100
    ORDER BY wickets DESC LIMIT 15
    """
    return hub_con.execute(bowl_q).df()

@st.cache_data
def get_venue_heatmap():
    heatmap_q = """
    SELECT t.team as team, m.venue as venue,
           ROUND(AVG(CASE WHEN m.winner = t.team THEN 1.0 ELSE 0.0 END) * 100, 0) as win_pct,
           COUNT(m.match_id) as matches
    FROM main_gold.fact_matches m
    JOIN main_silver.slv_match_teams t ON m.match_id = t.match_id
    GROUP BY t.team, m.venue HAVING matches >= 5
    ORDER BY win_pct DESC
    """
    return hub_con.execute(heatmap_q).df()

@st.cache_data
def get_exciting_matches():
    excitement_q = """
    SELECT m.match_id, m.match_date, m.event_name, m.venue, m.team_1, m.winner,
           m.result_margin,
           i1.total_runs as target,
           i2.total_runs as chased,
           ABS(i1.total_runs - i2.total_runs) as margin_runs
    FROM main_gold.fact_matches m
    JOIN main_gold.fact_innings i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
    JOIN main_gold.fact_innings i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
    WHERE m.result_type IN ('runs', 'wickets')
    ORDER BY margin_runs ASC
    LIMIT 20
    """
    return hub_con.execute(excitement_q).df()

@st.cache_data
def get_highest_scores():
    top_bat_q = """
    SELECT batter, SUM(runs_batter) as innings_runs, m.event_name, m.match_date, team_1, winner
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    GROUP BY d.batter, d.match_id, d.innings_number, m.event_name, m.match_date, m.team_1, m.winner
    ORDER BY innings_runs DESC LIMIT 10
    """
    return hub_con.execute(top_bat_q).df()

@st.cache_data
def get_best_bowling():
    top_bowl_q = """
    SELECT bowler, SUM(is_wicket) as wickets, SUM(runs_batter + runs_extras) as runs_conceded,
           m.event_name, m.match_date
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    GROUP BY d.bowler, d.match_id, d.innings_number, m.event_name, m.match_date
    HAVING wickets >= 3
    ORDER BY wickets DESC, runs_conceded ASC LIMIT 10
    """
    return hub_con.execute(top_bowl_q).df()

# ── AI Analyst Functions ───────────────────────────────────────────────────
def rewrite_query_with_llm(user_query: str, history: list) -> str:
    if not history: return user_query
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

    for _ in range(4): # Agent loop limit
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json={"model": MODEL_ID, "messages": messages, "max_tokens": 800, "temperature": 0.2}, timeout=20)
            reply = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"
            
        messages.append({"role": "assistant", "content": reply})
        sql_match = re.search(r'<SQL>(.*?)</SQL>', reply, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
            db_result = execute_sql(sql_query)
            
            st.session_state["sessions"][st.session_state["current_session_id"]]["messages"].append({
                "role": "tool", 
                "content": sql_query,
                "result": db_result
            })
            
            with st_placeholder.container():
                st.info(f"💾 Agent Executing SQL:\n```sql\n{sql_query}\n```")

            messages.append({"role": "user", "content": f"[DATABASE RESULT]\n{db_result}"})
            continue 
        else:
            return reply
    return "⚠️ Agent loop limit exceeded without a final answer."

# ── Navigation & Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5c/ICC_Cricket_World_Cup_Logo.svg/512px-ICC_Cricket_World_Cup_Logo.svg.png", width=120)
st.sidebar.markdown("## 🏏 CricketIQ")
st.sidebar.markdown("*Analytics Platform*")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio("Navigation", ["🤖 AI Analyst", "📊 Intelligence Hub", "💬 Chat History"])

# ── AI Analyst View ────────────────────────────────────────────────────────
if app_mode == "🤖 AI Analyst":
    st.title("🤖 AI Analyst")
    st.markdown("Ask anything about T20 cricket. I will write SQL and analyze the database for you.")
    
    # Interactive Sandbox UI
    with st.expander("🛠️ Interactive SQL Sandbox", expanded=False):
        st.info("Write your own SQL query against the main_gold schema to explore the database directly.")
        with st.form("sandbox_form"):
            default_query = "SELECT * FROM main_gold.fact_matches LIMIT 5"
            edited_sql = st.text_area("SQL Query", value=default_query, height=120)
            if st.form_submit_button("▶️ Run Query"):
                try:
                    df = hub_con.execute(edited_sql).df()
                    st.success(f"Query returned {len(df)} rows.")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"SQL Error: {e}")

    st.markdown("---")
    
    current_session = st.session_state["sessions"][st.session_state["current_session_id"]]
    msgs = current_session["messages"]

    if not msgs:
        st.markdown("<div class='section-header'>💡 Sample Questions</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        sample_qs = [
            "What is India's win rate in T20Is?",
            "Show India vs Pakistan head to head record",
            "Who are the top 10 run scorers?",
            "What is the avg score at Eden Gardens?",
            "Tell me about Virat Kohli's stats",
        ]
        for i, q in enumerate(sample_qs):
            if cols[i % 3].button(q, key=f"sample_{q[:20]}"):
                st.session_state["pending_query"] = q
                st.rerun()

    for i, msg in enumerate(msgs):
        if msg["role"] == "tool":
            with st.chat_message("tool", avatar="🛠️"):
                with st.expander("💾 Generated SQL Query", expanded=False):
                    st.markdown("**Query:**")
                    st.code(msg["content"], language="sql")
                    if "result" in msg:
                        st.markdown("**Result Preview:**")
                        st.markdown(msg["result"])
        else:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask about teams, players, venues, predictions...")
    if st.session_state["pending_query"]:
        user_input = st.session_state["pending_query"]
        st.session_state["pending_query"] = ""

    if user_input:
        if len(msgs) == 0:
            st.session_state["sessions"][st.session_state["current_session_id"]]["title"] = user_input[:30] + "..."
            
        st.session_state["sessions"][st.session_state["current_session_id"]]["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        status_placeholder = st.empty()
        with st.spinner("🧠 Agent is thinking..."):
            history_for_rewrite = msgs[:-1]
            standalone_query = rewrite_query_with_llm(user_input, history_for_rewrite)
            if standalone_query != user_input and standalone_query.lower() != user_input.lower():
                status_placeholder.caption(f"*Contextualized Query: '{standalone_query}'*")

            reply = agent_loop(standalone_query, history_for_rewrite, status_placeholder)

        st.session_state["sessions"][st.session_state["current_session_id"]]["messages"].append({"role": "assistant", "content": reply})
        st.rerun()

# ── Intelligence Hub View ──────────────────────────────────────────────────
elif app_mode == "📊 Intelligence Hub":
    st.title("📊 Intelligence Hub")
    
    st.markdown("""<div class='metric-card'>
    Explore pre-match predictions, venue behaviors, and strategic toss recommendations. Use the sub-tabs below to navigate different personas.
    </div>""", unsafe_allow_html=True)
    
    # Global KPIs (Power BI style)
    kpis = get_global_kpis()
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Total T20I Matches", f"{int(kpis['total_matches']):,}")
    g2.metric("Total Runs Scored", f"{int(kpis['total_runs']):,}")
    g3.metric("Total Wickets Taken", f"{int(kpis['total_wickets']):,}")
    g4.metric("Players Tracked", f"{int(kpis['total_players']):,}")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Match Analyst", "👨‍💼 Coach View", "📊 Management", "📺 Fan / Media"])
    
    teams_list = get_teams()
    venues_list = get_venues()
    
    with tab1:
        st.markdown("<div class='section-header'>Pre-Match Win Predictor</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            team1 = st.selectbox("Team 1 (Toss Winner)", teams_list, index=teams_list.index("India") if "India" in teams_list else 0)
        with c2:
            team2 = st.selectbox("Team 2", [t for t in teams_list if t != team1])
        with c3:
            venue = st.selectbox("Match Venue", venues_list)

        toss_decision = st.radio("Toss Decision", ["Bat", "Field"], horizontal=True)

        if st.button("🔮 Predict Match", type="primary"):
            h2h_rate = get_h2h_rate(team1)
            venue_avg = get_venue_avg(venue)
            team1_form = get_team_form(team1)
            
            win_prob = None
            if champion:
                feats = pd.DataFrame([{
                    "toss_bat": 1 if toss_decision == "Bat" else 0,
                    "venue_avg_1st_inns_runs": venue_avg,
                    "team_1_h2h_win_rate": h2h_rate,
                    "team_1_form_last5": team1_form,
                    "team_2_form_last5": 1 - team1_form,
                }])
                win_prob = float(champion["model"].predict_proba(feats)[0][1])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win Probability", f"{win_prob*100:.1f}%" if win_prob else "N/A", delta=f"{(win_prob-0.5)*100:+.1f}% vs 50/50" if win_prob else None)
            m2.metric("Venue Avg 1st Innings", f"{venue_avg:.0f} runs")
            m3.metric("H2H Win Rate", f"{h2h_rate*100:.1f}%")
            m4.metric("Last 5-Match Form", f"{team1_form*100:.1f}%")

            if win_prob:
                fig = go.Figure(go.Bar(
                    x=[team1, team2],
                    y=[win_prob * 100, (1 - win_prob) * 100],
                    marker_color=["#38bdf8", "#da3633"],
                    text=[f"{win_prob*100:.1f}%", f"{(1-win_prob)*100:.1f}%"],
                    textposition="auto"
                ))
                fig.update_layout(title="Win Probability %", template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
    with tab2:
        st.markdown("<div class='section-header'>Phase Run Rates & Risk Analysis</div>", unsafe_allow_html=True)
        v_sel = st.selectbox("Analyze Venue", venues_list, key="coach_v")
        phase_df = get_phase_data(v_sel)
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            if not phase_df.empty:
                fig2 = px.bar(phase_df, x="phase", y="run_rate", title=f"Run Rate by Phase",
                              color="run_rate", color_continuous_scale="Blues", template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
        with c_p2:
            if not phase_df.empty:
                fig3 = px.bar(phase_df, x="phase", y="wicket_rate", title="Wicket Rate by Phase",
                              color="wicket_rate", color_continuous_scale="Reds", template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)
                
        toss_df = get_toss_recommendation(v_sel)
        if not toss_df.empty:
            toss_df["win_pct"] = (toss_df["toss_wins"] / toss_df["total"] * 100).round(1)
            best = toss_df.loc[toss_df["win_pct"].idxmax()]
            st.success(f"🪙 **Toss Recommendation**: Choose to **{best['toss_decision'].upper()}** — {best['win_pct']:.0f}% win conversion at this venue.")

    with tab3:
        st.markdown("<div class='section-header'>Talent & Form Analytics</div>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("🏏 Top Run Scorers (Last 2 Years)")
            bat_df = get_top_batters()
            fig4 = px.bar(bat_df, x="batter", y="runs", color="sr", 
                          title="Runs (colored by Strike Rate)",
                          color_continuous_scale="Viridis", template="plotly_dark")
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)

        with col_m2:
            st.subheader("🎯 Top Wicket Takers (Last 2 Years)")
            bowl_df = get_top_bowlers()
            fig5 = px.bar(bowl_df, x="bowler", y="wickets", color="econ",
                          title="Wickets (colored by Economy)",
                          color_continuous_scale="RdYlGn_r", template="plotly_dark")
            fig5.update_layout(height=350)
            st.plotly_chart(fig5, use_container_width=True)

        st.subheader("📍 Team vs Venue Win Heatmap")
        hm_df = get_venue_heatmap()
        if not hm_df.empty:
            top_teams = hm_df.groupby("team")["matches"].sum().nlargest(12).index.tolist()
            top_venues = hm_df.groupby("venue")["matches"].sum().nlargest(15).index.tolist()
            hm_pivot = hm_df[hm_df["team"].isin(top_teams) & hm_df["venue"].isin(top_venues)]\
                .pivot_table(index="team", columns="venue", values="win_pct")
            if not hm_pivot.empty:
                fig6 = px.imshow(hm_pivot, title="Win % by Team × Venue (min 5 matches)",
                                 color_continuous_scale="RdYlGn", template="plotly_dark", aspect="auto")
                st.plotly_chart(fig6, use_container_width=True)

    with tab4:
        st.markdown("<div class='section-header'>Entertainment Analytics</div>", unsafe_allow_html=True)
        
        st.subheader("🔥 Most Exciting T20I Matches (Close Chases)")
        ex_df = get_exciting_matches()
        if not ex_df.empty:
            ex_df["excitement_score"] = 200 - ex_df["margin_runs"]
            ex_df["excitement_score"] = ex_df["excitement_score"].clip(lower=0, upper=200)
            ex_df["label"] = ex_df["event_name"].fillna("T20I") + " " + ex_df["match_date"].astype(str).str[:4]
            fig7 = px.bar(ex_df, x="label", y="excitement_score",
                          color="excitement_score", color_continuous_scale="Inferno",
                          title="Top 20 Most Exciting Matches (lower margin = more exciting)",
                          template="plotly_dark")
            st.plotly_chart(fig7, use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.subheader("💥 Highest Individual Scores")
            top_bat_df = get_highest_scores()
            st.dataframe(top_bat_df[["batter", "innings_runs", "event_name", "match_date"]].rename(columns={
                "batter": "Batter", "innings_runs": "Runs", "event_name": "Tournament", "match_date": "Date"
            }), use_container_width=True, hide_index=True)

        with col_f2:
            st.subheader("🎯 Best Bowling Spells")
            top_bowl_df = get_best_bowling()
            st.dataframe(top_bowl_df[["bowler", "wickets", "runs_conceded", "event_name", "match_date"]].rename(columns={
                "bowler": "Bowler", "wickets": "Wickets", "runs_conceded": "Runs", "event_name": "Tournament", "match_date": "Date"
            }), use_container_width=True, hide_index=True)

# ── Chat History View ───────────────────────────────────────────────────────
elif app_mode == "💬 Chat History":
    st.title("💬 Chat History")
    st.markdown("Review your previous conversations, generated SQL queries, and insights.")
    
    sessions = st.session_state["sessions"]
    
    if st.button("➕ Start New Conversation", type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state["current_session_id"] = new_id
        st.session_state["sessions"][new_id] = {"title": "New Chat", "timestamp": pd.Timestamp.now(), "messages": []}
        st.rerun()
        
    st.markdown("---")
    
    if not sessions:
        st.info("No chat history available. Go to the AI Analyst tab to start asking questions!")
    else:
        # Sort sessions by timestamp descending (newest first)
        sorted_sessions = sorted(sessions.items(), key=lambda x: x[1].get("timestamp", pd.Timestamp.min), reverse=True)
        
        for sess_id, s_data in sorted_sessions:
            msgs = s_data["messages"]
            if not msgs: continue # Skip empty new chats
            
            with st.expander(f"📝 {s_data['title']} ({s_data['timestamp'].strftime('%Y-%m-%d %H:%M')})", expanded=False):
                if st.button("Load this chat", key=f"load_{sess_id}"):
                    st.session_state["current_session_id"] = sess_id
                    # We might want to switch tabs automatically but Streamlit doesn't natively support programmatic string radio changes without session state hacks.
                    st.success("Loaded! Please switch back to '🤖 AI Analyst' in the sidebar to continue the chat.")
                    
                st.markdown("---")
                for msg in msgs:
                    if msg["role"] == "user":
                        st.markdown(f"**🧑 User:** {msg['content']}")
                    elif msg["role"] == "tool":
                        st.markdown("**🛠️ Executed SQL:**")
                        st.code(msg['content'], language="sql")
                        if "result" in msg:
                            st.markdown("**Preview:**")
                            st.caption(msg['result'])
                    elif msg["role"] == "assistant":
                        st.markdown(f"**🤖 AI:** {msg['content']}")
                        st.markdown("---")
