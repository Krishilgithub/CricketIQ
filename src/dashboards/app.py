"""
ICC T20 WC 2026 Predictor — Streamlit Dashboard Application.

Multi-page app: Data Quality, EDA, Persona Dashboards, ML Results, GenAI, Optimization.

Usage:
    streamlit run src/dashboards/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root is on path ──
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    st.set_page_config(
        page_title="🏏 ICC T20 WC 2026 Predictor",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 50%, #0d4f3c 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .main-header h1 {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .main-header p {
            color: rgba(255, 255, 255, 0.7);
            font-size: 1rem;
            margin: 0.5rem 0 0 0;
        }

        .kpi-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #818cf8;
            line-height: 1.1;
        }
        .kpi-label {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.5);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.5rem;
        }

        .section-divider {
            border: none;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            margin: 2rem 0;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 12px;
            padding: 1rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar Navigation ──
    with st.sidebar:
        st.markdown("## 🏏 Navigation")
        page = st.radio(
            "Go to",
            [
                "🏠 Home",
                "🔍 Data Quality",
                "📊 EDA Explorer",
                "🏏 Coach Dashboard",
                "📈 Analyst Dashboard",
                "🎯 Selector Dashboard",
                "📺 Broadcaster Dashboard",
                "🌍 Fan / ICC Dashboard",
                "🤖 ML Models",
                "💬 GenAI Chat",
                "⚡ Optimization",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            "<small style='color: rgba(255,255,255,0.4);'>"
            "Built with Streamlit • DuckDB • Cricsheet Data"
            "</small>",
            unsafe_allow_html=True,
        )

    # ── Page Router ──
    if page == "🏠 Home":
        render_home()
    elif page == "🔍 Data Quality":
        from src.dashboards.data_quality import render_data_quality
        render_data_quality()
    elif page == "📊 EDA Explorer":
        from src.dashboards.eda import render_eda
        render_eda()
    elif page == "🏏 Coach Dashboard":
        from src.dashboards.persona_coach import render_coach_dashboard
        render_coach_dashboard()
    elif page == "📈 Analyst Dashboard":
        from src.dashboards.persona_analyst import render_analyst_dashboard
        render_analyst_dashboard()
    elif page == "🎯 Selector Dashboard":
        from src.dashboards.persona_selector import render_selector_dashboard
        render_selector_dashboard()
    elif page == "📺 Broadcaster Dashboard":
        from src.dashboards.persona_broadcaster import render_broadcaster_dashboard
        render_broadcaster_dashboard()
    elif page == "🌍 Fan / ICC Dashboard":
        from src.dashboards.persona_fan import render_fan_dashboard
        render_fan_dashboard()
    elif page == "🤖 ML Models":
        from src.dashboards.ml_dashboard import render_ml_dashboard
        render_ml_dashboard()
    elif page == "💬 GenAI Chat":
        from src.dashboards.genai_dashboard import render_genai_dashboard
        render_genai_dashboard()
    elif page == "⚡ Optimization":
        from src.dashboards.optimization_dashboard import render_optimization_dashboard
        render_optimization_dashboard()


def render_home():
    """Home page with project overview."""
    st.markdown("""
    <div class="main-header">
        <h1>🏏 ICC Men's T20 World Cup 2026 — Outcome Prediction</h1>
        <p>Multi-Source T20 Intelligence System • Medallion Architecture • DuckDB</p>
    </div>
    """, unsafe_allow_html=True)

    # Load quick stats
    try:
        from src.warehouse.schema import get_connection
        conn = get_connection()

        match_count = conn.execute(
            "SELECT COUNT(*) FROM silver.matches"
        ).fetchone()[0]
        delivery_count = conn.execute(
            "SELECT COUNT(*) FROM silver.deliveries"
        ).fetchone()[0]
        player_count = conn.execute(
            "SELECT COUNT(*) FROM gold.dim_player"
        ).fetchone()[0]
        venue_count = conn.execute(
            "SELECT COUNT(*) FROM gold.dim_venue"
        ).fetchone()[0]
        team_count = conn.execute(
            "SELECT COUNT(*) FROM gold.dim_team"
        ).fetchone()[0]
        source_counts = conn.execute("""
            SELECT source, COUNT(DISTINCT match_id) as cnt
            FROM silver.matches GROUP BY source ORDER BY cnt DESC
        """).fetchall()
        conn.close()

        cols = st.columns(5)
        with cols[0]:
            st.metric("Total Matches", f"{match_count:,}")
        with cols[1]:
            st.metric("Deliveries", f"{delivery_count:,}")
        with cols[2]:
            st.metric("Players", f"{player_count:,}")
        with cols[3]:
            st.metric("Venues", f"{venue_count:,}")
        with cols[4]:
            st.metric("Teams", f"{team_count:,}")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Data Sources")
            for source, cnt in source_counts:
                st.markdown(f"- **{source.upper()}**: {cnt:,} matches")

        with col2:
            st.markdown("### 🏗️ Architecture")
            st.markdown("""
            ```
            Data Sources → Bronze (Raw) → Silver (Cleaned) → Gold (Analytics)
                                                                    ↓
                                                    ┌───────────────┼──────────────┐
                                                    ↓               ↓              ↓
                                              Dashboards       ML Models      GenAI/RAG
            ```
            """)

    except Exception as e:
        st.warning(
            f"⚠️ Could not load warehouse data. Run the ETL pipeline first:\n\n"
            f"```bash\npython src/etl/run_pipeline.py\n```\n\n"
            f"Error: {e}"
        )

    st.markdown("### 📌 Quick Links")
    link_cols = st.columns(4)
    with link_cols[0]:
        st.info("🔍 **Data Quality**\nProfile & validate data")
    with link_cols[1]:
        st.info("📊 **EDA Explorer**\nExplore patterns & trends")
    with link_cols[2]:
        st.info("🤖 **ML Models**\nPredictions & clustering")
    with link_cols[3]:
        st.info("💬 **GenAI Chat**\nAsk questions in plain English")


if __name__ == "__main__":
    main()
