import sys
from pathlib import Path
import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px

# ── Ensure project root is on path ──
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import DUCKDB_PATH

st.set_page_config(
    page_title="🏏 ICC T20 WC 2026 Predictor — Executive Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 50%, #0d4f3c 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #1a3a5c;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a3a5c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_connection():
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


def main():
    st.markdown('<div class="main-header"><h1>🏏 ICC T20 World Cup 2026 — Executive Dashboard</h1><p>Integrated Data Warehouse & Prediction Insights</p></div>', unsafe_allow_html=True)

    try:
        conn = get_db_connection()
        
        # KEY METRICS
        st.subheader("📊 Key Warehouse Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        matches_count = conn.execute("SELECT COUNT(*) FROM silver.matches").fetchone()[0]
        deliveries_count = conn.execute("SELECT COUNT(*) FROM silver.deliveries").fetchone()[0]
        players_count = conn.execute("SELECT COUNT(*) FROM silver.players").fetchone()[0]
        venues_count = conn.execute("SELECT COUNT(*) FROM gold.dim_venue").fetchone()[0]
        
        col1.markdown(f'<div class="metric-card"><div>Total Matches</div><div class="metric-value">{matches_count:,}</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div>Total Deliveries</div><div class="metric-value">{deliveries_count:,}</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div>T20 Players</div><div class="metric-value">{players_count:,}</div></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-card"><div>Unique Venues</div><div class="metric-value">{venues_count:,}</div></div>', unsafe_allow_html=True)
        
        st.write("---")

        # EXPLORATORY DATA ANALYSIS (EDA)
        st.subheader("📈 Exploratory Data Analysis")
        tab1, tab2, tab3 = st.tabs(["Top Venues", "Win Margins", "Run Rate Trends"])
        
        with tab1:
            st.markdown("##### Most Frequent T20 Venues")
            venue_df = conn.execute("""
                SELECT venue, COUNT(*) as matches_hosted 
                FROM silver.matches 
                GROUP BY venue 
                ORDER BY matches_hosted DESC LIMIT 10
            """).df()
            fig1 = px.bar(venue_df, x='venue', y='matches_hosted', color='matches_hosted', 
                         color_continuous_scale='Viridis', labels={'venue': 'Venue', 'matches_hosted': 'Matches'})
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            st.markdown("##### Win Margin Distribution (Runs vs Wickets)")
            margin_df = conn.execute("""
                SELECT result_type, result_margin 
                FROM silver.matches 
                WHERE result_margin IS NOT NULL AND result_type IN ('runs', 'wickets')
            """).df()
            fig2 = px.box(margin_df, x='result_type', y='result_margin', color='result_type',
                         labels={'result_type': 'Victory Type', 'result_margin': 'Margin'},
                         points="all")
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.markdown("##### Run Rate over Overs (Phase Analysis)")
            rr_df = conn.execute("""
                SELECT over, AVG(total_runs) * 6 as avg_run_rate 
                FROM silver.deliveries 
                GROUP BY over 
                ORDER BY over
            """).df()
            fig3 = px.line(rr_df, x='over', y='avg_run_rate', markers=True, 
                          labels={'over': 'Over Number', 'avg_run_rate': 'Average Run Rate'},
                          title="Average Run Rate Progression per Over")
            st.plotly_chart(fig3, use_container_width=True)

        st.write("---")
        
        # PREDICTIVE / ML PREVIEW (Mock / Preview data if true models not trained yet)
        st.subheader("🤖 Upcoming Match Win Probabilities")
        st.info("Dynamic Model Predictions module.")
        st.dataframe(pd.DataFrame({
            "Match Date": ["2026-03-20", "2026-03-21", "2026-03-22"],
            "Team A": ["India", "Australia", "England"],
            "Team B": ["Pakistan", "New Zealand", "South Africa"],
            "Venue": ["Wankhede", "MCG", "Lord's"],
            "Win Prob (Team A)": ["52%", "48%", "55%"],
            "Win Prob (Team B)": ["48%", "52%", "45%"]
        }), use_container_width=True)

    except Exception as e:
        st.error(f"Error connecting to Data Warehouse: {e}")
        st.info("Make sure the DuckDB warehouse is populated by running the ETL pipeline.")


if __name__ == "__main__":
    main()
