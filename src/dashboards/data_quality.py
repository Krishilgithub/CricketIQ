"""
Data Quality Dashboard — Profiling, missing values, outliers, schema checks.
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_conn():
    from src.warehouse.schema import get_connection
    return get_connection()


def render_data_quality():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Data Quality Monitor</h1>
        <p>Profile, validate, and monitor data across Bronze → Silver → Gold layers</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        conn = get_conn()
    except Exception as e:
        st.error(f"Cannot connect to warehouse: {e}")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Layer Overview", "🕳️ Missing Values", "📏 Outlier Analysis", "📊 Column Profiling"
    ])

    with tab1:
        _render_layer_overview(conn)
    with tab2:
        _render_missing_values(conn)
    with tab3:
        _render_outlier_analysis(conn)
    with tab4:
        _render_column_profiling(conn)

    conn.close()


def _render_layer_overview(conn):
    """Show record counts and freshness across layers."""
    st.markdown("### 📋 Medallion Architecture — Layer Health")

    layers = {
        "Bronze": [
            ("raw_deliveries", "bronze.raw_deliveries"),
            ("raw_matches", "bronze.raw_matches"),
            ("raw_player_registry", "bronze.raw_player_registry"),
        ],
        "Silver": [
            ("deliveries", "silver.deliveries"),
            ("matches", "silver.matches"),
            ("players", "silver.players"),
        ],
        "Gold": [
            ("dim_player", "gold.dim_player"),
            ("dim_team", "gold.dim_team"),
            ("dim_venue", "gold.dim_venue"),
            ("dim_date", "gold.dim_date"),
            ("dim_tournament", "gold.dim_tournament"),
            ("fact_match_results", "gold.fact_match_results"),
            ("fact_batting_innings", "gold.fact_batting_innings"),
            ("fact_bowling_innings", "gold.fact_bowling_innings"),
            ("fact_innings_summary", "gold.fact_innings_summary"),
        ],
    }

    for layer_name, tables in layers.items():
        emoji = {"Bronze": "🥉", "Silver": "🥈", "Gold": "🥇"}[layer_name]
        st.markdown(f"#### {emoji} {layer_name} Layer")

        rows = []
        for table_label, table_fqn in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_fqn}").fetchone()[0]
                col_count = len(
                    conn.execute(f"DESCRIBE {table_fqn}").fetchall()
                )
                rows.append({
                    "Table": table_label,
                    "Records": f"{count:,}",
                    "Columns": col_count,
                    "Status": "✅ OK" if count > 0 else "⚠️ Empty",
                })
            except Exception:
                rows.append({
                    "Table": table_label,
                    "Records": "—",
                    "Columns": "—",
                    "Status": "❌ Missing",
                })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Source breakdown
    st.markdown("#### 📡 Records by Data Source")
    try:
        source_df = conn.execute("""
            SELECT source,
                   COUNT(DISTINCT match_id) AS matches,
                   COUNT(*) AS deliveries
            FROM silver.deliveries
            GROUP BY source ORDER BY matches DESC
        """).df()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                source_df, x="source", y="matches",
                color="source", title="Matches by Source",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                template="plotly_dark", showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                source_df, x="source", y="deliveries",
                color="source", title="Deliveries by Source",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                template="plotly_dark", showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("No delivery data found.")


def _render_missing_values(conn):
    """Analyze missing / null values in key tables."""
    st.markdown("### 🕳️ Missing Value Analysis")

    table_choice = st.selectbox(
        "Select table to analyze",
        ["silver.deliveries", "silver.matches", "gold.dim_player",
         "gold.dim_venue", "gold.fact_innings_summary"],
        key="dq_missing_table",
    )

    try:
        columns = conn.execute(f"DESCRIBE {table_choice}").fetchall()
        total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_choice}").fetchone()[0]

        if total_rows == 0:
            st.warning("Table is empty.")
            return

        missing_data = []
        for col_info in columns:
            col_name = col_info[0]
            null_count = conn.execute(
                f'SELECT COUNT(*) FROM {table_choice} WHERE "{col_name}" IS NULL'
            ).fetchone()[0]
            missing_data.append({
                "Column": col_name,
                "Type": col_info[1],
                "Null Count": null_count,
                "Null %": round(null_count / total_rows * 100, 2),
                "Fill Rate": f"{round((1 - null_count / total_rows) * 100, 1)}%",
            })

        df = pd.DataFrame(missing_data).sort_values("Null %", ascending=False)

        # Summary metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Rows", f"{total_rows:,}")
        with cols[1]:
            full_cols = len(df[df["Null Count"] == 0])
            st.metric("Fully Populated Columns", f"{full_cols}/{len(df)}")
        with cols[2]:
            avg_fill = df["Null %"].mean()
            st.metric("Avg Null %", f"{avg_fill:.1f}%")

        # Bar chart of missing %
        df_plot = df[df["Null Count"] > 0].head(20)
        if not df_plot.empty:
            fig = px.bar(
                df_plot, x="Column", y="Null %",
                title=f"Missing Values — {table_choice}",
                color="Null %",
                color_continuous_scale="OrRd",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values in this table!")

        st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error analyzing table: {e}")


def _render_outlier_analysis(conn):
    """Detect outliers using IQR method on numeric columns."""
    st.markdown("### 📏 Outlier Detection (IQR Method)")

    metric_choice = st.selectbox(
        "Select metric to analyze",
        [
            "Innings Total Runs",
            "Batting Strike Rate",
            "Bowling Economy Rate",
            "Batter Runs per Innings",
        ],
        key="dq_outlier_metric",
    )

    source_filter = st.multiselect(
        "Filter by source", ["t20i", "ipl", "bbl", "cpl"],
        default=["t20i"], key="dq_outlier_source"
    )

    if not source_filter:
        st.warning("Select at least one source.")
        return

    source_clause = ", ".join(f"'{s}'" for s in source_filter)

    try:
        if metric_choice == "Innings Total Runs":
            query = f"""
                SELECT total_runs AS value, match_id
                FROM gold.fact_innings_summary
                WHERE source IN ({source_clause}) AND total_runs IS NOT NULL
            """
            label = "Total Runs"
        elif metric_choice == "Batting Strike Rate":
            query = f"""
                SELECT batter_runs * 100.0 / NULLIF(balls_faced, 0) AS value, match_id
                FROM gold.fact_batting_innings
                WHERE source IN ({source_clause})
                  AND balls_faced >= 10
            """
            label = "Strike Rate"
        elif metric_choice == "Bowling Economy Rate":
            query = f"""
                SELECT runs_conceded * 6.0 / NULLIF(balls_bowled, 0) AS value, match_id
                FROM gold.fact_bowling_innings
                WHERE source IN ({source_clause})
                  AND balls_bowled >= 12
            """
            label = "Economy Rate"
        else:
            query = f"""
                SELECT runs_scored AS value, match_id
                FROM gold.fact_batting_innings
                WHERE source IN ({source_clause})
                  AND balls_faced >= 1
            """
            label = "Runs Scored"

        df = conn.execute(query).df()

        if df.empty or df["value"].isna().all():
            st.info("No data available for this metric. Make sure the Gold layer is populated.")
            return

        df = df.dropna(subset=["value"])

        q1 = df["value"].quantile(0.25)
        q3 = df["value"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df["is_outlier"] = (df["value"] < lower) | (df["value"] > upper)
        outlier_count = df["is_outlier"].sum()

        cols = st.columns(4)
        with cols[0]:
            st.metric("Q1", f"{q1:.1f}")
        with cols[1]:
            st.metric("Q3", f"{q3:.1f}")
        with cols[2]:
            st.metric("IQR", f"{iqr:.1f}")
        with cols[3]:
            st.metric("Outliers", f"{outlier_count} ({outlier_count / len(df) * 100:.1f}%)")

        # Box plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=df["value"], name=label, marker_color="#818cf8", boxpoints="outliers"))
        fig.update_layout(
            title=f"{label} Distribution — Outlier Detection",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Histogram with outlier boundaries
        fig2 = px.histogram(
            df, x="value", nbins=50, title=f"{label} Histogram",
            color_discrete_sequence=["#818cf8"],
        )
        fig2.add_vline(x=lower, line_dash="dash", line_color="red",
                       annotation_text=f"Lower: {lower:.1f}")
        fig2.add_vline(x=upper, line_dash="dash", line_color="red",
                       annotation_text=f"Upper: {upper:.1f}")
        fig2.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error in outlier analysis: {e}")


def _render_column_profiling(conn):
    """Statistical profiling of selected table columns."""
    st.markdown("### 📊 Column Profiling")

    table_choice = st.selectbox(
        "Select table",
        ["silver.deliveries", "silver.matches", "gold.fact_innings_summary",
         "gold.fact_batting_innings", "gold.fact_bowling_innings"],
        key="dq_profile_table",
    )

    try:
        # Use DuckDB SUMMARIZE
        summary = conn.execute(f"SUMMARIZE {table_choice}").df()

        if summary.empty:
            st.info("Table is empty or cannot be summarized.")
            return

        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Numeric column distributions
        columns = conn.execute(f"DESCRIBE {table_choice}").fetchall()
        numeric_cols = [
            c[0] for c in columns
            if any(t in c[1].upper() for t in ["INT", "DOUBLE", "FLOAT", "DECIMAL", "BIGINT"])
        ]

        if numeric_cols:
            st.markdown("#### Numeric Column Distributions")
            selected_col = st.selectbox("Select column", numeric_cols, key="dq_profile_col")

            data = conn.execute(
                f'SELECT "{selected_col}" AS val FROM {table_choice} WHERE "{selected_col}" IS NOT NULL'
            ).df()

            if not data.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(
                        data, x="val", nbins=50,
                        title=f"Distribution of {selected_col}",
                        color_discrete_sequence=["#818cf8"],
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(
                        data, y="val",
                        title=f"Box Plot of {selected_col}",
                        color_discrete_sequence=["#818cf8"],
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                stats = data["val"].describe()
                st.dataframe(
                    pd.DataFrame(stats).T.round(2),
                    use_container_width=True, hide_index=True,
                )

    except Exception as e:
        st.error(f"Error profiling table: {e}")
