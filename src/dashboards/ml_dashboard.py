"""
ML Dashboard — Model results, comparisons, SHAP, predictions, clustering viz.
"""

import sys
import pickle
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import ML_MODEL_DIR


def render_ml_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Machine Learning Models</h1>
        <p>Match outcome prediction, score regression, player clustering, win probability</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Match Outcome", "📊 Score Prediction",
        "👥 Player Clusters", "📈 Win Probability", "🔮 Predict a Match"
    ])

    with tab1:
        _render_match_outcome()
    with tab2:
        _render_score_prediction()
    with tab3:
        _render_clustering()
    with tab4:
        _render_win_probability()
    with tab5:
        _render_predict_match()


def _render_match_outcome():
    st.markdown("### 🏆 Match Outcome Classification")

    model_path = ML_MODEL_DIR / "match_outcome_model.pkl"
    if not model_path.exists():
        st.warning("⚠️ No trained model found. Train first:")
        st.code("python src/ml/match_outcome.py", language="bash")

        if st.button("🚀 Train Match Outcome Models Now", key="train_match"):
            with st.spinner("Training models... This may take a minute."):
                try:
                    from src.warehouse.schema import get_connection
                    from src.ml.feature_engineering import build_match_features
                    from src.ml.match_outcome import train_match_outcome_models

                    conn = get_connection()
                    features = build_match_features(conn, source="t20i")
                    conn.close()

                    if not features.empty:
                        results = train_match_outcome_models(features)
                        st.success(f"✅ Trained {len(results) - 3} models!")
                        _display_classification_results(results)
                    else:
                        st.error("No features built. Run ETL pipeline first.")
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        st.success(f"✅ Model loaded: **{saved['model_name']}**")
        st.markdown(f"**Features used:** {len(saved['feature_cols'])}")
        st.dataframe(
            pd.DataFrame({"Feature": saved["feature_cols"]}),
            use_container_width=True, hide_index=True,
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")


def _display_classification_results(results):
    """Display results from training run."""
    rows = []
    for name, res in results.items():
        if name.startswith("_"):
            continue
        rows.append({
            "Model": name,
            "Accuracy": f"{res['accuracy']:.4f}",
            "F1 Score": f"{res['f1_score']:.4f}",
            "ROC AUC": f"{res['roc_auc']:.4f}",
            "CV Mean": f"{res['cv_mean']:.4f}",
            "CV Std": f"{res['cv_std']:.4f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Feature importance
    from src.ml.match_outcome import get_feature_importance
    imp_df = get_feature_importance(results)
    if not imp_df.empty:
        fig = px.bar(
            imp_df.sort_values("Importance", ascending=False).head(20),
            x="Feature", y="Importance", color="Model",
            title="Feature Importance (Top 20)", barmode="group",
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_score_prediction():
    st.markdown("### 📊 Score Prediction (Regression)")

    model_path = ML_MODEL_DIR / "score_prediction_model.pkl"
    if not model_path.exists():
        st.warning("⚠️ No trained model found.")
        if st.button("🚀 Train Score Prediction Models", key="train_score"):
            with st.spinner("Training..."):
                try:
                    from src.warehouse.schema import get_connection
                    from src.ml.feature_engineering import build_innings_features
                    from src.ml.score_prediction import train_score_models

                    conn = get_connection()
                    innings = build_innings_features(conn, source="t20i")
                    conn.close()

                    if not innings.empty:
                        results = train_score_models(innings)
                        _display_regression_results(results)
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        st.success(f"✅ Model: **{saved['model_name']}**")
        st.markdown(f"Features: {', '.join(saved['feature_cols'])}")
    except Exception as e:
        st.error(f"Error: {e}")


def _display_regression_results(results):
    rows = []
    for name, res in results.items():
        if name.startswith("_"):
            continue
        rows.append({
            "Model": name,
            "MAE": f"{res['mae']:.2f}",
            "RMSE": f"{res['rmse']:.2f}",
            "R²": f"{res['r2']:.4f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Actual vs Predicted scatter
    best = results[results["_best_model"]]
    fig = px.scatter(
        x=best["y_test"], y=best["y_pred"],
        labels={"x": "Actual Score", "y": "Predicted Score"},
        title=f"Actual vs Predicted — {results['_best_model']}",
        color_discrete_sequence=["#818cf8"],
    )
    fig.add_shape(type="line", x0=50, x1=250, y0=50, y1=250,
                  line=dict(dash="dash", color="grey"))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_clustering():
    st.markdown("### 👥 Player Clustering")

    model_path = ML_MODEL_DIR / "player_clustering_model.pkl"
    if not model_path.exists():
        st.warning("⚠️ No clustering model found.")
        if st.button("🚀 Run Player Clustering", key="train_cluster"):
            with st.spinner("Clustering..."):
                try:
                    from src.warehouse.schema import get_connection
                    from src.ml.feature_engineering import build_player_features
                    from src.ml.player_clustering import cluster_players

                    conn = get_connection()
                    players = build_player_features(conn, source="t20i")
                    conn.close()

                    if not players.empty:
                        n_clusters = st.sidebar.slider("Number of Clusters", 3, 8, 5)
                        results = cluster_players(players, n_clusters=n_clusters)
                        if results:
                            _display_clustering_results(results)
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        st.success("✅ Clustering model loaded")
        st.json(saved["cluster_labels"])
    except Exception as e:
        st.error(f"Error: {e}")


def _display_clustering_results(results):
    df = results["clustered_df"]

    # PCA scatter
    fig = px.scatter(
        df, x="pca_1", y="pca_2",
        color="archetype",
        hover_name="player_name",
        title="Player Clusters (PCA)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster distribution
    col1, col2 = st.columns(2)
    with col1:
        dist = df["archetype"].value_counts().reset_index()
        dist.columns = ["Archetype", "Count"]
        fig = px.pie(dist, names="Archetype", values="Count",
                     title="Cluster Distribution", hole=0.4)
        fig.update_layout(template="plotly_dark",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Elbow plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results["k_range"], y=results["silhouettes"],
            mode="lines+markers", name="Silhouette",
            line=dict(color="#818cf8"),
        ))
        fig.update_layout(
            title="Silhouette Score vs K", xaxis_title="K", yaxis_title="Silhouette",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[["player_name", "archetype", "avg_runs", "avg_sr", "total_wickets", "avg_economy"]].head(50),
        use_container_width=True, hide_index=True,
    )


def _render_win_probability():
    st.markdown("### 📈 Win Probability Model")

    model_path = ML_MODEL_DIR / "win_probability_model.pkl"
    if not model_path.exists():
        st.warning("⚠️ No model found.")
        if st.button("🚀 Train Win Probability Model", key="train_wp"):
            with st.spinner("Training (this processes ball-level data)..."):
                try:
                    from src.warehouse.schema import get_connection
                    from src.ml.win_probability import (
                        build_ball_level_features, train_win_probability_model,
                    )

                    conn = get_connection()
                    ball_df = build_ball_level_features(conn, source="t20i")
                    conn.close()

                    if not ball_df.empty:
                        results = train_win_probability_model(ball_df)
                        st.success(f"✅ Best model: {results['best_model']}")
                        st.metric("AUC", f"{results['lr']['auc']:.4f}")
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        st.success(f"✅ Model: **{saved['model_name']}**")
    except Exception as e:
        st.error(f"Error: {e}")


def _render_predict_match():
    st.markdown("### 🔮 Predict a Match")

    model_path = ML_MODEL_DIR / "match_outcome_model.pkl"
    if not model_path.exists():
        st.warning("Train the match outcome model first.")
        return

    try:
        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        from src.warehouse.schema import get_connection
        conn = get_connection()

        teams = conn.execute("""
            SELECT team_name FROM gold.dim_team
            WHERE is_icc_full_member = TRUE ORDER BY team_name
        """).fetchall()
        team_list = [t[0] for t in teams]
        conn.close()

        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1", team_list, key="pred_t1")
        with col2:
            team2 = st.selectbox("Team 2", team_list, index=1, key="pred_t2")

        if team1 == team2:
            st.warning("Select different teams.")
            return

        col1, col2 = st.columns(2)
        with col1:
            toss_winner = st.selectbox("Toss Winner", [team1, team2], key="pred_toss")
            toss_decision = st.selectbox("Toss Decision", ["bat", "field"], key="pred_dec")
        with col2:
            t1_wr = st.slider(f"{team1} Recent Win Rate", 0.0, 1.0, 0.5, key="pred_t1wr")
            t2_wr = st.slider(f"{team2} Recent Win Rate", 0.0, 1.0, 0.5, key="pred_t2wr")

        if st.button("🏏 Predict!", key="pred_btn"):
            feature_values = {col: 0 for col in saved["feature_cols"]}
            feature_values["toss_winner_is_team1"] = 1 if toss_winner == team1 else 0
            feature_values["toss_elected_bat"] = 1 if toss_decision == "bat" else 0
            feature_values["team1_recent_win_rate"] = t1_wr
            feature_values["team2_recent_win_rate"] = t2_wr
            feature_values["win_rate_diff"] = t1_wr - t2_wr
            feature_values["venue_avg_1st_score"] = 155
            feature_values["venue_avg_2nd_score"] = 145
            feature_values["team1_avg_score"] = 155
            feature_values["team2_avg_score"] = 155
            feature_values["team1_avg_rr"] = 8.0
            feature_values["team2_avg_rr"] = 8.0
            feature_values["h2h_team1_win_rate"] = 0.5

            X = np.array([[feature_values.get(c, 0) for c in saved["feature_cols"]]])

            if "Logistic" in saved["model_name"]:
                X = saved["scaler"].transform(X)

            prob = saved["model"].predict_proba(X)[0]

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #818cf8;">{prob[1]*100:.1f}%</div>
                    <div class="kpi-label">{team1} Win Probability</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #f472b6;">{prob[0]*100:.1f}%</div>
                    <div class="kpi-label">{team2} Win Probability</div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
