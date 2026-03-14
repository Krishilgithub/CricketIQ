"""
src/monitoring/mlops_dashboard.py
───────────────────────────────────
CricketIQ MLOps Monitoring Dashboard (Streamlit).

Provides a live view of:
  - Feature freshness SLA status
  - Drift detection results
  - Retraining trigger conditions
  - Model performance history
  - One-click retraining

Run:
  streamlit run src/monitoring/mlops_dashboard.py --server.port 8504
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pickle
import json
from glob import glob
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.monitoring.freshness_sla import run_sla_checks
from src.monitoring.retrain_trigger import check_retrain_conditions
from src.config import get_config, resolve_path

st.set_page_config(
    page_title="CricketIQ MLOps",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1117; }
    .status-pass { color: #2ea043; font-weight: bold; }
    .status-fail { color: #da3633; font-weight: bold; }
    .status-warn { color: #d29922; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

cfg = get_config()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📡 CricketIQ MLOps")
st.sidebar.markdown("*Model & Data Health Monitor*")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Updated**: {datetime.now().strftime('%H:%M:%S')}")

if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Links**")
st.sidebar.markdown("- [MLflow UI](http://localhost:5050)")
st.sidebar.markdown("- [API Docs](http://localhost:8000/docs)")

# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("📡 CricketIQ MLOps Dashboard")
st.markdown("*Real-time model and data health monitoring*")

# ── Section 1: SLA Status ─────────────────────────────────────────────────────
st.header("🏁 Feature Freshness SLAs")

with st.spinner("Running SLA checks..."):
    sla_report = run_sla_checks()

overall_color = "🟢" if "PASS" in sla_report["overall_status"] else "🔴"
st.markdown(f"### Overall Status: {overall_color} {sla_report['overall_status']}")

cols = st.columns(len(sla_report["checks"]))
for i, check in enumerate(sla_report["checks"]):
    emoji = "✅" if "PASS" in check["status"] else ("⚠️" if "WARN" in check["status"] else "❌")
    age = f"{check.get('age_days', 'N/A')} days" if check.get('age_days') is not None else "N/A"
    cols[i].metric(
        label=check["asset"].split("(")[0].strip(),
        value=f"{emoji} {check['status'].split()[0]}",
        delta=f"Age: {age}",
        delta_color="normal" if "PASS" in check["status"] else "inverse",
    )

# Show remediation hints
failed = [c for c in sla_report["checks"] if c.get("remediation")]
if failed:
    with st.expander("⚠️ Remediation Actions Required"):
        for c in failed:
            st.warning(f"**{c['asset']}**: {c['remediation']}")

# ── Section 2: Retraining Trigger Status ────────────────────────────────────
st.header("⚡ Retraining Trigger Conditions")

with st.spinner("Evaluating retraining conditions..."):
    trigger_status = check_retrain_conditions(cfg)

t1, t2, t3 = st.columns(3)
for col, (name, cond) in zip([t1, t2, t3], trigger_status["conditions"].items()):
    icon = "🔴 TRIGGERED" if cond["triggered"] else "✅ OK"
    col.metric(
        label=name.replace("_", " ").title(),
        value=icon,
        delta=cond["value"],
        delta_color="inverse" if cond["triggered"] else "normal",
    )

if trigger_status["should_retrain"]:
    st.error("⚡ **Retraining recommended!** One or more conditions are triggered.")
    if st.button("🚀 Trigger Retraining Now"):
        with st.spinner("Training models... This may take 1-2 minutes."):
            from src.monitoring.retrain_trigger import trigger_retraining_if_needed
            success = trigger_retraining_if_needed(cfg)
        if success:
            st.success("✅ Retraining complete! New champion model saved.")
        else:
            st.error("❌ Retraining failed. Check logs for details.")
else:
    st.success("✅ No retraining needed at this time.")

# ── Section 3: Champion Model Info ──────────────────────────────────────────
st.header("🏆 Champion Model")

models_dir = str(resolve_path(cfg["paths"]["models_dir"]))
champion_path = os.path.join(models_dir, "champion_model.pkl")

if os.path.exists(champion_path):
    with open(champion_path, "rb") as f:
        champion = pickle.load(f)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Name", champion["name"].replace("_", " ").title())
    m2.metric("Log Loss", f"{champion['metrics']['log_loss_mean']:.4f}")
    m3.metric("Brier Score", f"{champion['metrics']['brier_score_mean']:.4f}")
    m4.metric("Features", len(champion["features"]))

    with st.expander("Feature List"):
        for f_name in champion["features"]:
            st.markdown(f"- `{f_name}`")
else:
    st.warning("No champion model found. Run `python -m src.models.train_model` first.")

# ── Section 4: Drift Report History ─────────────────────────────────────────
st.header("📊 Drift Report History")

reports_dir = str(resolve_path(cfg["paths"]["reports_dir"]))
summary_files = sorted(glob(os.path.join(reports_dir, "drift_summary_*.json")), reverse=True)

if summary_files:
    drift_records = []
    for f in summary_files[:10]:
        try:
            with open(f) as fp:
                d = json.load(fp)
            drift_records.append({
                "timestamp": d.get("timestamp", "")[:16],
                "data_drift": d.get("data_drift", {}).get("drift_detected", False),
                "target_drift": d.get("target_drift", {}).get("drift_detected", False),
                "win_rate_shift": d.get("target_drift", {}).get("shift", 0.0),
                "alert": d.get("alert", False),
            })
        except Exception:
            continue

    if drift_records:
        df_drift = pd.DataFrame(drift_records)
        st.dataframe(
            df_drift.style.applymap(
                lambda v: "background-color: #3d1f1f" if v is True else "",
                subset=["data_drift", "target_drift", "alert"]
            ),
            use_container_width=True,
        )

        fig = px.bar(df_drift, x="timestamp", y="win_rate_shift",
                     color="alert", color_discrete_map={True: "#da3633", False: "#2ea043"},
                     title="Win Rate Shift Over Monitoring Runs (Target Drift Proxy)",
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No drift reports yet. Run `python -m src.monitoring.drift_monitor` to generate your first report.")
    if st.button("▶️ Run Drift Monitor Now"):
        with st.spinner("Running drift detection (this may take ~30s)..."):
            from src.monitoring.drift_monitor import run_full_monitoring
            parquet = str(resolve_path(cfg["paths"]["training_dataset"]))
            result = run_full_monitoring(parquet, reports_dir)
        status_icon = "🔴 ALERT" if result["alert"] else "✅ Clean"
        st.info(f"Drift check complete: {status_icon}")
        st.rerun()
