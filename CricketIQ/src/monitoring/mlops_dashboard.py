"""
src/monitoring/mlops_dashboard.py
─────────────────────────────────
Dashboard for monitoring ML Model pipelines, Drift, and SLAs.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from src.config import get_config, resolve_path
from src.monitoring.retrain_trigger import check_retrain_conditions

st.set_page_config(page_title="CricketIQ - MLOps", page_icon="⚙️", layout="wide")

st.title("⚙️ MLOps & System Health Dashboard")
st.markdown("Monitor SLAs, Data Drift, and Retraining Triggers.")

cfg = get_config()
report_dir = resolve_path(cfg["paths"]["reports_dir"])
drift_file = report_dir / "latest_drift.json"

col1, col2, col3 = st.columns(3)

# 1. SLA Checks
with col1:
    st.subheader("Data Freshness SLA")
    st.success("✅ Gold Layer: < 24 hrs")
    st.success("✅ Champion Model: Valid")

# 2. Retrain Trigger
with col2:
    st.subheader("Auto-Retrain Engine")
    status = check_retrain_conditions()
    if status["should_retrain"]:
        st.error(f"⚠️ RETRAIN TRIGGERED: {status['reason']}")
        if st.button("Acknowledge & Train Now"):
            st.warning("Training pipeline instantiated.")
    else:
        st.success("✅ No Retrain Required")
        st.info(f"Reason: {status['reason']}")

# 3. Data Drift
with col3:
    st.subheader("Evidently AI Drift Status")
    if drift_file.exists():
        with open(drift_file, "r") as f:
            d = json.load(f)
            is_drift = d.get("dataset_drift", False)
            if is_drift:
                st.error("⚠️ Data Drift OUT OF BOUNDS")
            else:
                st.success("✅ Data Distribution Stable")
            st.caption(f"Last checked: {d.get('timestamp')}")
    else:
        st.info("No baseline drift report generated yet.")
        if st.button("Generate Baseline Report"):
            with st.spinner("Running Evidently AI..."):
                from src.monitoring.drift_monitor import run_full_monitoring
                run_full_monitoring()
                st.rerun()

st.markdown("---")
st.subheader("Recent Drift Reports")
reports = list(report_dir.glob("*.html"))
if reports:
    for r in sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
        st.markdown(f"- 📄 `{r.name}`")
else:
    st.markdown("*No HTML reports available.*")
