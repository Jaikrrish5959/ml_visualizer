"""1_Regression.py — Predict · Results · Learn the algorithms."""
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import streamlit as st

from utils.explainers_regression import REGRESSION_EXPLAINERS
from utils.ui import BLUE, ORANGE, SAFE_PALETTE

st.set_page_config(page_title="Regression", page_icon="📈", layout="wide")
st.title("📈 Regression")
st.caption("Predict hourly bike-rental count (dataset: `hour.csv`)")

ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_READY = os.path.exists(os.path.join(DATA_DIR, "reg_results.json"))

# ── Feature metadata ──────────────────────────────────────────────────────────
REG_FEATURES = ["season","yr","mnth","day","hr","holiday","weekday",
                 "workingday","weathersit","temp","atemp","hum","windspeed"]

FEATURE_META = {
    "season":     {"label": "Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)", "type": "select",
                   "options": [1, 2, 3, 4], "default": 2},
    "yr":         {"label": "Year (0=2011, 1=2012)", "type": "select", "options": [0, 1], "default": 1},
    "mnth":       {"label": "Month (1–12)", "type": "int", "min": 1, "max": 12, "default": 6},
    "day":        {"label": "Day of month (1–31)", "type": "int", "min": 1, "max": 31, "default": 15},
    "hr":         {"label": "Hour of day (0–23)", "type": "int", "min": 0, "max": 23, "default": 8},
    "holiday":    {"label": "Holiday? (0=No, 1=Yes)", "type": "select", "options": [0, 1], "default": 0},
    "weekday":    {"label": "Weekday (0=Sun … 6=Sat)", "type": "int", "min": 0, "max": 6, "default": 1},
    "workingday": {"label": "Working day? (0=No, 1=Yes)", "type": "select", "options": [0, 1], "default": 1},
    "weathersit": {"label": "Weather (1=Clear, 2=Mist, 3=Light rain, 4=Heavy rain)",
                   "type": "select", "options": [1, 2, 3, 4], "default": 1},
    "temp":       {"label": "Normalized temperature (0.0–1.0)", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
    "atemp":      {"label": "Normalized feeling temperature (0.0–1.0)", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
    "hum":        {"label": "Normalized humidity (0.0–1.0)", "type": "float", "min": 0.0, "max": 1.0, "default": 0.6},
    "windspeed":  {"label": "Normalized wind speed (0.0–1.0)", "type": "float", "min": 0.0, "max": 1.0, "default": 0.2},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
section = st.sidebar.radio("Section", ["🎯 Predict", "📊 Results", "📚 Learn"])
model_name = st.sidebar.selectbox("Model", list(REGRESSION_EXPLAINERS.keys()))

# ════════════════════════════════════════════════════════════════════════════
# 🎯 PREDICT
# ════════════════════════════════════════════════════════════════════════════
if section == "🎯 Predict":
    st.subheader("🎯 Predict Bike Rental Count")
    if not MODELS_READY:
        st.warning("Run `python train_models.py` first to enable predictions.")
        st.stop()

    @st.cache_resource
    def get_reg_model(name):
        fname = f"reg_{name.replace(' ', '_')}.pkl"
        return joblib.load(os.path.join(MODELS_DIR, fname))

    st.markdown(f"**Using model:** `{model_name}`")
    st.markdown("Fill in the inputs below and click **Predict**.")

    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(REG_FEATURES):
        meta = FEATURE_META[feat]
        col = cols[i % 3]
        if meta["type"] == "select":
            values[feat] = col.selectbox(meta["label"], meta["options"],
                                         index=meta["options"].index(meta["default"]),
                                         key=f"reg_inp_{feat}")
        elif meta["type"] == "int":
            values[feat] = col.number_input(meta["label"], min_value=meta["min"],
                                             max_value=meta["max"], value=meta["default"],
                                             step=1, key=f"reg_inp_{feat}")
        else:
            values[feat] = col.slider(meta["label"], meta["min"], meta["max"],
                                       meta["default"], key=f"reg_inp_{feat}")

    if st.button("🚀 Predict", type="primary"):
        pipe = get_reg_model(model_name)
        # Same engineered features as regression.ipynb
        row = json.load(open(os.path.join(DATA_DIR, "reg_input_template.json")))
        row.update(values)
        row["rush_hour"] = int(row["workingday"] == 1 and row["hr"] in [7, 8, 9, 16, 17, 18, 19])
        row["temp_hum_interaction"] = row["temp"] * row["hum"]
        row["is_weekend"] = int(row["weekday"] in [0, 6])
        X_in = pd.DataFrame([row])
        pred = pipe.predict(X_in)[0]
        st.success(f"**Predicted bike rentals: {pred:.0f}**")
        st.caption(f"Model: {model_name}")

# ════════════════════════════════════════════════════════════════════════════
# 📊 RESULTS
# ════════════════════════════════════════════════════════════════════════════
elif section == "📊 Results":
    st.subheader("📊 Model Comparison")
    if not MODELS_READY:
        st.warning("Run `python train_models.py` first.")
        st.stop()

    reg_res = json.load(open(os.path.join(DATA_DIR, "reg_results.json")))
    df = pd.DataFrame(reg_res).T.reset_index().rename(columns={"index": "Algorithm"})
    df = df.sort_values("R2", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Metric cards for best model
    best = df.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🥇 Best Model", best["Algorithm"])
    m2.metric("R²",   f"{best['R2']:.4f}")
    m3.metric("RMSE", f"{best['RMSE']:.2f}")
    m4.metric("MAE",  f"{best['MAE']:.2f}")

    def _sync_from_table():
        rows = st.session_state["reg_results_table"]["selection"]["rows"]
        if rows:
            st.session_state["reg_detail_model"] = df.iloc[rows[0]]["Algorithm"]

    st.divider()
    st.markdown("👆 **Tick the checkbox at the left of a row** (it appears when you hover) "
                "or pick a model from the dropdown below to see its predicted-vs-actual and residual plots.")
    st.dataframe(
        df.style.background_gradient(subset=["R2"], cmap="Blues")
                .format({"R2": "{:.4f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
        use_container_width=True, height=400, hide_index=True,
        on_select=_sync_from_table, selection_mode="single-row", key="reg_results_table",
    )

    sel_name = st.selectbox(
        "🔎 Show details for", df["Algorithm"].tolist(), key="reg_detail_model",
        format_func=lambda n: f"#{df.index[df['Algorithm'] == n][0] + 1}  {n}",
    )
    sel = df[df["Algorithm"] == sel_name].iloc[0]

    # ── Details for the selected model ────────────────────────────────────────
    st.divider()
    st.subheader(f"🔎 {sel_name}" + ("  ·  🥇 best model" if sel_name == best["Algorithm"] else f"  ·  rank #{sel['Rank']}"))
    d1, d2, d3 = st.columns(3)
    d1.metric("R²",   f"{sel['R2']:.4f}",   f"{sel['R2'] - best['R2']:+.4f} vs best")
    d2.metric("RMSE", f"{sel['RMSE']:.2f}", f"{sel['RMSE'] - best['RMSE']:+.2f} vs best", delta_color="inverse")
    d3.metric("MAE",  f"{sel['MAE']:.2f}",  f"{sel['MAE'] - best['MAE']:+.2f} vs best", delta_color="inverse")

    @st.cache_resource
    def get_reg_model(name):
        fname = f"reg_{name.replace(' ', '_')}.pkl"
        return joblib.load(os.path.join(MODELS_DIR, fname))

    X_te = pd.read_pickle(os.path.join(DATA_DIR, "X_reg_test.pkl"))
    y_te = np.load(os.path.join(DATA_DIR, "y_reg_test.npy"))

    @st.cache_data
    def reg_test_predictions(name):
        return get_reg_model(name).predict(X_te)

    preds = reg_test_predictions(sel_name)
    residuals = y_te - preds

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te, y=preds, mode="markers",
                                  marker=dict(color=BLUE, opacity=0.4, size=5), name="predicted"))
        mn, mx = y_te.min(), y_te.max()
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                  line=dict(color=ORANGE, width=2, dash="dash"), name="perfect"))
        fig.update_layout(title=f"Predicted vs Actual — {sel_name}",
                           xaxis_title="Actual cnt", yaxis_title="Predicted cnt", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=preds, y=residuals, mode="markers",
                                   marker=dict(color=ORANGE, opacity=0.4, size=5)))
        fig2.add_hline(y=0, line=dict(color="gray", dash="dash"))
        fig2.update_layout(title=f"Residuals — {sel_name}",
                            xaxis_title="Predicted", yaxis_title="Residual", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    st.divider()
    imp_model = json.load(open(os.path.join(DATA_DIR, "reg_feature_importance_model.json")))["model"]
    st.markdown(f"#### 🌳 {imp_model} Feature Importances")
    imp_df = pd.read_csv(os.path.join(DATA_DIR, "reg_feature_importance.csv"))
    fig3 = px.bar(imp_df, x="importance", y="feature", orientation="h",
                   color="importance", color_continuous_scale="Blues",
                   title=f"Top 15 Feature Importances ({imp_model})")
    fig3.update_layout(yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 📚 LEARN
# ════════════════════════════════════════════════════════════════════════════
else:
    st.subheader(f"📚 Learn: {model_name}")
    REGRESSION_EXPLAINERS[model_name]()
    if MODELS_READY:
        reg_res = json.load(open(os.path.join(DATA_DIR, "reg_results.json")))
        scores = reg_res.get(model_name, {})
        if scores:
            st.markdown("**On our data:**")
            c1, c2, c3 = st.columns(3)
            c1.metric("R²",   f"{scores['R2']:.4f}")
            c2.metric("RMSE", f"{scores['RMSE']:.2f}")
            c3.metric("MAE",  f"{scores['MAE']:.2f}")
