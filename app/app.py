"""app.py — Home / Overview page."""
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="ML Visualizer · 23CSE301",
    page_icon="📊",
    layout="wide",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_READY = os.path.exists(os.path.join(DATA_DIR, "reg_results.json"))

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Interactive ML Visualizer")
st.caption("23CSE301 — Machine Learning capstone project (B.Tech CSE)")

st.markdown(
    """
This app covers **22 algorithms** across three tracks — Regression, Classification and Clustering.
Every algorithm gets a plain-English explainer card, an interactive chart you can play with,
and — once models are trained — live predictions on real data.

Use the **sidebar** to navigate between pages.
"""
)

if not MODELS_READY:
    st.warning(
        "⚠️  Trained models not found.  "
        "Run `python train_models.py` once to train and save all models, "
        "then refresh this page."
    )
    st.info(
        "Track pages: **Regression** (10 algorithms), **Classification** (10 algorithms), "
        "**Clustering** (2 algorithms), and a **Playground** to tune hyperparameters live.  "
        "The *Learn the algorithms* cards are fully functional right now!"
    )
    st.stop()

# ── Dataset summaries ─────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Datasets")

@st.cache_data
def load_hour():
    return pd.read_csv(os.path.join(ROOT, "hour.csv"))

@st.cache_data
def load_mkt():
    df = pd.read_csv(os.path.join(ROOT, "marketing_campaign.csv"), sep=";")
    df["Age"]            = 2015 - df["Year_Birth"]
    df["TotalSpend"]     = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
                             df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"])
    df["TotalPurchases"] = (df["NumDealsPurchases"] + df["NumWebPurchases"] +
                             df["NumCatalogPurchases"] + df["NumStorePurchases"])
    return df.dropna(subset=["Income"])

hour = load_hour()
mkt  = load_mkt()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Regression dataset — `hour.csv`**")
    st.markdown(f"- **{hour.shape[0]:,}** rows · **{hour.shape[1]}** columns")
    st.markdown(f"- Target: `cnt` (hourly bike rentals)")
    st.markdown(f"- Missing values: **{hour.isnull().sum().sum()}**")
with c2:
    st.markdown("**Classification dataset — `marketing_campaign.csv`**")
    st.markdown(f"- **{mkt.shape[0]:,}** rows · **{mkt.shape[1]}** columns")
    st.markdown(f"- Target: `Response` · Classes: 0 (no) / 1 (yes)")
    st.markdown(f"- Class balance: {(mkt['Response']==1).mean()*100:.1f}% positive")
with c3:
    st.markdown("**Clustering dataset — marketing spend**")
    st.markdown("- Same `marketing_campaign.csv`")
    st.markdown("- Features: all customer attributes + engineered totals (imputed, scaled, one-hot)")
    st.markdown("- No labels used during training")

# ── EDA plots ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Quick EDA")

col1, col2, col3 = st.columns(3)

with col1:
    fig = px.histogram(hour, x="cnt", nbins=50,
                       title="Bike rentals per hour (target distribution)",
                       color_discrete_sequence=[BLUE])
    fig.update_layout(height=320, xaxis_title="cnt", yaxis_title="count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    avg_hr = hour.groupby("hr")["cnt"].mean().reset_index()
    fig = px.bar(avg_hr, x="hr", y="cnt", title="Average rentals by hour of day",
                 color_discrete_sequence=[ORANGE])
    fig.update_layout(height=320, xaxis_title="Hour", yaxis_title="avg cnt")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    resp_counts = mkt["Response"].value_counts().reset_index()
    resp_counts.columns = ["Response", "Count"]
    resp_counts["Label"] = resp_counts["Response"].map({0: "No (0)", 1: "Yes (1)"})
    fig = px.pie(resp_counts, names="Label", values="Count",
                 title="Campaign response distribution",
                 color_discrete_sequence=[BLUE, ORANGE])
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

# ── Results summary ───────────────────────────────────────────────────────────
st.divider()
st.subheader("🏆 Model Results Summary")

tab_reg, tab_clf, tab_clu = st.tabs(["Regression", "Classification", "Clustering"])

with tab_reg:
    reg_res = json.load(open(os.path.join(DATA_DIR, "reg_results.json")))
    df_reg = pd.DataFrame(reg_res).T.reset_index().rename(columns={"index": "Algorithm"})
    df_reg = df_reg.sort_values("R2", ascending=False).reset_index(drop=True)
    df_reg.index += 1
    st.dataframe(
        df_reg.style.background_gradient(subset=["R2"], cmap="Blues")
                    .format({"R2": "{:.4f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
        use_container_width=True, height=400,
    )

with tab_clf:
    clf_res = json.load(open(os.path.join(DATA_DIR, "clf_results.json")))
    df_clf = pd.DataFrame(clf_res).T.reset_index().rename(columns={"index": "Algorithm"})
    df_clf = df_clf.sort_values("F1", ascending=False).reset_index(drop=True)
    df_clf.index += 1
    st.dataframe(
        df_clf.style.background_gradient(subset=["F1"], cmap="Oranges")
                    .format({c: "{:.4f}" for c in df_clf.columns if c != "Algorithm"},
                             na_rep="—"),
        use_container_width=True, height=400,
    )

with tab_clu:
    clu_res = json.load(open(os.path.join(DATA_DIR, "clu_results.json")))
    df_clu = pd.DataFrame(clu_res).T.reset_index().rename(columns={"index": "Algorithm"})
    st.dataframe(
        df_clu.style.format({c: "{:.4f}" for c in df_clu.columns if c != "Algorithm"}),
        use_container_width=True,
    )

st.divider()
st.info(
    "Track pages: **Regression** (10 algorithms) · **Classification** (10 algorithms) · "
    "**Clustering** (2 algorithms) · **Playground** to tune live."
)
