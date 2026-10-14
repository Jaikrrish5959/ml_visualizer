"""2_Classification.py — Predict · Results · Learn the algorithms."""
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils.explainers_classification import CLASSIFICATION_EXPLAINERS
from utils.ui import BLUE, ORANGE, SAFE_PALETTE

st.set_page_config(page_title="Classification", page_icon="🧩", layout="wide")
st.title("🧩 Classification")
st.caption("Predict marketing campaign response (dataset: `marketing_campaign.csv`)")

ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_READY = os.path.exists(os.path.join(DATA_DIR, "clf_results.json"))

# Inputs asked in the form; every other feature (individual Mnt*/Num*/AcceptedCmp*
# columns etc.) is filled with the training-set median/mode, see classification_2.ipynb.
CLF_FEATURES = ["Age", "Income", "Total_Spend", "Total_Purchases", "Total_Campaign_Acceptances",
                 "Recency", "Kidhome", "Teenhome", "NumWebVisitsMonth", "Customer_Tenure_Days"]
CLF_CATEGORICAL = ["Education", "Marital_Status"]

FEATURE_META = {
    "Age":                        {"label": "Customer Age",                "min": 18,  "max": 90,    "default": 45,    "step": 1},
    "Income":                     {"label": "Annual Income ($)",           "min": 1000.0, "max": 200000.0, "default": 55000.0, "step": 1000.0},
    "Total_Spend":                {"label": "Total Spend ($)",             "min": 0,   "max": 2600,  "default": 400,   "step": 10},
    "Total_Purchases":            {"label": "Total Purchases (web+catalog+store)", "min": 0, "max": 45, "default": 12, "step": 1},
    "Total_Campaign_Acceptances": {"label": "Campaigns Accepted (0–5)",    "min": 0,   "max": 5,     "default": 0,     "step": 1},
    "Recency":                    {"label": "Days since last purchase",    "min": 0,   "max": 99,    "default": 30,    "step": 1},
    "Kidhome":                    {"label": "Kids at home",                "min": 0,   "max": 3,     "default": 0,     "step": 1},
    "Teenhome":                   {"label": "Teens at home",               "min": 0,   "max": 3,     "default": 0,     "step": 1},
    "NumWebVisitsMonth":          {"label": "Web visits per month",        "min": 0,   "max": 20,    "default": 5,     "step": 1},
    "Customer_Tenure_Days":       {"label": "Customer tenure (days)",      "min": 0,   "max": 1100,  "default": 500,   "step": 1},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
section    = st.sidebar.radio("Section", ["🎯 Predict", "📊 Results", "📚 Learn"])
model_name = st.sidebar.selectbox("Model", list(CLASSIFICATION_EXPLAINERS.keys()))

# ════════════════════════════════════════════════════════════════════════════
# 🎯 PREDICT
# ════════════════════════════════════════════════════════════════════════════
if section == "🎯 Predict":
    st.subheader("🎯 Will the customer respond to the campaign?")
    if not MODELS_READY:
        st.warning("Run `python train_models.py` first to enable predictions.")
        st.stop()

    @st.cache_resource
    def get_clf_model(name):
        fname = f"clf_{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        return joblib.load(os.path.join(MODELS_DIR, fname))

    st.markdown(f"**Using model:** `{model_name}`")

    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(CLF_FEATURES):
        meta = FEATURE_META[feat]
        col = cols[i % 3]
        if isinstance(meta["default"], float):
            values[feat] = col.number_input(
                meta["label"], min_value=float(meta["min"]), max_value=float(meta["max"]),
                value=float(meta["default"]), step=float(meta["step"]), key=f"clf_inp_{feat}")
        else:
            values[feat] = col.number_input(
                meta["label"], min_value=int(meta["min"]), max_value=int(meta["max"]),
                value=int(meta["default"]), step=int(meta["step"]), key=f"clf_inp_{feat}")

    template = json.load(open(os.path.join(DATA_DIR, "clf_input_template.json")))
    categories = json.load(open(os.path.join(DATA_DIR, "clf_categories.json")))
    cat_cols = st.columns(3)
    for i, feat in enumerate(CLF_CATEGORICAL):
        opts = categories[feat]
        values[feat] = cat_cols[i].selectbox(feat.replace("_", " "), opts,
                                             index=opts.index(template[feat]), key=f"clf_inp_{feat}")
    st.caption("Other features (per-category spend, purchase channels, individual campaigns, …) "
               "use the typical (median) customer from the training set.")

    if st.button("🚀 Predict", type="primary"):
        pipe = get_clf_model(model_name)
        row = dict(template)
        row.update(values)
        X_in = pd.DataFrame([row])
        pred = pipe.predict(X_in)[0]
        if hasattr(pipe.named_steps.get("model", pipe), "predict_proba"):
            try:
                proba = pipe.predict_proba(X_in)[0]
                conf = max(proba) * 100
            except Exception:
                conf = None
        else:
            conf = None

        if pred == 1:
            msg = f"✅ **Will respond** to the campaign"
        else:
            msg = f"❌ **Will NOT respond** to the campaign"

        if conf is not None:
            st.success(f"{msg}  ({conf:.0f}% confidence)")
        else:
            st.success(msg)

# ════════════════════════════════════════════════════════════════════════════
# 📊 RESULTS
# ════════════════════════════════════════════════════════════════════════════
elif section == "📊 Results":
    st.subheader("📊 Model Comparison")
    if not MODELS_READY:
        st.warning("Run `python train_models.py` first.")
        st.stop()

    @st.cache_resource
    def get_clf_model(name):
        fname = f"clf_{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        return joblib.load(os.path.join(MODELS_DIR, fname))

    clf_res = json.load(open(os.path.join(DATA_DIR, "clf_results.json")))
    df = pd.DataFrame(clf_res).T.reset_index().rename(columns={"index": "Algorithm"})
    df = df.sort_values("F1", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    best = df.iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🥇 Best Model", best["Algorithm"])
    m2.metric("Accuracy",  f"{best['Accuracy']:.4f}")
    m3.metric("Precision", f"{best['Precision']:.4f}")
    m4.metric("Recall",    f"{best['Recall']:.4f}")
    m5.metric("Weighted F1", f"{best['F1']:.4f}")

    def _sync_from_table():
        rows = st.session_state["clf_results_table"]["selection"]["rows"]
        if rows:
            st.session_state["clf_detail_model"] = df.iloc[rows[0]]["Algorithm"]

    st.divider()
    st.markdown("👆 **Tick the checkbox at the left of a row** (it appears when you hover) "
                "or pick a model from the dropdown below to see its confusion matrix and ROC curve.")
    st.dataframe(
        df.style.background_gradient(subset=["F1"], cmap="Oranges")
                .format({c: "{:.4f}" for c in df.columns if c not in ("Algorithm", "Rank")}, na_rep="—"),
        use_container_width=True, height=420, hide_index=True,
        on_select=_sync_from_table, selection_mode="single-row", key="clf_results_table",
    )
    st.caption("Ranked by weighted F1 (primary metric for the 85:15 class imbalance). "
               "Precision / Recall / F1 are weighted averages across both classes.")

    sel_name = st.selectbox(
        "🔎 Show details for", df["Algorithm"].tolist(), key="clf_detail_model",
        format_func=lambda n: f"#{df.index[df['Algorithm'] == n][0] + 1}  {n}",
    )
    sel = df[df["Algorithm"] == sel_name].iloc[0]

    # ── Details for the selected model ────────────────────────────────────────
    st.divider()
    st.subheader(f"🔎 {sel_name}" + ("  ·  🥇 best model" if sel_name == best["Algorithm"] else f"  ·  rank #{sel['Rank']}"))
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Accuracy",    f"{sel['Accuracy']:.4f}",  f"{sel['Accuracy'] - best['Accuracy']:+.4f} vs best")
    d2.metric("Precision",   f"{sel['Precision']:.4f}", f"{sel['Precision'] - best['Precision']:+.4f} vs best")
    d3.metric("Recall",      f"{sel['Recall']:.4f}",    f"{sel['Recall'] - best['Recall']:+.4f} vs best")
    d4.metric("Weighted F1", f"{sel['F1']:.4f}",        f"{sel['F1'] - best['F1']:+.4f} vs best")
    d5.metric("ROC-AUC",     f"{sel['ROC_AUC']:.4f}",   f"{sel['ROC_AUC'] - best['ROC_AUC']:+.4f} vs best")

    X_te = pd.read_pickle(os.path.join(DATA_DIR, "X_clf_test.pkl"))
    y_te = np.load(os.path.join(DATA_DIR, "y_clf_test.npy"))

    @st.cache_data
    def clf_test_outputs(name):
        pipe = get_clf_model(name)
        return pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]

    preds, y_prob = clf_test_outputs(sel_name)
    cm = confusion_matrix(y_te, preds)

    c1, c2 = st.columns(2)
    with c1:
        z = cm.tolist()
        hover = [
            [f"Predicted No, Actually No<br>{z[0][0]}",
             f"Predicted Yes, Actually No<br>{z[0][1]}"],
            [f"Predicted No, Actually Yes<br>{z[1][0]}",
             f"Predicted Yes, Actually Yes<br>{z[1][1]}"],
        ]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            text=hover, hoverinfo="text",
            colorscale="Blues", showscale=False,
        ))
        fig.update_layout(title=f"Confusion Matrix — {sel_name}", height=380)
        for i in range(2):
            for j in range(2):
                fig.add_annotation(x=j, y=i, text=str(z[i][j]),
                                    showarrow=False, font=dict(size=18, color="white" if z[i][j]>cm.max()/2 else "black"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        fig2 = go.Figure()
        if sel_name != best["Algorithm"]:
            _, best_prob = clf_test_outputs(best["Algorithm"])
            bfpr, btpr, _ = roc_curve(y_te, best_prob)
            fig2.add_trace(go.Scatter(x=bfpr, y=btpr, mode="lines",
                                       name=f"{best['Algorithm']} (best) AUC = {auc(bfpr, btpr):.3f}",
                                       line=dict(color=BLUE, width=2, dash="dot"), opacity=0.6))
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{sel_name} AUC = {roc_auc:.3f}",
                                   line=dict(color=ORANGE, width=3)))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   line=dict(color="gray", dash="dash"), name="random"))
        fig2.update_layout(title=f"ROC Curve — {sel_name}", xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate", height=380,
                            legend=dict(x=0.35, y=0.05))
        st.plotly_chart(fig2, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    st.caption(
        f"On the {len(y_te)} test customers, {sel_name} correctly caught **{tp}** of {tp + fn} responders "
        f"and raised **{fp}** false alarms. "
        "AUC (Area Under the Curve): closer to 1.0 = better, 0.5 = random guessing."
    )

# ════════════════════════════════════════════════════════════════════════════
# 📚 LEARN
# ════════════════════════════════════════════════════════════════════════════
else:
    st.subheader(f"📚 Learn: {model_name}")
    CLASSIFICATION_EXPLAINERS[model_name]()
    if MODELS_READY:
        clf_res = json.load(open(os.path.join(DATA_DIR, "clf_results.json")))
        scores = clf_res.get(model_name, {})
        if scores:
            st.markdown("**On our data:**")
            cols = st.columns(5)
            cols[0].metric("Accuracy",  f"{scores['Accuracy']:.4f}")
            cols[1].metric("Precision", f"{scores['Precision']:.4f}")
            cols[2].metric("Recall",    f"{scores['Recall']:.4f}")
            cols[3].metric("Weighted F1", f"{scores['F1']:.4f}")
            cols[4].metric("ROC-AUC",   f"{scores['ROC_AUC']:.4f}" if scores["ROC_AUC"] else "—")
