"""3_Clustering.py — Results · Learn the algorithms (no real-time predict for clustering)."""
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage

from utils.explainers_clustering import CLUSTERING_EXPLAINERS
from utils.ui import SAFE_PALETTE

st.set_page_config(page_title="Clustering", page_icon="🔵", layout="wide")
st.title("🔵 Clustering")
st.caption("Segment customers using all customer features, `Response` dropped (dataset: `marketing_campaign.csv`)")

ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_READY = os.path.exists(os.path.join(DATA_DIR, "clu_results.json"))

section  = st.sidebar.radio("Section", ["📊 Results", "📚 Learn"])
algo_key = st.sidebar.selectbox("Algorithm", list(CLUSTERING_EXPLAINERS.keys()))

# ════════════════════════════════════════════════════════════════════════════
# 📊 RESULTS
# ════════════════════════════════════════════════════════════════════════════
if section == "📊 Results":
    st.subheader("📊 Clustering Results")
    if not MODELS_READY:
        st.warning("Run `python train_models.py` first.")
        st.stop()

    clu_res = json.load(open(os.path.join(DATA_DIR, "clu_results.json")))
    df = pd.DataFrame(clu_res).T.reset_index().rename(columns={"index": "Algorithm"})
    st.dataframe(
        df.style.format({c: "{:.4f}" for c in df.columns if c != "Algorithm"}),
        use_container_width=True,
    )
    st.caption(
        "Silhouette: higher = better (max 1). "
        "Davies-Bouldin: lower = better. "
        "Calinski-Harabasz: higher = better."
    )

    X_pca      = np.load(os.path.join(DATA_DIR, "clu_X_pca.npy"))
    km_labels  = np.load(os.path.join(DATA_DIR, "clu_km_labels.npy"))
    agg_labels = np.load(os.path.join(DATA_DIR, "clu_agg_labels.npy"))

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for i in np.unique(km_labels):
            mask = km_labels == i
            fig.add_trace(go.Scatter(x=X_pca[mask, 0], y=X_pca[mask, 1], mode="markers",
                                      name=f"Cluster {i}",
                                      marker=dict(color=SAFE_PALETTE[i % len(SAFE_PALETTE)], size=5, opacity=0.7)))
        fig.update_layout(title="K-Means clusters (PCA 2D)", height=400,
                           xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        for i, cl in enumerate(np.unique(agg_labels)):
            mask = agg_labels == cl
            fig2.add_trace(go.Scatter(x=X_pca[mask, 0], y=X_pca[mask, 1], mode="markers",
                                       name=f"Cluster {cl}",
                                       marker=dict(color=SAFE_PALETTE[i % len(SAFE_PALETTE)], size=5, opacity=0.7)))
        fig2.update_layout(title="Agglomerative clusters (PCA 2D)", height=400,
                            xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig2, use_container_width=True)

    # Elbow curve
    st.divider()
    elbow = json.load(open(os.path.join(DATA_DIR, "clu_elbow.json")))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=elbow["k"], y=elbow["inertia"],
                               mode="lines+markers", line=dict(color=SAFE_PALETTE[0], width=3)))
    fig3.add_vline(x=3, line=dict(color="red", dash="dash"), annotation_text="k=3 chosen")
    fig3.update_layout(title="Elbow Curve (K-Means)", xaxis_title="k (clusters)",
                        yaxis_title="Inertia", height=380)
    st.plotly_chart(fig3, use_container_width=True)

    # Dendrogram on a sample
    st.divider()
    st.markdown("#### Dendrogram (Agglomerative, sample of 60 points)")
    X_scaled = np.load(os.path.join(DATA_DIR, "clu_X_scaled.npy"))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_scaled), 60, replace=False)
    X_samp = X_scaled[sample_idx]
    Z = linkage(X_samp, method="ward")
    ddata = dendrogram(Z, no_plot=True)

    fig4 = go.Figure()
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        fig4.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                   line=dict(color=SAFE_PALETTE[1]), showlegend=False))
    fig4.update_layout(title="Dendrogram (Ward linkage, 60-point sample)",
                        xaxis=dict(visible=False), yaxis_title="distance", height=420)
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 📚 LEARN
# ════════════════════════════════════════════════════════════════════════════
else:
    st.subheader(f"📚 Learn: {algo_key}")
    CLUSTERING_EXPLAINERS[algo_key]()
    if MODELS_READY:
        clu_res = json.load(open(os.path.join(DATA_DIR, "clu_results.json")))
        scores = clu_res.get(algo_key, {})
        if scores:
            st.markdown("**On our data:**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Silhouette",         f"{scores['Silhouette']:.4f}")
            c2.metric("Davies-Bouldin",     f"{scores['Davies_Bouldin']:.4f}")
            c3.metric("Calinski-Harabasz",  f"{scores['Calinski_Harabasz']:.1f}")
