"""One explainer function per clustering algorithm."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans

from utils.ui import explainer_card, SAFE_PALETTE
from utils.toy_data import toy_unlabeled_blobs


def kmeans():
    explainer_card("Pick k centres, assign each dot to the nearest centre, move centres to the middle, repeat.")
    X = toy_unlabeled_blobs()
    c1, c2 = st.columns(2)
    k = c1.slider("k", 2, 6, 3, key="km_k")
    step = c2.radio("Step", ["Assign", "Move centres"], key="km_step", horizontal=True)

    rng = np.random.RandomState(42)
    centres = X[rng.choice(len(X), k, replace=False)]
    labels = np.argmin(((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
    if step == "Move centres":
        centres = np.array([X[labels == i].mean(axis=0) if (labels == i).any() else centres[i] for i in range(k)])
        labels = np.argmin(((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)

    fig = go.Figure()
    for i in range(k):
        mask = labels == i
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"cluster {i}",
                                  marker=dict(color=SAFE_PALETTE[i % len(SAFE_PALETTE)], size=7)))
    fig.add_trace(go.Scatter(x=centres[:, 0], y=centres[:, 1], mode="markers", name="centres",
                              marker=dict(color="black", size=18, symbol="x")))
    fig.update_layout(title=f"K-Means — {step}", height=460)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        inertias = []
        for kk in range(1, 9):
            inertias.append(KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X).inertia_)
        fig2 = go.Figure(go.Scatter(x=list(range(1, 9)), y=inertias, mode="lines+markers"))
        fig2.add_vline(x=k, line=dict(color="red", dash="dash"))
        fig2.update_layout(title="Elbow curve (chosen k in red)", height=460,
                            xaxis_title="k", yaxis_title="inertia")
        st.plotly_chart(fig2, use_container_width=True)


def agglomerative():
    explainer_card("Start with every dot alone and keep merging the closest groups.")
    X = toy_unlabeled_blobs(n=40)
    link_method = st.selectbox("Linkage", ["single", "complete", "average", "ward"], index=3, key="agg_link")
    Z = linkage(X, method=link_method)

    max_dist = Z[:, 2].max()
    cut = st.slider("Cut-line height", 0.0, float(max_dist), float(max_dist) * 0.5, key="agg_cut")
    labels = fcluster(Z, t=cut, criterion="distance")

    ddata = dendrogram(Z, no_plot=True)
    fig = go.Figure()
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="gray"), showlegend=False))
    fig.add_hline(y=cut, line=dict(color="red", dash="dash"))
    fig.update_layout(title=f"Dendrogram ({link_method} linkage)", height=420,
                       xaxis=dict(visible=False), yaxis_title="distance")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure()
        for i, cl in enumerate(np.unique(labels)):
            mask = labels == cl
            fig2.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"cluster {cl}",
                                       marker=dict(color=SAFE_PALETTE[i % len(SAFE_PALETTE)], size=8)))
        fig2.update_layout(title=f"Resulting clusters ({len(np.unique(labels))})", height=420)
        st.plotly_chart(fig2, use_container_width=True)


CLUSTERING_EXPLAINERS = {
    "K-Means": kmeans,
    "Agglomerative Hierarchical": agglomerative,
}
