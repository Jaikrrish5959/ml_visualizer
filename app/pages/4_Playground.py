"""Hyperparameter sliders that retrain toy models live and redraw charts + metrics."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.ui import SAFE_PALETTE
from utils.toy_data import toy_moons, toy_unlabeled_blobs

st.set_page_config(page_title="Playground", page_icon="🎛️", layout="wide")
st.title("🎛️ Playground")
st.caption("Tune a hyperparameter and watch the model retrain instantly on toy data.")

model_choice = st.selectbox("Model", ["KNN", "Decision Tree", "SVM", "K-Means"])


def boundary_fig(model, X, y, title):
    h = 0.05
    xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, h),
                          np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=np.linspace(xx.min(), xx.max(), Z.shape[1]),
                              y=np.linspace(yy.min(), yy.max(), Z.shape[0]), z=Z,
                              showscale=False, opacity=0.3, colorscale=[[0, "#0072B2"], [1, "#E69F00"]],
                              contours=dict(coloring="heatmap")))
    for cls, color in zip(np.unique(y), ["#0072B2", "#E69F00"]):
        mask = y == cls
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"class {cls}",
                                  marker=dict(color=color, size=7)))
    fig.update_layout(title=title, height=460)
    return fig


if model_choice in ("KNN", "Decision Tree", "SVM"):
    X, y = toy_moons()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_choice == "KNN":
        k = st.slider("k", 1, 30, 5)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_choice == "Decision Tree":
        depth = st.slider("max_depth", 1, 15, 3)
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    else:
        C = st.slider("C", 0.01, 20.0, 1.0)
        model = SVC(C=C, kernel="rbf")

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(boundary_fig(model, X, y, model_choice), use_container_width=True)
    with c2:
        st.metric("Test accuracy", f"{acc:.2%}")

else:
    X = toy_unlabeled_blobs()
    k = st.slider("k (clusters)", 2, 8, 3)
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)

    fig = go.Figure()
    for i in range(k):
        mask = model.labels_ == i
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"cluster {i}",
                                  marker=dict(color=SAFE_PALETTE[i % len(SAFE_PALETTE)], size=7)))
    fig.add_trace(go.Scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1], mode="markers",
                              name="centres", marker=dict(color="black", size=16, symbol="x")))
    fig.update_layout(title="K-Means", height=460)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Inertia", f"{model.inertia_:.1f}")
