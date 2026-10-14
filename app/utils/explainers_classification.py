"""One explainer function per classification algorithm."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier

from utils.ui import explainer_card, BLUE, ORANGE
from utils.toy_data import toy_blobs, toy_moons, toy_circles

CLASS_COLORS = [BLUE, ORANGE]


def _mesh(X, h=0.05):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def _boundary_figure(model, X, y, title, extra_traces=None):
    xx, yy = _mesh(X)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=np.arange(xx.min(), xx.max(), (xx.max() - xx.min()) / xx.shape[1]),
                              y=np.arange(yy.min(), yy.max(), (yy.max() - yy.min()) / yy.shape[0]),
                              z=Z, showscale=False, opacity=0.35,
                              colorscale=[[0, BLUE], [1, ORANGE]], contours=dict(coloring="heatmap")))
    for cls, color in zip(np.unique(y), CLASS_COLORS):
        mask = y == cls
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"class {cls}",
                                  marker=dict(color=color, size=8, line=dict(width=1, color="white"))))
    if extra_traces:
        for t in extra_traces:
            fig.add_trace(t)
    fig.update_layout(title=title, height=460, xaxis_title="feature 1", yaxis_title="feature 2")
    return fig


def logistic_regression():
    explainer_card("Straight dividing line that gives a probability for each class.")
    X, y = toy_blobs()
    model = LogisticRegression().fit(X, y)

    c1, c2 = st.columns(2)
    px = c1.slider("new point — feature 1", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()), key="logreg_x")
    py = c2.slider("new point — feature 2", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()), key="logreg_y")

    proba = model.predict_proba([[px, py]])[0]
    pred_class = np.argmax(proba)
    point_trace = go.Scatter(x=[px], y=[py], mode="markers", name="new point",
                              marker=dict(color="red", size=14, symbol="x"))
    fig = _boundary_figure(model, X, y, "Logistic Regression", extra_traces=[point_trace])
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Predicted: **class {pred_class}** ({proba[pred_class]*100:.0f}% confidence)")

    zs = np.linspace(-6, 6, 100)
    sig = 1 / (1 + np.exp(-zs))
    fig2 = go.Figure(go.Scatter(x=zs, y=sig, mode="lines", line=dict(color=ORANGE, width=3)))
    fig2.update_layout(title="The sigmoid curve (turns a score into a probability)", height=250)
    st.plotly_chart(fig2, use_container_width=True)


def knn_classifier():
    explainer_card("A new point takes the majority colour of its k nearest neighbours.")
    X, y = toy_blobs()
    c1, c2, c3, c4 = st.columns(4)
    k = c1.slider("k", 1, 15, 5, key="knnc_k")
    metric = c2.radio("distance", ["euclidean", "manhattan"], key="knnc_metric", horizontal=True)
    px = c3.slider("new x", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()), key="knnc_x")
    py = c4.slider("new y", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()), key="knnc_y")

    model = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X, y)
    pred = model.predict([[px, py]])[0]

    if metric == "euclidean":
        dists = np.sqrt(((X - [px, py]) ** 2).sum(axis=1))
    else:
        dists = np.abs(X - [px, py]).sum(axis=1)
    nearest = np.argsort(dists)[:k]
    votes = {c: int((y[nearest] == c).sum()) for c in np.unique(y)}

    fig = go.Figure()
    for cls, color in zip(np.unique(y), CLASS_COLORS):
        mask = y == cls
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"class {cls}",
                                  marker=dict(color=color, size=7, opacity=0.5)))
    for i in nearest:
        fig.add_trace(go.Scatter(x=[px, X[i, 0]], y=[py, X[i, 1]], mode="lines",
                                  line=dict(color="gray", width=1, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=X[nearest, 0], y=X[nearest, 1], mode="markers", name="k nearest",
                              marker=dict(color="black", size=11, symbol="circle-open", line=dict(width=2))))
    fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers", name="new point",
                              marker=dict(color=CLASS_COLORS[pred], size=16, symbol="star")))
    fig.update_layout(title="KNN Classifier", height=460)
    st.plotly_chart(fig, use_container_width=True)
    tally = " vs ".join(f"{v} (class {c})" for c, v in votes.items())
    st.success(f"Votes: {tally} → predicted **class {pred}**")


def naive_bayes():
    explainer_card("Picks the most likely class from probabilities, assuming features don't affect each other.")
    X, y = toy_blobs()
    model = GaussianNB().fit(X, y)

    x_point = st.slider("Move the point along feature 1", float(X[:, 0].min()), float(X[:, 0].max()),
                         float(X[:, 0].mean()), key="nb_x")
    y_fixed = float(X[:, 1].mean())
    proba = model.predict_proba([[x_point, y_fixed]])[0]

    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    fig = go.Figure()
    for cls, color in zip(np.unique(y), CLASS_COLORS):
        mask = y == cls
        mu, sigma = X[mask, 0].mean(), X[mask, 0].std()
        bell = np.exp(-((xs - mu) ** 2) / (2 * sigma ** 2))
        fig.add_trace(go.Scatter(x=xs, y=bell, mode="lines", name=f"class {cls} curve", line=dict(color=color, width=3)))
    fig.add_vline(x=x_point, line=dict(color="black", dash="dash"))
    fig.update_layout(title="Naive Bayes — per-class bell curves on feature 1", height=380)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Bar(x=[f"class {c}" for c in np.unique(y)], y=proba, marker_color=CLASS_COLORS))
    fig2.update_layout(title="Class probabilities", height=280, yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)


def decision_tree_classifier():
    explainer_card("A flowchart of yes/no questions.")
    X, y = toy_blobs()
    max_depth = st.slider("max_depth", 1, 8, 3, key="dtc_depth")
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(X, y)
    fig = _boundary_figure(model, X, y, "Decision Tree — coloured regions = answers")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"The tree asks up to {max_depth} yes/no questions before deciding.")


def svm_svc():
    explainer_card("Finds the dividing line with the widest gap between classes.")
    c1, c2, c3 = st.columns(3)
    C = c1.slider("C", 0.1, 20.0, 1.0, key="svc_C")
    kernel = c2.radio("kernel", ["linear", "rbf"], key="svc_kernel", horizontal=True)
    use_circles = c3.checkbox("use circular data", value=(kernel == "rbf"), key="svc_circles")

    X, y = toy_circles() if use_circles else toy_blobs()
    model = SVC(C=C, kernel=kernel).fit(X, y)

    sv_trace = go.Scatter(x=model.support_vectors_[:, 0], y=model.support_vectors_[:, 1], mode="markers",
                           name="support vectors", marker=dict(color="black", size=12, symbol="circle-open", line=dict(width=2)))
    fig = _boundary_figure(model, X, y, "SVM (SVC)", extra_traces=[sv_trace])
    st.plotly_chart(fig, use_container_width=True)


def random_forest_classifier():
    explainer_card("Many trees vote; the majority wins.")
    X, y = toy_blobs()
    n_estimators = st.slider("n_estimators", 3, 10, 5, key="rfc_n")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=4, random_state=42).fit(X, y)
    fig = _boundary_figure(model, X, y, "Random Forest — combined vote")
    st.plotly_chart(fig, use_container_width=True)

    new_point = X[0]
    votes = [tree.predict([new_point])[0] for tree in model.estimators_[:5]]
    st.caption("First 5 trees' votes for one sample point: " + ", ".join(f"class {v}" for v in votes)
               + f" → final: class {model.predict([new_point])[0]}")


def adaboost():
    explainer_card("Simple rules added one by one; misclassified dots get more attention.")
    X, y = toy_moons()
    rounds = st.slider("Boosting rounds", 1, 50, 5, key="ada_rounds")
    model = AdaBoostClassifier(n_estimators=rounds, random_state=42).fit(X, y)
    preds = model.predict(X)
    wrong = preds != y

    xx, yy = _mesh(X)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(z=Z, x=np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, Z.shape[1]),
                              y=np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, Z.shape[0]),
                              showscale=False, opacity=0.3, colorscale=[[0, BLUE], [1, ORANGE]],
                              contours=dict(coloring="heatmap")))
    fig.add_trace(go.Scatter(x=X[~wrong, 0], y=X[~wrong, 1], mode="markers", name="correct",
                              marker=dict(color=BLUE, size=7)))
    fig.add_trace(go.Scatter(x=X[wrong, 0], y=X[wrong, 1], mode="markers", name="misclassified (bigger)",
                              marker=dict(color="red", size=14, symbol="x")))
    fig.update_layout(title=f"AdaBoost after {rounds} rounds", height=460)
    st.plotly_chart(fig, use_container_width=True)


def gradient_boosting_classifier():
    explainer_card("Each new tree fixes the remaining errors.")
    X, y = toy_moons()
    rounds = st.slider("rounds", 1, 100, 10, key="gbc_rounds")
    accs = []
    for r in range(1, rounds + 1, max(1, rounds // 10)):
        m = GradientBoostingClassifier(n_estimators=r, random_state=42).fit(X, y)
        accs.append((r, m.score(X, y)))
    model = GradientBoostingClassifier(n_estimators=rounds, random_state=42).fit(X, y)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_boundary_figure(model, X, y, f"Boundary after {rounds} rounds"), use_container_width=True)
    with c2:
        rs, a = zip(*accs)
        fig2 = go.Figure(go.Scatter(x=rs, y=a, mode="lines+markers", line=dict(color=ORANGE)))
        fig2.update_layout(title="Accuracy vs rounds", height=460, xaxis_title="rounds", yaxis_title="accuracy")
        st.plotly_chart(fig2, use_container_width=True)


def bagging():
    explainer_card("Same tree trained on many random copies of the data, then a vote.")
    X, y = toy_blobs()
    rng = np.random.RandomState(42)
    fig_rows = st.columns(3)
    boundaries = []
    for i in range(3):
        idx = rng.choice(len(X), len(X), replace=True)
        tree = DecisionTreeClassifier(max_depth=4, random_state=i).fit(X[idx], y[idx])
        boundaries.append(tree)
        with fig_rows[i]:
            st.plotly_chart(_boundary_figure(tree, X[idx], y[idx], f"Sample {i+1}"), use_container_width=True)

    model = BaggingClassifier(n_estimators=20, random_state=42).fit(X, y)
    st.plotly_chart(_boundary_figure(model, X, y, "Combined bagging boundary (smoother)"), use_container_width=True)


def mlp_classifier():
    explainer_card("Layers of connected neurons that learn curved boundaries.")
    c1, c2 = st.columns(2)
    hidden = c1.slider("hidden neurons", 1, 20, 6, key="mlp_hidden")
    activation = c2.selectbox("activation", ["relu", "tanh", "logistic"], key="mlp_act")

    X, y = toy_moons()
    model = MLPClassifier(hidden_layer_sizes=(hidden,), activation=activation, max_iter=3000, random_state=42).fit(X, y)
    st.plotly_chart(_boundary_figure(model, X, y, f"MLP ({hidden} hidden neurons, {activation})"), use_container_width=True)

    weights = model.coefs_[0]
    fig2 = go.Figure()
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            w = weights[i, j]
            fig2.add_trace(go.Scatter(x=[0, 1], y=[i, j], mode="lines", showlegend=False,
                                       line=dict(color=BLUE if w > 0 else ORANGE, width=max(0.5, abs(w) * 3))))
    fig2.add_trace(go.Scatter(x=[0]*weights.shape[0], y=list(range(weights.shape[0])), mode="markers",
                               marker=dict(color="black", size=10), name="inputs"))
    fig2.add_trace(go.Scatter(x=[1]*weights.shape[1], y=list(range(weights.shape[1])), mode="markers",
                               marker=dict(color="black", size=10), name="hidden neurons"))
    fig2.update_layout(title="Network diagram (line thickness = weight strength)", height=350,
                        xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig2, use_container_width=True)


CLASSIFICATION_EXPLAINERS = {
    "Logistic Regression": logistic_regression,
    "KNN": knn_classifier,
    "Naive Bayes": naive_bayes,
    "Decision Tree": decision_tree_classifier,
    "SVM (SVC)": svm_svc,
    "Random Forest": random_forest_classifier,
    "AdaBoost": adaboost,
    "Gradient Boosting": gradient_boosting_classifier,
    "Bagging": bagging,
    "MLP (Neural Network)": mlp_classifier,
}
