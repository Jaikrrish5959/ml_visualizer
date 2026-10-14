"""One explainer function per regression algorithm. Each renders its own card."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

from utils.ui import explainer_card, BLUE, ORANGE, SAFE_PALETTE
from utils.toy_data import toy_regression_1d, toy_regression_curved, toy_regression_multifeature


def _scatter_with_line(x, y, line_x=None, line_y=None, errors=False, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data", marker=dict(color=BLUE, size=7)))
    if line_x is not None:
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="fit", line=dict(color=ORANGE, width=3)))
    if errors and line_x is not None:
        preds = np.interp(x, line_x, line_y)
        for xi, yi, pi in zip(x, y, preds):
            fig.add_trace(go.Scatter(x=[xi, xi], y=[yi, pi], mode="lines",
                                      line=dict(color="gray", width=1, dash="dot"), showlegend=False))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y", height=420, legend=dict(orientation="h"))
    return fig


def linear_regression():
    explainer_card("Draws the one straight line closest to all the dots.")
    x, y = toy_regression_1d()
    c1, c2 = st.columns(2)
    slope = c1.slider("Your slope", -5.0, 10.0, 3.0, 0.1, key="lr_slope")
    intercept = c2.slider("Your intercept", -20.0, 20.0, 5.0, 0.5, key="lr_intercept")

    model = LinearRegression().fit(x.reshape(-1, 1), y)
    best_line = model.predict(np.linspace(x.min(), x.max(), 50).reshape(-1, 1))
    your_line = slope * np.linspace(x.min(), x.max(), 50) + intercept
    xs = np.linspace(x.min(), x.max(), 50)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data", marker=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=xs, y=best_line, mode="lines", name="best fit", line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=xs, y=your_line, mode="lines", name="your line",
                              line=dict(color="green", width=2, dash="dash")))
    fig.update_layout(title="Linear Regression", xaxis_title="x", yaxis_title="y", height=420)
    st.plotly_chart(fig, use_container_width=True)

    your_preds = slope * x + intercept
    your_mse = np.mean((y - your_preds) ** 2)
    best_mse = np.mean((y - model.predict(x.reshape(-1, 1))) ** 2)
    st.caption(f"Your line error (MSE): **{your_mse:.2f}**  |  Best-fit error (MSE): **{best_mse:.2f}**")


def _coef_bar_chart(coefs, names, title):
    fig = go.Figure(go.Bar(x=names, y=coefs, marker_color=BLUE))
    fig.update_layout(title=title, yaxis_title="coefficient value", height=380)
    return fig


def ridge():
    explainer_card("Straight line with a penalty that stops any feature weight from getting too big.")
    X, y, names = toy_regression_multifeature()
    alpha = st.slider("alpha (penalty strength)", 0.0, 20.0, 1.0, 0.5, key="ridge_alpha")
    model = Ridge(alpha=alpha).fit(X, y)
    st.plotly_chart(_coef_bar_chart(model.coef_, names, f"Ridge coefficients (alpha={alpha})"), use_container_width=True)
    st.caption("Notice bars shrink toward zero as alpha grows, but never disappear completely.")


def lasso():
    explainer_card("Like Ridge, but can shrink weights to exactly zero, which removes features.")
    X, y, names = toy_regression_multifeature()
    alpha = st.slider("alpha (penalty strength)", 0.001, 2.0, 0.05, 0.01, key="lasso_alpha")
    ridge_model = Ridge(alpha=alpha * 10).fit(X, y)
    lasso_model = Lasso(alpha=alpha).fit(X, y)
    c1, c2 = st.columns(2)
    c1.plotly_chart(_coef_bar_chart(ridge_model.coef_, names, "Ridge (for comparison)"), use_container_width=True)
    c2.plotly_chart(_coef_bar_chart(lasso_model.coef_, names, f"Lasso (alpha={alpha})"), use_container_width=True)
    st.caption("Lasso bars can hit exactly zero — that feature is effectively removed.")


def elastic_net():
    explainer_card("A blend of Ridge and Lasso.")
    X, y, names = toy_regression_multifeature()
    c1, c2 = st.columns(2)
    alpha = c1.slider("alpha", 0.001, 2.0, 0.05, 0.01, key="en_alpha")
    l1_ratio = c2.slider("l1_ratio (0=Ridge, 1=Lasso)", 0.0, 1.0, 0.5, 0.05, key="en_l1")
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X, y)
    st.plotly_chart(_coef_bar_chart(model.coef_, names, f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})"),
                     use_container_width=True)


def polynomial_regression():
    explainer_card("Bends the line into a curve.")
    x, y = toy_regression_curved()
    degree = st.slider("Polynomial degree", 1, 10, 2, key="poly_degree")

    n = len(x)
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(n * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]

    poly = PolynomialFeatures(degree=degree)
    Xp = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(Xp[train_idx], y[train_idx])

    xs = np.linspace(x.min(), x.max(), 200)
    curve = model.predict(poly.transform(xs.reshape(-1, 1)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data", marker=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=xs, y=curve, mode="lines", name=f"degree {degree} fit", line=dict(color=ORANGE, width=3)))
    fig.update_layout(title="Polynomial Regression", height=420)
    st.plotly_chart(fig, use_container_width=True)

    train_err = np.mean((y[train_idx] - model.predict(Xp[train_idx])) ** 2)
    test_err = np.mean((y[test_idx] - model.predict(Xp[test_idx])) ** 2)
    st.caption(f"Train error: **{train_err:.2f}**  |  Test error: **{test_err:.2f}**  "
               f"— a much bigger test error than train error means overfitting.")


def decision_tree_regressor():
    explainer_card("Splits data using yes/no questions and predicts each group's average.")
    x, y = toy_regression_curved()
    max_depth = st.slider("max_depth", 1, 10, 3, key="dtr_depth")
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42).fit(x.reshape(-1, 1), y)
    xs = np.linspace(x.min(), x.max(), 300)
    preds = model.predict(xs.reshape(-1, 1))
    fig = _scatter_with_line(x, y, xs, preds, title="Decision Tree Regressor (staircase prediction)")
    st.plotly_chart(fig, use_container_width=True)


def random_forest_regressor():
    explainer_card("Many trees on random samples, answers averaged.")
    x, y = toy_regression_curved()
    n_estimators = st.slider("n_estimators", 1, 50, 10, key="rfr_n")
    xs = np.linspace(x.min(), x.max(), 300)
    rng = np.random.RandomState(42)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data", marker=dict(color=BLUE)))
    n_show = min(n_estimators, 15)
    all_preds = []
    for i in range(n_estimators):
        idx = rng.choice(len(x), len(x), replace=True)
        tree = DecisionTreeRegressor(max_depth=4, random_state=i).fit(x[idx].reshape(-1, 1), y[idx])
        preds = tree.predict(xs.reshape(-1, 1))
        all_preds.append(preds)
        if i < n_show:
            fig.add_trace(go.Scatter(x=xs, y=preds, mode="lines", line=dict(color="lightgray", width=1),
                                      showlegend=False))
    avg_preds = np.mean(all_preds, axis=0)
    fig.add_trace(go.Scatter(x=xs, y=avg_preds, mode="lines", name="forest average", line=dict(color=ORANGE, width=3)))
    fig.update_layout(title="Random Forest Regressor", height=420)
    st.plotly_chart(fig, use_container_width=True)


def gradient_boosting_regressor():
    explainer_card("Trees built one after another, each fixing earlier mistakes.")
    x, y = toy_regression_curved()
    rounds = st.slider("Boosting rounds", 1, 50, 5, key="gbr_rounds")
    learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1, 0.01, key="gbr_lr")

    model = GradientBoostingRegressor(n_estimators=rounds, learning_rate=learning_rate, random_state=42)
    model.fit(x.reshape(-1, 1), y)
    xs = np.linspace(x.min(), x.max(), 300)
    preds = model.predict(xs.reshape(-1, 1))

    residuals = y - model.predict(x.reshape(-1, 1))
    c1, c2 = st.columns(2)
    with c1:
        fig = _scatter_with_line(x, y, xs, preds, title=f"Fit after {rounds} rounds")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure(go.Bar(x=list(range(len(residuals))), y=residuals, marker_color=ORANGE))
        fig2.update_layout(title="Remaining errors (residuals)", height=420)
        st.plotly_chart(fig2, use_container_width=True)


def svr():
    explainer_card("Fits a tube around the line; only dots outside the tube matter.")
    x, y = toy_regression_1d()
    c1, c2, c3 = st.columns(3)
    epsilon = c1.slider("epsilon (tube width)", 0.1, 10.0, 2.0, 0.1, key="svr_eps")
    kernel = c2.radio("kernel", ["linear", "rbf"], key="svr_kernel", horizontal=True)
    C = c3.slider("C", 0.1, 50.0, 10.0, key="svr_C")

    model = SVR(kernel=kernel, epsilon=epsilon, C=C).fit(x.reshape(-1, 1), y)
    xs = np.linspace(x.min(), x.max(), 200)
    preds = model.predict(xs.reshape(-1, 1))
    outside = np.abs(y - model.predict(x.reshape(-1, 1))) > epsilon

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=preds + epsilon, mode="lines", line=dict(width=0),
                              fillcolor="rgba(230,159,0,0.15)", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=preds - epsilon, mode="lines", line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(230,159,0,0.15)", name="epsilon tube"))
    fig.add_trace(go.Scatter(x=xs, y=preds, mode="lines", name="fit", line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=x[~outside], y=y[~outside], mode="markers", name="inside tube",
                              marker=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=x[outside], y=y[outside], mode="markers", name="outside tube (matters)",
                              marker=dict(color="red", size=10, symbol="x")))
    fig.update_layout(title="SVR", height=450)
    st.plotly_chart(fig, use_container_width=True)


def knn_regressor():
    explainer_card("Predicts the average value of the k closest dots.")
    x, y = toy_regression_1d()
    c1, c2 = st.columns(2)
    k = c1.slider("k", 1, 20, 5, key="knnr_k")
    new_x = c2.slider("new point x", float(x.min()), float(x.max()), float(x.mean()), key="knnr_x")

    model = KNeighborsRegressor(n_neighbors=k).fit(x.reshape(-1, 1), y)
    pred = model.predict([[new_x]])[0]
    dists = np.abs(x - new_x)
    nearest_idx = np.argsort(dists)[:k]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data", marker=dict(color="lightgray")))
    fig.add_trace(go.Scatter(x=x[nearest_idx], y=y[nearest_idx], mode="markers", name=f"{k} nearest",
                              marker=dict(color=BLUE, size=10)))
    for i in nearest_idx:
        fig.add_trace(go.Scatter(x=[new_x, x[i]], y=[pred, y[i]], mode="lines",
                                  line=dict(color="gray", width=1, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=[new_x], y=[pred], mode="markers", name="prediction",
                              marker=dict(color=ORANGE, size=16, symbol="star")))
    fig.update_layout(title="KNN Regressor", height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Predicted value: **{pred:.2f}** (average of the {k} nearest neighbours)")


REGRESSION_EXPLAINERS = {
    "Linear Regression": linear_regression,
    "Ridge": ridge,
    "Lasso": lasso,
    "ElasticNet": elastic_net,
    "Polynomial Regression": polynomial_regression,
    "Decision Tree Regressor": decision_tree_regressor,
    "Random Forest Regressor": random_forest_regressor,
    "Gradient Boosting Regressor": gradient_boosting_regressor,
    "SVR": svr,
    "KNN Regressor": knn_regressor,
}
