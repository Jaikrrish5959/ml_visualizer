"""Small cached synthetic datasets used only for the 'Learn the algorithms' pictures."""
import numpy as np
import streamlit as st
from sklearn.datasets import make_blobs, make_moons, make_circles

RNG = 42


@st.cache_data
def toy_regression_1d(n=150):
    rng = np.random.RandomState(RNG)
    x = rng.uniform(0, 10, n)
    y = 3 * x + 5 + rng.normal(0, 4, n)
    return x, y


@st.cache_data
def toy_regression_curved(n=150):
    rng = np.random.RandomState(RNG)
    x = rng.uniform(-3, 3, n)
    y = x**2 + rng.normal(0, 1.5, n)
    return x, y


@st.cache_data
def toy_regression_multifeature(n=150, n_features=5):
    rng = np.random.RandomState(RNG)
    X = rng.normal(0, 1, (n, n_features))
    true_w = np.array([3, -2, 0.5, 0, 1.5])[:n_features]
    y = X @ true_w + rng.normal(0, 0.5, n)
    return X, y, [f"feat_{i+1}" for i in range(n_features)]


@st.cache_data
def toy_blobs(n=200, centers=2):
    X, y = make_blobs(n_samples=n, centers=centers, cluster_std=1.3, random_state=RNG)
    return X, y


@st.cache_data
def toy_moons(n=200, noise=0.2):
    X, y = make_moons(n_samples=n, noise=noise, random_state=RNG)
    return X, y


@st.cache_data
def toy_circles(n=200, noise=0.1, factor=0.5):
    X, y = make_circles(n_samples=n, noise=noise, factor=factor, random_state=RNG)
    return X, y


@st.cache_data
def toy_unlabeled_blobs(n=200, centers=3):
    X, _ = make_blobs(n_samples=n, centers=centers, cluster_std=1.1, random_state=RNG)
    return X
