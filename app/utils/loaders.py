"""Cached loading of trained models and their saved results/metadata."""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")


@st.cache_resource
def load_model(filename: str):
    return joblib.load(os.path.join(MODELS_DIR, filename))


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, filename))


@st.cache_data
def load_npy(filename: str) -> np.ndarray:
    return np.load(os.path.join(DATA_DIR, filename))


def regression_model_file(name: str) -> str:
    return f"reg_{name.replace(' ', '_')}.pkl"


def classification_model_file(name: str) -> str:
    return f"clf_{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"


def clustering_model_file(name: str) -> str:
    return f"clu_{name.replace(' ', '_')}.pkl"


REGRESSION_ALGOS = [
    "Linear Regression", "Ridge", "Lasso", "ElasticNet", "Polynomial Regression",
    "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor",
    "SVR", "KNN Regressor",
]

CLASSIFICATION_ALGOS = [
    "Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "SVM (SVC)",
    "Random Forest", "AdaBoost", "Gradient Boosting", "Bagging", "MLP (Neural Network)",
]

CLUSTERING_ALGOS = ["K-Means", "Agglomerative Hierarchical"]
