"""Shared layout helpers so every page looks consistent."""
import streamlit as st

BLUE = "#0072B2"
ORANGE = "#E69F00"
SAFE_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def explainer_card(simple_words: str):
    st.info(f"**In simple words:** {simple_words}")


def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)
