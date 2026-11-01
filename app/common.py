"""Shared helpers for every page: path setup, cached data access, chart styling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets pages `import rci` without installing the package

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from rci import config, db  # noqa: E402

PROCESSED = config.PROCESSED_DIR
ACCENT = "#0F766E"
SEGMENT_COLORS = {
    "Champions": "#0F766E",
    "Promising": "#3B82F6",
    "At risk": "#F59E0B",
    "Hibernating": "#94A3B8",
}
MODEL_COLORS = {
    "Recency rule": "#94A3B8",
    "Logistic regression": "#3B82F6",
    "HistGradientBoosting": "#0F766E",
    "LightGBM": "#0F766E",
    "PyTorch MLP": "#F59E0B",
}
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [ACCENT, "#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899", "#64748B"]


def setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · Retail Customer Intelligence", page_icon=":material/storefront:",
                       layout="wide")
    if not (PROCESSED / "sales.parquet").exists():
        st.error("Data files not found. Build them first with `make data` (see the README).")
        st.stop()


@st.cache_resource(show_spinner="Loading data...")
def get_con():
    """One in-memory DuckDB shared by all visitors. Small memory limit for free hosting."""
    return db.connect(memory_limit="900MB", threads=2)


@st.cache_data(show_spinner=False)
def run_query(name: str, **params) -> pd.DataFrame:
    return db.query(get_con(), name, **params)


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict | None:
    path = PROCESSED / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame | None:
    path = PROCESSED / name
    return pd.read_parquet(path) if path.exists() else None


def show_sql(*names: str, label: str = "SQL behind this") -> None:
    """Show the real query text so a reader can see exactly how the numbers were produced."""
    with st.expander(label):
        for n in names:
            st.code(db.all_queries()[n], language="sql")


def gbp(x: float, decimals: int = 1) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"£{x / 1e6:.{decimals}f}M"
    if abs(x) >= 1e3:
        return f"£{x / 1e3:.{decimals}f}k"
    return f"£{x:,.0f}"


def style(fig, height: int = 380, legend_top: bool = True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10))
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None))
    return fig
