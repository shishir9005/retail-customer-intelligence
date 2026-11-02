"""Turn the sales history into a model-ready table for one "pretend today" (a cutoff date).

The heavy lifting is SQL (sql/features.sql). This file marks labels that cannot be observed yet and
stacks several cutoffs into one training table. The model uses no product-category features: the
categories (clustered over all products, dates included) stay on the Products page and in the analyst.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from rci import config, db

BASE_FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "tenure_days",
    "avg_lines_per_order", "orders_last_30d", "orders_last_90d", "revenue_last_90d",
    "n_products", "n_returns", "avg_gap_days", "is_uk",
]
# Counts and money are heavily skewed, so linear models and the MLP see log(1 + x) of them.
# Tree models do not care about scale, so they get the raw numbers.
LOG_FEATURES = [c for c in BASE_FEATURES if c != "is_uk"]


def feature_columns(con) -> list[str]:
    """The model's input columns: the 13 base features. `con` is kept so callers need not change."""
    return list(BASE_FEATURES)


def label_is_observable(last_day: dt.date, cutoff: dt.date, horizon: int = config.HORIZON_DAYS) -> bool:
    """True when the whole label window [cutoff, cutoff + horizon days) lies inside the data.

    The last day of data is only partly recorded (orders stop at midday), so the window must end on
    or before that day. One rule, used by the feature builder, the model script and the segment
    validation, so they can never disagree about which labels are real.
    """
    return cutoff + dt.timedelta(days=horizon) <= last_day


def build_snapshot(con, cutoff: dt.date | str, horizon: int = config.HORIZON_DAYS) -> pd.DataFrame:
    """One row per customer who had ordered before `cutoff`.

    Columns: customer_id, snapshot, the features, and `label`
    (1 = bought again within `horizon` days after cutoff). The label is NaN when the horizon runs
    past the end of the data, because then we cannot know the answer yet.
    """
    cutoff = pd.Timestamp(cutoff).date()
    df = db.query(con, "features", cutoff=cutoff, horizon=horizon)

    if not label_is_observable(db.data_end(con), cutoff, horizon):
        df["label"] = np.nan
    df.insert(1, "snapshot", pd.Timestamp(cutoff))
    return df


def build_dataset(con, cutoffs: list[str], horizon: int = config.HORIZON_DAYS) -> pd.DataFrame:
    """Stack several 'pretend todays' into one table (a customer can appear once per cutoff)."""
    return pd.concat([build_snapshot(con, c, horizon) for c in cutoffs], ignore_index=True)


def log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """log(1 + x) on the skewed columns; used for logistic regression and the MLP."""
    out = X.copy()
    out[LOG_FEATURES] = np.log1p(out[LOG_FEATURES].clip(lower=0))
    return out
