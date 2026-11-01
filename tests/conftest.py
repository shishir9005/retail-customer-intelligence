"""Shared test data: a tiny retail history small enough to check by hand."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from rci import db

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def _lines(rows):
    cols = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id", "country"]
    df = pd.DataFrame(rows, columns=cols)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df["quantity"] = df["quantity"].astype("int32")
    return df


def make_frames():
    """Customers (cutoff 2011-04-01, horizon 90 days -> label window Apr 1 to Jun 29):
      1 UK     orders on Jan 10, Feb 10 (before) and May 10 (after)  -> label 1, 2 orders, spend 28
      2 France order on Jan 20 only                                  -> label 0
      3 UK     orders on Mar 5 (before) and Apr 20 (after)           -> label 1
      4 UK     first order Jul 15 (after the cutoff)                 -> not in the snapshot
      guest    order on Feb 1 (customer_id NULL)                     -> never a customer
    A return by customer 1 on Mar 1. The data runs to Jul 15, so 90-day labels are observable."""
    sales = _lines([
        ("1001", "85123A", 2, "2011-01-10 10:00", 3.0, 1, "United Kingdom"),
        ("1001", "22423", 1, "2011-01-10 10:00", 10.0, 1, "United Kingdom"),
        ("1002", "85123A", 4, "2011-02-10 11:00", 3.0, 1, "United Kingdom"),
        ("1003", "22423", 1, "2011-05-10 12:00", 10.0, 1, "United Kingdom"),
        ("2001", "85123A", 10, "2011-01-20 09:00", 3.0, 2, "France"),
        ("3001", "22423", 1, "2011-03-05 14:00", 10.0, 3, "United Kingdom"),
        ("3002", "85123A", 2, "2011-04-20 15:00", 3.0, 3, "United Kingdom"),
        ("4001", "22423", 3, "2011-07-15 16:00", 10.0, 4, "United Kingdom"),
        ("9001", "85123A", 5, "2011-02-01 13:00", 3.0, None, "United Kingdom"),
    ])
    returns = _lines([("C1004", "85123A", -1, "2011-03-01 10:00", 3.0, 1, "United Kingdom")])
    products = pd.DataFrame({"stock_code": ["22423", "85123A"],
                             "description": ["REGENCY CAKESTAND 3 TIER", "WHITE HANGING HEART T-LIGHT HOLDER"]})
    return sales, returns, products


def write_frames(directory: Path, sales, returns, products, categories=None, segments=None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sales.to_parquet(directory / "sales.parquet", index=False)
    returns.to_parquet(directory / "returns.parquet", index=False)
    products.to_parquet(directory / "products.parquet", index=False)
    if categories is not None:
        categories.to_parquet(directory / "product_categories.parquet", index=False)
    if segments is not None:
        segments.to_parquet(directory / "segments.parquet", index=False)


@pytest.fixture
def frames():
    return make_frames()


@pytest.fixture
def con(tmp_path, frames):
    write_frames(tmp_path, *frames)
    return db.connect(tmp_path)


CUTOFF = dt.date(2011, 4, 1)


@pytest.fixture(scope="session")
def real_data_available() -> bool:
    return (PROCESSED / "sales.parquet").exists()


@pytest.fixture(scope="session")
def real_con(real_data_available):
    if not real_data_available:
        pytest.skip("processed data not built")
    return db.connect(PROCESSED)
