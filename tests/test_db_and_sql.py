"""The SQL files must stay valid, and the SQL layer must agree with plain pandas."""
import datetime as dt

import pandas as pd
import pytest

from rci import db, quality

PARAMS = {
    "top_products": {"n": 3},
    "top_countries": {"n": 3},
    "rfm": {"as_of": dt.date(2011, 6, 1)},
    "features": {"cutoff": dt.date(2011, 4, 1), "horizon": 90},
    "category_revenue": {"cutoff": dt.date(2011, 4, 1)},
}


def test_load_queries_splits_named_blocks(tmp_path):
    f = tmp_path / "x.sql"
    f.write_text("-- header\n-- name: a\nSELECT 1;\n\n-- name: b\nSELECT 2\n")
    assert db.load_queries(f) == {"a": "SELECT 1", "b": "SELECT 2"}


@pytest.mark.parametrize("name", sorted(db.all_queries()))
def test_every_named_query_runs(con, name):
    out = db.query(con, name, **PARAMS.get(name, {}))
    assert isinstance(out, pd.DataFrame)


def test_headline_matches_pandas(con, frames):
    sales, returns, _ = frames
    h = db.query(con, "headline").iloc[0]
    assert h["revenue"] == round(sales["revenue"].sum())
    assert h["orders"] == sales["invoice"].nunique()
    assert h["customers"] == sales["customer_id"].nunique()
    assert h["avg_order_value"] == pytest.approx(sales["revenue"].sum() / sales["invoice"].nunique(), abs=0.01)
    assert h["guest_revenue_pct"] == pytest.approx(100 * 15 / sales["revenue"].sum(), abs=0.1)


def test_return_rate_matches_pandas(con, frames):
    sales, returns, _ = frames
    expected = round(100 * abs(returns["revenue"].sum()) / sales["revenue"].sum(), 1)
    assert db.query(con, "return_rate")["return_pct"][0] == expected


def test_monthly_revenue_matches_pandas(con, frames):
    sales, *_ = frames
    got = db.query(con, "monthly_revenue").set_index("month")["revenue"]
    want = sales.groupby(sales["invoice_date"].dt.to_period("M").dt.to_timestamp())["revenue"].sum().round()
    assert got.round().to_dict() == want.to_dict()


def test_new_vs_returning_hand_check(con):
    nv = db.query(con, "new_vs_returning").set_index("month")
    assert nv.loc[pd.Timestamp("2011-01-01"), "new_customers"] == 2          # customers 1 and 2
    assert nv.loc[pd.Timestamp("2011-02-01"), "returning_customers"] == 1    # customer 1 again
    assert nv.loc[pd.Timestamp("2011-05-01"), "returning_customers"] == 1    # customer 1 again


def test_cohort_retention_hand_check(con):
    c = db.query(con, "cohort_retention")
    jan = c[c["cohort_month"] == pd.Timestamp("2011-01-01")].set_index("months_since_first")
    assert jan.loc[0, "cohort_size"] == 2 and jan.loc[0, "retention_pct"] == 100.0
    assert jan.loc[1, "retention_pct"] == 50.0       # only customer 1 returned in Feb
    assert jan.loc[4, "retention_pct"] == 50.0       # and again in May


def test_rfm_hand_check(con):
    r = db.query(con, "rfm", as_of=dt.date(2011, 6, 1)).set_index("customer_id")
    assert r.loc[1, "frequency"] == 3 and r.loc[1, "monetary"] == 38.0
    assert r.loc[1, "recency_days"] == (dt.date(2011, 6, 1) - dt.date(2011, 5, 10)).days
    assert 4 not in r.index                           # first order is after the as_of date


def test_customers_table_uses_the_same_rfm_definition(con):
    c = con.execute("SELECT customer_id, frequency, monetary FROM customers ORDER BY 1").df().set_index("customer_id")
    assert c.loc[4, "frequency"] == 1                 # as_of = day after the last order, so everyone is included


def test_quality_checks_pass_on_clean_data(con):
    assert all(r["passed"] for r in quality.run_checks(con))


def test_quality_checks_catch_bad_data(tmp_path, frames):
    sales, returns, products = frames
    bad = sales.copy()
    bad.loc[0, "quantity"] = -3                       # a negative-quantity sale
    bad.loc[1, "revenue"] = 999.0                     # revenue no longer equals quantity x price
    bad.loc[2, "invoice"] = "C1002"                   # a cancellation hiding in sales
    from conftest import write_frames
    write_frames(tmp_path, bad, returns, products)
    failed = {r["check"] for r in quality.run_checks(db.connect(tmp_path)) if not r["passed"]}
    assert len(failed) >= 3


# ── the real data ────────────────────────────────────────────────────────────
def test_real_data_headline_matches_pandas(real_con):
    from conftest import PROCESSED
    sales = pd.read_parquet(PROCESSED / "sales.parquet")
    h = db.query(real_con, "headline").iloc[0]
    assert h["revenue"] == pytest.approx(sales["revenue"].sum(), abs=1)
    assert h["orders"] == sales["invoice"].nunique()
    assert h["customers"] == sales["customer_id"].nunique()
    assert all(r["passed"] for r in quality.run_checks(real_con))
