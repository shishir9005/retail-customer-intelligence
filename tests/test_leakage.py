"""Permanent leakage audit, run on the REAL data.

These tests grew out of one-off audit scripts. They answer, with evidence, "can anything from the
period being predicted reach the training features?":

  1. Features at a cutoff are unchanged when EVERY later row is deleted (all six cutoffs).
  2. The SQL features agree with an independent pandas implementation (the boundary arithmetic is right).
  3. The feature SQL can only read `sales` and `returns`.
  4. Fitted scalers see training rows only.
  5. Labels that are not fully observed are REFUSED (they used to be cast to garbage integers silently).
  6. Segments are refitted from earlier data only when they are validated against later behaviour.
  7. The step that runs over all dates before the split (price-median cleaning) is measured to have no effect on
     identified customers' rows, and the features are checked not to depend on the category file (which is clustered
     over all products, dates included): the snapshots are identical with and without it, and none of them has a
     category column.

tests/test_no_leakage.py holds the small hand-checkable versions of 1 on a toy dataset.
"""
from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np
import pandas as pd
import pytest

from rci import config, db, features, model, pipeline, segments, sql_guard

from conftest import PROCESSED, write_frames

ALL_CUTOFFS = [(split, c) for split, cutoffs in config.SPLITS.items() for c in cutoffs]
CUTOFF_IDS = [f"{split}-{c}" for split, c in ALL_CUTOFFS]
VALID_CUTOFF = config.SPLITS["valid"][0]
TEST_CUTOFF = config.SPLITS["test"][0]

# Whole-number features must match exactly; money features are published rounded to the penny.
EXACT_COLUMNS = ["recency_days", "frequency", "tenure_days", "orders_last_30d", "orders_last_90d",
                 "n_products", "n_returns", "is_uk"]


# ── shared helpers ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def real_frames(real_data_available):
    if not real_data_available:
        pytest.skip("processed data not built")
    return {name: pd.read_parquet(PROCESSED / f"{name}.parquet")
            for name in ("sales", "returns", "products", "product_categories")}


def future_deleted_con(directory, frames, cutoff):
    """A database in which every sale and return on or after `cutoff` has been physically removed."""
    ts = pd.Timestamp(cutoff)
    write_frames(directory,
                 frames["sales"][frames["sales"]["invoice_date"] < ts],
                 frames["returns"][frames["returns"]["invoice_date"] < ts],
                 frames["products"], frames["product_categories"])
    return db.connect(directory)


def strip_rounding(sql: str) -> str:
    """The feature SQL with its ROUND(x, n) calls removed, to compare values before display rounding."""
    out = re.sub(r"ROUND\(([^,()]+(?:\([^()]*\))?[^,()]*),\s*\d\)", r"\1", sql)
    assert "ROUND(" not in out
    return out


# ═════════════════════ 1. the future cannot move the features ═════════════════════
@pytest.mark.slow
@pytest.mark.parametrize("split,cutoff", ALL_CUTOFFS, ids=CUTOFF_IDS)
def test_features_do_not_change_when_every_later_row_is_deleted(split, cutoff, real_con, real_frames, tmp_path):
    cut_con = future_deleted_con(tmp_path, real_frames, cutoff)
    full = features.build_snapshot(real_con, cutoff).set_index("customer_id")
    cut = features.build_snapshot(cut_con, cutoff).set_index("customer_id")
    cols = features.feature_columns(real_con)

    assert full.index.equals(cut.index), "the set of customers depends on rows after the cutoff"
    pd.testing.assert_frame_equal(full[EXACT_COLUMNS], cut[EXACT_COLUMNS])

    # Published money values are ROUND(x, 2). A different summation order (the future-deleted database
    # has different row sets) can flip a value that sits exactly on a .xx5 tie by one penny. That is
    # display rounding, not information: it must stay within one penny...
    published_gap = (full[cols].astype(float) - cut[cols].astype(float)).abs().to_numpy().max()
    assert published_gap <= 0.0100001

    # ...and with the rounding removed the two databases must agree to floating-point noise.
    raw_sql = strip_rounding(db.all_queries()["features"])
    params = {"cutoff": pd.Timestamp(cutoff).date(), "horizon": config.HORIZON_DAYS}
    raw_full = real_con.cursor().execute(raw_sql, params).df().set_index("customer_id").drop(columns="label")
    raw_cut = cut_con.cursor().execute(raw_sql, params).df().set_index("customer_id").drop(columns="label")
    assert (raw_full.astype(float) - raw_cut.astype(float)).abs().to_numpy().max() < 1e-6


# ═════════════════════ 2. an independent implementation agrees ═════════════════════
def pandas_reference(sales: pd.DataFrame, cutoff: str, horizon: int) -> pd.DataFrame:
    """Recency, frequency, spend, 90-day activity, product count and the label, derived from raw rows in pandas."""
    c = pd.Timestamp(cutoff)
    ids = sales[sales["customer_id"].notna()].copy()
    ids["customer_id"] = ids["customer_id"].astype("int64")
    hist = ids[ids["invoice_date"] < c]
    inv = hist.groupby(["customer_id", "invoice"]).agg(day=("invoice_date", "min"), rev=("revenue", "sum")).reset_index()
    inv["days_ago"] = (c.normalize() - inv["day"].dt.normalize()).dt.days
    g = inv.groupby("customer_id")
    recent = inv[inv["days_ago"] <= 90].groupby("customer_id")
    ref = pd.DataFrame({"recency_days": g["days_ago"].min(), "tenure_days": g["days_ago"].max(),
                        "frequency": g.size(), "monetary": g["rev"].sum()})
    ref["orders_last_90d"] = recent.size().reindex(ref.index, fill_value=0)
    ref["revenue_last_90d"] = recent["rev"].sum().reindex(ref.index, fill_value=0.0)
    ref["n_products"] = hist.groupby("customer_id")["stock_code"].nunique()
    window_end = c.normalize() + pd.Timedelta(days=horizon)
    returned = set(ids[(ids["invoice_date"] >= c) & (ids["invoice_date"] < window_end)]["customer_id"])
    ref["label"] = ref.index.isin(returned).astype(int)
    return ref


@pytest.mark.parametrize("cutoff", [VALID_CUTOFF, TEST_CUTOFF], ids=["valid", "test"])
def test_sql_features_match_an_independent_pandas_implementation(cutoff, real_con, real_frames):
    ref = pandas_reference(real_frames["sales"], cutoff, config.HORIZON_DAYS)
    sql = features.build_snapshot(real_con, cutoff).set_index("customer_id")

    assert set(sql.index) == set(ref.index), "customers differ: someone first ordered on/after the cutoff, or went missing"
    sql = sql.loc[ref.index]
    for col in ["recency_days", "tenure_days", "frequency", "orders_last_90d", "n_products", "label"]:
        assert (sql[col].astype(int) == ref[col].astype(int)).all(), col
    for col in ["monetary", "revenue_last_90d"]:
        np.testing.assert_allclose(sql[col].astype(float), ref[col].astype(float), atol=0.011, err_msg=col)


# ═════════════════════ 3. the feature SQL can only read sales and returns ═════════════════════
def test_feature_sql_reads_only_the_event_tables():
    """Never `customers`, `segments` or the all-time `products` statistics (which aggregate over every date)."""
    sql = db.all_queries()["features"].replace("$cutoff", "DATE '2011-01-01'").replace("$horizon", "90")
    tables, table_functions = sql_guard.parse_references(sql)
    assert tables == {"sales", "returns"}
    assert not table_functions


# ═════════════════════ 4. fitted scalers only see training rows ═════════════════════
def synthetic_features(n: int, seed: int, scale: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.gamma(2.0, scale, n) for c in features.BASE_FEATURES})
    X["is_uk"] = rng.integers(0, 2, n)
    return X


def test_logistic_regression_scaler_is_fitted_on_training_rows_only():
    X_train = synthetic_features(400, seed=0, scale=50.0)
    X_later = synthetic_features(400, seed=1, scale=5000.0)       # a very different "future"
    y = (X_train["recency_days"] < X_train["recency_days"].median()).astype(int).to_numpy()
    lr = model.fit_logreg(X_train, y)
    scaler = lr.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, features.log_transform(X_train).to_numpy().mean(axis=0))
    pooled = features.log_transform(pd.concat([X_train, X_later])).to_numpy().mean(axis=0)
    assert not np.allclose(scaler.mean_, pooled), "scaler statistics should not include any later data"


# ═════════════════════ 5. unobserved labels are refused, not silently cast ═════════════════════
def test_labels_as_int_accepts_observed_labels():
    out = model.labels_as_int(pd.DataFrame({"label": [0.0, 1.0, 1.0, 0.0]}), "train")
    assert out.dtype.kind == "i" and out.tolist() == [0, 1, 1, 0]


def test_labels_as_int_refuses_nan_and_names_the_split():
    with pytest.raises(ValueError, match=r"valid: 2 of 4 labels are NaN"):
        model.labels_as_int(pd.DataFrame({"label": [1.0, np.nan, 0.0, np.nan]}), "valid")


@pytest.mark.parametrize("bad", [[0, 2], [0.5, 1.0], [-1, 1]])
def test_labels_as_int_refuses_values_that_are_not_zero_or_one(bad):
    with pytest.raises(ValueError, match="0 or 1"):
        model.labels_as_int(pd.DataFrame({"label": bad}), "train")


def test_labels_as_int_refuses_an_empty_split():
    with pytest.raises(ValueError, match="no rows"):
        model.labels_as_int(pd.DataFrame({"label": []}), "test")


def test_label_observability_boundary():
    last_day = dt.date(2011, 12, 9)                                 # orders stop midday on this day
    assert features.label_is_observable(last_day, dt.date(2011, 9, 10), 90)       # window ends exactly on the last day
    assert not features.label_is_observable(last_day, dt.date(2011, 9, 11), 90)   # one day too late


def test_check_labels_observable_names_the_offending_cutoff():
    last_day = dt.date(2011, 12, 9)
    model.check_labels_observable({"train": ["2010-01-01"], "test": ["2011-09-10"]}, last_day)
    with pytest.raises(ValueError, match=r"test cutoff 2011-09-11"):
        model.check_labels_observable({"train": ["2010-01-01"], "test": ["2011-09-11"]}, last_day)


def test_configured_splits_have_fully_observed_labels(real_con):
    model.check_labels_observable(config.SPLITS, db.data_end(real_con))


def test_a_real_snapshot_with_unobserved_labels_cannot_reach_the_model(real_con):
    last_day = db.data_end(real_con)
    late = last_day - dt.timedelta(days=30)                         # its 90-day window runs 60 days past the data
    snap = features.build_snapshot(real_con, late)
    assert snap["label"].isna().all()
    with pytest.raises(ValueError, match="NaN"):
        model.labels_as_int(snap, "train")


def test_snapshot_labels_are_observed_exactly_up_to_the_boundary(real_con):
    last_day = db.data_end(real_con)
    ok = features.build_snapshot(real_con, last_day - dt.timedelta(days=90))
    late = features.build_snapshot(real_con, last_day - dt.timedelta(days=89))
    assert ok["label"].notna().all() and late["label"].isna().all()


# ═════════════════════ 6. segments are validated without hindsight ═════════════════════
def test_rfm_rows_come_back_in_a_fixed_order(real_con):
    """KMeans picks its start centres by row position. Without ORDER BY, DuckDB's grouping order changed
    between connections, so segments moved by a customer or two from run to run."""
    as_of = dt.date(2011, 12, 10)
    a = db.query(real_con, "rfm", as_of=as_of)["customer_id"].to_numpy()
    b = db.query(db.connect(PROCESSED), "rfm", as_of=as_of)["customer_id"].to_numpy()
    assert np.array_equal(a, b) and np.array_equal(a, np.sort(a))


@pytest.mark.slow
@pytest.mark.parametrize("cutoff", [VALID_CUTOFF, TEST_CUTOFF], ids=["valid", "test"])
def test_segment_fit_uses_only_orders_before_the_cutoff(cutoff, real_con, real_frames, tmp_path):
    """Refit on the full database and on one with every later row deleted: the fitted segments must agree."""
    cut_con = future_deleted_con(tmp_path, real_frames, cutoff)
    m_full, k_full, rfm_full, _ = segments.fit_at_cutoff(real_con, cutoff)
    m_cut, k_cut, rfm_cut, _ = segments.fit_at_cutoff(cut_con, cutoff)

    assert np.array_equal(rfm_full["customer_id"].to_numpy(), rfm_cut["customer_id"].to_numpy())
    assert k_full == k_cut and m_full.names == m_cut.names
    np.testing.assert_allclose(m_full.scaler.mean_, m_cut.scaler.mean_, atol=1e-6)
    np.testing.assert_allclose(m_full.scaler.scale_, m_cut.scaler.scale_, atol=1e-6)
    np.testing.assert_allclose(m_full.kmeans.cluster_centers_, m_cut.kmeans.cluster_centers_, atol=1e-4)


def test_segment_validation_refuses_a_window_past_the_data(real_con):
    with pytest.raises(ValueError, match="label window"):
        segments.validate_at_cutoffs(real_con, [("test", "2011-11-20")])


def committed_segments() -> dict:
    path = PROCESSED / "segments.json"
    if not path.exists():
        pytest.skip("segments not built")
    return json.loads(path.read_text())


def test_committed_segment_validation_covers_every_cutoff_and_is_consistent():
    seg = committed_segments()
    val = seg["validation"]
    assert [e["cutoff"] for e in val["by_cutoff"]] == [c for _, c in ALL_CUTOFFS]
    assert val["primary_cutoff"] == TEST_CUTOFF == seg["validation_cutoff"]
    for e in val["by_cutoff"]:
        assert sum(s["customers"] for s in e["segments"]) == e["customers"]
        assert all(0.0 <= s["repurchase_rate"] <= 1.0 for s in e["segments"])
    primary = {s["segment"]: s for s in val["by_cutoff"][-1]["segments"]}
    for s in seg["segments"]:                                        # the table on the dashboard uses these numbers
        if s["segment"] in primary:
            assert s["repurchase_rate_next_90d"] == primary[s["segment"]]["repurchase_rate"]


def test_committed_segment_validation_can_be_reproduced(real_con):
    committed = {s["segment"]: s for s in committed_segments()["validation"]["by_cutoff"][-1]["segments"]}
    fresh = {s["segment"]: s for s in segments.validate_at_cutoffs(real_con, [("test", TEST_CUTOFF)])[0]["segments"]}
    assert set(fresh) == set(committed)
    for name, s in fresh.items():
        assert s["repurchase_rate"] == pytest.approx(committed[name]["repurchase_rate"], abs=0.01), name
        assert abs(s["customers"] - committed[name]["customers"]) <= 10, name


def test_the_extreme_segments_stay_ordered_in_every_period():
    """The check that exposed the weak spot: Champions are best and Hibernating worst at every cutoff.
    (The two middle segments are allowed to swap places; that is reported, not hidden.)"""
    for e in committed_segments()["validation"]["by_cutoff"]:
        ordered = [s["segment"] for s in e["segments"]]
        assert ordered[0] == "Champions" and ordered[-1] == "Hibernating", (e["cutoff"], ordered)


# ═════════════════════ 7. steps that run over all dates: measured, not assumed ═════════════════════
def rows_entering_price_rule(raw: pd.DataFrame) -> pd.DataFrame:
    """The sales rows exactly as they reach cleaning rule 6 (mirrors steps 1-5 of pipeline.clean)."""
    df = raw.rename(columns=pipeline.COLUMN_MAP)[list(pipeline.COLUMN_MAP.values())].copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates()
    df = df[~df["invoice"].str.startswith("C")]
    df = df[df["stock_code"].str.match(pipeline.PRODUCT_CODE)]
    return df[(df["quantity"] > 0) & (df["unit_price"] > 0)]


@pytest.mark.slow
@pytest.mark.skipif(not config.RAW_XLSX.exists(), reason="the raw Excel is git-ignored; run `make data` to download it")
def test_price_median_rule_never_changes_an_identified_customers_rows():
    """pipeline.py computes a per-product median price over ALL dates to delete typo prices. Using only
    earlier prices instead changes a handful of decisions at each cutoff, but they must all be guest
    checkouts (no customer ID), which no customer feature or label ever reads."""
    df = rows_entering_price_rule(pipeline.read_raw(config.RAW_XLSX))
    ratio = config.PRICE_OUTLIER_RATIO
    removed_global = (df["unit_price"] > ratio * df.groupby("stock_code")["unit_price"].transform("median")).to_numpy()

    report = json.loads((PROCESSED / "cleaning_report.json").read_text())
    shipped = next(s["removed"] for s in report["steps"] if s["rule"].startswith("Price >"))
    assert int(removed_global.sum()) == shipped, "this test no longer mirrors pipeline.clean"

    for split, c in ALL_CUTOFFS:
        before = (df["invoice_date"] < pd.Timestamp(c)).to_numpy()
        sub = df[before]
        pre_median = sub.groupby("stock_code")["unit_price"].median()
        removed_pre = (sub["unit_price"] > ratio * sub["stock_code"].map(pre_median)).to_numpy()
        flipped = removed_pre != removed_global[before]
        touched = sub[flipped]["customer_id"].dropna()
        assert len(touched) == 0, f"{split} {c}: {touched.nunique()} identified customers' rows depend on later prices"


@pytest.mark.slow
def test_the_features_do_not_depend_on_the_category_file(real_frames, tmp_path):
    """products.py clusters ALL products, including ones first sold after the training cutoffs, so nothing the model
    reads may depend on that file. Build the real database twice, with and without product_categories.parquet: the
    snapshot at every cutoff must be exactly the same, and none of it may be a category column.

    threads=1 is required: with several threads DuckDB adds the money values in a varying order, so avg_order_value
    can differ by a penny between two builds of the very same data."""
    sales, returns, products, cats = (real_frames[k] for k in ("sales", "returns", "products", "product_categories"))
    write_frames(tmp_path / "with_categories", sales, returns, products, cats)
    write_frames(tmp_path / "without_categories", sales, returns, products)
    with_cats = db.connect(tmp_path / "with_categories", threads=1)
    without = db.connect(tmp_path / "without_categories", threads=1)
    # the test is only meaningful if the categories really are visible in one database and absent from the other
    assert with_cats.cursor().execute("SELECT COUNT(category) FROM sales").fetchone()[0] > 0
    assert without.cursor().execute("SELECT COUNT(category) FROM sales").fetchone()[0] == 0

    assert features.feature_columns(with_cats) == features.BASE_FEATURES
    assert not [c for c in features.feature_columns(with_cats) if c.startswith("cat_")]
    for split, cutoff in ALL_CUTOFFS:
        a, b = features.build_snapshot(with_cats, cutoff), features.build_snapshot(without, cutoff)
        assert not [c for c in a.columns if c.startswith("cat_")], f"{split} {cutoff}: a category column is in the snapshot"
        pd.testing.assert_frame_equal(a, b)


def test_metrics_json_lists_the_features_the_model_was_trained_on(real_con):
    path = PROCESSED / "metrics.json"
    if not path.exists():
        pytest.skip("model not built")
    report = json.loads(path.read_text())
    assert report["features"] == features.feature_columns(real_con)
    assert len(report["features"]) == 13 and not [c for c in report["features"] if c.startswith("cat_")]
    assert sorted(r["feature"] for r in report["importance"]) == sorted(report["features"])   # the importances cover exactly those
