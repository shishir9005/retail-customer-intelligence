"""The most important ML test: features for a cutoff may only use data from BEFORE it."""
import datetime as dt

import pandas as pd
import pytest

from rci import config, db, features
from rci.model import check_labels_observable, check_split_chain

from conftest import CUTOFF, write_frames


def snapshot(con, cutoff=CUTOFF):
    return features.build_snapshot(con, cutoff, horizon=90).set_index("customer_id")


def test_snapshot_contains_only_customers_seen_before_the_cutoff(con):
    snap = snapshot(con)
    assert sorted(snap.index) == [1, 2, 3]            # customer 4 first orders in July; the guest is never a customer


def test_hand_checked_feature_values(con):
    snap = snapshot(con)
    one = snap.loc[1]
    assert one["frequency"] == 2                      # invoices 1001 and 1002; invoice 1003 is in the future
    assert one["monetary"] == 28.0                    # 16 + 12
    assert one["recency_days"] == (CUTOFF - dt.date(2011, 2, 10)).days
    assert one["n_products"] == 2
    assert one["n_returns"] == 1                      # the Mar 1 return is before the cutoff
    assert one["is_uk"] == 1 and snap.loc[2]["is_uk"] == 0
    assert one["avg_gap_days"] == 31                  # Jan 10 -> Feb 10


def test_labels_look_only_at_the_next_90_days(con):
    snap = snapshot(con)
    assert snap["label"].to_dict() == {1: 1, 2: 0, 3: 1}


def test_label_is_unknown_when_the_window_runs_past_the_data(con):
    snap = features.build_snapshot(con, dt.date(2011, 6, 1), horizon=90)   # window ends Aug 30, data ends Jul 15
    assert snap["label"].isna().all()


def test_features_do_not_change_when_the_future_is_rewritten(tmp_path, frames):
    """Rewrite EVERYTHING on or after the cutoff (bigger, different, extra orders). The features of the
    customers in the snapshot must be identical; only the label may differ."""
    sales, returns, products = frames
    base_dir, alt_dir = tmp_path / "base", tmp_path / "alt"
    write_frames(base_dir, sales, returns, products)

    future = sales["invoice_date"] >= pd.Timestamp(CUTOFF)
    altered = sales.copy()
    altered.loc[future, "quantity"] = (altered.loc[future, "quantity"] * 100).astype("int32")
    altered.loc[future, "revenue"] = altered.loc[future, "revenue"] * 100
    extra = altered[future].assign(invoice=lambda d: d["invoice"] + "X", customer_id=pd.array([2] * future.sum(), dtype="Int64"))
    altered = pd.concat([altered, extra], ignore_index=True)
    alt_returns = returns.copy()
    late_return = alt_returns.assign(invoice_date=pd.Timestamp("2011-05-01"), invoice="C9999")
    write_frames(alt_dir, altered, pd.concat([alt_returns, late_return], ignore_index=True), products)

    a = snapshot(db.connect(base_dir)).drop(columns=["label"])
    b = snapshot(db.connect(alt_dir)).drop(columns=["label"])
    pd.testing.assert_frame_equal(a, b)


def test_label_does_change_when_the_future_changes(tmp_path, frames):
    """Control for the test above: removing customer 1's May order must flip their label to 0."""
    sales, returns, products = frames
    pruned = sales[sales["invoice"] != "1003"]
    write_frames(tmp_path, pruned, returns, products)
    snap = snapshot(db.connect(tmp_path))
    assert snap.loc[1, "label"] == 0


def test_log_transform_does_not_touch_flags():
    X = pd.DataFrame({c: [0.0, 9.0] for c in features.BASE_FEATURES})
    out = features.log_transform(X)
    assert out["is_uk"].tolist() == [0.0, 9.0]
    assert out["monetary"].iloc[1] == pytest.approx(2.302585, abs=1e-5)


# ── the real train / valid / test layout ─────────────────────────────────────
def test_configured_splits_do_not_overlap():
    check_split_chain(config.SPLITS)


def test_overlapping_splits_are_caught():
    bad = {"train": ["2011-01-01", "2011-06-01"], "valid": ["2011-07-01"], "test": ["2011-12-01"]}
    with pytest.raises(ValueError, match="after valid starts"):
        check_split_chain(bad)


def test_test_labels_are_fully_observable(real_con):
    last = db.data_end(real_con)
    for cutoff in config.SPLITS["test"]:
        assert features.label_is_observable(last, pd.Timestamp(cutoff).date(), config.HORIZON_DAYS)


# ── the three-quarter backtest folds ─────────────────────────────────────────
FOLDS = list(config.BACKTEST_FOLDS)


def test_the_backtest_has_three_folds_and_the_last_is_the_main_split():
    assert FOLDS == ["2011-03", "2011-06", "2011-09"]
    assert [config.BACKTEST_FOLDS[f]["test"] for f in FOLDS] == [["2011-03-09"], ["2011-06-09"], ["2011-09-09"]]
    assert config.BACKTEST_FOLDS["2011-09"] == config.SPLITS


@pytest.mark.parametrize("fold", FOLDS)
def test_every_backtest_fold_passes_the_split_chain_check(fold):
    check_split_chain(config.BACKTEST_FOLDS[fold])


@pytest.mark.parametrize("fold", FOLDS)
def test_every_backtest_fold_has_fully_observed_labels(fold, real_con):
    check_labels_observable(config.BACKTEST_FOLDS[fold], db.data_end(real_con))


def test_each_fold_trains_on_everything_the_previous_fold_saw():
    """An expanding window: a later fold's training cutoffs are the previous fold's training plus its validation cutoff."""
    for earlier, later in zip(FOLDS, FOLDS[1:]):
        a, b = config.BACKTEST_FOLDS[earlier], config.BACKTEST_FOLDS[later]
        assert b["train"] == a["train"] + a["valid"]
