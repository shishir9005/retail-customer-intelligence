"""Refund lines of sales that cleaning removed (rules 6 and 7), and only those.

Cleaning deletes sale lines with typo prices or absurd quantities. Their refunds must go too, or the
refund would have nothing to offset. But the matching must be strict: a refund is dropped only when it
EXACTLY reverses a removed sale line. Partial refunds and credits stay.

Part 1 tests `refunds_of_removed_sales` on small hand-built cases. Part 2 checks the shipped data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rci import db, pipeline, quality

from conftest import PROCESSED, write_frames

COLUMNS = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id", "country"]
UK, FR = "United Kingdom", "France"


def lines(*rows, index=None) -> pd.DataFrame:
    """Sale or refund lines: (invoice, stock_code, quantity, date, unit_price, customer_id, country)."""
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["quantity"] = df["quantity"].astype("int64")
    df["customer_id"] = pd.array([r[5] for r in rows], dtype="Int64")
    if index is not None:
        df.index = index
    return df


def dropped(returns: pd.DataFrame, removed: pd.DataFrame) -> list:
    return list(pipeline.refunds_of_removed_sales(returns, removed))


def partial(returns: pd.DataFrame, removed: pd.DataFrame) -> list:
    return list(pipeline.partial_refunds_of_removed_sales(returns, removed))


# the removed sale used by most cases: customer 12346 bought 74,215 units at £1.04 (a rule-7 removal)
HUGE_SALE = ("S1", "23166", 74215, "2011-01-10 10:00", 1.04, 12346, UK)


# ═════════════════════ Part 1: the matching rule ═════════════════════
def test_1_exact_refund_of_a_huge_quantity_sale_is_dropped():
    removed = lines(HUGE_SALE)
    returns = lines(("C1", "23166", -74215, "2011-01-18 09:00", 1.04, 12346, UK))
    assert dropped(returns, removed) == [0]


def test_2_exact_refund_of_a_price_typo_sale_is_dropped():
    removed = lines(("S2", "84016", 1, "2010-12-01 10:00", 25.52, None, UK))          # a guest, 1 x £25.52
    returns = lines(("C2", "84016", -1, "2010-12-20 10:00", 25.52, None, UK))
    assert dropped(returns, removed) == [0]


def test_3_same_customer_and_product_but_different_price_is_kept():
    """The 15098 pattern: the refund reverses a different (kept) sale at £4.95, not the £649.50 typo."""
    removed = lines(("S3", "22502", 60, "2011-02-01 10:00", 649.50, 15098, UK))
    returns = lines(("C3", "22502", -60, "2011-02-10 10:00", 4.95, 15098, UK))
    assert dropped(returns, removed) == []


def test_4_a_lump_credit_with_no_matching_sale_is_kept_and_is_not_called_partial():
    """The 16422 pattern: -1 x £57.60 credits; no sale at £57.60 exists."""
    removed = lines(("S4", "17003", 720, "2011-03-01 10:00", 0.08, 16422, UK))
    returns = lines(("C4", "17003", -1, "2011-03-05 10:00", 57.60, 16422, UK))
    assert dropped(returns, removed) == []
    assert partial(returns, removed) == []                                             # different price, so not 'partial'


def test_5_a_partial_refund_is_kept_and_reported():
    removed = lines(("S5", "23166", 1000, "2011-01-10 10:00", 1.04, 12346, UK))
    returns = lines(("C5", "23166", -400, "2011-01-20 10:00", 1.04, 12346, UK))
    assert dropped(returns, removed) == []
    assert partial(returns, removed) == [0]


def test_6_two_identical_refunds_for_one_removed_sale_drop_exactly_one():
    removed = lines(HUGE_SALE)
    returns = lines(("C6a", "23166", -74215, "2011-01-18 09:00", 1.04, 12346, UK),
                    ("C6b", "23166", -74215, "2011-01-25 09:00", 1.04, 12346, UK))
    assert dropped(returns, removed) == [0]                                            # the earliest one
    assert partial(returns.drop(index=[0]), removed) == []                             # the survivor has the same quantity: not partial


def test_6b_earliest_refund_is_chosen_regardless_of_row_order():
    removed = lines(HUGE_SALE)
    returns = lines(("late", "23166", -74215, "2011-02-20 09:00", 1.04, 12346, UK),
                    ("early", "23166", -74215, "2011-01-12 09:00", 1.04, 12346, UK))
    assert dropped(returns, removed) == [1]


def test_6c_two_identical_removed_sales_and_one_refund_drop_one_refund():
    removed = lines(HUGE_SALE, ("S1b", "23166", 74215, "2011-01-11 10:00", 1.04, 12346, UK))
    returns = lines(("C6", "23166", -74215, "2011-01-18 09:00", 1.04, 12346, UK))
    assert dropped(returns, removed) == [0]


def test_6d_two_sales_and_two_refunds_pair_up_one_to_one():
    removed = lines(HUGE_SALE, ("S1b", "23166", 74215, "2011-01-11 10:00", 1.04, 12346, UK))
    returns = lines(("C6a", "23166", -74215, "2011-01-18 09:00", 1.04, 12346, UK),
                    ("C6b", "23166", -74215, "2011-01-25 09:00", 1.04, 12346, UK),
                    ("C6c", "23166", -74215, "2011-02-25 09:00", 1.04, 12346, UK))
    assert dropped(returns, removed) == [0, 1]                                         # two pairs; the third refund stays


def test_7_a_refund_dated_before_the_sale_is_kept():
    removed = lines(("S7", "23166", 5, "2011-02-01 10:00", 1.04, 12346, UK))
    returns = lines(("C7", "23166", -5, "2011-01-15 10:00", 1.04, 12346, UK))
    assert dropped(returns, removed) == []


def test_7b_a_refund_at_the_same_moment_as_the_sale_counts_as_on_or_after():
    removed = lines(("S7", "23166", 5, "2011-02-01 10:00", 1.04, 12346, UK))
    returns = lines(("C7", "23166", -5, "2011-02-01 10:00", 1.04, 12346, UK))
    assert dropped(returns, removed) == [0]


def test_8_in_a_multi_line_cancellation_only_the_matching_line_is_dropped():
    removed = lines(("S8", "84016", 1, "2010-12-01 10:00", 25.52, None, UK))
    returns = lines(("C8", "22423", -2, "2010-12-20 10:00", 9.95, None, UK),          # same invoice, an ordinary refund
                    ("C8", "84016", -1, "2010-12-20 10:00", 25.52, None, UK))          # same invoice, the matching line
    assert dropped(returns, removed) == [1]


def test_9_a_guest_never_matches_an_identified_customer_in_either_direction():
    guest_sale = lines(("S9", "23166", 10, "2011-01-10 10:00", 1.04, None, UK))
    customer_sale = lines(("S9", "23166", 10, "2011-01-10 10:00", 1.04, 12346, UK))
    customer_refund = lines(("C9", "23166", -10, "2011-01-20 10:00", 1.04, 12346, UK))
    guest_refund = lines(("C9", "23166", -10, "2011-01-20 10:00", 1.04, None, UK))
    assert dropped(customer_refund, guest_sale) == []
    assert dropped(guest_refund, customer_sale) == []
    assert dropped(guest_refund, guest_sale) == [0]                                    # two guests do count as equal


def test_10_float_noise_in_the_price_still_matches_but_a_genuine_price_difference_does_not():
    removed = lines(HUGE_SALE)
    noisy = lines(("C10", "23166", -74215, "2011-01-18 09:00", 1.0400000000001, 12346, UK))
    noisy_below = lines(("C10", "23166", -74215, "2011-01-18 09:00", 1.0399999999999, 12346, UK))
    one_penny_more = lines(("C10", "23166", -74215, "2011-01-18 09:00", 1.05, 12346, UK))
    assert dropped(noisy, removed) == [0]
    assert dropped(noisy_below, removed) == [0]
    assert dropped(one_penny_more, removed) == []


@pytest.mark.parametrize("field,value", [("stock_code", "23167"), ("customer_id", 12347), ("country", FR), ("quantity", -74214)])
def test_any_single_difference_prevents_a_match(field, value):
    removed = lines(HUGE_SALE)
    refund = dict(zip(COLUMNS, ("C11", "23166", -74215, "2011-01-18 09:00", 1.04, 12346, UK)))
    refund[field] = value
    assert dropped(lines(tuple(refund.values())), removed) == []


def test_labels_not_positions_are_returned_so_clean_can_drop_the_right_rows():
    removed = lines(HUGE_SALE, index=[7])
    returns = lines(("C1", "23166", -5, "2011-01-18 09:00", 1.04, 12346, UK),
                    ("C2", "23166", -74215, "2011-01-18 09:00", 1.04, 12346, UK), index=[100, 205])
    assert dropped(returns, removed) == [205]


def test_empty_inputs_return_an_empty_index():
    empty = pd.DataFrame(columns=COLUMNS)
    some = lines(HUGE_SALE)
    assert len(pipeline.refunds_of_removed_sales(some, empty)) == 0
    assert len(pipeline.refunds_of_removed_sales(empty, some)) == 0
    assert len(pipeline.partial_refunds_of_removed_sales(some, empty)) == 0


def test_the_report_adds_money_in_integer_pence_and_rounds_once():
    three = lines(("C541433", "23166", -74215, "2011-01-18", 1.04, 12346, UK),
                  ("C581484", "23843", -80995, "2011-12-05", 2.08, 16446, UK),
                  ("C508859", "84016", -1, "2010-12-20", 25.52, None, UK))
    summary = pipeline._refund_summary(three, list_invoices=True)
    assert summary == {"lines": 3, "refund_pence": 24_567_872, "refund": 245678.72,
                       "invoices": ["C508859", "C541433", "C581484"]}
    assert pipeline._refund_summary(three.iloc[0:0]) == {"lines": 0, "refund_pence": 0, "refund": 0.0}


# ═════════════════════ the new quality check ═════════════════════
def test_there_are_eleven_quality_checks():
    assert len(quality.CHECKS) == 11


def test_an_oversized_refund_is_caught_by_the_new_quality_check(tmp_path, frames):
    sales, returns, products = frames
    assert all(r["passed"] for r in quality.run_checks(_connect(tmp_path / "ok", sales, returns, products)))

    huge = returns.assign(invoice="C9999", quantity=np.int32(-60_000), revenue=-180000.0)
    bad_returns = pd.concat([returns, huge], ignore_index=True)
    results = quality.run_checks(_connect(tmp_path / "bad", sales, bad_returns, products))
    failed = [r for r in results if not r["passed"]]
    assert [r["check"] for r in failed] == ["No return line above the extreme-quantity limit"]
    assert failed[0]["offending_rows"] == 1


def _connect(directory, sales, returns, products):
    write_frames(directory, sales, returns, products)
    return db.connect(directory)


# ═════════════════════ Part 2: the shipped data ═════════════════════
def pence(con, sql: str) -> int:
    return int(con.cursor().execute(sql).fetchone()[0])


def test_real_the_three_refunds_of_removed_sales_are_absent(real_con):
    cur = real_con.cursor()
    assert cur.execute("SELECT COUNT(*) FROM returns WHERE invoice IN ('C541433', 'C581484')").fetchone()[0] == 0
    assert cur.execute("SELECT COUNT(*) FROM returns WHERE invoice = 'C508859' AND stock_code = '84016'").fetchone()[0] == 0


def test_real_legitimate_refunds_are_still_there(real_con):
    rows = real_con.cursor().execute(
        "SELECT invoice, stock_code, customer_id, quantity, unit_price FROM returns "
        "WHERE invoice IN ('C556448', 'C557624', 'C558726', 'C559739') ORDER BY invoice").fetchall()
    assert rows == [("C556448", "22502", 15098, -60, 4.95),
                    ("C557624", "17003", 16422, -1, 57.6),
                    ("C558726", "17003", 16422, -1, 57.6),
                    ("C559739", "17003", 16422, -1, 57.6)]


def test_real_refund_totals_are_exact_in_integer_pence(real_con):
    refunds = pence(real_con, "SELECT SUM(ROUND(revenue * 100)::BIGINT) FROM returns")
    sales = pence(real_con, "SELECT SUM(ROUND(revenue * 100)::BIGINT) FROM sales")
    assert refunds == -47_074_725                                   # -£470,747.25
    assert sales == 1_935_375_851                                   # £19,353,758.51, unchanged by this task
    assert sales + refunds == 1_888_301_126                         # net revenue £18,883,011.26
    assert pence(real_con, "SELECT COUNT(*) FROM returns") == 17_911
    assert pence(real_con, "SELECT COUNT(*) FROM sales") == 1_003_243
    assert pence(real_con, "SELECT -SUM(quantity) FROM returns") == 312_528
    assert round(100 * -refunds / sales, 4) == 2.4323


def test_real_net_revenue_by_year(real_con):
    by_year = {y: pence(real_con, f"""SELECT SUM(p) FROM (
                  SELECT ROUND(revenue * 100)::BIGINT AS p, invoice_date FROM sales
                  UNION ALL SELECT ROUND(revenue * 100)::BIGINT, invoice_date FROM returns) WHERE year(invoice_date) = {y}""")
               for y in (2010, 2011)}
    assert by_year == {2010: 913_123_639, 2011: 897_352_602}        # £9,131,236.39 and £8,973,526.02


def test_real_return_rate_kpi_is_2_4(real_con):
    assert db.query(real_con, "return_rate")["return_pct"][0] == 2.4


def test_real_no_refund_is_beyond_the_quantity_limit_and_all_eleven_checks_pass(real_con):
    results = quality.run_checks(real_con)
    assert len(results) == 11 and all(r["passed"] for r in results)


def test_real_the_cleaning_report_records_exactly_these_three_refunds():
    report = json.loads((PROCESSED / "cleaning_report.json").read_text())
    notes = report["summary"]
    assert notes["refunds_of_removed_sales"] == {"lines": 3, "refund_pence": 24_567_872, "refund": 245678.72,
                                                 "invoices": ["C508859", "C541433", "C581484"]}
    assert notes["partial_refunds_of_removed_sales"] == {"lines": 0, "refund_pence": 0, "refund": 0.0}
    assert notes["returns_rows"] == 17_911
    assert len(report["steps"]) == 7                                # the refund drop is not a sales-stream step
    assert len(report["checks"]) == 11 and all(c["passed"] for c in report["checks"])
