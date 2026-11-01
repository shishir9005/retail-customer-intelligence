"""Each cleaning rule is tested on a row built to trigger exactly that rule."""
import numpy as np
import pandas as pd
import pytest

from rci import pipeline


def raw_frame():
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    d = "2011-01-05 10:00"
    rows = [
        # normal sales of product 22423 (median price 10) and 85123A
        ("1001", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2, d, 3.0, 1.0, "United Kingdom"),
        ("1001", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2, d, 3.0, 1.0, "United Kingdom"),   # exact duplicate
        ("1002", "22423", "REGENCY CAKESTAND 3 TIER", 1, d, 10.0, 2.0, "France"),
        ("1003", "22423", "REGENCY CAKESTAND 3 TIER", 1, d, 10.0, 3.0, "France"),
        ("1004", "22423", "REGENCY CAKESTAND 3 TIER", 1, d, 10.0, 3.0, "France"),
        ("1005", "22423", "Regency Cakestand 3 Tier", 1, d, 10.0, 3.0, "France"),                       # other spelling
        ("C1006", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", -1, d, 3.0, 1.0, "United Kingdom"),    # cancellation
        ("1007", "POST", "POSTAGE", 1, d, 15.0, 1.0, "United Kingdom"),                                  # non-product
        ("1008", "BANK CHARGES", "Bank Charges", 1, d, 15.0, None, "United Kingdom"),                    # non-product
        ("1009", "22423", "REGENCY CAKESTAND 3 TIER", -5, d, 10.0, 2.0, "France"),                       # quantity <= 0
        ("1010", "22423", "REGENCY CAKESTAND 3 TIER", 1, d, 0.0, 2.0, "France"),                         # price 0
        ("1011", "22423", "REGENCY CAKESTAND 3 TIER", 1, d, 500.0, 2.0, "France"),                       # 50x median
        ("1012", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 60000, d, 3.0, 4.0, "United Kingdom"),  # huge quantity
        ("1013", "85123a ", "WHITE HANGING HEART T-LIGHT HOLDER", 1, d, 3.0, None, "United Kingdom"),     # guest, messy code
    ]
    return pd.DataFrame(rows, columns=cols).assign(InvoiceDate=lambda x: pd.to_datetime(x["InvoiceDate"]))


def steps_by_rule(res):
    return {s["rule"].split(" >")[0].split(" <=")[0]: s for s in res.steps}


def test_every_rule_removes_the_intended_rows():
    res = pipeline.clean(raw_frame())
    removed = [s["removed"] for s in res.steps]
    # duplicates, cancellations, non-product (POST + BANK CHARGES), qty<=0, price<=0, price outlier, huge quantity
    assert removed == [1, 1, 2, 1, 1, 1, 1]


def test_final_tables():
    res = pipeline.clean(raw_frame())
    assert len(res.sales) == 6                   # 1001, 1002, 1003, 1004, 1005 and the guest 1013
    assert len(res.returns) == 1
    assert (res.sales["quantity"] > 0).all() and (res.sales["unit_price"] > 0).all()
    assert (res.returns["quantity"] < 0).all()


def test_messy_stock_code_is_normalised_and_guest_rows_are_kept():
    res = pipeline.clean(raw_frame())
    guest = res.sales[res.sales["customer_id"].isna()]
    assert guest["stock_code"].tolist() == ["85123A"]          # '85123a ' -> '85123A'
    assert res.notes["guest_line_share"] == 1 / 6


def test_revenue_is_quantity_times_price():
    res = pipeline.clean(raw_frame())
    assert np.allclose(res.sales["revenue"], res.sales["quantity"] * res.sales["unit_price"])


def test_product_description_is_the_most_common_spelling():
    res = pipeline.clean(raw_frame())
    desc = res.products.set_index("stock_code")["description"]
    assert desc["22423"] == "REGENCY CAKESTAND 3 TIER"        # 3 votes vs 1 for the other spelling


def test_steps_account_for_every_row():
    raw = raw_frame()
    res = pipeline.clean(raw)
    # rows the sales stream lost = sum of removed counts; what is left is the sales table
    assert len(raw) - sum(s["removed"] for s in res.steps) == len(res.sales)


# ── refunds of removed sales (see also tests/test_refunds.py) ────────────────
def raw_with_refunds():
    """raw_frame() plus five cancellation lines dated the day after the sales:
      C1014  an exact refund of the removed 60,000-unit sale 1012            -> dropped
      C1015  TWO lines. Line 1 exactly refunds the removed 50x-median-price sale 1011 -> dropped
             Line 2 is an ordinary credit (85123A, customer 2)               -> stays: lines are matched one by one, never whole invoices
      C1016  a refund of a KEPT sale (22423, customer 3, £10)                 -> stays
      C1017  59,000 of the removed 60,000 units: a partial refund             -> stays, and is listed
    """
    d = "2011-01-06 09:00"
    base = raw_frame()
    extra = pd.DataFrame([
        ("C1014", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", -60000, d, 3.0, 4.0, "United Kingdom"),
        ("C1015", "22423", "REGENCY CAKESTAND 3 TIER", -1, d, 500.0, 2.0, "France"),
        ("C1015", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", -1, d, 3.0, 2.0, "France"),
        ("C1016", "22423", "REGENCY CAKESTAND 3 TIER", -1, d, 10.0, 3.0, "France"),
        ("C1017", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", -59000, d, 3.0, 4.0, "United Kingdom"),
    ], columns=base.columns).assign(InvoiceDate=lambda x: pd.to_datetime(x["InvoiceDate"]))
    return pd.concat([base, extra], ignore_index=True)


def test_refunds_of_removed_sales_are_dropped_and_everything_else_is_kept():
    res = pipeline.clean(raw_with_refunds())
    assert sorted(res.returns["invoice"]) == ["C1006", "C1015", "C1016", "C1017"]
    assert res.notes["returns_rows"] == 4
    assert res.notes["refunds_of_removed_sales"] == {
        "lines": 2, "refund_pence": 18_050_000, "refund": 180500.0, "invoices": ["C1014", "C1015"]}   # 60000 x 300p + 1 x 50000p


def test_only_the_matching_line_of_a_multi_line_cancellation_is_dropped():
    res = pipeline.clean(raw_with_refunds())
    survivor = res.returns[res.returns["invoice"] == "C1015"]
    assert survivor["stock_code"].tolist() == ["85123A"]          # the 22423 line went, its sibling stayed


def test_a_partial_refund_is_kept_and_listed_in_the_report():
    res = pipeline.clean(raw_with_refunds())
    assert res.notes["partial_refunds_of_removed_sales"] == {"lines": 1, "refund_pence": 17_700_000, "refund": 177000.0}   # 59000 x 300p


def test_refund_handling_never_changes_the_sales_stream_or_the_steps_table():
    plain, with_refunds = pipeline.clean(raw_frame()), pipeline.clean(raw_with_refunds())
    pd.testing.assert_frame_equal(plain.sales, with_refunds.sales)
    assert len(with_refunds.steps) == 7                          # the refund drop is reported in `notes`, not as a step
    assert [s["removed"] for s in with_refunds.steps] == [1, 6, 2, 1, 1, 1, 1]   # only the cancellation count grows (1 -> 6)


def test_plain_frame_reports_no_refunds_of_removed_sales():
    res = pipeline.clean(raw_frame())
    assert res.notes["refunds_of_removed_sales"] == {"lines": 0, "refund_pence": 0, "refund": 0.0, "invoices": []}
    assert res.notes["partial_refunds_of_removed_sales"]["lines"] == 0


def test_rule_one_explanation_describes_the_overlap_without_hard_coded_counts():
    why = pipeline.clean(raw_frame()).steps[0]["why"]
    assert "overlapping" in why and "inside one invoice" in why
    assert not any(ch.isdigit() for ch in why.replace("1-9 Dec 2010", ""))


# ── a GUEST refund, end to end through clean() ───────────────────────────────
def raw_with_guest_refunds():
    """raw_frame() plus a removed GUEST sale (a £400 price typo, rule 6) and three refunds:
      C1021  the guest's exact refund of that sale                                -> dropped
      C1022  a guest refund at £500, the price of CUSTOMER 2's removed sale 1011   -> stays (a guest never takes a customer's sale)
      C1023  customer 2's refund at £400, the price of the GUEST's removed sale    -> stays (a customer never takes a guest's sale)
    """
    base = raw_frame()
    name = "REGENCY CAKESTAND 3 TIER"
    sold, refunded = "2011-01-05 10:00", "2011-01-06 09:00"
    extra = pd.DataFrame([
        ("1020", "22423", name, 1, sold, 400.0, None, "France"),
        ("C1021", "22423", name, -1, refunded, 400.0, None, "France"),
        ("C1022", "22423", name, -1, refunded, 500.0, None, "France"),
        ("C1023", "22423", name, -1, refunded, 400.0, 2.0, "France"),
    ], columns=base.columns).assign(InvoiceDate=lambda x: pd.to_datetime(x["InvoiceDate"]))
    return pd.concat([base, extra], ignore_index=True)


def test_a_guest_refund_exactly_reversing_a_removed_guest_sale_is_dropped_by_clean():
    res = pipeline.clean(raw_with_guest_refunds())
    assert "C1021" not in set(res.returns["invoice"])
    assert res.notes["refunds_of_removed_sales"] == {"lines": 1, "refund_pence": 40_000, "refund": 400.0, "invoices": ["C1021"]}


def test_a_guest_refund_never_takes_an_identified_customers_removed_sale_through_clean():
    res = pipeline.clean(raw_with_guest_refunds())
    kept = res.returns[res.returns["invoice"] == "C1022"]
    assert len(kept) == 1 and kept["customer_id"].isna().all()


def test_an_identified_refund_never_takes_a_guests_removed_sale_through_clean():
    res = pipeline.clean(raw_with_guest_refunds())
    kept = res.returns[res.returns["invoice"] == "C1023"]
    assert len(kept) == 1 and kept["customer_id"].tolist() == [2]


def test_guest_refund_handling_leaves_the_sales_stream_and_the_other_categories_alone():
    res = pipeline.clean(raw_with_guest_refunds())
    assert sorted(res.returns["invoice"]) == ["C1006", "C1022", "C1023"]
    pd.testing.assert_frame_equal(res.sales, pipeline.clean(raw_frame()).sales)       # the guest sale was removed by rule 6
    assert res.notes["partial_refunds_of_removed_sales"]["lines"] == 0
    assert res.notes["surplus_refunds_of_removed_sales"]["lines"] == 0


# ── the leftover exact refund, end to end ────────────────────────────────────
def raw_with_a_surplus_refund():
    """raw_frame() plus TWO identical exact refunds of the one removed 60,000-unit sale 1012."""
    base = raw_frame()
    name = "WHITE HANGING HEART T-LIGHT HOLDER"
    extra = pd.DataFrame([
        ("C1030", "85123A", name, -60000, "2011-01-06 09:00", 3.0, 4.0, "United Kingdom"),
        ("C1031", "85123A", name, -60000, "2011-01-07 09:00", 3.0, 4.0, "United Kingdom"),
    ], columns=base.columns).assign(InvoiceDate=lambda x: pd.to_datetime(x["InvoiceDate"]))
    return pd.concat([base, extra], ignore_index=True)


def test_the_second_identical_refund_is_kept_and_reported_as_surplus():
    res = pipeline.clean(raw_with_a_surplus_refund())
    assert sorted(res.returns["invoice"]) == ["C1006", "C1031"]       # C1030 was paired with the sale; C1031 is left over
    assert res.notes["refunds_of_removed_sales"]["invoices"] == ["C1030"]
    assert res.notes["surplus_refunds_of_removed_sales"] == {"lines": 1, "refund_pence": 18_000_000, "refund": 180000.0}
    assert res.notes["partial_refunds_of_removed_sales"]["lines"] == 0


def test_a_sub_penny_price_on_a_removed_sale_stops_cleaning_with_a_clear_error():
    base = raw_frame()
    sub_penny = pd.DataFrame([("1040", "22423", "REGENCY CAKESTAND 3 TIER", 1, "2011-01-05 10:00", 500.004, 2.0, "France")],
                             columns=base.columns).assign(InvoiceDate=lambda x: pd.to_datetime(x["InvoiceDate"]))
    with pytest.raises(ValueError, match="not a whole number of pence"):
        pipeline.clean(pd.concat([base, sub_penny], ignore_index=True))      # removed by rule 6 (>20x the median), so it is checked
