"""Each cleaning rule is tested on a row built to trigger exactly that rule."""
import numpy as np
import pandas as pd

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
