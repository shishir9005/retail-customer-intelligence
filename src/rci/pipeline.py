"""Download the UCI Online Retail II workbook and turn it into clean Parquet files.

Every cleaning rule is one small named step that records how many rows it removed. The numbers go
to `cleaning_report.json` (and the README / dashboard), so each rule can be explained and defended.

Run:  python -m rci.pipeline
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rci import config

COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# A real product code is 5 digits plus up to 3 letters (85123A, 15056BL) or a DCGS gift-set code.
# Everything else (POST, DOT, M, D, BANK CHARGES, AMAZONFEE, gift_0001_20, ...) is a fee or an
# accounting line, not something a customer bought.
PRODUCT_CODE = r"^(?:\d{5}[A-Z]{0,3}|DCGS[A-Z0-9]*)$"


@dataclass
class CleanResult:
    sales: pd.DataFrame
    returns: pd.DataFrame
    products: pd.DataFrame
    steps: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


def download_raw(dest: Path = config.RAW_DIR) -> Path:
    """Fetch the UCI zip (CC BY 4.0, no login) and unpack the Excel file. Skips if already there."""
    import requests

    xlsx = dest / config.RAW_XLSX.name
    if xlsx.exists():
        return xlsx
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "online_retail_ii.zip"
    print(f"Downloading {config.UCI_ZIP_URL} ...")
    with requests.get(config.UCI_ZIP_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return xlsx


def read_raw(xlsx: Path) -> pd.DataFrame:
    """Both sheets (2009-10 and 2010-11) stacked into one frame. calamine is a fast Excel reader."""
    sheets = pd.read_excel(xlsx, sheet_name=None, engine="calamine")
    return pd.concat(sheets.values(), ignore_index=True)


GUEST = "<guest>"  # stands in for a missing customer ID when matching; it can never equal a real ID


def _refund_match_frame(lines: pd.DataFrame, quantity_sign: int) -> pd.DataFrame:
    """The fields a sale and its refund must agree on, with the price as integer pence.

    Cleaned prices are whole pence, so rounding `unit_price * 100` removes float noise (about 1e-11)
    and lets prices be compared exactly. `quantity_sign=-1` flips a refund's negative quantity so it
    can be compared directly with the sale it reverses.
    """
    return pd.DataFrame({
        "stock_code": lines["stock_code"].astype(str),
        "customer": lines["customer_id"].astype("Int64").astype("string").fillna(GUEST),
        "country": lines["country"].astype(str),
        "pence": np.rint(lines["unit_price"].astype("float64") * 100).astype("int64"),
        "quantity": lines["quantity"].astype("int64") * quantity_sign,
        "date": lines["invoice_date"],
    }, index=lines.index)


def refunds_of_removed_sales(returns: pd.DataFrame, removed: pd.DataFrame) -> pd.Index:
    """Index labels of the refund LINES that exactly reverse a sale line that cleaning removed.

    A refund line matches a removed sale line when they have the same product, customer (two guests
    count as equal, a guest never equals a real customer), country and price in whole pence, the
    refund quantity is exactly minus the sale quantity, and the refund is dated on or after the sale.
    Matching is one-to-one: each sale takes the earliest still-unused refund it is eligible for, going
    through the sales oldest first. Lines are matched individually, never whole invoices.

    Partial refunds (different quantity) and credits with no exact counterpart never match, so they
    are kept.
    """
    if returns.empty or removed.empty:
        return returns.index[:0]
    refunds = _refund_match_frame(returns, -1).sort_values("date", kind="stable")
    sales = _refund_match_frame(removed, +1).sort_values("date", kind="stable")

    pool: dict[tuple, list] = {}  # match key -> [(refund date, refund label)], oldest first
    for r in refunds.itertuples():
        pool.setdefault((r.stock_code, r.customer, r.country, r.pence, r.quantity), []).append((r.date, r.Index))

    matched = []
    for s in sales.itertuples():
        candidates = pool.get((s.stock_code, s.customer, s.country, s.pence, s.quantity), [])
        for position, (refund_date, label) in enumerate(candidates):
            if refund_date >= s.date:
                matched.append(label)
                del candidates[position]
                break
    return pd.Index(matched, dtype=returns.index.dtype) if matched else returns.index[:0]


def partial_refunds_of_removed_sales(returns: pd.DataFrame, removed: pd.DataFrame) -> pd.Index:
    """Refund lines with the same customer, product and price as a removed sale but a DIFFERENT quantity.

    These are kept; they are only listed in the cleaning report for transparency. Pass the returns that
    are left after `refunds_of_removed_sales` has been dropped.
    """
    if returns.empty or removed.empty:
        return returns.index[:0]
    refunds = _refund_match_frame(returns, -1)
    sales = _refund_match_frame(removed, +1)
    sold: dict[tuple, set] = {}
    for s in sales.itertuples():
        sold.setdefault((s.stock_code, s.customer, s.pence), set()).add(s.quantity)
    flagged = [r.Index for r in refunds.itertuples()
               if (r.stock_code, r.customer, r.pence) in sold and r.quantity not in sold[(r.stock_code, r.customer, r.pence)]]
    return pd.Index(flagged, dtype=returns.index.dtype) if flagged else returns.index[:0]


def _refund_summary(lines: pd.DataFrame, list_invoices: bool = False) -> dict:
    """Count and value of refund lines. Money is summed in integer pence and rounded once at the end."""
    pence = int((lines["quantity"].astype("int64").abs() * np.rint(lines["unit_price"].astype("float64") * 100).astype("int64")).sum())
    out = {"lines": int(len(lines)), "refund_pence": pence, "refund": round(pence / 100, 2)}
    if list_invoices:
        out["invoices"] = sorted({str(i) for i in lines["invoice"]})
    return out


def clean(raw: pd.DataFrame) -> CleanResult:
    """Apply the cleaning rules in order. Pure function: no files, easy to unit-test."""
    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    steps: list[dict] = []

    def log(rule: str, why: str, before: int, after: int) -> None:
        steps.append({"rule": rule, "why": why, "removed": before - after, "remaining": after})

    # Normalise text first so that "85123a " and "85123A" count as the same product.
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = (
        df["description"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    )
    df["customer_id"] = df["customer_id"].astype("Int64")
    n_raw = len(df)

    # 1. Exact duplicates. Most come from the two sheets overlapping; the rest repeat inside one invoice.
    before = len(df)
    df = df.drop_duplicates()
    log("Exact duplicate rows",
        "Most duplicates come from the two sheets overlapping on 1-9 Dec 2010; the rest are lines repeated "
        "inside one invoice.", before, len(df))

    # 2. Cancellations (invoice starts with "C") are real returns, not sales. Keep them in `returns`.
    is_cancel = df["invoice"].str.startswith("C")
    returns = df[is_cancel].copy()
    before = len(df)
    df = df[~is_cancel].copy()
    log("Cancellation invoices moved to returns",
        "Invoices starting with 'C' are returns. They would subtract from revenue and confuse "
        "order counts, so they live in their own table.", before, len(df))

    # 3. Non-product stock codes (postage, fees, manual adjustments, test rows, gift vouchers).
    before = len(df)
    df = df[df["stock_code"].str.match(PRODUCT_CODE)]
    log("Non-product stock codes",
        "POST, DOT, M, D, BANK CHARGES, AMAZONFEE, ADJUST, TEST001, gift vouchers... are fees or "
        "bookkeeping lines, not products.", before, len(df))
    returns = returns[returns["stock_code"].str.match(PRODUCT_CODE)]

    # 4. Zero or negative quantity on a normal invoice = stock write-off / adjustment.
    before = len(df)
    df = df[df["quantity"] > 0]
    log("Quantity <= 0", "Negative quantities on non-cancel invoices are stock adjustments "
        "('damaged', 'lost'), not purchases.", before, len(df))

    # 5. Price <= 0: free samples and adjustments carry no revenue.
    before = len(df)
    df = df[df["unit_price"] > 0]
    log("Unit price <= 0", "Free items and manual adjustments carry no revenue information.",
        before, len(df))
    returns = returns[(returns["unit_price"] > 0) & (returns["quantity"] < 0)]

    # 6. Extreme prices: the same product sold at more than 20x its usual price is a typo.
    before = len(df)
    median_price = df.groupby("stock_code")["unit_price"].transform("median")
    keep = df["unit_price"] <= config.PRICE_OUTLIER_RATIO * median_price
    removed_by_price, df = df[~keep], df[keep]  # the removed lines are kept aside to find their refunds
    log(f"Price > {config.PRICE_OUTLIER_RATIO}x the product's median price",
        "Likely box-vs-unit or typing errors, e.g. a car flag with a median of £0.42 sold at "
        "£1,157. Expensive furniture is kept because its own median is also high. Only 0.2% of "
        "revenue is affected.", before, len(df))

    # 7. Extreme quantities: two single lines of 74,215 and 80,995 units were cancelled at once.
    before = len(df)
    keep = df["quantity"] <= config.MAX_LINE_QUANTITY
    removed_by_quantity, df = df[~keep], df[keep]
    log(f"Quantity > {config.MAX_LINE_QUANTITY:,} in one line",
        "Both such lines (74,215 jars, 80,995 birdies) were cancelled the same day and would add "
        "£245k of fake revenue.", before, len(df))

    # 8. Refunds of the sale lines removed by rules 6 and 7. Deleting a sale but keeping its refund
    # would leave a refund with nothing to offset, understating revenue and inflating the return rate.
    # Only LINES that exactly reverse a removed sale are dropped (see refunds_of_removed_sales);
    # partial refunds and credits stay. This is reported in `notes`, not in `steps`, because `steps`
    # counts the sales stream.
    removed = pd.concat([removed_by_price, removed_by_quantity])
    dropped = refunds_of_removed_sales(returns, removed)
    dropped_lines = returns.loc[dropped]
    returns = returns.drop(index=dropped)
    partial_lines = returns.loc[partial_refunds_of_removed_sales(returns, removed)]

    df = df.sort_values(["invoice_date", "invoice", "stock_code"]).reset_index(drop=True)
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    returns = returns.sort_values(["invoice_date", "invoice", "stock_code"]).reset_index(drop=True)
    returns["revenue"] = (returns["quantity"] * returns["unit_price"]).round(2)

    # Canonical description per stock code = the most common one (646 codes have several).
    products = (
        df.groupby("stock_code")["description"]
        .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "UNKNOWN")
        .rename("description")
        .reset_index()
    )

    # Guest checkouts (no customer ID) are kept: they are real revenue. Customer-level analysis
    # filters them out in SQL (`WHERE customer_id IS NOT NULL`).
    notes = {
        "raw_rows": n_raw,
        "sales_rows": len(df),
        "returns_rows": len(returns),
        "refunds_of_removed_sales": _refund_summary(dropped_lines, list_invoices=True),
        "partial_refunds_of_removed_sales": _refund_summary(partial_lines),
        "guest_line_share": float(df["customer_id"].isna().mean()),
        "guest_revenue_share": float(df.loc[df["customer_id"].isna(), "revenue"].sum() / df["revenue"].sum()),
        "customers": int(df["customer_id"].nunique()),
        "orders": int(df["invoice"].nunique()),
        "products": int(df["stock_code"].nunique()),
        "countries": int(df["country"].nunique()),
        "first_date": str(df["invoice_date"].min()),
        "last_date": str(df["invoice_date"].max()),
    }
    sales = df[["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id",
                "country", "revenue"]]
    returns = returns[["invoice", "stock_code", "quantity", "invoice_date", "unit_price",
                       "customer_id", "country", "revenue"]]
    return CleanResult(sales, returns, products, steps, notes)


def write_outputs(res: CleanResult, out_dir: Path = config.PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(index=False, compression="zstd")
    res.sales.astype({"quantity": "int32"}).to_parquet(out_dir / "sales.parquet", **kw)
    res.returns.astype({"quantity": "int32"}).to_parquet(out_dir / "returns.parquet", **kw)
    res.products.to_parquet(out_dir / "products.parquet", **kw)
    report = {"steps": res.steps, "summary": res.notes}
    (out_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    xlsx = download_raw()
    print("Reading Excel (about 15 seconds) ...")
    raw = read_raw(xlsx)
    res = clean(raw)
    write_outputs(res)
    print(f"{'rule':<46}{'removed':>10}{'remaining':>12}")
    print(f"{'raw rows':<46}{'':>10}{len(raw):>12,}")
    for s in res.steps:
        print(f"{s['rule']:<46}{s['removed']:>10,}{s['remaining']:>12,}")
    print(f"returns kept: {len(res.returns):,}")
    r, p = res.notes["refunds_of_removed_sales"], res.notes["partial_refunds_of_removed_sales"]
    print(f"refund lines of removed sales dropped: {r['lines']} (£{r['refund']:,.2f}) {r['invoices']}; "
          f"partial refunds kept and listed: {p['lines']}")
    print(json.dumps(res.notes, indent=2))


if __name__ == "__main__":
    main()
