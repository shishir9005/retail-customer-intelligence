"""Data-quality checks on the FINAL tables (after cleaning). Each check is one SQL query.

A check passes when its query returns 0 offending rows. The results are saved into
`cleaning_report.json` and shown on the dashboard, so a reader can see the data was verified.

Run:  python -m rci.quality
"""
from __future__ import annotations

import json

import duckdb

from rci import config, db
from rci.pipeline import PRODUCT_CODE

# (name, SQL that counts offending rows)
CHECKS: list[tuple[str, str]] = [
    ("No missing invoice, stock code, date, quantity, price or country in sales",
     """SELECT COUNT(*) FROM sales WHERE invoice IS NULL OR stock_code IS NULL OR invoice_date IS NULL
        OR quantity IS NULL OR unit_price IS NULL OR country IS NULL"""),
    ("Every sale has quantity > 0 and unit price > 0",
     "SELECT COUNT(*) FROM sales WHERE quantity <= 0 OR unit_price <= 0"),
    ("Every return has quantity < 0",
     "SELECT COUNT(*) FROM returns WHERE quantity >= 0"),
    ("revenue = quantity x unit price (to the penny)",
     "SELECT COUNT(*) FROM sales WHERE ABS(revenue - quantity * unit_price) > 0.01"),
    ("No invoice starts with 'C' in sales (cancellations live in returns)",
     "SELECT COUNT(*) FROM sales WHERE invoice LIKE 'C%'"),
    ("Every stock code looks like a product code",
     f"SELECT COUNT(*) FROM sales WHERE NOT regexp_matches(stock_code, '{PRODUCT_CODE}')"),
    ("Every sold stock code has a product record",
     "SELECT COUNT(*) FROM sales s LEFT JOIN products p USING (stock_code) WHERE p.stock_code IS NULL"),
    ("No invoice is split across two customers",
     """SELECT COUNT(*) FROM (SELECT invoice FROM sales WHERE customer_id IS NOT NULL
        GROUP BY invoice HAVING COUNT(DISTINCT customer_id) > 1)"""),
    ("Dates fall inside the dataset's published range (Dec 2009 - Dec 2011)",
     """SELECT COUNT(*) FROM sales WHERE invoice_date < TIMESTAMP '2009-12-01'
        OR invoice_date >= TIMESTAMP '2011-12-10'"""),
    ("No sale line above the extreme-quantity limit",
     f"SELECT COUNT(*) FROM sales WHERE quantity > {config.MAX_LINE_QUANTITY}"),
    ("No return line above the extreme-quantity limit",
     f"SELECT COUNT(*) FROM returns WHERE quantity < -{config.MAX_LINE_QUANTITY}"),
]


def run_checks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    results = []
    for name, sql in CHECKS:
        offending = con.execute(sql).fetchone()[0]
        results.append({"check": name, "offending_rows": int(offending), "passed": offending == 0})
    return results


def main() -> None:
    con = db.connect()
    results = run_checks(con)
    for r in results:
        print(("PASS" if r["passed"] else "FAIL"), f"({r['offending_rows']:>6})", r["check"])
    path = config.PROCESSED_DIR / "cleaning_report.json"
    report = json.loads(path.read_text())
    report["checks"] = results
    path.write_text(json.dumps(report, indent=2))
    if not all(r["passed"] for r in results):
        raise SystemExit("data-quality checks failed")


if __name__ == "__main__":
    main()
