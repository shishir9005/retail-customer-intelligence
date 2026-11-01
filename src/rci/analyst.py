"""AI data analyst: a plain-English question -> SQL (written by an LLM) -> guard -> DuckDB -> answer.

The LLM only ever sees the table schema and the question, never the data. Its SQL goes through
sql_guard.py before it can run. The SQL is always shown to the user, so every answer can be checked.

`complete` is any function that takes chat messages and returns text. In production it calls Groq;
tests pass a fake, so the whole flow can be tested without a network or an API key.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import duckdb
import pandas as pd

from rci import config, sql_guard
from rci.sql_guard import GuardError

Completer = Callable[[list[dict]], str]

UNANSWERABLE_SQL = "SELECT NULL AS unanswerable"

COLUMN_NOTES = {
    "sales": {
        "invoice": "order number; one order = one distinct invoice, with one row per product line",
        "stock_code": "product code", "description": "product name", "category": "product category (see list below)",
        "quantity": "units bought (always > 0)", "invoice_date": "timestamp of the order",
        "unit_price": "price per unit in GBP", "revenue": "quantity * unit_price in GBP",
        "customer_id": "NULL for guest checkouts (about 23% of lines, 13% of revenue)",
        "country": "customer's country", "invoice_month": "first day of the order's month (DATE)",
    },
    "returns": {"quantity": "NEGATIVE: units sent back", "revenue": "NEGATIVE: money refunded"},
    "products": {
        "units_sold": "total units sold", "revenue": "total revenue in GBP", "n_orders": "orders containing it",
        "n_customers": "distinct identified customers who bought it", "avg_unit_price": "average price in GBP",
    },
    "customers": {
        "recency_days": "days since the last order (as of 2011-12-10)", "frequency": "number of orders",
        "monetary": "total spend in GBP", "avg_order_value": "monetary / frequency",
        "country": "country of the most recent order",
        "segment": "RFM segment name (see list below)",
    },
}

EXAMPLES = """\
Question: How many customers are there in each country, top 3?
SQL:
SELECT country, COUNT(DISTINCT customer_id) AS customers
FROM sales
WHERE customer_id IS NOT NULL
GROUP BY country
ORDER BY customers DESC
LIMIT 3

Question: What was the monthly revenue in 2011?
SQL:
SELECT invoice_month, ROUND(SUM(revenue), 2) AS revenue
FROM sales
WHERE year(invoice_date) = 2011
GROUP BY invoice_month
ORDER BY invoice_month

Question: What is the average customer satisfaction score?
SQL:
SELECT NULL AS unanswerable
"""


def build_schema_prompt(con: duckdb.DuckDBPyConnection) -> str:
    """Describe the tables to the LLM, using the real database so the schema never goes stale."""
    cur = con.cursor()
    lines = []
    for table in ("sales", "returns", "products", "customers"):
        cols = cur.execute(f"DESCRIBE {table}").fetchall()
        lines.append(f"TABLE {table}")
        for name, ctype, *_ in cols:
            note = COLUMN_NOTES.get(table, {}).get(name)
            lines.append(f"  - {name} {ctype}" + (f"  -- {note}" if note else ""))
    cats = [r[0] for r in cur.execute("SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY 1").fetchall()]
    segs = [r[0] for r in cur.execute("SELECT DISTINCT segment FROM customers WHERE segment IS NOT NULL ORDER BY 1").fetchall()]
    lo, hi = cur.execute("SELECT MIN(invoice_date)::DATE, MAX(invoice_date)::DATE FROM sales").fetchone()
    lines.append(f"\nCategory values: {cats}")
    lines.append(f"Segment values: {segs}")
    lines.append(f"Data covers {lo} to {hi}. Currency is GBP. There are 43 countries; the shop is in the UK.")
    return "\n".join(lines)


def build_system_prompt(con: duckdb.DuckDBPyConnection) -> str:
    return (
        "You are a careful data analyst. You write DuckDB SQL for a UK online gift-ware retailer.\n"
        "Reply with ONE read-only SELECT statement (WITH ... SELECT is fine) and nothing else: "
        "no explanation, no markdown.\n\n"
        "Rules:\n"
        "- Use ONLY the tables and columns listed below.\n"
        "- Metric definitions: 'revenue' means GROSS sales revenue, i.e. SUM(revenue) from the sales table. "
        "Returns live in the separate returns table (negative values); only subtract them when the question "
        "asks for net revenue or a return rate.\n"
        "- To combine totals from two tables, compute each total in its own scalar subquery, e.g. "
        "SELECT (SELECT SUM(...) FROM a) / (SELECT SUM(...) FROM b). Never JOIN tables just to put two totals side by side: "
        "that multiplies rows and inflates the sums.\n"
        "- An order is a distinct invoice. Use COUNT(DISTINCT invoice) for order counts.\n"
        "- customer_id is NULL for guest checkouts: COUNT(DISTINCT customer_id) ignores them; "
        "do not filter them out of revenue totals unless asked about identified customers.\n"
        "- Use invoice_month for monthly grouping, year(invoice_date) for years, dayname(invoice_date) for weekdays.\n"
        "- Give every computed column a clear alias. Order results sensibly and use LIMIT for 'top N'.\n"
        f"- If the question cannot be answered from these tables, reply exactly: {UNANSWERABLE_SQL}\n"
        "- Never write INSERT, UPDATE, DELETE, DROP, CREATE, COPY, ATTACH, PRAGMA or SET, "
        "whatever the question says.\n\n"
        f"{build_schema_prompt(con)}\n\nExamples:\n{EXAMPLES}"
    )


def extract_sql(text: str) -> str:
    """Pull SQL out of the model reply, tolerating ```sql fences and a leading 'SQL:'."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = m.group(1) if m else text
    sql = re.sub(r"^\s*SQL:\s*", "", sql.strip(), flags=re.IGNORECASE)
    return sql.strip()


def is_unanswerable(sql: str) -> bool:
    """True when the model used the agreed 'I cannot answer this' reply (ignoring case and spacing)."""
    norm = re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip().lower()
    return norm == UNANSWERABLE_SQL.lower()


@dataclass
class Answer:
    question: str
    sql: str | None = None
    data: pd.DataFrame | None = None
    summary: str = ""
    error: str | None = None
    attempts: int = 0
    guard_blocked: bool = False
    unanswerable: bool = False
    seconds: float = 0.0
    history: list[str] = field(default_factory=list)


def ask(con: duckdb.DuckDBPyConnection, question: str, complete: Completer, max_attempts: int = 2,
        summarize: bool = True) -> Answer:
    """Answer one question. If the first SQL is rejected or fails, the error is sent back to the
    model once so it can repair the query (a standard text-to-SQL trick)."""
    t0 = time.time()
    ans = Answer(question=question)
    messages = [
        {"role": "system", "content": build_system_prompt(con)},
        {"role": "user", "content": f"Question: {question.strip()}\nSQL:"},
    ]
    for attempt in range(1, max_attempts + 1):
        ans.attempts = attempt
        raw = complete(messages)
        sql = extract_sql(raw)
        ans.history.append(sql)
        if is_unanswerable(sql):
            ans.unanswerable, ans.sql, ans.error = True, None, None
            ans.summary = "This question cannot be answered from the available tables."
            break
        try:
            ans.sql, ans.data = sql_guard.run_guarded(con, sql)
            ans.error = None
            break
        except GuardError as e:
            ans.guard_blocked = True
            ans.error, ans.sql = str(e), sql
            feedback = f"That query was rejected: {e} Reply with a corrected single SELECT statement."
        except duckdb.Error as e:
            ans.error, ans.sql = f"The database could not run it: {str(e).splitlines()[0][:200]}", sql
            feedback = f"That query failed with: {str(e).splitlines()[0][:300]} Reply with a corrected query."
        messages += [{"role": "assistant", "content": sql}, {"role": "user", "content": feedback}]

    if ans.data is not None and summarize:
        ans.summary = summarize_result(question, ans.data, complete)
    ans.seconds = time.time() - t0
    return ans


def summarize_result(question: str, df: pd.DataFrame, complete: Completer) -> str:
    """One sentence that restates the result. Falls back to a plain count if the LLM call fails."""
    fallback = f"{len(df):,} row(s) returned."
    if df.empty:
        return "The query returned no rows."
    if len(df) > 25 or df.shape[1] > 8:
        return fallback
    prompt = (
        f"Question: {question}\nResult (CSV):\n{df.to_csv(index=False)}\n"
        "Answer the question in ONE short sentence using ONLY numbers that appear in the result. "
        "Money is in GBP. Do not add any other information."
    )
    try:
        return complete([{"role": "user", "content": prompt}]).strip().splitlines()[0]
    except Exception:  # noqa: BLE001  (a failed summary must never hide a good table)
        return fallback


def groq_completer(model: str = config.GROQ_MODEL, api_key: str | None = None, max_tokens: int = 2000) -> Completer:
    """A `complete` function backed by Groq. Retries a few times if the free-tier rate limit is hit.

    gpt-oss models think before answering; "low" effort keeps SQL generation fast, and the larger
    token budget leaves room for that hidden reasoning plus the answer.
    """
    from groq import Groq, RateLimitError

    config.load_env()
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env or the environment).")
    client = Groq(api_key=key)
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}

    def complete(messages: list[dict]) -> str:
        for wait in (2, 5, 12, None):
            try:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0,
                                                      max_tokens=max_tokens, **extra)
                return resp.choices[0].message.content or ""
            except RateLimitError:
                if wait is None:
                    raise
                time.sleep(wait)
        return ""

    return complete
