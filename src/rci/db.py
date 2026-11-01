"""DuckDB access: build the tables from Parquet, and run the named SQL queries in sql/*.sql.

DuckDB is a SQL engine that runs inside Python, so no database server is needed. We load the small
cleaned Parquet files into in-memory tables, then LOCK the connection so it can no longer read or
write files. That lock is one of the safety layers for the AI analyst.
"""
from __future__ import annotations

import datetime as dt
import re
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from rci import config

_NAME = re.compile(r"^--\s*name:\s*(\w+)\s*$")


def load_queries(path: Path) -> dict[str, str]:
    """Split a .sql file into named blocks. A block starts at a line like `-- name: headline`."""
    blocks: dict[str, list[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        m = _NAME.match(line.strip())
        if m:
            current = m.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return {k: "\n".join(v).strip().rstrip(";").strip() for k, v in blocks.items()}


@lru_cache(maxsize=1)
def all_queries() -> dict[str, str]:
    """Every named query from every sql file except schema.sql, keyed by name."""
    merged: dict[str, str] = {}
    for f in sorted(config.SQL_DIR.glob("*.sql")):
        if f.name == "schema.sql":
            continue
        for name, sql in load_queries(f).items():
            if name in merged:
                raise ValueError(f"duplicate query name {name!r} in {f.name}")
            merged[name] = sql
    return merged


def _sql_path(p: Path) -> str:
    return str(p).replace("'", "''")


def connect(
    processed_dir: Path = config.PROCESSED_DIR,
    memory_limit: str = "1GB",
    threads: int | None = None,
    lock: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB with the tables sales, returns, products, customers.

    `product_categories.parquet` and `segments.parquet` are optional (they are produced by later
    pipeline steps); when absent the category / segment columns are simply NULL.
    """
    cfg: dict = {"memory_limit": memory_limit}
    if threads:
        cfg["threads"] = threads
    con = duckdb.connect(":memory:", config=cfg)

    def src(name: str, file: str, empty_sql: str | None = None) -> None:
        path = processed_dir / file
        if path.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{_sql_path(path)}')")
        elif empty_sql:
            con.execute(f"CREATE VIEW {name} AS {empty_sql}")
        else:
            raise FileNotFoundError(f"{path} not found. Run `python -m rci.pipeline` first.")

    src("src_sales", "sales.parquet")
    src("src_returns", "returns.parquet")
    src("src_products", "products.parquet")
    src("src_categories", "product_categories.parquet",
        "SELECT NULL::VARCHAR AS stock_code, NULL::VARCHAR AS category LIMIT 0")
    src("src_segments", "segments.parquet",
        "SELECT NULL::BIGINT AS customer_id, NULL::VARCHAR AS segment LIMIT 0")

    schema = load_queries(config.SQL_DIR / "schema.sql")
    for name in ("sales", "returns", "products"):
        con.execute(schema[name])

    # "today" for recency = the day after the last order in the data
    last = con.execute("SELECT MAX(invoice_date)::DATE FROM sales").fetchone()[0]
    as_of = last + dt.timedelta(days=1)
    rfm_sql = load_queries(config.SQL_DIR / "rfm.sql")["rfm"]
    con.execute(schema["customers"].replace("RFM_QUERY", rfm_sql), {"as_of": as_of})

    for v in ("src_sales", "src_returns", "src_products", "src_categories", "src_segments"):
        con.execute(f"DROP VIEW {v}")
    if lock:
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
    return con


def query(con: duckdb.DuckDBPyConnection, name: str, **params) -> pd.DataFrame:
    """Run a named query from sql/*.sql and return a DataFrame.

    A fresh cursor per call keeps this safe when Streamlit serves several sessions at once.
    """
    sql = all_queries()[name]
    return con.cursor().execute(sql, params or None).df()


def data_end(con: duckdb.DuckDBPyConnection) -> dt.date:
    """Last day with data. A 90-day label is only observable if cutoff + 90 days <= this + 1."""
    return con.cursor().execute("SELECT MAX(invoice_date)::DATE FROM sales").fetchone()[0]
