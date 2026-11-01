"""The SQL guard is the safety boundary for LLM-written SQL, so it is tested with attacks as well as good queries."""
import duckdb
import pytest

from rci import sql_guard
from rci.sql_guard import GuardError, run_guarded, validate_sql, wrap_with_limit

GOOD = [
    "SELECT * FROM sales",
    "SELECT country, SUM(revenue) FROM sales GROUP BY country ORDER BY 2 DESC LIMIT 5;",
    "WITH t AS (SELECT * FROM sales) SELECT COUNT(*) FROM t",
    "SELECT s.invoice, p.description FROM sales s JOIN products p USING (stock_code)",
    "SELECT * FROM (SELECT customer_id FROM customers) x",
    "SELECT COUNT(*) FROM returns",
    "SELECT * FROM range(3)",
    "SELECT description FROM products WHERE description LIKE '%read_csv(%'",   # text inside a literal is not a call
    "SELECT 1 -- a trailing comment",
    "SELECT * FROM main.sales",                                    # the default schema may be written out
    "SELECT * FROM generate_series(1, 3)",
    "WITH x AS (SELECT 1 AS a) SELECT * FROM x JOIN sales ON true",
    "SELECT s.invoice, c.segment FROM sales s JOIN customers c USING (customer_id)",   # regression: USING joins
]

BAD = {
    "drop": "DROP TABLE sales",
    "delete": "DELETE FROM sales",
    "insert": "INSERT INTO sales SELECT * FROM sales",
    "update": "UPDATE sales SET revenue = 0",
    "create": "CREATE TABLE x AS SELECT 1",
    "copy_out": "COPY sales TO '/tmp/leak.csv'",
    "attach": "ATTACH '/tmp/other.db'",
    "install": "INSTALL httpfs",
    "pragma": "PRAGMA database_list",
    "set": "SET enable_external_access = true",
    "explain": "EXPLAIN SELECT 1",
    "two_statements": "SELECT 1; DROP TABLE sales",
    "two_selects": "SELECT 1; SELECT 2",
    "read_csv": "SELECT * FROM read_csv('/etc/passwd')",
    "read_csv_quoted": "SELECT * FROM \"read_csv\"('/etc/passwd')",
    "read_parquet": "SELECT * FROM read_parquet('/tmp/x.parquet')",
    "glob": "SELECT * FROM glob('/*')",
    "file_literal": "SELECT * FROM '/etc/passwd'",
    "duckdb_settings": "SELECT * FROM duckdb_settings()",
    "unknown_table": "SELECT * FROM secrets",
    "unknown_in_join": "SELECT * FROM sales JOIN passwords ON true",
    "unknown_in_cte": "WITH a AS (SELECT * FROM hidden) SELECT * FROM a",
    "comment_hides_nothing": "SELECT 1; /* harmless */ DROP TABLE sales",
    "information_schema": "SELECT * FROM information_schema.tables",
    "duckdb_tables": "SELECT * FROM duckdb_tables()",
    "other_catalog": "SELECT * FROM other.main.sales",
    "cte_shadow": "WITH sales AS (SELECT * FROM secrets) SELECT * FROM sales",
    "scalar_subquery": "SELECT (SELECT COUNT(*) FROM secrets) AS n",
    "where_subquery": "SELECT * FROM sales WHERE customer_id IN (SELECT id FROM users)",
    "pivot_source": "SELECT * FROM secrets PIVOT (SUM(x) FOR y IN ('a'))",
    "table_function_in_join": "SELECT * FROM sales, read_json('/tmp/x.json')",
    "garbage": "SELEC nonsense FROM",
    "empty": "   ",
    "too_long": "SELECT " + "1," * 3000 + "1",
}


@pytest.mark.parametrize("sql", GOOD)
def test_good_queries_pass(sql):
    out = validate_sql(sql)
    assert not out.endswith(";")


@pytest.mark.parametrize("name", BAD)
def test_bad_queries_are_rejected(name):
    with pytest.raises(GuardError):
        validate_sql(BAD[name])


def test_rejection_messages_are_user_safe():
    with pytest.raises(GuardError, match="Only read-only SELECT"):
        validate_sql("DROP TABLE sales")
    with pytest.raises(GuardError, match="Unknown or forbidden table"):
        validate_sql("SELECT * FROM secrets")


def test_row_cap_wraps_query_and_survives_trailing_comment():
    wrapped = wrap_with_limit("SELECT 1 -- note", 5)
    assert wrapped.rstrip().endswith("LIMIT 5")
    duckdb.connect().execute(wrapped).fetchall()          # the comment must not swallow the closing bracket


def test_row_cap_is_enforced_and_order_is_kept(con):
    sql, df = run_guarded(con, "SELECT range AS n FROM range(100) ORDER BY n DESC", max_rows=7)
    assert len(df) == 7
    assert df["n"].tolist() == [99, 98, 97, 96, 95, 94, 93]


def test_run_guarded_returns_real_results(con):
    sql, df = run_guarded(con, "SELECT SUM(revenue) AS total FROM sales")
    assert round(df["total"][0], 2) == 6 + 10 + 12 + 10 + 30 + 10 + 6 + 30 + 15


def test_slow_query_is_stopped_by_the_timeout(con):
    with pytest.raises(GuardError, match="stopped"):
        run_guarded(con, "SELECT COUNT(*) FROM range(1000000000) a, range(1000000000) b", timeout_s=0.5)


def test_connection_is_locked_even_if_the_guard_were_bypassed(con):
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM read_csv('/etc/hosts')").fetchall()
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access = true")
    with pytest.raises(duckdb.Error):
        con.execute("COPY sales TO '/tmp/rci_should_not_exist.csv'")


def test_allowed_tables_match_the_database(con):
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert sql_guard.ALLOWED_TABLES == tables
