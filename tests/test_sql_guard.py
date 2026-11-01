"""The SQL guard is the safety boundary for LLM-written SQL, so it is tested with attacks as well as good queries."""
import json
import warnings

import duckdb
import numpy as np
import pandas as pd
import pytest

from rci import config, sql_guard
from rci.sql_guard import GuardError, run_guarded, validate_sql, wrap_with_limit

from conftest import PROCESSED, ROOT

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
    # WITH queries that must keep working under scope-aware parsing
    "WITH t AS (SELECT * FROM sales) SELECT * FROM (SELECT COUNT(*) AS n FROM t) x",                     # a WITH name inside a sub-query
    "WITH a AS (SELECT * FROM sales), b AS (SELECT * FROM a) SELECT COUNT(*) FROM b",                    # a chain
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
    "WITH t AS (SELECT 1 AS a) SELECT a FROM t UNION ALL SELECT a FROM t",                               # both sides of UNION ALL
    "WITH a AS (WITH b AS (SELECT 1 AS x) SELECT x FROM b) SELECT * FROM a",                             # a WITH inside a WITH body
    "WITH t AS MATERIALIZED (SELECT * FROM sales LIMIT 5) SELECT * FROM t",
    "WITH sales AS (SELECT 1 AS a) SELECT * FROM sales",                                                 # shadowing an allowed table is harmless
    "SELECT [1, 2] AS xs",                                                                               # a list literal calls list_value, not list
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
    # a WITH name must never hide a read of a real table
    "with_dotted_information_schema": 'WITH "information_schema.tables" AS (SELECT * FROM information_schema.tables) SELECT * FROM "information_schema.tables"',
    "with_sqlite_master": "WITH sqlite_master AS (SELECT 1 AS a) SELECT * FROM sqlite_master",
    "with_duckdb_tables": "WITH duckdb_tables AS (SELECT 1 AS a) SELECT * FROM duckdb_tables",
    "with_x_then_qualified_x": "WITH x AS (SELECT 1 AS a) SELECT * FROM main.x",
    "with_in_one_union_branch_hides_nothing_in_the_other":
        "SELECT * FROM (WITH secrets AS (SELECT 1 AS a) SELECT a FROM secrets) UNION ALL SELECT a FROM secrets",
    "inner_with_name_is_not_visible_outside_its_own_body":
        "WITH a AS (WITH b AS (SELECT 1 AS x) SELECT x FROM b) SELECT * FROM b",
    "reserved_name_in_one_union_branch": "SELECT 1 AS a UNION ALL (WITH duckdb_tables AS (SELECT 2 AS a) SELECT a FROM duckdb_tables)",
    "self_reference_sqlite_master": "WITH sqlite_master AS (SELECT * FROM sqlite_master) SELECT * FROM sqlite_master",
    "self_reference_duckdb_tables": "WITH duckdb_tables AS (SELECT * FROM duckdb_tables) SELECT * FROM duckdb_tables",
    "self_reference_pragma_database_list": "WITH pragma_database_list AS (SELECT * FROM pragma_database_list) SELECT * FROM pragma_database_list",
    "forward_reference": "WITH a AS (SELECT * FROM tables), tables AS (SELECT 1 AS x) SELECT * FROM a",
    "recursive_forward_reference": "WITH RECURSIVE a AS (SELECT * FROM tables), tables AS (SELECT 1 AS x) SELECT * FROM a",
    "upper_case_quoted_reserved_name": 'WITH "SQLITE_MASTER" AS (SELECT 1 AS a) SELECT * FROM "SQLITE_MASTER"',
    # functions that can build one enormous value are refused before the query runs
    "string_agg": "SELECT string_agg(description, ',') FROM sales",
    "group_concat": "SELECT group_concat(description) FROM sales",
    "listagg": "SELECT listagg(description, ',') FROM sales",
    "list": "SELECT list(description) FROM sales",
    "array_agg": "SELECT array_agg(description) FROM sales",
    "histogram": "SELECT histogram(description) FROM sales",
    "repeat": "SELECT repeat('x', 1000000)",
    "schema_qualified_string_agg": "SELECT main.string_agg(description, ',') FROM sales",
    "the_huge_aggregate": "SELECT string_agg(description, ',') FROM sales, range(4)",
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


# ═════════════════════ every GOOD query also runs end to end ═════════════════════
@pytest.mark.parametrize("sql", GOOD)
def test_good_queries_also_execute_through_the_streaming_path(con, sql):
    safe, df = run_guarded(con, sql)
    assert isinstance(df, pd.DataFrame) and safe == validate_sql(sql)


# ═════════════════════ reserved WITH names ═════════════════════
def test_the_reserved_with_names_cover_the_system_views_an_attacker_would_want():
    for name in ("sqlite_master", "duckdb_tables", "pg_settings", "tables"):
        assert name in sql_guard.RESERVED_WITH_NAMES
    assert not (sql_guard.RESERVED_WITH_NAMES & sql_guard.ALLOWED_TABLES)       # our own tables are never reserved


def test_a_reserved_with_name_gets_the_documented_message():
    with pytest.raises(GuardError) as error:
        validate_sql("WITH sqlite_master AS (SELECT 1 AS a) SELECT * FROM sqlite_master")
    assert str(error.value) == "The name 'sqlite_master' can't be used for a WITH query."


@pytest.mark.parametrize("name", ["a.b", "duckdb_anything", "sqlite_anything", "pragma_anything", "pg_anything",
                                  "information_schema_x", "DuckDB_Mixed_Case"])
def test_with_names_with_a_dot_or_a_system_prefix_are_refused(name):
    with pytest.raises(GuardError, match="can't be used for a WITH query"):
        validate_sql(f'WITH "{name}" AS (SELECT 1 AS a) SELECT * FROM "{name}"')


def test_names_given_as_reserved_are_refused_unless_they_are_allowed_tables():
    with pytest.raises(GuardError, match="'secrets'"):
        validate_sql("WITH secrets AS (SELECT 1 AS a) SELECT * FROM secrets", reserved_names={"secrets"})
    validate_sql("WITH sales AS (SELECT 1 AS a) SELECT * FROM sales", reserved_names={"sales"})   # allowed table: harmless


def test_run_guarded_reserves_the_names_of_the_connections_own_views():
    """A WITH that borrows the name of a view on the data connection must not run. Only the reserved-name rule
    catches this one: the query reads no table, because the name resolves to the WITH."""
    plain = duckdb.connect()
    plain.execute("CREATE TABLE sales AS SELECT 1 AS a")
    plain.execute("CREATE VIEW hidden_view AS SELECT 42 AS secret")
    with pytest.raises(GuardError, match="'hidden_view'"):
        run_guarded(plain, "WITH hidden_view AS (SELECT 1 AS a) SELECT * FROM hidden_view")
    assert run_guarded(plain, "WITH sales AS (SELECT 2 AS a) SELECT * FROM sales")[1]["a"][0] == 2


# ═════════════════════ the function deny-list ═════════════════════
DENIED = ["string_agg", "group_concat", "listagg", "list", "array_agg", "histogram", "repeat"]


@pytest.mark.parametrize("name", DENIED)
def test_each_denied_function_is_refused_with_a_helpful_message(name):
    with pytest.raises(GuardError, match=r"counts or a top-N"):
        validate_sql(f"SELECT {name}(description) FROM sales")


def test_a_denied_function_is_refused_before_the_query_is_ever_sent_to_the_database(con):
    sent = []

    class SpyCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        def sql(self, query, *args, **kwargs):
            sent.append(query)
            return self._cursor.sql(query, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class SpyConnection:
        def cursor(self):
            return SpyCursor(con.cursor())

    with pytest.raises(GuardError):
        run_guarded(SpyConnection(), "SELECT string_agg(description, ',') FROM sales, range(4)")
    assert sent == []                                               # nothing reached the engine
    run_guarded(SpyConnection(), "SELECT 1 AS n")
    assert len(sent) == 1                                           # while a harmless query does


# ═════════════════════ result-size caps: refused while streaming, before pandas ═════════════════════
OVER_THE_CAPS = {
    "total": ("SELECT lpad('x', 5000, 'x') AS s FROM range(1000)", "larger than 2 MB"),
    "one_cell": ("SELECT lpad('x', 50000, 'x') AS big", "single value"),
    "one_cell_just_over": ("SELECT lpad('x', 10001, 'x') AS big", "single value"),
    "binary": ("SELECT lpad('x', 50000, 'x')::BLOB AS b", "single value"),
    "in_a_list": ("SELECT [lpad('x', 20000, 'x')] AS v", "single value"),
    "in_a_struct": ("SELECT {'a': lpad('x', 20000, 'x')} AS v", "single value"),
    "in_a_map_value": ("SELECT MAP {'k': lpad('x', 20000, 'x')} AS v", "single value"),
    "in_a_map_key": ("SELECT MAP {lpad('x', 20000, 'x'): 1} AS v", "single value"),
    "deeply_nested": ("SELECT [{'a': [lpad('x', 20000, 'x')]}] AS v", "single value"),
    "list_items": ("SELECT range(5000) AS r", "list"),
    "list_just_over": ("SELECT range(1001) AS r", "list"),
    "list_in_list": ("SELECT [range(5000)] AS r", "list"),
    "list_in_struct": ("SELECT {'a': range(5000)} AS r", "list"),
}

WITHIN_THE_CAPS = [
    "SELECT lpad('x', 10000, 'x') AS s",                       # exactly the per-cell limit
    "SELECT range(1000) AS r",                                 # exactly the list-item limit
    "SELECT [range(1000)] AS r",
    "SELECT NULL::VARCHAR AS s FROM range(10)",                # an all-null column has no length
    "SELECT NULL::VARCHAR[] AS s FROM range(10)",
    "SELECT lpad('x', 1000, 'x') AS s FROM range(1000)",       # about 1 MB in total, under the 2 MB cap
]


def test_the_limits_are_the_documented_ones():
    assert (config.ANALYST_MAX_RESULT_BYTES, config.ANALYST_MAX_CELL_BYTES, config.ANALYST_MAX_LIST_ITEMS,
            config.ANALYST_BATCH_ROWS, config.ANALYST_SUMMARY_MAX_CHARS) == (2_000_000, 10_000, 1_000, 256, 4_000)


@pytest.mark.parametrize("name", list(OVER_THE_CAPS))
def test_results_over_a_cap_are_refused(con, name):
    sql, expected = OVER_THE_CAPS[name]
    with pytest.raises(GuardError, match=expected):
        run_guarded(con, sql)


@pytest.mark.parametrize("sql", WITHIN_THE_CAPS)
def test_results_exactly_at_a_cap_are_allowed(con, sql):
    assert len(run_guarded(con, sql)[1]) >= 1


@pytest.mark.parametrize("name", list(OVER_THE_CAPS))
def test_an_over_cap_result_never_reaches_pandas(con, monkeypatch, name):
    def reached_pandas(*args, **kwargs):
        raise AssertionError("an over-cap result reached the pandas conversion")

    monkeypatch.setattr(sql_guard, "_to_dataframe", reached_pandas)
    with pytest.raises(GuardError):
        run_guarded(con, OVER_THE_CAPS[name][0])


def test_a_result_within_the_caps_does_go_through_the_conversion(con, monkeypatch):
    calls, original = [], sql_guard._to_dataframe
    monkeypatch.setattr(sql_guard, "_to_dataframe", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    run_guarded(con, "SELECT 1 AS n")
    assert calls == [1]


def test_the_total_cap_stops_the_stream_early_instead_of_reading_it_all(con):
    """5 MB of results, 256 rows per batch: the cap trips on the second batch, long before the end."""
    seen = []
    original = sql_guard._check_batch
    sql_guard._check_batch = lambda batch: (seen.append(batch.num_rows), original(batch))[1]
    try:
        with pytest.raises(GuardError, match="larger than 2 MB"):
            run_guarded(con, OVER_THE_CAPS["total"][0])
    finally:
        sql_guard._check_batch = original
    assert sum(seen) <= 512


def test_a_wide_1000_row_result_is_allowed(real_con):
    _, df = run_guarded(real_con, "SELECT * FROM sales LIMIT 1000")
    assert len(df) == 1000 and df.shape[1] == 11


def test_an_empty_result_keeps_its_columns_and_dtypes(real_con):
    sql = "SELECT * FROM sales WHERE 1 = 0"
    _, got = run_guarded(real_con, sql)
    want = real_con.cursor().execute(wrap_with_limit(sql)).df()
    assert len(got) == 0
    assert list(got.columns) == list(want.columns)
    assert [str(t) for t in got.dtypes] == [str(t) for t in want.dtypes]


# ═════════════════════ identical to the plain `.df()` path ═════════════════════
def canonical_cell(value):
    """A hashable, comparable form of one cell, so lists, structs, maps and NaNs compare sensibly."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return tuple(canonical_cell(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), canonical_cell(v)) for k, v in value.items()))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sort_key(row) -> str:
    """Rows are ordered by their text, with floats cut to 6 significant digits so that last-digit noise cannot reorder them."""
    def short(cell):
        if isinstance(cell, float):
            return float(f"{cell:.6g}")
        return tuple(short(c) for c in cell) if isinstance(cell, tuple) else cell
    return repr(tuple(short(c) for c in row))


def canonical_rows(df: pd.DataFrame, ordered: bool) -> list:
    rows = [tuple(canonical_cell(v) for v in row) for row in df.itertuples(index=False, name=None)]
    return rows if ordered else sorted(rows, key=_sort_key)


def same_cell(a, b) -> bool:
    """Exact equality, except floats: DuckDB adds numbers on several threads, so two runs of the same query can
    differ in the last digits (about 1e-16 relative). That noise is not a difference in the result."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same_cell(x, y) for x, y in zip(a, b))
    return a == b


def assert_same_as_plain_df(connection, sql: str) -> None:
    safe, got = run_guarded(connection, sql)
    want = connection.cursor().execute(wrap_with_limit(safe)).df()               # today's path
    assert list(got.columns) == list(want.columns)
    assert [str(t) for t in got.dtypes] == [str(t) for t in want.dtypes]
    ordered = " order by " in " ".join(safe.lower().split())                     # only an ORDER BY fixes the row order
    got_rows, want_rows = canonical_rows(got, ordered), canonical_rows(want, ordered)
    assert len(got_rows) == len(want_rows)
    for g, w in zip(got_rows, want_rows):
        assert same_cell(g, w), f"{g!r} != {w!r}"


REFERENCE = [(q["id"], q["gold_sql"]) for q in json.loads((ROOT / "evals" / "analyst_questions.json").read_text())["questions"]]
_saved_path = PROCESSED / "analyst_eval.json"
SAVED = [(r["id"], r["generated_sql"]) for r in json.loads(_saved_path.read_text())["results"] if r.get("generated_sql")] \
    if _saved_path.exists() else []

TYPED = """SELECT i,
    (i * 1.25)::DECIMAL(10,2) AS dec,
    (i::HUGEINT * 100000000000000000000) AS hug,
    DATE '2011-03-04' + i::INTEGER AS d,
    TIMESTAMP '2011-03-04 05:06:07' + to_seconds(i) AS ts,
    [i, i + 1] AS lst,
    {'a': i, 'b': 'x'} AS st,
    MAP {'k': i} AS mp,
    ('b' || i::VARCHAR)::BLOB AS bl,
    CASE WHEN i = 1 THEN NULL ELSE i END AS nul
FROM range(3) t(i) ORDER BY i"""


def test_there_are_thirty_reference_and_thirty_saved_queries():
    assert len(REFERENCE) == 30 and len(SAVED) == 30


@pytest.mark.parametrize("qid,sql", REFERENCE, ids=[q for q, _ in REFERENCE])
def test_reference_queries_give_the_same_result_as_the_plain_df_path(real_con, qid, sql):
    assert_same_as_plain_df(real_con, sql)


@pytest.mark.parametrize("qid,sql", SAVED, ids=[q for q, _ in SAVED])
def test_saved_generated_queries_give_the_same_result_as_the_plain_df_path(real_con, qid, sql):
    assert_same_as_plain_df(real_con, sql)


def test_a_query_with_every_awkward_type_gives_the_same_result_as_the_plain_df_path(con):
    assert_same_as_plain_df(con, TYPED)
    _, df = run_guarded(con, TYPED)
    assert str(df["d"].dtype).startswith("datetime64")                           # dates stay datetimes (Table.to_pandas would give objects)


# ═════════════════════ no warnings ═════════════════════
def test_no_warning_is_emitted_for_ordinary_empty_and_nested_results(con):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                           # any warning becomes a failure
        for sql in ("SELECT 1 AS n", "SELECT * FROM sales WHERE 1 = 0", TYPED, "SELECT range(10) AS r"):
            run_guarded(con, sql)
