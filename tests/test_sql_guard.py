"""The SQL guard is the safety boundary for LLM-written SQL, so it is tested with attacks as well as good queries."""
import json
import threading
import time
import types
import warnings

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
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
    "SELECT * FROM sales, range(3)",                               # range as a TABLE function stays allowed
    "SELECT * FROM range(3) AS t(i)",
    "WITH x AS (SELECT 1 AS a) SELECT * FROM x JOIN sales ON true",
    "SELECT s.invoice, c.segment FROM sales s JOIN customers c USING (customer_id)",   # regression: USING joins
    # WITH queries that must keep working under scope-aware parsing
    "WITH t AS (SELECT * FROM sales) SELECT * FROM (SELECT COUNT(*) AS n FROM t) x",                     # a WITH name inside a sub-query
    "WITH a AS (SELECT * FROM sales), b AS (SELECT * FROM a) SELECT COUNT(*) FROM b",                    # a chain
    "WITH RECURSIVE a AS (SELECT 1 AS n) SELECT * FROM a",                                               # the keyword alone is harmless: nothing recurses
    "WITH t AS (SELECT 1 AS a) SELECT a FROM t UNION ALL SELECT a FROM t",                               # both sides of UNION ALL
    "WITH a AS (WITH b AS (SELECT 1 AS x) SELECT x FROM b) SELECT * FROM a",                             # a WITH inside a WITH body
    "WITH t AS MATERIALIZED (SELECT * FROM sales LIMIT 5) SELECT * FROM t",
    "WITH sales AS (SELECT 1 AS a) SELECT * FROM sales",                                                 # shadowing an allowed table is harmless
    "SELECT [1, 2] AS xs",                                                                               # a list literal calls list_value, not list
]

# Task 2b: functions that can build one enormous value, and recursive WITH queries
NEW_DENIALS = {
    "range_as_a_scalar": "SELECT range(3)",
    "generate_series_as_a_scalar": "SELECT generate_series(1, 3)",
    "scalar_range_inside_unnest": "SELECT * FROM unnest(range(3))",
    "scalar_range_inside_a_range_argument": "SELECT * FROM range((SELECT len(range(5))))",
    "list_comprehension_over_range": "SELECT [x FOR x IN range(10)]",
    "quoted_upper_case_lpad": 'SELECT "LPAD"(\'x\', 5, \'x\')',
    "schema_qualified_lpad": "SELECT main.lpad('x', 5, 'x')",
    "window_form_json_group_array": "SELECT json_group_array(invoice) OVER () FROM sales",
    "list_resize": "SELECT list_resize([1], 5)",
    "array_resize": "SELECT array_resize([1], 5)",
    "json_group_array": "SELECT json_group_array(description) FROM sales",
    "json_group_object": "SELECT json_group_object(invoice, description) FROM sales",
    "lpad": "SELECT lpad('x', 5, 'x')",
    "rpad": "SELECT rpad('x', 5, 'x')",
    "printf": "SELECT printf('%5s', 'x')",
    "format": "SELECT format('{}', 'x')",
}
RECURSIVE = {
    "recursive_counter": "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
    "recursive_string_doubling":
        "WITH RECURSIVE r(i, s) AS (SELECT 0, 'x' UNION ALL SELECT i + 1, s || s FROM r WHERE i < 26) SELECT length(s) FROM r",
    "recursive_nested_inside_another_with":
        "WITH outer_q AS (WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT n FROM r) SELECT * FROM outer_q",
    "recursive_in_one_union_branch":
        "SELECT 1 AS n UNION ALL (WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT n FROM r)",
}

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
    **NEW_DENIALS,
    **RECURSIVE,
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
DENIED = ["string_agg", "group_concat", "listagg", "list", "array_agg", "histogram", "repeat",                  # Task 2
          "range", "generate_series", "list_resize", "array_resize", "json_group_array", "json_group_object",   # Task 2b
          "lpad", "rpad", "printf", "format"]
NEWLY_DENIED = DENIED[7:]


@pytest.mark.parametrize("name", DENIED)
def test_each_denied_function_is_refused_with_a_helpful_message(name):
    with pytest.raises(GuardError, match=r"counts or a top-N"):
        validate_sql(f"SELECT {name}(description) FROM sales")


def test_the_deny_list_has_exactly_seventeen_names():
    assert sql_guard.DENIED_FUNCTIONS == frozenset(DENIED) and len(DENIED) == 17


@pytest.mark.parametrize("name", ["range", "generate_series"])
def test_range_and_generate_series_are_allowed_as_table_functions_but_refused_as_scalars(name):
    for allowed in (f"SELECT * FROM {name}(1, 3)", f"SELECT * FROM {name}(1, 3) AS t(i)", f"SELECT * FROM sales, {name}(1, 3)"):
        validate_sql(allowed)
    for refused in (f"SELECT {name}(1, 3)", f"SELECT * FROM unnest({name}(1, 3))", f"SELECT * FROM range((SELECT len({name}(1, 3))))"):
        with pytest.raises(GuardError, match="counts or a top-N"):
            validate_sql(refused)


def test_the_function_reader_skips_only_the_name_a_table_function_owns():
    names = lambda q: sql_guard._function_names(sql_guard._syntax_tree(q))
    assert names("SELECT * FROM range(3)") == set()
    assert names("SELECT * FROM range(3) AS t(i)") == set()
    assert names("SELECT range(3)") == {"range"}
    assert names("SELECT * FROM unnest(range(3))") == {"range"}                          # unnest is owned; its argument is scalar
    assert names("SELECT * FROM range((SELECT len(range(5))))") == {"len", "range"}
    assert names("SELECT [x FOR x IN range(10)]") == {"list_apply", "range"}
    assert names("SELECT json_group_array(invoice) OVER () FROM sales") == {"json_group_array"}
    # parse_references keeps its meaning: the table functions are still reported there
    assert sql_guard.parse_references("SELECT * FROM range(3)") == (set(), {"range"})


# ═════════════════════ recursive WITH ═════════════════════
@pytest.mark.parametrize("name", list(RECURSIVE))
def test_recursive_with_queries_are_refused_with_the_documented_message(name):
    with pytest.raises(GuardError, match="Recursive WITH queries are not allowed, because they can grow without limit."):
        validate_sql(RECURSIVE[name])


# ═════════════════════ refused before the query is ever sent to the database ═════════════════════
class SpyReader:
    def __init__(self, reader, events):
        self._reader, self._events = reader, events

    def __iter__(self):
        return iter(self._reader)

    def close(self):
        self._events.append("reader.close")
        self._reader.close()

    def __getattr__(self, name):
        return getattr(self._reader, name)


class SpyRelation:
    def __init__(self, relation, events):
        self._relation, self._events = relation, events

    def to_arrow_reader(self, *args, **kwargs):
        reader = self._relation.to_arrow_reader(*args, **kwargs)     # raises instead when the error is in the first 2048-row vector
        self._events.append("reader.created")
        return SpyReader(reader, self._events)

    def __getattr__(self, name):
        return getattr(self._relation, name)


class SpyCursor:
    def __init__(self, cursor, events):
        self._cursor, self._events = cursor, events

    def sql(self, query, *args, **kwargs):
        self._events.append("sql")
        return SpyRelation(self._cursor.sql(query, *args, **kwargs), self._events)

    def close(self):
        self._events.append("cursor.close")
        self._cursor.close()

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class SpyConnection:
    """Wraps a connection and records what run_guarded does with its cursor, reader and timer."""

    def __init__(self, connection, events):
        self._connection, self._events = connection, events

    def cursor(self):
        return SpyCursor(self._connection.cursor(), self._events)


@pytest.fixture
def events(monkeypatch):
    """The event log, with sql_guard's timer replaced by one that records start and cancel."""
    log = []

    class SpyTimer(threading.Timer):
        def start(self):
            log.append("timer.start")
            super().start()

        def cancel(self):
            log.append("timer.cancel")
            super().cancel()

    monkeypatch.setattr(sql_guard, "threading", types.SimpleNamespace(Timer=SpyTimer))
    return log


def assert_cleaned_up(log: list, ran: bool, reader: bool = True) -> None:
    """Every outcome must cancel the timer, then close the reader (if one was made) and the cursor, exactly once.

    `reader` says whether a reader is expected to exist. DuckDB runs the first 2048 rows while the reader is being
    created, so an error in those rows (and an interrupt of a long aggregate) is raised before any reader exists.
    Either way, every reader that was created must have been closed.
    """
    assert log.count("cursor.close") == 1 and log.count("timer.cancel") == 1
    assert log.count("reader.close") == log.count("reader.created")
    if ran:
        assert log.count("timer.start") == 1 and log.count("sql") == 1
        assert log.count("reader.created") == (1 if reader else 0)
        if reader:
            assert log.index("timer.cancel") < log.index("reader.close") < log.index("cursor.close")
        else:
            assert log.index("timer.cancel") < log.index("cursor.close")
    else:
        assert "timer.start" not in log and "reader.created" not in log and "sql" not in log
        assert log.index("timer.cancel") < log.index("cursor.close")


SPY_CASES = [*NEW_DENIALS, *RECURSIVE, "string_agg", "group_concat", "listagg", "list", "array_agg", "histogram", "repeat",
             "schema_qualified_string_agg", "the_huge_aggregate"]


@pytest.mark.parametrize("name", SPY_CASES)
def test_a_denied_query_never_reaches_the_database(con, events, name):
    with pytest.raises(GuardError):
        run_guarded(SpyConnection(con, events), BAD[name])
    assert_cleaned_up(events, ran=False)                           # no sql, no reader, the timer never started


def test_a_harmless_query_does_reach_the_database(con, events):
    run_guarded(SpyConnection(con, events), "SELECT 1 AS n")
    assert events.count("sql") == 1
    assert_cleaned_up(events, ran=True)


# ═════════════════════ result-size caps: refused while streaming, before pandas ═════════════════════
# The big values are built by SETUP SQL that runs outside the guard (denied functions are fine there). The guarded
# query is then a plain SELECT of the stored value.
def connection_with(value_sql: str, rows: int = 1) -> duckdb.DuckDBPyConnection:
    """A plain connection whose allowed-name table `sales` has one column `v` holding `value_sql` in `rows` rows."""
    connection = duckdb.connect()
    connection.execute(f"CREATE TABLE sales AS SELECT {value_sql} AS v FROM range({rows})")
    return connection


READ = "SELECT v FROM sales"

OVER_THE_CAPS = {                                              # name: (value built by setup SQL, rows, expected message)
    "total": ("repeat('x', 5000)", 1000, "larger than 2 MB"),
    "one_cell": ("repeat('x', 50000)", 1, "single value"),
    "one_cell_just_over": ("repeat('x', 10001)", 1, "single value"),
    "binary": ("repeat('x', 50000)::BLOB", 1, "single value"),
    "in_a_list": ("[repeat('x', 20000)]", 1, "single value"),
    "in_a_struct": ("{'a': repeat('x', 20000)}", 1, "single value"),
    "in_a_map_value": ("MAP {'k': repeat('x', 20000)}", 1, "single value"),
    "in_a_map_key": ("MAP {repeat('x', 20000): 1}", 1, "single value"),
    "deeply_nested": ("[{'a': [repeat('x', 20000)]}]", 1, "single value"),
    "in_a_union": ("union_value(k := repeat('x', 50000))", 1, "single value"),
    "bignum": ("repeat('9', 25000)::BIGNUM", 1, "single value"),          # an Arrow extension type, about 10,385 bytes
    "list_items": ("range(5000)", 1, "list"),
    "list_just_over": ("range(1001)", 1, "list"),
    "list_in_list": ("[range(5000)]", 1, "list"),
    "list_in_struct": ("{'a': range(5000)}", 1, "list"),
    "fixed_size_array": ("CAST(range(2000) AS INTEGER[2000])", 1, "list"),
}

WITHIN_THE_CAPS = {
    "exactly_the_cell_limit": ("repeat('x', 10000)", 1),
    "exactly_the_list_limit": ("range(1000)", 1),
    "list_in_list_at_the_limit": ("[range(1000)]", 1),
    "fixed_size_array_at_the_limit": ("CAST(range(1000) AS INTEGER[1000])", 1),
    "union_at_the_cell_limit": ("union_value(k := repeat('x', 10000))", 1),
    "all_null_strings": ("NULL::VARCHAR", 10),
    "all_null_lists": ("NULL::VARCHAR[]", 10),
    "about_one_megabyte_in_total": ("repeat('x', 1000)", 1000),     # under the 2 MB cap
}


def test_the_limits_are_the_documented_ones():
    assert (config.ANALYST_MAX_RESULT_BYTES, config.ANALYST_MAX_CELL_BYTES, config.ANALYST_MAX_LIST_ITEMS,
            config.ANALYST_BATCH_ROWS, config.ANALYST_SUMMARY_MAX_CHARS) == (2_000_000, 10_000, 1_000, 256, 4_000)


@pytest.mark.parametrize("name", list(OVER_THE_CAPS))
def test_results_over_a_cap_are_refused(name):
    value, rows, expected = OVER_THE_CAPS[name]
    with pytest.raises(GuardError, match=expected):
        run_guarded(connection_with(value, rows), READ)


@pytest.mark.parametrize("name", list(WITHIN_THE_CAPS))
def test_results_exactly_at_a_cap_are_allowed(name):
    value, rows = WITHIN_THE_CAPS[name]
    assert len(run_guarded(connection_with(value, rows), READ)[1]) == rows


@pytest.mark.parametrize("name", list(OVER_THE_CAPS))
def test_an_over_cap_result_never_reaches_pandas(monkeypatch, name):
    def reached_pandas(*args, **kwargs):
        raise AssertionError("an over-cap result reached the pandas conversion")

    monkeypatch.setattr(sql_guard, "_to_dataframe", reached_pandas)
    value, rows, _ = OVER_THE_CAPS[name]
    with pytest.raises(GuardError):
        run_guarded(connection_with(value, rows), READ)


def test_a_result_within_the_caps_does_go_through_the_conversion(con, monkeypatch):
    calls, original = [], sql_guard._to_dataframe
    monkeypatch.setattr(sql_guard, "_to_dataframe", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    run_guarded(con, "SELECT 1 AS n")
    assert calls == [1]


def test_the_total_cap_stops_the_stream_early_instead_of_reading_it_all(monkeypatch):
    """5 MB of results, 256 rows per batch: the cap trips on the second batch, long before the end."""
    seen, original = [], sql_guard._check_batch
    monkeypatch.setattr(sql_guard, "_check_batch", lambda batch: (seen.append(batch.num_rows), original(batch))[1])
    value, rows, expected = OVER_THE_CAPS["total"]
    with pytest.raises(GuardError, match=expected):
        run_guarded(connection_with(value, rows), READ)
    assert sum(seen) <= 512


# ═════════════════════ Arrow type coverage: synthetic arrays straight into _check_array ═════════════════════
LIMIT = config.ANALYST_MAX_CELL_BYTES


def dense_union(n):
    return pa.UnionArray.from_dense(pa.array([0, 1], pa.int8()), pa.array([0, 0], pa.int32()),
                                    [pa.array(["x" * n]), pa.array([1])], ["s", "i"])


def sparse_union(n):
    return pa.UnionArray.from_sparse(pa.array([0, 1], pa.int8()), [pa.array(["x" * n, None]), pa.array([None, 1])], ["s", "i"])


OPAQUE = pa.opaque(pa.binary(), "bignum", "DuckDB")           # the type DuckDB uses to hand a BIGNUM to Arrow

SYNTHETIC = {                                                  # builders that put one string/bytes value of n bytes in the array
    "dense_union": dense_union,
    "sparse_union": sparse_union,
    "string_view": lambda n: pa.array(["x" * n, None], pa.string_view()),
    "binary_view": lambda n: pa.array([b"x" * n, None], pa.binary_view()),
    "fixed_size_binary": lambda n: pa.array([b"x" * n, None], pa.binary(n)),
    "dictionary": lambda n: pa.DictionaryArray.from_arrays(pa.array([0, None, 1], pa.int8()), pa.array(["x" * n, "y"])),
    "extension_over_binary": lambda n: pa.ExtensionArray.from_storage(OPAQUE, pa.array([b"x" * n, None], pa.binary())),
}

ALL_NULL = {                                                   # arrays with no value at all, even where the declared width is huge
    "dense_union": pa.UnionArray.from_dense(pa.array([0, 1], pa.int8()), pa.array([0, 0], pa.int32()),
                                            [pa.array([None], pa.string()), pa.array([None], pa.int32())], ["s", "i"]),
    "sparse_union": pa.UnionArray.from_sparse(pa.array([0, 1], pa.int8()), [pa.array([None, None], pa.string()), pa.array([None, None], pa.int32())], ["s", "i"]),
    "string_view": pa.array([None, None], pa.string_view()),
    "binary_view": pa.array([None, None], pa.binary_view()),
    "fixed_size_binary": pa.array([None, None], pa.binary(LIMIT * 2)),
    "dictionary": pa.DictionaryArray.from_arrays(pa.array([None, None], pa.int8()), pa.array(["y"])),
    "extension_over_binary": pa.ExtensionArray.from_storage(OPAQUE, pa.array([None, None], pa.binary())),
}


@pytest.mark.parametrize("kind", list(SYNTHETIC))
def test_an_oversized_value_is_refused_in_every_arrow_type(kind):
    with pytest.raises(GuardError, match="single value"):
        sql_guard._check_array(SYNTHETIC[kind](LIMIT + 1), "v")


@pytest.mark.parametrize("kind", list(SYNTHETIC))
def test_a_value_exactly_at_the_limit_is_allowed_in_every_arrow_type(kind):
    sql_guard._check_array(SYNTHETIC[kind](LIMIT), "v")


@pytest.mark.parametrize("kind", list(ALL_NULL))
def test_arrays_with_only_nulls_are_allowed_in_every_arrow_type(kind):
    sql_guard._check_array(ALL_NULL[kind], "v")


def test_view_types_are_cast_first_because_binary_length_has_no_kernel_for_them():
    """pc.binary_length raises ArrowNotImplementedError on a view type; the guard must refuse cleanly instead."""
    import pyarrow.compute as pc
    with pytest.raises(pa.ArrowNotImplementedError):
        pc.binary_length(pa.array(["x"], pa.string_view()))
    with pytest.raises(GuardError):
        sql_guard._check_array(pa.array(["x" * (LIMIT + 1)], pa.string_view()), "v")


def test_a_fixed_size_binary_wider_than_the_limit_is_refused_only_when_it_holds_a_value():
    with pytest.raises(GuardError, match="single value"):
        sql_guard._check_array(pa.array([b"\x00" * (LIMIT + 1)], pa.binary(LIMIT + 1)), "v")
    sql_guard._check_array(pa.array([None], pa.binary(LIMIT + 1)), "v")


def bignum_arrow_array(digits: int) -> pa.Array:
    """A BIGNUM column exactly as DuckDB delivers it to Arrow (read straight from the database, not through the guard)."""
    reader = connection_with(f"repeat('9', {digits})::BIGNUM").cursor().sql(READ).to_arrow_reader(256)
    return reader.read_all().column(0).chunk(0)


def test_duckdb_hands_a_bignum_to_arrow_as_an_extension_type_over_binary():
    """This is why _check_array unwraps extension types: without it, a BIGNUM matched no branch and was never measured."""
    array = bignum_arrow_array(25000)
    assert isinstance(array.type, pa.BaseExtensionType) and array.storage.type == pa.binary()
    assert len(array.storage[0].as_py()) > config.ANALYST_MAX_CELL_BYTES


def test_a_real_bignum_over_the_cell_limit_is_refused_and_one_under_it_is_not():
    with pytest.raises(GuardError, match="single value"):
        sql_guard._check_array(bignum_arrow_array(25000), "v")                 # 10,385 bytes
    sql_guard._check_array(bignum_arrow_array(24000), "v")                     # 9,969 bytes
    sql_guard._check_array(pa.ExtensionArray.from_storage(OPAQUE, pa.array([None], pa.binary())), "v")


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


QUESTIONS = json.loads((ROOT / "evals" / "analyst_questions.json").read_text())["questions"]
REFERENCE = [(q["id"], q["gold_sql"]) for q in QUESTIONS]
_saved_path = PROCESSED / "analyst_eval.json"
_saved_rows = json.loads(_saved_path.read_text())["results"] if _saved_path.exists() else []
# only rows whose FINAL query ran: a correctly blocked attempt or a safety case is never a guard regression
SAVED = [(r["id"], r["generated_sql"]) for r in _saved_rows if r.get("generated_sql") and r.get("error") is None]

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


def test_there_is_one_reference_query_per_question_and_one_saved_query_per_executed_result():
    assert len(REFERENCE) == len(QUESTIONS) == 40
    assert len(SAVED) == sum(1 for r in _saved_rows if r["executed"])


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
        for sql in ("SELECT 1 AS n", "SELECT * FROM sales WHERE 1 = 0", TYPED, "SELECT [1, 2, 3] AS r"):
            run_guarded(con, sql)


# ═════════════════════ execution and cleanup ═════════════════════
def test_a_slow_streaming_query_is_stopped_within_seconds_and_never_reaches_pandas(real_con, monkeypatch):
    def reached_pandas(*args, **kwargs):
        raise AssertionError("a timed-out query reached the pandas conversion")

    monkeypatch.setattr(sql_guard, "_to_dataframe", reached_pandas)
    started = time.time()
    with pytest.raises(GuardError, match="stopped"):
        run_guarded(real_con, "SELECT a.invoice FROM sales a, sales b WHERE hash(a.invoice || b.invoice) % 1000000007 = 3", timeout_s=2)
    assert time.time() - started < 4.5                              # about the 2 s timeout, not the minutes the query would take


def test_cleanup_after_a_successful_query(con, events):
    run_guarded(SpyConnection(con, events), "SELECT 1 AS n")
    assert_cleaned_up(events, ran=True)


def test_cleanup_after_a_size_rejection(events):
    value, rows, expected = OVER_THE_CAPS["total"]
    with pytest.raises(GuardError, match=expected):
        run_guarded(SpyConnection(connection_with(value, rows), events), READ)
    assert_cleaned_up(events, ran=True)


def test_cleanup_after_a_timeout(con, events):
    with pytest.raises(GuardError, match="stopped"):
        run_guarded(SpyConnection(con, events), "SELECT COUNT(*) FROM range(1000000000) a, range(1000000000) b", timeout_s=0.5)
    assert_cleaned_up(events, ran=True, reader=False)              # the aggregate is interrupted while the reader is being created


def test_cleanup_after_a_deny_list_rejection(con, events):
    with pytest.raises(GuardError):
        run_guarded(SpyConnection(con, events), "SELECT lpad('x', 5, 'x')")
    assert_cleaned_up(events, ran=False)                           # cursor closed, no reader was created, the timer never started


def test_cleanup_after_a_conversion_error(con, events, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(sql_guard, "_to_dataframe", broken)
    with pytest.raises(RuntimeError, match="conversion failed"):
        run_guarded(SpyConnection(con, events), "SELECT 1 AS n")
    assert_cleaned_up(events, ran=True)


LATE_ERROR = "SELECT CASE WHEN i < 600 THEN i ELSE error('late') END AS x FROM range(2000) t(i)"
# the failing row is in the THIRD 2048-row vector, so the reader exists and the error is raised while iterating it
LATER_ERROR = "SELECT i FROM range(10000) t(i) WHERE (CASE WHEN i < 5000 THEN i ELSE error('late') END) % 10 = 0"


def test_a_runtime_error_in_the_first_vector_is_a_duckdb_error_and_everything_is_cleaned_up(con, events):
    with pytest.raises(duckdb.Error) as error:
        run_guarded(SpyConnection(con, events), LATE_ERROR)
    assert not isinstance(error.value, (GuardError, pa.ArrowException))   # the repair loop catches duckdb.Error
    assert "late" in str(error.value)
    assert_cleaned_up(events, ran=True, reader=False)                     # raised while the reader was being created


def test_a_runtime_error_while_iterating_the_reader_is_a_duckdb_error_and_everything_is_cleaned_up(con, events):
    with pytest.raises(duckdb.Error) as error:
        run_guarded(SpyConnection(con, events), LATER_ERROR)
    assert not isinstance(error.value, (GuardError, OSError, pa.ArrowException))
    assert "late" in str(error.value)
    assert_cleaned_up(events, ran=True, reader=True)                      # this time a reader existed and was closed


def test_the_raw_reader_raises_an_oserror_for_that_late_error_which_run_guarded_converts(con):
    """Pins down WHY _read_batches exists: DuckDB itself delivers this error as an OSError, not a duckdb.Error."""
    reader = con.cursor().sql(sql_guard.wrap_with_limit(LATER_ERROR)).to_arrow_reader(256)
    with pytest.raises(OSError, match="late"):
        list(reader)
    with pytest.raises(duckdb.Error) as error:
        run_guarded(con, LATER_ERROR)
    assert error.value.__cause__ is not None and isinstance(error.value.__cause__, OSError)


# ═════════════════════ the eval statements must not be affected ═════════════════════
def recorded_eval_statements() -> set:
    """Every reference and generated SQL string stored in the repository's eval files."""
    found = set()
    paths = [ROOT / "evals" / "analyst_questions.json", *sorted((ROOT / "evals" / "history").glob("*.json")), PROCESSED / "analyst_eval.json"]

    def collect(node):
        if isinstance(node, dict):
            ran = node.get("error") is None                   # a generated statement counts only if its final query ran
            for key, value in node.items():
                if (key == "gold_sql" or (key == "generated_sql" and ran)) and isinstance(value, str) and value.strip():
                    found.add(value.strip())
                else:
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    for path in paths:
        if path.exists():
            collect(json.loads(path.read_text()))
    return found


def test_every_recorded_eval_statement_still_passes_the_guard():
    statements = recorded_eval_statements()
    assert len(statements) >= 80
    for sql in sorted(statements):
        try:
            validate_sql(sql)
        except GuardError as error:
            pytest.fail(f"{error} <- {sql[:100]}")


# ═════════════════════ the documentation says what the guard does not guarantee ═════════════════════
SAFETY_NETS = ("The DuckDB memory limit and the timeout are safety nets, not hard isolation. The memory limit does not cap "
               "every allocation (large lists can exceed it), and the timer cannot interrupt a single long-running function.")
ABUSE_CONTROLS = ("The question limits (15 per session, 40 per hour across the app) protect the free LLM quota and slow down "
                  "repeated abuse. They do not protect against a single expensive query: one query is enough to cause a large "
                  "memory spike.")
ISOLATION = ("The robust protection would be process isolation: running each analyst query in a separate worker process with an "
             "operating-system memory cap, killed outright when it runs too long. That is outside the scope of this project.")


def squashed(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("where", ["module docstring", "Ask page"])
def test_the_limits_of_the_guard_are_stated_plainly(where):
    text = squashed(sql_guard.__doc__ if where == "module docstring" else (ROOT / "app" / "pages" / "4_Ask_the_Data.py").read_text())
    for statement in (SAFETY_NETS, ABUSE_CONTROLS, ISOLATION):
        assert statement in text
    for name in NEWLY_DENIED:
        assert name in text, f"{name} is not listed in the {where}"
    assert "Recursive WITH" in text or "recursive WITH" in text
