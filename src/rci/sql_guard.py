"""The safety layer between an LLM and the database.

An LLM can write wrong SQL, or SQL that a malicious question talked it into ("ignore your rules and
drop the table"). We never run its output directly. Each query passes these layers:

  1. PARSE        DuckDB's own parser must accept it. It must be exactly ONE statement, and that
                  statement's type must be SELECT (this rejects DROP, INSERT, COPY, ATTACH, PRAGMA, SET...).
  2. TABLES       Every table it reads (found by walking DuckDB's syntax tree of the query) must be on
                  the allow-list: sales, returns, products, customers. A name defined with WITH counts only
                  inside the WITH that defines it, a qualified name (main.x) is always a real table, and a
                  WITH may not reuse the name of a system or catalog view (sqlite_master, duckdb_tables...),
                  so a WITH can never be used to disguise a read of one.
  3. FUNCTIONS    Only harmless table functions (range, generate_series, unnest) are allowed; file-reading
                  and system functions (read_csv, glob, duckdb_settings, ...) are refused, because they
                  are SELECTs that can still read the server's files. Functions that can build one enormous
                  value (string_agg, group_concat, listagg, list, array_agg, histogram, repeat) are refused
                  before the query runs: a size cap on the result cannot stop the engine building it first.
  4. ROW CAP      The query is wrapped in `SELECT * FROM (...) LIMIT n`, so it can never return a flood.
  5. RESULT SIZE  The result streams in batches of 256 rows and each batch is checked BEFORE it is kept:
                  2 MB in total, 10 KB in any single value (also inside lists, structs and maps) and 1,000
                  items in any list. Nothing over those limits reaches pandas, Streamlit or the LLM.
  6. TIMEOUT      A timer interrupts any query that runs too long (e.g. an accidental cross join).
  7. LOCK         db.connect() already switched off all file access and locked the settings, so even if
                  layers 1-3 had a hole the database could not touch the disk.
"""
from __future__ import annotations

import json
import re
import threading

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from rci import config

ALLOWED_TABLES = frozenset({"sales", "returns", "products", "customers"})
ALLOWED_TABLE_FUNCTIONS = frozenset({"range", "generate_series", "unnest"})  # harmless row generators
DENIED_FUNCTIONS = frozenset({"string_agg", "group_concat", "listagg", "list", "array_agg", "histogram", "repeat"})
RESERVED_WITH_PREFIXES = ("duckdb_", "sqlite_", "pragma_", "pg_", "information_schema")
MAX_SQL_CHARS = 4000

# A separate, empty DuckDB used only to PARSE queries into a syntax tree. It holds no data, so
# parsing can never touch the real tables, and it needs no tables to exist to parse a query.
_PARSER = duckdb.connect(":memory:")

# The names of DuckDB's own views and tables (sqlite_master, duckdb_tables, pg_settings, tables, columns...),
# read once from the empty parser connection. A WITH query may not borrow any of them.
RESERVED_WITH_NAMES = frozenset(
    str(name).lower() for (name,) in _PARSER.execute(
        "SELECT view_name FROM duckdb_views() UNION SELECT table_name FROM duckdb_tables()").fetchall()
)

# An identifier followed by "(" is a function call. These families read files or leak system info.
_BLOCKED_CALL = re.compile(
    r"\b(?:read_\w+|\w+_scan|glob|sniff_csv|query|query_table|duckdb_\w+|pragma_\w+|"
    r"parquet_\w+|getenv|current_setting|load_extension|install_extension)\s*\(",
    re.IGNORECASE,
)


class GuardError(ValueError):
    """The query was rejected before it ran. The message is safe to show to the user."""


# ── reading the syntax tree ──────────────────────────────────────────────────
def _syntax_tree(sql: str) -> dict:
    """DuckDB's own parse of a SELECT, as nested dicts. Parsed once per query; needs no tables to exist."""
    try:
        tree = json.loads(_PARSER.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    except duckdb.Error as e:
        raise GuardError(f"This is not valid SQL: {_short(e)}") from e
    if tree.get("error"):
        raise GuardError(f"This is not valid SQL: {str(tree.get('error_message', ''))[:200]}")
    return tree


def _table_name(node: dict) -> tuple[str, bool]:
    """(lower-cased name, is it qualified?). A table in a non-default schema or catalog is reported as
    'catalog.schema.table' so it can never match an allowed name; main.sales is the allowed table `sales`."""
    table = str(node["table_name"]).lower()
    schema, catalog = str(node.get("schema_name", "")).lower(), str(node.get("catalog_name", "")).lower()
    qualified = bool(schema or catalog)
    if schema not in ("", "main") or catalog not in ("", "memory"):
        table = ".".join(part for part in (catalog, schema, table) if part)
    return table, qualified


def _read_tables(tree: dict) -> tuple[set[str], set[str]]:
    """(tables read, table functions called), with WITH names handled by SCOPE.

    A node that defines WITH names makes them visible only inside its own subtree (the WITH bodies and
    the query that follows). An UNQUALIFIED table counts as a WITH reference only when its name is in
    scope there; a qualified one (main.x, other.main.x) is always a real table.
    """
    tables: set[str] = set()
    table_functions: set[str] = set()

    def walk(node, scope: frozenset) -> None:
        if isinstance(node, dict):
            defined = (node.get("cte_map") or {}).get("map") or []
            if defined:
                scope = scope | {str(entry["key"]).lower() for entry in defined}
            kind = node.get("type")
            if kind == "BASE_TABLE":
                name, qualified = _table_name(node)
                if qualified or name not in scope:
                    tables.add(name)
            elif kind == "TABLE_FUNCTION":
                table_functions.add(str(node.get("function", {}).get("function_name", "?")).lower())
            for value in node.values():
                walk(value, scope)
        elif isinstance(node, list):
            for value in node:
                walk(value, scope)

    walk(tree, frozenset())
    return tables, table_functions


def _with_names(tree: dict) -> set[str]:
    """Every name the query defines with WITH, anywhere in it, lower-cased."""
    names: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for entry in (node.get("cte_map") or {}).get("map") or []:
                names.add(str(entry["key"]).lower())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(tree)
    return names


def _function_names(tree: dict) -> set[str]:
    """Every function the query calls, by its bare lower-cased name (so main.string_agg is string_agg)."""
    names: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if "function_name" in node:
                names.add(str(node["function_name"]).lower())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(tree)
    return names


def parse_references(sql: str) -> tuple[set[str], set[str]]:
    """Return (tables read, table functions called) by walking DuckDB's syntax tree of the query.

    WITH names are handled by scope (see `_read_tables`), not by subtracting them from the whole result.
    """
    return _read_tables(_syntax_tree(sql))


def _strip_comments_and_strings(sql: str) -> str:
    """Remove comments and 'string literals' so that text such as LIKE '%read_csv(%' is not
    mistaken for a function call. Double quotes are dropped so "read_csv"( is still caught."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return sql.replace('"', "")


def _unusable_with_name(name: str, allowed: set[str], reserved: set[str]) -> bool:
    """A WITH name is refused when it could be mistaken for, or could hide, a real catalog object."""
    return (
        "." in name
        or name.startswith(RESERVED_WITH_PREFIXES)
        or name in RESERVED_WITH_NAMES
        or (name in reserved and name not in allowed)  # reusing an ALLOWED table's name is harmless
    )


def validate_sql(sql: str, allowed_tables=ALLOWED_TABLES, reserved_names=frozenset()) -> str:
    """Return the cleaned SQL (no trailing semicolon) or raise GuardError.

    `reserved_names` are extra names a WITH may not use (run_guarded passes the data connection's own
    table and view names); DuckDB's built-in view names are always reserved.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise GuardError("The query is empty.")
    sql = sql.strip().rstrip(";").strip()
    if len(sql) > MAX_SQL_CHARS:
        raise GuardError("The query is too long.")

    # 1. parse: one statement, and it must be a SELECT
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as e:
        raise GuardError(f"This is not valid SQL: {_short(e)}") from e
    if len(statements) != 1:
        raise GuardError("Only a single SQL statement is allowed.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise GuardError("Only read-only SELECT queries are allowed.")

    # 2 + 3. tables, WITH names and functions, all read from one syntax tree (not guessed from the text)
    tree = _syntax_tree(sql)
    tables, table_functions = _read_tables(tree)
    allowed = {t.lower() for t in allowed_tables}

    if table_functions - ALLOWED_TABLE_FUNCTIONS or _BLOCKED_CALL.search(_strip_comments_and_strings(sql)):
        raise GuardError("This query uses a file or system function, which is not allowed.")
    denied = sorted(_function_names(tree) & DENIED_FUNCTIONS)
    if denied:
        raise GuardError(f"The function '{denied[0]}' is not allowed because it can build one enormous value. "
                         "Ask for counts or a top-N instead.")
    reserved = {n.lower() for n in reserved_names}
    for name in sorted(_with_names(tree)):
        if _unusable_with_name(name, allowed, reserved):
            raise GuardError(f"The name '{name}' can't be used for a WITH query.")
    unknown = tables - allowed
    if unknown:
        raise GuardError(f"Unknown or forbidden table(s): {', '.join(sorted(unknown))}. "
                         f"Allowed: {', '.join(sorted(allowed_tables))}.")
    return sql


def wrap_with_limit(sql: str, max_rows: int = config.ANALYST_MAX_ROWS) -> str:
    """Force a row cap. The newline keeps a trailing `-- comment` from swallowing the bracket."""
    return f"SELECT * FROM (\n{sql}\n) AS guarded LIMIT {int(max_rows)}"


# ── result size: checked batch by batch, before anything is kept ─────────────
def _check_array(array: pa.Array, column: str) -> None:
    """Refuse an oversized value anywhere inside this column, however deeply it is nested."""
    kind = array.type
    if pa.types.is_dictionary(kind):
        _check_array(array.dictionary, column)
    elif pa.types.is_string(kind) or pa.types.is_large_string(kind) or pa.types.is_binary(kind) or pa.types.is_large_binary(kind):
        longest = pc.max(pc.binary_length(array)).as_py() or 0          # None (all values null) counts as 0
        if longest > config.ANALYST_MAX_CELL_BYTES:
            raise GuardError(f"A single value in column '{column}' is larger than {config.ANALYST_MAX_CELL_BYTES:,} bytes. "
                             "Ask for something shorter, such as a count or the first few characters.")
    elif pa.types.is_map(kind):
        _check_array(array.keys, column)
        _check_array(array.items, column)
    elif pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
        longest = pc.max(pc.list_value_length(array)).as_py() or 0
        if longest > config.ANALYST_MAX_LIST_ITEMS:
            raise GuardError(f"A list in column '{column}' has more than {config.ANALYST_MAX_LIST_ITEMS:,} items. "
                             "Ask for a count or a top-N instead.")
        _check_array(array.flatten(), column)
    elif pa.types.is_struct(kind):
        for i in range(kind.num_fields):
            _check_array(array.field(i), column)


def _check_batch(batch: pa.RecordBatch) -> None:
    for column, array in zip(batch.schema.names, batch.columns):
        _check_array(array, column)


def _to_dataframe(cursor, batches: list, schema: pa.Schema) -> pd.DataFrame:
    """Accepted Arrow batches -> pandas, through DuckDB itself on THIS cursor.

    That gives the same dtypes as `.df()` (dates stay datetimes). `pa.Table.to_pandas()` would turn dates into
    objects, and `duckdb.from_arrow` would use a module-wide default connection shared by every visitor.
    The schema is passed because an empty result produces zero batches.
    """
    return cursor.from_arrow(pa.Table.from_batches(batches, schema=schema)).df()


def _timeout_error(timeout_s: float) -> GuardError:
    return GuardError(f"The query ran longer than {timeout_s:.0f} seconds and was stopped.")


def _catalog_names(cursor) -> set[str]:
    """Names of every table and view on the data connection, lower-cased."""
    rows = cursor.execute("SELECT table_name FROM duckdb_tables() UNION SELECT view_name FROM duckdb_views()").fetchall()
    return {str(name).lower() for (name,) in rows}


def run_guarded(con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = config.ANALYST_MAX_ROWS,
                timeout_s: float = config.ANALYST_TIMEOUT_S) -> tuple[str, pd.DataFrame]:
    """Validate, cap, stream with limits and a timeout. Returns (the validated SQL, the result)."""
    cur = con.cursor()
    timer = threading.Timer(timeout_s, cur.interrupt)
    reader = None
    try:
        safe = validate_sql(sql, reserved_names=_catalog_names(cur) - ALLOWED_TABLES)
        timer.start()
        # cur.sql(...) streams. cur.execute(...).to_arrow_reader(...) would warn and copy every row first.
        reader = cur.sql(wrap_with_limit(safe, max_rows)).to_arrow_reader(config.ANALYST_BATCH_ROWS)
        batches, total_bytes = [], 0
        for batch in reader:
            total_bytes += batch.nbytes                       # check BEFORE keeping the batch
            if total_bytes > config.ANALYST_MAX_RESULT_BYTES:
                raise GuardError(f"The result is larger than {config.ANALYST_MAX_RESULT_BYTES / 1_000_000:g} MB. "
                                 "Ask for fewer rows or columns, a count or a top-N instead.")
            _check_batch(batch)
            batches.append(batch)
        df = _to_dataframe(cur, batches, reader.schema)
    except duckdb.InterruptException as e:
        raise _timeout_error(timeout_s) from e
    except (pa.ArrowException, OSError) as e:
        if "interrupt" in str(e).lower():                    # an interrupt can surface as an Arrow error while streaming
            raise _timeout_error(timeout_s) from e
        raise
    finally:
        timer.cancel()
        for resource in (reader, cur):
            if resource is not None:
                try:
                    resource.close()
                except Exception:  # noqa: BLE001  (cleanup must never hide the real error)
                    pass
    return safe, df


def _short(e: Exception) -> str:
    return str(e).splitlines()[0][:200]
