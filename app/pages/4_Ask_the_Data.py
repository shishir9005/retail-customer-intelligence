"""AI data analyst: plain-English question -> LLM writes SQL -> guard -> DuckDB -> answer (SQL always shown)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find app/common.py when run on its own

import os

import common
import pandas as pd
import plotly.express as px
import streamlit as st
from rci import analyst, config
from rci.ratelimit import SlidingWindowLimiter

common.setup("Ask the Data")
st.title("Ask the data")
st.write(
    f"Type a question in plain English. An LLM (`{config.GROQ_MODEL}` on Groq) writes SQL from the **table schema only**: it never "
    "sees the data. A guard then checks the SQL is a single read-only query on known tables, and DuckDB runs it. "
    "The SQL is always shown, so every answer can be checked."
)

MAX_PER_SESSION = 15
EXAMPLES = [
    "What are the top 5 products by revenue?",
    "How many customers are in each segment?",
    "Which country has the highest revenue outside the United Kingdom?",
    "What was the revenue in each month of 2011?",
    "What percentage of revenue comes from guest checkouts?",
    "Which segment generates the most revenue?",
]


def api_key() -> str | None:
    config.load_env()
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:  # noqa: BLE001  (no secrets file when running locally without one)
        return None


@st.cache_resource
def answer_cache() -> dict:
    return {}


@st.cache_resource
def global_limiter() -> SlidingWindowLimiter:
    return SlidingWindowLimiter(max_calls=40, window_s=3600)  # protects the free Groq quota for all visitors


def auto_chart(df: pd.DataFrame):
    """A bar chart when the result is one label column plus numbers; a line when the label is a date."""
    if not 2 <= len(df) <= 40 or not 2 <= df.shape[1] <= 4:
        return None
    x, ys = df.columns[0], [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if not ys or pd.api.types.is_numeric_dtype(df[x]):
        return None
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        return px.line(df, x=x, y=ys, markers=True)
    return px.bar(df, x=x, y=ys, barmode="group")


def render(ans: analyst.Answer) -> None:
    if ans.unanswerable:
        st.info(ans.summary)
        return
    if ans.data is None:
        st.error(ans.error or "The question could not be answered.")
        if ans.sql:
            st.code(ans.sql, language="sql")
        return
    st.markdown(f"**{ans.summary}**")
    st.code(ans.sql, language="sql")
    st.dataframe(ans.data, hide_index=True)
    fig = auto_chart(ans.data)
    if fig is not None:
        st.plotly_chart(common.style(fig, height=340))
    notes = []
    if ans.attempts > 1:
        notes.append("The first query failed or was rejected, so the model was shown the error and repaired it.")
    if len(ans.data) >= config.ANALYST_MAX_ROWS:
        notes.append(f"Results are capped at {config.ANALYST_MAX_ROWS:,} rows.")
    if notes:
        st.caption(" ".join(notes))


def use_example(q: str) -> None:
    st.session_state["question"] = q
    st.session_state["pending"] = q


# ── ask ───────────────────────────────────────────────────────────────────────
key = api_key()
if not key:
    st.info("This demo needs a `GROQ_API_KEY` (a free key from console.groq.com) in `.env` or in the app's secrets to answer live questions. "
            "The measured accuracy and the safety design are shown below.")
else:
    st.write("**Try an example**")
    cols = st.columns(3)
    for i, q in enumerate(EXAMPLES):
        cols[i % 3].button(q, key=f"example{i}", on_click=use_example, args=(q,))
    with st.form("ask_form"):
        typed = st.text_input("Your question", key="question", placeholder="e.g. Which 5 customers spent the most?")
        submitted = st.form_submit_button("Ask", type="primary")
    question = st.session_state.pop("pending", None) or (typed if submitted else None)

    st.session_state.setdefault("n_asked", 0)
    if question and question.strip():
        cache = answer_cache()
        norm = " ".join(question.lower().split())
        if norm in cache:
            render(cache[norm])
            st.caption("Cached answer: identical questions are not sent to the LLM again.")
        elif st.session_state["n_asked"] >= MAX_PER_SESSION:
            st.warning(f"This session has used its {MAX_PER_SESSION} live questions. Cached examples still work.")
        elif not global_limiter().try_acquire():
            st.warning("The shared free LLM quota is used up for this hour. Please try again later, or use an example.")
        else:
            st.session_state["n_asked"] += 1
            with st.spinner("Writing SQL and running it..."):
                try:
                    ans = analyst.ask(common.get_con(), question.strip(), analyst.groq_completer(api_key=key))
                except Exception as e:  # noqa: BLE001  (network / quota problems must not crash the page)
                    st.error(f"The language model could not be reached ({type(e).__name__}). Try again in a moment.")
                else:
                    if len(cache) < 300:
                        cache[norm] = ans
                    render(ans)
        st.caption(f"{st.session_state['n_asked']} of {MAX_PER_SESSION} live questions used in this session.")

st.divider()

# ── how accurate is it ────────────────────────────────────────────────────────
st.subheader("How accurate is it?")
ev = common.load_json("analyst_eval.json")
if ev is None:
    st.write("The evaluation has not been run yet. Run `make eval`.")
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Execution accuracy", f"{ev['accuracy']:.0%}", help=f"{ev['n_correct']} of {ev['n_questions']} questions")
    lv = ev["by_level"]
    for col, level in zip((c2, c3, c4), ("easy", "medium", "hard")):
        col.metric(level.title(), f"{lv[level]['correct']}/{lv[level]['total']}")
    c5.metric("Safety cases", f"{ev['safety_handled']}/{ev['safety_total']}", help="Must return no data")
    st.caption(
        f"{ev['n_questions']} fixed questions, each with a hand-written gold SQL. A question counts as correct only when the "
        "analyst's query returns the same result as the gold query (column order, extra columns and small rounding are ignored; "
        f"wrong numbers are not). Model: {ev['model']}. Safety cases include a missing column, a prompt-injection "
        "'DROP TABLE' and a file-read attempt: all must return no data."
    )
    detail = pd.DataFrame(ev["results"])[["id", "level", "question", "correct", "attempts"]].rename(
        columns={"id": "ID", "level": "Level", "question": "Question", "correct": "Correct", "attempts": "Attempts"})
    with st.expander("Per-question results"):
        st.dataframe(detail, hide_index=True)
        for r in ev["results"]:
            if not r["correct"]:
                st.markdown(f"**{r['id']} (wrong):** {r['question']}")
                st.code(r.get("generated_sql") or r.get("error") or "(no SQL)", language="sql")

# ── how it stays safe ─────────────────────────────────────────────────────────
with st.expander("How the SQL guard keeps this safe"):
    st.markdown(
        """
An LLM can write wrong SQL, or SQL that a hostile question talked it into. Its output is never run directly:

1. **Parse.** DuckDB's own parser must accept it, as **exactly one statement of type SELECT** (this rejects DROP, INSERT, COPY, ATTACH, PRAGMA, SET...).
2. **Tables.** Only `sales`, `returns`, `products` and `customers` may be read. A WITH query can't reuse the name of a system view (such as `sqlite_master` or `duckdb_tables`), so it can't be used to disguise a read of one.
3. **Functions.** File-reading and system functions (`read_csv`, `glob`, `duckdb_settings`...) are refused, and so are the functions that can build one enormous value: `string_agg`, `group_concat`, `listagg`, `list`, `array_agg`, `histogram` and `repeat`.
4. **Row cap.** The query is wrapped in `SELECT * FROM (...) LIMIT 1000`.
5. **Result size.** Results stream in small batches and are refused as soon as they pass 2 MB in total, 10 KB in any single value (also inside lists, structs and maps) or 1,000 items in a list.
6. **Timeout.** A timer interrupts any query running longer than 15 seconds.
7. **Locked database.** The connection has file access switched off and its settings locked, so even a missed case cannot touch the disk.

One repair attempt: if the SQL fails or is rejected, the error is sent back to the model once. Identical questions are cached,
and each session and the whole app have a question limit, so visitors cannot burn the free LLM quota.
        """
    )
