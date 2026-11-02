"""Smoke tests: every page renders without an exception, and the eval set stays valid on the real data."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rci import analyst, db, sql_guard

from conftest import PROCESSED, ROOT

APP = ROOT / "app"
PAGES = [APP / "Overview.py", *sorted((APP / "pages").glob("*.py"))]

pytestmark = pytest.mark.skipif(not (PROCESSED / "sales.parquet").exists(), reason="processed data not built")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_renders_without_error(page, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("RCI_ENV_FILE", "/nonexistent")
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_overview_shows_the_headline_numbers():
    at = AppTest.from_file(str(APP / "Overview.py"), default_timeout=120).run()
    labels = [m.label for m in at.metric]
    assert labels[:3] == ["Revenue", "Orders", "Identified customers"]


def test_ab_planner_flags_a_segment_that_is_too_small():
    at = AppTest.from_file(str(APP / "pages" / "1_Customers_and_Segments.py"), default_timeout=120).run()
    assert any("too small" in w.value for w in at.warning)          # 5-point uplift needs more than 1,445 customers
    at.slider[0].set_value(15.0).run()                              # a big effect is easy to detect
    assert any("Feasible" in s.value for s in at.success)


def fake_groq(**_):
    """Stands in for Groq: answers the 'top 5 products' question and summary prompts."""
    def complete(messages):
        last = messages[-1]["content"]
        if "Result (CSV)" in last:
            return "The best seller is the Regency cakestand."
        return ("SELECT description, ROUND(SUM(revenue), 0) AS revenue FROM sales "
                "GROUP BY description ORDER BY revenue DESC LIMIT 5")
    return complete


def test_ask_page_answers_an_example_with_a_fake_llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(analyst, "groq_completer", fake_groq)
    at = AppTest.from_file(str(APP / "pages" / "4_Ask_the_Data.py"), default_timeout=120).run()
    assert not at.exception
    at.button(key="example0").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("SELECT description" in c.value for c in at.code)    # the SQL is always shown
    assert any("Regency" in m.value for m in at.markdown)           # the summary sentence
    assert len(at.dataframe) >= 1


def test_ask_page_without_a_key_explains_instead_of_failing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("RCI_ENV_FILE", "/nonexistent")
    at = AppTest.from_file(str(APP / "pages" / "4_Ask_the_Data.py"), default_timeout=120).run()
    assert not at.exception
    assert any("GROQ_API_KEY" in i.value for i in at.info)


# ── the evaluation set ───────────────────────────────────────────────────────
QUESTIONS = json.loads((ROOT / "evals" / "analyst_questions.json").read_text())
FROZEN_GOLD_SHA256 = "fcd7c2eaecae4e8b721d9f837093ee60063524c53ae2b53206509c3e28e9c5b0"


@pytest.mark.parametrize("q", QUESTIONS["questions"], ids=lambda q: q["id"])
def test_gold_sql_is_valid_and_returns_rows(q, real_con):
    sql, df = sql_guard.run_guarded(real_con, q["gold_sql"])
    assert len(df) >= 1


def test_eval_ids_are_unique_and_levels_are_known():
    ids = [q["id"] for q in QUESTIONS["questions"] + QUESTIONS["safety_cases"] + QUESTIONS["guard_cases"]]
    assert len(ids) == len(set(ids))
    assert {q["level"] for q in QUESTIONS["questions"]} == {"easy", "medium", "hard"}
    assert Counter(q["set"] for q in QUESTIONS["questions"]) == {"dev": 22, "heldout": 8, "exploratory": 10}


def test_the_gold_sql_is_frozen():
    """The questions were fixed before the live run. The hash covers each question's id, text, gold SQL, set, level and
    order_matters (False when absent), so moving a question between the headline and exploratory sets, or changing how
    its result is compared, is caught too. Change any of them only if a gold SQL is wrong: re-score the saved SQL
    offline, and say so in the commit message."""
    frozen = json.dumps([[q["id"], q["question"], q["gold_sql"], q["set"], q["level"], q.get("order_matters", False)]
                         for q in QUESTIONS["questions"]])
    assert hashlib.sha256(frozen.encode()).hexdigest() == FROZEN_GOLD_SHA256


HEADLINE_SETS = ("dev", "heldout")


def test_saved_eval_report_is_consistent():
    path = PROCESSED / "analyst_eval.json"
    if not path.exists():
        pytest.skip("analyst eval not run")
    rep = json.loads(path.read_text())
    rows = rep["results"]
    headline = [r for r in rows if r["set"] in HEADLINE_SETS]
    assert [r["id"] for r in rows] == [q["id"] for q in QUESTIONS["questions"]] and len(rows) == 40
    for row, q in zip(rows, QUESTIONS["questions"]):                      # each row must belong to the question it claims
        assert (row["set"], row["level"], row["gold_sql"]) == (q["set"], q["level"], q["gold_sql"]), row["id"]
    assert {r["set"] for r in rows} == {"dev", "heldout", "exploratory"}

    # the headline is the dev + held-out questions only; the exploratory set never enters it
    assert rep["n_questions"] == len(headline) == 30
    assert rep["n_executed"] == sum(r["executed"] for r in headline)
    assert rep["n_correct"] == sum(r["correct"] for r in headline)
    assert rep["n_correct"] <= rep["n_executed"] <= rep["n_questions"]
    assert rep["accuracy"] == pytest.approx(rep["n_correct"] / rep["n_questions"], abs=1e-3)
    for level, v in rep["by_level"].items():
        in_level = [r for r in headline if r["level"] == level]
        assert (v["correct"], v["total"]) == (sum(r["correct"] for r in in_level), len(in_level))
    assert sum(v["total"] for v in rep["by_level"].values()) == rep["n_questions"]

    # one breakdown per set present, covering every row exactly once
    assert set(rep["by_set"]) == {"dev", "heldout", "exploratory"}
    for name, v in rep["by_set"].items():
        in_set = [r for r in rows if r["set"] == name]
        assert v == {"correct": sum(r["correct"] for r in in_set), "executed": sum(r["executed"] for r in in_set), "total": len(in_set)}
        assert v["correct"] <= v["executed"] <= v["total"]
    assert sum(v["total"] for v in rep["by_set"].values()) == len(rows)
    assert sum(rep["by_set"][s]["total"] for s in HEADLINE_SETS) == rep["n_questions"]
    for r in rows:
        assert not r["correct"] or r["executed"]
        assert not r["executed"] or (r["error"] is None and r["generated_sql"])

    # safety: mutually exclusive buckets that add up to "handled"
    safety = rep["safety"]
    assert rep["safety_total"] == len(safety) == len(QUESTIONS["safety_cases"])
    assert rep["safety_handled"] == sum(r["handled"] for r in safety)
    assert (rep["safety_refused_by_model"], rep["safety_blocked_by_guard"], rep["safety_failed_other"]) == tuple(
        sum(r["bucket"] == b for r in safety) for b in ("refused_by_model", "blocked_by_guard", "failed_other"))
    assert rep["safety_refused_by_model"] + rep["safety_blocked_by_guard"] + rep["safety_failed_other"] == rep["safety_handled"]
    assert all(r["handled"] == (r["bucket"] != "leaked") for r in safety)

    # guard cases: raw attack SQL, no LLM
    assert rep["guard_cases_total"] == len(rep["guard_cases"]) == len(QUESTIONS["guard_cases"])
    assert rep["guard_cases_blocked"] == sum(c["blocked"] for c in rep["guard_cases"])


def test_the_ask_page_reports_the_exploratory_set_on_its_own_line(monkeypatch):
    path = PROCESSED / "analyst_eval.json"
    if not path.exists():
        pytest.skip("analyst eval not run")
    rep = json.loads(path.read_text())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("RCI_ENV_FILE", "/nonexistent")
    at = AppTest.from_file(str(APP / "pages" / "4_Ask_the_Data.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].label == "Execution accuracy" and at.metric[0].value == f"{rep['accuracy']:.0%}"      # the 30-question headline
    assert at.metric[0].help == f"{rep['n_correct']} of {rep['n_questions']} questions"
    table = next(d.value for d in at.dataframe if "Set" in d.value.columns)
    assert len(table) == 40 and set(table["Set"]) == {"dev", "heldout", "exploratory"}
    exploratory = rep["by_set"]["exploratory"]
    lines = [c.value for c in at.caption if "xploratory" in c.value]
    assert lines and f"{exploratory['correct']}/{exploratory['total']}" in lines[0]
    assert any(f"{rep['n_questions']} fixed questions" in c.value for c in at.caption)                         # the headline caption
    assert any(f"{rep['guard_cases_blocked']} of {rep['guard_cases_total']} raw attack queries" in c.value for c in at.caption)


def test_the_committed_model_report_is_consistent():
    path = PROCESSED / "metrics.json"
    if not path.exists():
        pytest.skip("model not trained")
    m = json.loads(path.read_text())
    for name, d in m["models"].items():
        for metric, v in d.items():
            assert v["ci_low"] <= v["value"] <= v["ci_high"], (name, metric)
    assert m["base_rate_test"] == pytest.approx(m["splits"]["test"]["repurchase_rate"], abs=1e-3)


def test_the_committed_backtest_is_consistent_with_the_main_table():
    path = PROCESSED / "metrics.json"
    if not path.exists():
        pytest.skip("model not trained")
    m = json.loads(path.read_text())
    bt = m["backtest"]
    assert bt["title"] == "Three-quarter backtest (MLP not included)"
    assert [f["label"] for f in bt["folds"]] == ["2011-03", "2011-06", "2011-09"]
    assert m["latest_quarter_only"] == ["PyTorch MLP"] and "PyTorch MLP" in m["models"]    # labelled, and kept out of `models`' metric schema
    assert "PyTorch MLP" not in bt["mean"] and all("PyTorch MLP" not in f["roc_auc"] for f in bt["folds"])
    assert set(bt["folds"][0]["roc_auc"]) == set(bt["models"]) == set(m["models"]) - {"PyTorch MLP"}
    for name in bt["models"]:                                                              # fold 3 IS the main split
        assert m["models"][name]["roc_auc"]["value"] == pytest.approx(bt["folds"][2]["roc_auc"][name], abs=0.002), name
        assert bt["mean"][name] == pytest.approx(sum(f["roc_auc"][name] for f in bt["folds"]) / 3)
    assert bt["folds"][2]["test_customers"] == m["test_customers"]
    gap = [f["roc_auc"][m["gbm_model"]] - f["roc_auc"]["Logistic regression"] for f in bt["folds"]]
    assert bt["gbm_minus_logreg"]["per_fold"] == pytest.approx(gap) and bt["gbm_minus_logreg"]["mean"] == pytest.approx(sum(gap) / 3)


def test_the_model_page_shows_the_backtest_and_labels_the_mlp_as_latest_quarter_only(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(str(APP / "pages" / "2_Repurchase_Model.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Three-quarter backtest (MLP not included)" in [s.value for s in at.subheader]
    text = " ".join(e.value for group in (at.markdown, at.caption, at.info, at.success, at.warning) for e in group)
    assert "the gain is real" not in text and "excludes zero" not in text
    assert "one test period" not in text
    tables = [d.value for d in at.dataframe]
    main = next(t for t in tables if "PyTorch MLP" in t.to_string())
    mlp_row = main[main["Model"] == "PyTorch MLP"].to_string()
    assert "latest quarter only (Sep–Dec 2011)" in mlp_row
    assert "latest quarter only" not in main[main["Model"] == "Logistic regression"].to_string()
    backtest = next(t for t in tables if "2011-03" in t.to_string() and "PyTorch MLP" not in t.to_string())
    assert {"2011-03", "2011-06", "2011-09", "Mean"} <= set(backtest.columns)


# ── the README results blocks are generated, never typed ─────────────────────
def test_the_readme_results_blocks_are_what_render_results_generates(monkeypatch):
    """Both generated blocks (RESULTS and RESULTS_ZH) must equal render_results.build(), apart from the test-count lines
    (they change whenever a test is added). count_tests is replaced so the test does not start a nested pytest run."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import render_results

    monkeypatch.setattr(render_results, "count_tests", lambda: 1)
    built = render_results.build()
    readme = (ROOT / "README.md").read_text()

    def block(tag: str) -> str:
        return re.search(rf"<!-- {tag}:START -->(.*?)<!-- {tag}:END -->", readme, re.S).group(1)

    def without_test_counts(text: str) -> str:
        return "\n".join(line for line in text.strip().splitlines() if "test cases (" not in line and "测试用例（" not in line)

    assert without_test_counts(block("RESULTS")) == without_test_counts(built["en"])
    assert without_test_counts(block("RESULTS_ZH")) == without_test_counts(built["zh"])
    # the fixed "Small differences" limitation quotes the backtest numbers and verdict: they must be the committed ones
    metrics = json.loads((PROCESSED / "metrics.json").read_text())
    bullet = next(line for line in readme.splitlines() if line.startswith("- **Small differences.**"))
    assert render_results.backtest_summary(metrics) in bullet and render_results.backtest_verdict(metrics) in bullet
