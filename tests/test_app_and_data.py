"""Smoke tests: every page renders without an exception, and the eval set stays valid on the real data."""
import json
import re
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


@pytest.mark.parametrize("q", QUESTIONS["questions"], ids=lambda q: q["id"])
def test_gold_sql_is_valid_and_returns_rows(q, real_con):
    sql, df = sql_guard.run_guarded(real_con, q["gold_sql"])
    assert len(df) >= 1


def test_eval_ids_are_unique_and_levels_are_known():
    ids = [q["id"] for q in QUESTIONS["questions"] + QUESTIONS["safety_cases"]]
    assert len(ids) == len(set(ids))
    assert {q["level"] for q in QUESTIONS["questions"]} == {"easy", "medium", "hard"}


def test_saved_eval_report_is_consistent():
    path = PROCESSED / "analyst_eval.json"
    if not path.exists():
        pytest.skip("analyst eval not run")
    rep = json.loads(path.read_text())
    assert rep["n_questions"] == len(QUESTIONS["questions"]) == len(rep["results"])
    assert rep["n_correct"] == sum(r["correct"] for r in rep["results"])
    assert rep["accuracy"] == pytest.approx(rep["n_correct"] / rep["n_questions"], abs=1e-3)


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
    tables = [d.value for d in at.dataframe]
    main = next(t for t in tables if "PyTorch MLP" in t.to_string())
    mlp_row = main[main["Model"] == "PyTorch MLP"].to_string()
    assert "latest quarter only (Sep–Dec 2011)" in mlp_row
    assert "latest quarter only" not in main[main["Model"] == "Logistic regression"].to_string()
    backtest = next(t for t in tables if "2011-03" in t.to_string() and "PyTorch MLP" not in t.to_string())
    assert {"2011-03", "2011-06", "2011-09", "Mean"} <= set(backtest.columns)
