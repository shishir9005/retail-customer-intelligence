"""Measure the AI analyst: how often does its SQL return the same RESULT as a hand-written gold SQL?

This is "execution accuracy", the standard text-to-SQL metric. Comparing results (not SQL text) is
fair because many different queries are correct. Column order, extra columns and rounding are
tolerated; wrong numbers or wrong row counts are not.

Run:  RCI_ENV_FILE=/path/to/.env python -m evals.run_analyst_eval
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

from rci import analyst, config, db, sql_guard

QUESTIONS = Path(__file__).with_name("analyst_questions.json")


# ── comparing two results ────────────────────────────────────────────────────
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))


def _cell_eq(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    if _is_num(a) and _is_num(b):
        return math.isclose(float(a), float(b), rel_tol=5e-3, abs_tol=0.01)
    sa, sb = str(a).strip().lower(), str(b).strip().lower()
    return sa == sb or sa.startswith(sb) or sb.startswith(sa)  # '2011-11' matches '2011-11-01 00:00:00'


def _sort_key(v):
    return (0, float(v), "") if _is_num(v) else (1, 0.0, str(v).lower())


def _cols_match(g: list, p: list, order_matters: bool) -> bool:
    if len(g) != len(p):
        return False
    if not order_matters:
        g, p = sorted(g, key=_sort_key), sorted(p, key=_sort_key)
    return all(_cell_eq(a, b) for a, b in zip(g, p))


def results_match(gold: pd.DataFrame, pred: pd.DataFrame, order_matters: bool = False) -> bool:
    """True when `pred` contains every column of `gold` with the same values on the same rows.

    Step 1: for each gold column find a predicted column with the same values (extra predicted
            columns and a different column order are fine).
    Step 2: project the prediction onto those columns and compare row by row (sorted first unless
            the question asks for an order).
    """
    if len(gold) != len(pred) or gold.shape[1] == 0:
        return False
    used: set[int] = set()
    mapping: list[int] = []
    for j in range(gold.shape[1]):
        gcol = gold.iloc[:, j].tolist()
        hit = next((i for i in range(pred.shape[1]) if i not in used and _cols_match(gcol, pred.iloc[:, i].tolist(), order_matters)), None)
        if hit is None:
            return False
        used.add(hit)
        mapping.append(hit)
    g_rows = [tuple(r) for r in gold.itertuples(index=False)]
    p_rows = [tuple(r) for r in pred.iloc[:, mapping].itertuples(index=False)]
    if not order_matters:
        g_rows, p_rows = sorted(g_rows, key=lambda r: [_sort_key(v) for v in r]), sorted(p_rows, key=lambda r: [_sort_key(v) for v in r])
    return all(all(_cell_eq(a, b) for a, b in zip(gr, pr)) for gr, pr in zip(g_rows, p_rows))


# ── running the evaluation ───────────────────────────────────────────────────
def run_eval(con, complete, questions: dict, pause_s: float = 0.0) -> dict:
    rows = []
    for q in questions["questions"]:
        gold_sql, gold = sql_guard.run_guarded(con, q["gold_sql"])
        ans = analyst.ask(con, q["question"], complete, summarize=False)
        ok = ans.data is not None and results_match(gold, ans.data, q.get("order_matters", False))
        rows.append({
            "id": q["id"], "level": q["level"], "set": q.get("set", "dev"), "question": q["question"], "correct": bool(ok),
            "attempts": ans.attempts, "guard_blocked": ans.guard_blocked, "unanswerable": ans.unanswerable,
            "error": ans.error, "generated_sql": ans.sql, "gold_sql": q["gold_sql"],
            "seconds": round(ans.seconds, 2),
        })
        print(f"{'PASS' if ok else 'FAIL'}  {q['id']} [{q['level']}] {q['question']}")
        time.sleep(pause_s)

    safety = []
    for s in questions["safety_cases"]:
        ans = analyst.ask(con, s["question"], complete, summarize=False)
        handled = ans.data is None or ans.unanswerable  # nothing came back to the user
        safety.append({"id": s["id"], "question": s["question"], "handled": bool(handled),
                       "unanswerable": ans.unanswerable, "guard_blocked": ans.guard_blocked,
                       "error": ans.error, "generated_sql": ans.sql})
        print(f"{'SAFE' if handled else 'LEAK'}  {s['id']} {s['question']}")
        time.sleep(pause_s)

    n = len(rows)
    by_level = {lv: [r["correct"] for r in rows if r["level"] == lv] for lv in ("easy", "medium", "hard")}
    return {
        "model": config.GROQ_MODEL,
        "n_questions": n,
        "n_correct": sum(r["correct"] for r in rows),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "by_level": {lv: {"correct": sum(v), "total": len(v)} for lv, v in by_level.items()},
        "by_set": {s: {"correct": sum(r["correct"] for r in rows if r["set"] == s),
                       "total": sum(1 for r in rows if r["set"] == s)} for s in ("dev", "heldout")},
        "repaired_after_error": sum(1 for r in rows if r["attempts"] > 1 and r["correct"]),
        "safety_handled": sum(s["handled"] for s in safety),
        "safety_total": len(safety),
        "results": rows,
        "safety": safety,
    }


def main() -> None:
    config.load_env()
    questions = json.loads(QUESTIONS.read_text())
    con = db.connect()
    complete = analyst.groq_completer()
    report = run_eval(con, complete, questions, pause_s=1.0)
    out = config.PROCESSED_DIR / "analyst_eval.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nExecution accuracy: {report['n_correct']}/{report['n_questions']} = {report['accuracy']:.0%}  "
          f"| by level {report['by_level']} | by set {report['by_set']} | safety {report['safety_handled']}/{report['safety_total']}")
    print(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
