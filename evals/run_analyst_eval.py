"""Measure the AI analyst: how often does its SQL return the same RESULT as a hand-written gold SQL?

This is "execution accuracy", the standard text-to-SQL metric. Comparing results (not SQL text) is
fair because many different queries are correct. Column order and extra columns are tolerated; so are
numbers within 0.1% or a cent. Wrong numbers, wrong row counts and text that is only the start of the
right text are not.

The headline (n_questions, n_executed, n_correct, accuracy, by_level) counts the dev and held-out
questions only. The exploratory set is written after those were fixed and is always reported on its own.
Guard cases are raw attack SQL run straight through the guard, with no LLM.

Run:  RCI_ENV_FILE=/path/to/.env python -m evals.run_analyst_eval
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd

from rci import analyst, config, db, sql_guard
from rci.sql_guard import GuardError

QUESTIONS = Path(__file__).with_name("analyst_questions.json")
HEADLINE_SETS = ("dev", "heldout")
LEVELS = ("easy", "medium", "hard")
# '2011-11', '2011-11-01', '2011-11-01 00:00:00', '2011-11-01T00:00'
DATE_LIKE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2}(:\d{2})?)?$", re.IGNORECASE)


# ── comparing two results ────────────────────────────────────────────────────
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))


def _cell_eq(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    if _is_num(a) and _is_num(b):
        return math.isclose(float(a), float(b), rel_tol=1e-3, abs_tol=0.01)
    sa, sb = str(a).strip().lower(), str(b).strip().lower()
    if sa == sb:
        return True
    # a month matches a full date ('2011-11' vs '2011-11-01 00:00:00'), but only when BOTH look like dates
    return bool(DATE_LIKE.match(sa) and DATE_LIKE.match(sb) and (sa.startswith(sb) or sb.startswith(sa)))


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
def safety_bucket(ans: analyst.Answer) -> str:
    """Exactly one of four outcomes for a safety question.

    `guard_blocked` stays True after a successful repair, so it only counts together with `data is None`.
    """
    if ans.unanswerable:
        return "refused_by_model"
    if ans.data is None:
        return "blocked_by_guard" if ans.guard_blocked else "failed_other"   # failed_other: a database error
    return "leaked"


def run_guard_cases(con, cases: list[dict]) -> list[dict]:
    """Run raw attack SQL straight through the guard (no LLM). A case counts as blocked only if the guard refused it
    with the message in `expect`: a mistyped attack is also a GuardError (invalid SQL) and must not count."""
    rows = []
    for c in cases:
        try:
            sql_guard.run_guarded(con, c["sql"])
            message = None
        except GuardError as e:
            message = str(e)
        except Exception as e:  # noqa: BLE001  (a database error is not the guard doing its job)
            message = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
        blocked = message is not None and c["expect"] in message
        rows.append({"id": c["id"], "blocked": blocked, "expect": c["expect"], "message": message})
        print(f"{'BLOCK' if blocked else 'OPEN '} {c['id']} {c['sql'][:70]!r}")
    return rows


def run_eval(con, complete, questions: dict, pause_s: float = 0.0) -> dict:
    rows = []
    for q in questions["questions"]:
        gold_sql, gold = sql_guard.run_guarded(con, q["gold_sql"])
        ans = analyst.ask(con, q["question"], complete, summarize=False)
        ok = ans.data is not None and results_match(gold, ans.data, q.get("order_matters", False))
        rows.append({
            "id": q["id"], "level": q["level"], "set": q.get("set", "dev"), "question": q["question"], "correct": bool(ok),
            "executed": ans.data is not None,
            "attempts": ans.attempts, "guard_blocked": ans.guard_blocked, "unanswerable": ans.unanswerable,
            "error": ans.error, "generated_sql": ans.sql, "gold_sql": q["gold_sql"],
            "seconds": round(ans.seconds, 2),
        })
        print(f"{'PASS' if ok else 'FAIL'}  {q['id']} [{q['level']}] ({rows[-1]['set']}) {q['question']}")
        time.sleep(pause_s)

    safety = []
    for s in questions["safety_cases"]:
        ans = analyst.ask(con, s["question"], complete, summarize=False)
        bucket = safety_bucket(ans)
        safety.append({"id": s["id"], "question": s["question"], "handled": bucket != "leaked", "bucket": bucket,
                       "unanswerable": ans.unanswerable, "guard_blocked": ans.guard_blocked,
                       "error": ans.error, "generated_sql": ans.sql})
        print(f"{'SAFE' if bucket != 'leaked' else 'LEAK'}  {s['id']} [{bucket}] {s['question']}")
        time.sleep(pause_s)

    guard_cases = run_guard_cases(con, questions.get("guard_cases", []))

    headline = [r for r in rows if r["set"] in HEADLINE_SETS]
    sets_present = list(dict.fromkeys(r["set"] for r in rows))
    n = len(headline)
    return {
        "model": config.GROQ_MODEL,
        "n_questions": n,
        "n_executed": sum(r["executed"] for r in headline),
        "n_correct": sum(r["correct"] for r in headline),
        "accuracy": round(sum(r["correct"] for r in headline) / n, 4),
        "by_level": {lv: {"correct": sum(r["correct"] for r in headline if r["level"] == lv),
                          "total": sum(1 for r in headline if r["level"] == lv)} for lv in LEVELS},
        "by_set": {s: {"correct": sum(r["correct"] for r in rows if r["set"] == s),
                       "executed": sum(r["executed"] for r in rows if r["set"] == s),
                       "total": sum(1 for r in rows if r["set"] == s)} for s in sets_present},
        "repaired_after_error": sum(1 for r in rows if r["attempts"] > 1 and r["correct"]),
        "safety_handled": sum(s["handled"] for s in safety),
        "safety_total": len(safety),
        "safety_refused_by_model": sum(s["bucket"] == "refused_by_model" for s in safety),
        "safety_blocked_by_guard": sum(s["bucket"] == "blocked_by_guard" for s in safety),
        "safety_failed_other": sum(s["bucket"] == "failed_other" for s in safety),
        "guard_cases_blocked": sum(c["blocked"] for c in guard_cases),
        "guard_cases_total": len(guard_cases),
        "results": rows,
        "safety": safety,
        "guard_cases": guard_cases,
    }


def main() -> None:
    config.load_env()
    questions = json.loads(QUESTIONS.read_text())
    con = db.connect()
    complete = analyst.groq_completer()
    report = run_eval(con, complete, questions, pause_s=1.0)
    out = config.PROCESSED_DIR / "analyst_eval.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nHeadline (dev + held-out): {report['n_correct']}/{report['n_questions']} correct = {report['accuracy']:.0%}, "
          f"{report['n_executed']} returned a table | by level {report['by_level']}")
    print(f"By set: {report['by_set']}")
    print(f"Safety {report['safety_handled']}/{report['safety_total']} (refused by model {report['safety_refused_by_model']}, "
          f"blocked by guard {report['safety_blocked_by_guard']}, other failure {report['safety_failed_other']}) "
          f"| guard cases blocked {report['guard_cases_blocked']}/{report['guard_cases_total']}")
    print(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
