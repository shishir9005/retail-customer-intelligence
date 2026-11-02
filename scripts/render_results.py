"""Fill the results blocks of README.md and write docs/CV_NUMBERS.md from the saved result files.

Every number comes from data/processed/*.json, so the documentation cannot drift from what was
measured. Re-run after retraining:   python scripts/render_results.py   (or `make results`)
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
LATEST = "Sep–Dec 2011"                                  # the latest test quarter (the main table)
COUNTRY_NOTE = "country labels (about 40 countries; includes Unspecified and European Community)"
CI_HAS_RUN = False        # set True only after a real CI run on GitHub; until then the docs say "configured"


def load(name: str) -> dict:
    return json.loads((P / name).read_text())


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def analyst_runs() -> list[str]:
    """One line per saved evaluation run, oldest first, ending with the current one."""
    runs = []
    for f in sorted((ROOT / "evals" / "history").glob("run*.json")) + [P / "analyst_eval.json"]:
        if f.exists():
            r = json.loads(f.read_text())
            runs.append(f"{r['n_correct']}/{r['n_questions']}")
    return runs


def count_tests() -> int | None:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True,
                         env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin"})
    m = re.search(r"(\d+) tests? collected", out.stdout) or re.search(r"^(\d+) tests?", out.stdout, re.M)
    return int(m.group(1)) if m else None


def count_test_functions() -> int:
    """Test functions in tests/ (a parametrized function counts once), read from the source: no nested pytest run."""
    return sum(1 for f in (ROOT / "tests").glob("test_*.py") for node in ast.walk(ast.parse(f.read_text()))
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"))


# ── wording that depends on what the backtest shows ─────────────────────────
def backtest_summary(met: dict) -> str:
    """The backtest means and the per-quarter gap, as one phrase (also quoted in README's 'Small differences' limitation)."""
    bt, gbm = met["backtest"], met["gbm_model"]
    gaps = ", ".join(f"{d:+.3f}" for d in bt["gbm_minus_logreg"]["per_fold"])
    return (f"logistic regression averages {bt['mean']['Logistic regression']:.3f} ROC-AUC and {gbm} {bt['mean'][gbm]:.3f}; "
            f"{gbm} minus logistic regression is {gaps} by quarter")


def gbm_advantage(met: dict) -> str:
    """'gbm', 'logreg' or 'same': a gradient-boosting advantage is claimed only if it leads on the mean AND in every quarter."""
    gap = met["backtest"]["gbm_minus_logreg"]
    if gap["mean"] > 0 and all(d > 0 for d in gap["per_fold"]):
        return "gbm"
    if gap["mean"] < 0 and all(d < 0 for d in gap["per_fold"]):
        return "logreg"
    return "same"


def backtest_verdict(met: dict) -> str:
    gbm, gap = met["gbm_model"], met["backtest"]["gbm_minus_logreg"]
    return {"gbm": f"{gbm} is ahead of logistic regression in every quarter and on average.",
            "logreg": f"Logistic regression is ahead of {gbm} in every quarter and on average.",
            "same": f"The two perform about the same: neither is ahead in every quarter, and the average difference is {gap['mean']:+.3f} ROC-AUC."
            }[gbm_advantage(met)]


def backtest_verdict_zh(met: dict) -> str:
    gbm = met["gbm_model"]
    return {"gbm": f"{gbm} 在每个季度都领先", "logreg": f"逻辑回归在每个季度都领先", "same": "两者表现相近，没有谁在每个季度都领先"}[gbm_advantage(met)]


def build() -> dict:
    clean, seg, met, cat = load("cleaning_report.json"), load("segments.json"), load("metrics.json"), load("categories.json")
    ev = load("analyst_eval.json") if (P / "analyst_eval.json").exists() else None
    s = clean["summary"]
    gbm = met["gbm_model"]
    best = met["models"][gbm]
    bt = met["backtest"]
    only_latest = set(met.get("latest_quarter_only", []))
    latest_label = met.get("latest_quarter_label", "latest quarter only")
    n_tests, n_funcs = count_tests(), count_test_functions()
    ci_en = "CI runs in GitHub Actions" if CI_HAS_RUN else "CI is configured in `.github/workflows/ci.yml`"
    ci_zh = "已在 GitHub Actions 上持续集成" if CI_HAS_RUN else "持续集成已配置在 `.github/workflows/ci.yml`"
    ref = s["refunds_of_removed_sales"]
    refunds = f"{ref['lines']} refund{'s' if ref['lines'] != 1 else ''} of deleted sale lines removed (£{ref['refund']:,.2f})"
    lr_mean, gb_mean, rec_mean = bt["mean"]["Logistic regression"], bt["mean"][gbm], bt["mean"]["Recency rule"]
    mlp = met["models"].get("PyTorch MLP", {}).get("roc_auc", {}).get("value", float("nan"))
    cap = pct(best["top20_capture"]["value"], 0)

    orders = [[x["segment"] for x in e["segments"]] for e in seg["validation"]["by_cutoff"]]
    n_cut = len(orders)
    seg_note = (f"Refitted at each of {n_cut} dates, "
                + (f"{orders[0][0]} always buys again most and {orders[0][-1]} least" if len({o[0] for o in orders}) == 1 and len({o[-1] for o in orders}) == 1
                   else "the best and worst segments change between dates")
                + ("; the two middle segments change places between periods." if len({tuple(o[1:-1]) for o in orders}) > 1 else "."))

    steps = "\n".join(f"| {st['rule']} | {st['removed']:,} | {st['remaining']:,} |" for st in clean["steps"])
    seg_rows = "\n".join(
        f"| {r['segment']} | {r['customers']:,} | {pct(r['customer_share'], 0)} | {pct(r['revenue_share'], 0)} | "
        f"{r['avg_recency_days']:.0f} | {r['avg_orders']:.1f} | £{r['avg_spend']:,.0f} | "
        f"{pct(r.get('repurchase_rate_next_90d', float('nan')), 0)} (of {r.get('validation_customers', 0):,}) |" for r in seg["segments"])

    def ci(d, pc=False):
        f = (lambda v: pct(v)) if pc else (lambda v: f"{v:.3f}")
        return f"{f(d['value'])} ({f(d['ci_low'])} to {f(d['ci_high'])})"

    model_rows = "\n".join(
        f"| {name}{f': {latest_label}' if name in only_latest else ''} | {ci(d['roc_auc'])} | {ci(d['pr_auc'])} | "
        f"{ci(d['top20_capture'], True)} | {ci(d['top20_precision'], True)} |" for name, d in met["models"].items())
    folds = bt["folds"]
    gap = bt["gbm_minus_logreg"]
    backtest_rows = "\n".join(
        [f"| {name} | " + " | ".join(f"{f['roc_auc'][name]:.3f}" for f in folds) + f" | {bt['mean'][name]:.3f} |" for name in bt["models"]]
        + [f"| {gbm} minus logistic regression | " + " | ".join(f"{d:+.3f}" for d in gap["per_fold"]) + f" | {gap['mean']:+.3f} |"])

    analyst = ""
    analyst_zh = ""
    if ev:
        by_set = ev["by_set"]
        expl = by_set.get("exploratory")
        analyst = (
            f"**{ev['n_correct']} of {ev['n_questions']} questions correct ({pct(ev['accuracy'], 0)})** with `{ev['model']}` "
            f"(dev set {by_set['dev']['correct']}/{by_set['dev']['total']}, held-out set {by_set['heldout']['correct']}/{by_set['heldout']['total']}). "
            f"{ev['n_executed']} of the {ev['n_questions']} returned a result table, and {ev['n_correct']} of those matched the "
            "hand-written gold query's result: execution accuracy compares results, not SQL text, with numbers within 0.1% or a cent.\n\n"
            + (f"Exploratory set, reported on its own line and not part of the headline: {expl['correct']}/{expl['total']} correct "
               f"({expl['executed']} returned a table). These {expl['total']} further questions were written after the {ev['n_questions']} "
               "above were fixed and were never used to change the prompt.\n\n" if expl else "")
            + f"Safety cases handled {ev['safety_handled']}/{ev['safety_total']} (refused by the model {ev['safety_refused_by_model']}, "
              f"blocked by the guard {ev['safety_blocked_by_guard']}, other failure {ev['safety_failed_other']}). Guard cases: "
              f"{ev['guard_cases_blocked']}/{ev['guard_cases_total']} raw attack queries, run through the guard with no LLM, were refused "
              "with the expected message.\n\n"
              f"Runs so far, oldest first: {', '.join(analyst_runs())}. The first run used 22 questions and an unrefined prompt "
              "(two failures were the model reading 'revenue' as net of returns instead of gross; the fix was to state the metric "
              "definitions in the prompt). The last run is scored with the stricter comparison. The model is not perfectly repeatable "
              "even at temperature 0, so quote the range, not the best run. The saved runs are in `evals/history/`."
        )
        analyst_zh = (
            f"- **AI 数据分析师**：用自然语言提问，LLM 生成 SQL，经过只读 SQL 防护后在 DuckDB 执行；在 {ev['n_questions']} 道评测题（开发集 + 留出集）上 "
            f"{ev['n_correct']} 道正确，执行准确率 {pct(ev['accuracy'], 0)}（{ev['n_executed']} 道返回了结果表）"
            + (f"；另有 {expl['total']} 道探索性题目单独统计（{expl['correct']}/{expl['total']} 正确），不计入上述数字" if expl else "")
            + f"；安全用例 {ev['safety_handled']}/{ev['safety_total']}；{ev['guard_cases_blocked']}/{ev['guard_cases_total']} 个防护用例（直接攻击 SQL，不经过 LLM）被拦截。\n")

    block_en = f"""\
### Data and cleaning

{s['raw_rows']:,} raw rows become **{s['sales_rows']:,}** clean sale lines ({s['customers']:,} identified customers, \
{s['orders']:,} orders, {s['products']:,} products, {s['countries']} {COUNTRY_NOTE}, {s['first_date'][:10]} to {s['last_date'][:10]}) \
plus {s['returns_rows']:,} return lines kept separately; {refunds}. {pct(s['guest_revenue_share'])} of revenue is guest checkouts \
with no customer ID (kept for revenue, excluded from customer analysis).

| Cleaning rule | Rows removed | Rows remaining |
|---|---:|---:|
{steps}

### Customer segments (RFM + KMeans, k = {seg['chosen_k']})

| Segment | Customers | % of customers | % of revenue | Days since last order | Orders | Avg spend | Bought again in next 90 days (segments refitted at {seg['validation_cutoff']}; size of that group) |
|---|---:|---:|---:|---:|---:|---:|---:|
{seg_rows}

Revenue shares are of identified customers' revenue. The last column is leak-free: the segments were refitted using only orders \
before {seg['validation_cutoff']}, so the group sizes there differ from the customer counts in the second column; those customers were then \
followed for 90 days. {seg_note}

### 90-day repurchase model, latest quarter ({LATEST}: test period starting {met['splits']['test']['cutoffs'][0]}, \
{met['test_customers']:,} customers, {pct(met['base_rate_test'], 0)} bought again)

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top 20% of list contains (share of returners) | Of top 20%, really return |
|---|---|---|---|---|
{model_rows}

**{bt['title']}**, ROC-AUC (each test quarter trains only on earlier cutoffs):

| Model | {' | '.join(f['label'] for f in folds)} | Mean |
|---|{'---:|' * (len(folds) + 1)}
{backtest_rows}

On the backtest, {backtest_summary(met)}. {backtest_verdict(met)} Contacting the top 20% by {gbm} score in the latest quarter reaches \
{cap} of all returning customers (a random list reaches 20%), and {pct(best['top20_precision']['value'], 0)} of those contacted do \
return, against {pct(met['base_rate_test'], 0)} overall. The PyTorch MLP is trained for the latest quarter only: it is not part of the \
backtest and is never compared with its means.{
    '' if met['lightgbm_used'] else ' This run used scikit-learn HistGradientBoosting because LightGBM could not load (macOS needs `brew install libomp`); the code switches to LightGBM automatically when it can.'}

### AI analyst

{analyst or 'Run `make eval` to measure it.'}
{f"Test suite: {n_tests} test cases ({n_funcs} test functions); {ci_en}." if n_tests else ''}
"""

    block_zh = f"""\
- **数据**：UCI Online Retail II，原始 {s['raw_rows']:,} 行 → 清洗后 {s['sales_rows']:,} 行订单明细（约 {s['orders'] / 10_000:.0f} 万个订单），{s['customers']:,} 位可识别客户，{s['countries']} 个国家标签（约 40 个国家，含 Unspecified 与 European Community）。
- **客户分群**：RFM + KMeans，k = {seg['chosen_k']}；{seg['segments'][0]['segment']} 群占 {pct(seg['segments'][0]['customer_share'], 0)} 的客户，贡献 {pct(seg['segments'][0]['revenue_share'], 0)} 的收入。
- **90 天复购预测**：按时间切分（不随机切分，并有防泄漏测试），并做了三个季度的回测（不含 MLP）：逻辑回归和 {gbm} 的平均 ROC-AUC 分别为 {lr_mean:.3f} 和 {gb_mean:.3f}（{backtest_verdict_zh(met)}）；在最新季度（2011 年 9–12 月），按分数排名前 20% 的客户覆盖 {cap} 的回购客户（随机选只有 20%）。PyTorch MLP 仅最新季度训练（ROC-AUC {mlp:.3f}），不参与回测。
""" + analyst_zh + (f"- **工程化**：{n_tests} 个测试用例（{n_funcs} 个测试函数）；{ci_zh}。\n" if n_tests else "")

    bullet3 = (f"Predicted 90-day repurchase with a time-based split, leakage tests and a three-quarter backtest, in which logistic regression "
               f"and gradient boosting averaged ROC-AUC {lr_mean:.3f} and {gb_mean:.3f} (recency rule {rec_mean:.3f}). On the latest quarter alone, "
               f"the gradient-boosting top 20% contained {cap} of returners; a PyTorch MLP trained for that quarter only reached {mlp:.3f} and is "
               "not part of the backtest.")
    cv = f"""# Numbers for the CV (generated from the saved results)

Fill the `[X]` placeholders in `CV_Common_Draft.md` with these, copied exactly as printed. Generated by `scripts/render_results.py`; do not edit by hand.

| CV placeholder | Value | Source |
|---|---|---|
| Customers segmented `[N]k` | {s['customers'] / 1000:.1f}k ({s['customers']:,}) | cleaning_report.json |
| Order lines | 1M+ order lines (~40k orders): {s['sales_rows']:,} clean lines, {s['orders']:,} orders, from {s['raw_rows']:,} raw rows | cleaning_report.json |
| Countries | {s['countries']} {COUNTRY_NOTE} | cleaning_report.json |
| Logistic regression backtest mean ROC-AUC `[LR backtest mean]` | {lr_mean:.3f} | metrics.json (`backtest.mean`) |
| Gradient boosting backtest mean ROC-AUC `[GB backtest mean]` | {gb_mean:.3f} | metrics.json (`backtest.mean`, `{gbm}`) |
| Recency rule backtest mean ROC-AUC `[recency backtest mean]` | {rec_mean:.3f} | metrics.json (`backtest.mean`) |
| Gradient boosting top 20% of scored customers contain `[GB top-20% capture]` (latest quarter) | {cap} of returners | metrics.json (`models`, `{gbm}`) |
| PyTorch MLP ROC-AUC `[MLP ROC-AUC]` ({latest_label}) | {mlp:.3f} | metrics.json (`models`) |
| Of the top 20%, really return (latest quarter) | {pct(best['top20_precision']['value'], 0)} (vs {pct(met['base_rate_test'], 0)} overall) | metrics.json |
| Gradient boosting ROC-AUC (latest quarter) | {best['roc_auc']['value']:.3f} | metrics.json (`{gbm}`) |
| Logistic regression ROC-AUC (latest quarter) | {met['models']['Logistic regression']['roc_auc']['value']:.3f} | metrics.json |
| Analyst accuracy `[X] of [N]` (dev + held-out) | {f"{ev['n_correct']} of {ev['n_questions']} ({ev['n_executed']} returned a table)" if ev else 'not run'}; runs so far: {', '.join(analyst_runs())} | analyst_eval.json, evals/history/ |
| Analyst exploratory set (own line) | {f"{ev['by_set']['exploratory']['correct']} of {ev['by_set']['exploratory']['total']}" if ev and 'exploratory' in ev['by_set'] else 'not run'} | analyst_eval.json |
| Analyst safety and guard cases | {f"safety {ev['safety_handled']} of {ev['safety_total']} (refused by the model {ev['safety_refused_by_model']}, blocked by the guard {ev['safety_blocked_by_guard']}, other failure {ev['safety_failed_other']}); guard cases {ev['guard_cases_blocked']} of {ev['guard_cases_total']}" if ev else 'not run'} | analyst_eval.json |
| Tests `[N]` | {n_tests} test cases ({n_funcs} test functions); {ci_en}{'' if CI_HAS_RUN else ' (no real CI run yet)'} | pytest --collect-only, tests/ |

## Wording that is true as measured

- CV bullet 3, copy exactly: {bullet3}
{"" if met['lightgbm_used'] else chr(10) + "- **Important:** this run used scikit-learn `HistGradientBoostingClassifier`, not LightGBM (LightGBM could not load without `libomp`). Write 'gradient boosting (scikit-learn)' on the CV, never 'LightGBM'." + chr(10)}
- On the backtest, {backtest_summary(met)}. {backtest_verdict(met)} Do not write that the model is 'highly accurate'{'' if gbm_advantage(met) == 'gbm' else " or that gradient boosting beats logistic regression"}: the simple recency rule averages {rec_mean:.3f} and is already strong.
- The PyTorch MLP ({mlp:.3f}) is {latest_label}: never average it with, or rank it against, the backtest means.
- Analyst: {f"{ev['n_correct']} of {ev['n_questions']} on the headline questions; " if ev else ''}say "about 90%" or give the range across runs ({', '.join(analyst_runs())}), not just the best run: the LLM varies slightly between runs. The exploratory set is its own line, never merged into the headline.
- Tests: write "{n_tests} test cases" and {'"CI runs in GitHub Actions"' if CI_HAS_RUN else '"CI configured in .github/workflows/ci.yml" (the repository has not run on GitHub yet; say "tests run in GitHub Actions" only after a real CI run)'}.
"""
    return {"en": block_en, "zh": block_zh, "cv": cv}


def replace_block(text: str, tag: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {tag}:START -->)(.*?)(<!-- {tag}:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md is missing the <!-- {tag}:START --> / <!-- {tag}:END --> markers")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def main() -> None:
    out = build()
    readme = ROOT / "README.md"
    text = readme.read_text()
    text = replace_block(text, "RESULTS", out["en"])
    text = replace_block(text, "RESULTS_ZH", out["zh"])
    readme.write_text(text)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "CV_NUMBERS.md").write_text(out["cv"])
    print("README.md and docs/CV_NUMBERS.md updated")


if __name__ == "__main__":
    main()
