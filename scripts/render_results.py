"""Fill the results blocks of README.md and write docs/CV_NUMBERS.md from the saved result files.

Every number comes from data/processed/*.json, so the documentation cannot drift from what was
measured. Re-run after retraining:   python scripts/render_results.py   (or `make results`)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


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


def build() -> dict:
    clean, seg, met, cat = load("cleaning_report.json"), load("segments.json"), load("metrics.json"), load("categories.json")
    ev = load("analyst_eval.json") if (P / "analyst_eval.json").exists() else None
    s = clean["summary"]
    gbm = met["gbm_model"]
    best = met["models"][gbm]
    diff = met["auc_diff_vs_logreg"][gbm]
    n_tests = count_tests()

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
        f"{pct(r.get('repurchase_rate_next_90d', float('nan')), 0)} |" for r in seg["segments"])

    def ci(d, pc=False):
        f = (lambda v: pct(v)) if pc else (lambda v: f"{v:.3f}")
        return f"{f(d['value'])} ({f(d['ci_low'])} to {f(d['ci_high'])})"

    model_rows = "\n".join(
        f"| {name} | {ci(d['roc_auc'])} | {ci(d['pr_auc'])} | {ci(d['top20_capture'], True)} | {ci(d['top20_precision'], True)} |"
        for name, d in met["models"].items())

    analyst = ""
    if ev:
        by_set = ev.get("by_set", {})
        analyst = (
            f"**{ev['n_correct']} of {ev['n_questions']} questions correct ({pct(ev['accuracy'], 0)})** with `{ev['model']}`"
            + (f" (dev set {by_set['dev']['correct']}/{by_set['dev']['total']}, held-out set "
               f"{by_set['heldout']['correct']}/{by_set['heldout']['total']})" if by_set else "")
            + f"; safety cases handled {ev['safety_handled']}/{ev['safety_total']}. Execution accuracy: the generated SQL must "
              "return the same result as a hand-written gold query.\n\n"
              f"Runs so far, oldest first: {', '.join(analyst_runs())}. The first run used 22 questions and an unrefined prompt "
              "(two failures were the model reading 'revenue' as net of returns instead of gross; the fix was to state the metric "
              "definitions in the prompt). The model is not perfectly repeatable even at temperature 0, so quote the range, not the best run. "
              "The saved runs are in `evals/history/`."
        )

    block_en = f"""\
### Data and cleaning

{s['raw_rows']:,} raw rows become **{s['sales_rows']:,}** clean sale lines ({s['customers']:,} identified customers, \
{s['orders']:,} orders, {s['products']:,} products, {s['countries']} countries, {s['first_date'][:10]} to {s['last_date'][:10]}) \
plus {s['returns_rows']:,} return lines kept separately. {pct(s['guest_revenue_share'])} of revenue is guest checkouts with no \
customer ID (kept for revenue, excluded from customer analysis).

| Cleaning rule | Rows removed | Rows remaining |
|---|---:|---:|
{steps}

### Customer segments (RFM + KMeans, k = {seg['chosen_k']})

| Segment | Customers | % of customers | % of revenue | Days since last order | Orders | Avg spend | Bought again in next 90 days |
|---|---:|---:|---:|---:|---:|---:|---:|
{seg_rows}

Revenue shares are of identified customers' revenue. The last column is leak-free: the segments were refitted using only orders \
before {seg['validation_cutoff']}, then customers were followed for 90 days. {seg_note}

### 90-day repurchase model (test period starting {met['splits']['test']['cutoffs'][0]}, {met['test_customers']:,} customers, \
{pct(met['base_rate_test'], 0)} bought again)

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top 20% of list contains (share of returners) | Of top 20%, really return |
|---|---|---|---|---|
{model_rows}

{gbm} beats logistic regression by {diff['value']:+.3f} ROC-AUC (95% bootstrap interval {diff['ci_low']:+.3f} to \
{diff['ci_high']:+.3f}): a real but small gain, because most of the signal is simple recency and frequency. Contacting the \
top 20% by {gbm} score reaches {pct(best['top20_capture']['value'], 0)} of all returning customers (a random list reaches 20%), \
and {pct(best['top20_precision']['value'], 0)} of those contacted do return, against {pct(met['base_rate_test'], 0)} overall.{
    '' if met['lightgbm_used'] else ' This run used scikit-learn HistGradientBoosting because LightGBM could not load (macOS needs `brew install libomp`); the code switches to LightGBM automatically when it can.'}

### AI analyst

{analyst or 'Run `make eval` to measure it.'}
{f"The unit-test suite has {n_tests} tests." if n_tests else ''}
"""

    block_zh = f"""\
- **数据**：UCI Online Retail II，原始 {s['raw_rows']:,} 行 → 清洗后 {s['sales_rows']:,} 行销售记录，{s['customers']:,} 位可识别客户，{s['countries']} 个国家。
- **客户分群**：RFM + KMeans，k = {seg['chosen_k']}；{seg['segments'][0]['segment']} 群占 {pct(seg['segments'][0]['customer_share'], 0)} 的客户，贡献 {pct(seg['segments'][0]['revenue_share'], 0)} 的收入。
- **90 天复购预测**：按时间切分（不随机切分，并有防泄漏测试）。{gbm} ROC-AUC {best['roc_auc']['value']:.3f}，逻辑回归 {met['models']['Logistic regression']['roc_auc']['value']:.3f}；按分数排名前 20% 的客户覆盖 {pct(best['top20_capture']['value'], 0)} 的回购客户（随机选只有 20%）。
""" + (f"- **AI 数据分析师**：用自然语言提问，LLM 生成 SQL，经过只读 SQL 防护后在 DuckDB 执行；在 {ev['n_questions']} 道评测题上执行准确率 {pct(ev['accuracy'], 0)}。\n" if ev else "") + (f"- **工程化**：{n_tests} 个自动化测试，GitHub Actions 持续集成。\n" if n_tests else "")

    cv = f"""# Numbers for the CV (generated from the saved results)

Fill the `[X]` placeholders in `CV_Common_Draft.md` with these. Generated by `scripts/render_results.py`; do not edit by hand.

| CV placeholder | Value | Source |
|---|---|---|
| Customers segmented `[N]k` | {s['customers'] / 1000:.1f}k ({s['customers']:,}) | cleaning_report.json |
| Real transactions | {s['raw_rows']:,} raw rows, {s['sales_rows']:,} after cleaning | cleaning_report.json |
| Countries | {s['countries']} | cleaning_report.json |
| Gradient boosting ROC-AUC | {best['roc_auc']['value']:.3f} | metrics.json (`{gbm}`) |
| Logistic regression ROC-AUC `[Y]` | {met['models']['Logistic regression']['roc_auc']['value']:.3f} | metrics.json |
| PyTorch MLP ROC-AUC `[Z]` | {met['models'].get('PyTorch MLP', {}).get('roc_auc', {}).get('value', float('nan')):.3f} | metrics.json |
| Top 20% of scored customers contain | {pct(best['top20_capture']['value'], 0)} of returners | metrics.json |
| Of the top 20%, really return | {pct(best['top20_precision']['value'], 0)} (vs {pct(met['base_rate_test'], 0)} overall) | metrics.json |
| Analyst accuracy | {f"{ev['n_correct']} of {ev['n_questions']} in the final run" if ev else 'not run'}; runs so far: {', '.join(analyst_runs())} | analyst_eval.json, evals/history/ |
| Tests in CI | {n_tests} | pytest --collect-only |

## Wording that is true as measured

- Predicted 90-day repurchase with a time-based split and a leakage test: **{gbm}** reached ROC-AUC **{best['roc_auc']['value']:.3f}** vs \
**{met['models']['Logistic regression']['roc_auc']['value']:.3f}** for logistic regression and **{met['models'].get('PyTorch MLP', {}).get('roc_auc', {}).get('value', float('nan')):.3f}** for a PyTorch MLP; \
the top 20% of scored customers contain **{pct(best['top20_capture']['value'], 0)}** of returners.
{"" if met['lightgbm_used'] else chr(10) + "- **Important:** this run used scikit-learn `HistGradientBoostingClassifier`, not LightGBM (LightGBM could not load without `libomp`). Write 'gradient-boosted trees (scikit-learn)' on the CV, or run `brew install libomp && make model && make results` to get real LightGBM numbers and update this file." + chr(10)}
- Analyst: say "about 90%" or give the range across runs ({', '.join(analyst_runs())}), not just the best run: the LLM varies slightly between runs.
- Do not write that the model is 'highly accurate': it beats logistic regression by {diff['value']:+.3f} AUC (a real but small gain) and the simple recency rule is already strong.
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
