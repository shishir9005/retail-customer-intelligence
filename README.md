# Retail Customer Intelligence + AI Data Analyst

Analytics a retail data team would build, on **1 million real transactions** from a UK online gift shop (UCI Online Retail II, 2009-2011, 43 countries):
a tested cleaning pipeline, SQL analytics in a dashboard, customer segments, a model that predicts who will buy again in the next 90 days,
product categories from sentence embeddings, and an **AI analyst that turns plain-English questions into SQL** behind a read-only guard.

Built with Python, DuckDB (SQL), pandas, scikit-learn, PyTorch, MiniLM embeddings, Groq (LLM) and Streamlit. Cost: 0 (public data, free tiers).

> 中文简介见文末 / Chinese summary at the bottom.

## Results

<!-- RESULTS:START -->
### Data and cleaning

1,067,371 raw rows become **1,003,243** clean sale lines (5,852 identified customers, 39,500 orders, 4,722 products, 43 countries, 2009-12-01 to 2011-12-09) plus 17,914 return lines kept separately. 13.3% of revenue is guest checkouts with no customer ID (kept for revenue, excluded from customer analysis).

| Cleaning rule | Rows removed | Rows remaining |
|---|---:|---:|
| Exact duplicate rows | 34,335 | 1,033,036 |
| Cancellation invoices moved to returns | 19,104 | 1,013,932 |
| Non-product stock codes | 4,633 | 1,009,299 |
| Quantity <= 0 | 3,390 | 1,005,909 |
| Unit price <= 0 | 2,571 | 1,003,338 |
| Price > 20x the product's median price | 93 | 1,003,245 |
| Quantity > 50,000 in one line | 2 | 1,003,243 |

### Customer segments (RFM + KMeans, k = 4)

| Segment | Customers | % of customers | % of revenue | Days since last order | Orders | Avg spend | Bought again in next 90 days |
|---|---:|---:|---:|---:|---:|---:|---:|
| Champions | 1,190 | 20% | 74% | 28 | 19.2 | £10,403 | 84% |
| At risk | 1,446 | 25% | 16% | 229 | 5.0 | £1,889 | 46% |
| Promising | 1,259 | 22% | 6% | 29 | 3.0 | £838 | 55% |
| Hibernating | 1,957 | 33% | 4% | 395 | 1.4 | £315 | 20% |

Revenue shares are of identified customers' revenue. The last column is leak-free: the segments were refitted using only orders before 2011-09-09, then customers were followed for 90 days. Refitted at each of 6 dates, Champions always buys again most and Hibernating least; the two middle segments change places between periods.

### 90-day repurchase model (test period starting 2011-09-09, 5,253 customers, 43% bought again)

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top 20% of list contains (share of returners) | Of top 20%, really return |
|---|---|---|---|---|
| Recency rule | 0.762 (0.748 to 0.774) | 0.681 (0.660 to 0.703) | 34.7% (33.4% to 35.7%) | 75.2% (72.2% to 77.7%) |
| Logistic regression | 0.782 (0.770 to 0.796) | 0.754 (0.736 to 0.770) | 38.2% (37.1% to 39.4%) | 82.9% (80.2% to 85.3%) |
| HistGradientBoosting | 0.795 (0.783 to 0.807) | 0.770 (0.754 to 0.786) | 39.2% (38.0% to 40.5%) | 85.1% (82.7% to 87.3%) |
| PyTorch MLP | 0.787 (0.775 to 0.800) | 0.765 (0.747 to 0.781) | 38.5% (37.3% to 39.7%) | 83.3% (80.7% to 85.9%) |

HistGradientBoosting beats logistic regression by +0.013 ROC-AUC (95% bootstrap interval +0.007 to +0.018): a real but small gain, because most of the signal is simple recency and frequency. Contacting the top 20% by HistGradientBoosting score reaches 39% of all returning customers (a random list reaches 20%), and 85% of those contacted do return, against 43% overall. This run used scikit-learn HistGradientBoosting because LightGBM could not load (macOS needs `brew install libomp`); the code switches to LightGBM automatically when it can.

### AI analyst

**28 of 30 questions correct (93%)** with `openai/gpt-oss-120b` (dev set 20/22, held-out set 8/8); safety cases handled 3/3. Execution accuracy: the generated SQL must return the same result as a hand-written gold query.

Runs so far, oldest first: 19/22, 27/30, 29/30, 28/30. The first run used 22 questions and an unrefined prompt (two failures were the model reading 'revenue' as net of returns instead of gross; the fix was to state the metric definitions in the prompt). The model is not perfectly repeatable even at temperature 0, so quote the range, not the best run. The saved runs are in `evals/history/`.
The unit-test suite has 225 tests.

<!-- RESULTS:END -->

## The six modules

| # | Module | What it does | Files |
|---|---|---|---|
| 1 | **Data pipeline** | Download, clean (7 named rules that each log how many rows they remove), write Parquet, run 10 data-quality checks | `src/rci/pipeline.py`, `quality.py` |
| 2 | **SQL analytics dashboard** | Revenue trend, top products and countries, new vs returning customers, monthly cohort retention with a window function. The SQL behind every chart is shown on the page | `sql/*.sql`, `app/Overview.py` |
| 3 | **Customer segments + A/B planner** | RFM, log + scale, KMeans (k by silhouette within 3-6), named segments checked against real later behaviour. A hand-written power analysis for a win-back email test | `segments.py`, `abtest.py` |
| 4 | **Repurchase model** | Logistic regression vs gradient boosting vs a PyTorch MLP, time-based split, bootstrap confidence intervals, "top 20% captures X%" | `features.py`, `model.py`, `mlp.py`, `metrics.py` |
| 5 | **Product categories** | MiniLM embeddings of descriptions, KMeans, categories named by top words, "similar products" by cosine similarity | `products.py` |
| 6 | **AI data analyst** | Question, then an LLM writes SQL from the schema only, then a six-layer SQL guard, then DuckDB. SQL always shown. 30-question evaluation | `analyst.py`, `sql_guard.py`, `evals/` |

## Architecture

```mermaid
flowchart LR
    Z[UCI zip] --> P[pipeline.py<br/>clean + checks] --> PQ[(data/processed<br/>Parquet)]
    PQ --> D[DuckDB tables<br/>sales / returns / products / customers]
    D --> K[SQL analytics<br/>sql/*.sql]
    D --> S[RFM + KMeans<br/>segments]
    D --> M[features.sql<br/>repurchase model]
    PQ --> E[MiniLM embeddings<br/>product categories]
    D --> A[AI analyst<br/>Groq to SQL to guard]
    K & S & M & E & A --> UI[Streamlit app<br/>Overview, Segments, Model, Products, Ask the Data]
```

Heavy work (reading the Excel file, embeddings, model training, the PyTorch MLP) runs **once, offline**. The deployed app only reads small files
(about 10 MB in `data/processed`), so it needs no PyTorch or embedding model and fits a free hosting tier.

## Key decisions (and why)

| Decision | Why |
|---|---|
| **Real, messy data** (UCI Online Retail II, CC BY 4.0) | Not a tutorial dataset: duplicates, returns, fee rows, typos in prices, 23% guest checkouts. The cleaning rules are the interesting part. |
| **DuckDB + SQL files** | Real SQL (joins, window functions, cohorts) with no database server. Queries live in `sql/` where they can be read, and the dashboard shows them. |
| **Guest checkouts kept for revenue, dropped for customer analysis** | They are 13% of revenue, so removing them would understate sales; but without an ID they cannot be a customer. |
| **Split by time, not randomly** | A random split would put a customer's future orders in training. Each "pretend today" uses only earlier data; `tests/test_no_leakage.py` rewrites the future and checks the features do not move. |
| **Baselines first** | A one-line recency rule and logistic regression set the bar that gradient boosting and the neural net must clear. |
| **PR-AUC and "top 20% captures X%"** | Marketing contacts a limited list, so the quality of the top of the ranking matters more than accuracy. |
| **Bootstrap intervals** | Says how much each number could move by luck; the difference between models is paired so it is fair. |
| **SQL guard in layers** | An LLM can be talked into dangerous SQL. Parse (one SELECT only), table allow-list, function deny-list, row cap, timeout, and a locked connection with file access off. Attack cases are unit-tested. |
| **Evaluation set with gold SQL** | "It seems to work" is not evidence. Execution accuracy compares results, not SQL text. |
| **No FastAPI / MLflow / Spark / vector DB** | Not needed here, and everything stays explainable line by line. |

## Run it

```bash
make setup      # Python 3.12 virtual environment + dependencies
make all        # download + clean data, categories, segments, model (about 4 minutes)
make app        # dashboard at http://localhost:8501
make test       # unit tests
make eval       # measure the AI analyst (needs GROQ_API_KEY in .env, free key at console.groq.com)
make results    # refresh the numbers in this README and docs/CV_NUMBERS.md
```

The cleaned Parquet files are committed, so `make app` works straight after cloning, without `make all`.
The raw 45 MB Excel file is downloaded by the pipeline and never committed.

macOS note: LightGBM needs the OpenMP runtime (`brew install libomp`). Without it `model.py` automatically uses scikit-learn's
`HistGradientBoostingClassifier` (same algorithm family) and the results above say which one ran.

### Deploy (Streamlit Community Cloud, free)

1. Push the repository to GitHub (keep `data/processed/`, which is small).
2. share.streamlit.io, New app, main file `app/Overview.py`.
3. In Advanced settings, Secrets, add `GROQ_API_KEY = "..."`. Without it the app still works; only live questions in "Ask the Data" are disabled.

The analyst is protected from visitors burning the free quota: identical questions are cached, each session gets 15 questions, and the whole app is limited to 40 per hour.
A `Dockerfile` is included (`docker build -t rci . && docker run -p 8501:8501 -e GROQ_API_KEY=... rci`); it has not been built in the environment this was developed in.

## Project layout

```
sql/            schema.sql, kpis.sql, cohorts.sql, rfm.sql, features.sql   (all analytics SQL)
src/rci/        pipeline, quality, db, segments, abtest, features, metrics, model, mlp,
                products, analyst, sql_guard, ratelimit
app/            Overview.py + pages/ (Segments, Model, Products, Ask the Data)
evals/          analyst_questions.json (30 questions + 3 safety cases), run_analyst_eval.py, history/
tests/          cleaning, SQL guard attacks, no-leakage, SQL vs pandas, A/B maths vs statsmodels, segments,
                metrics, analyst with a fake LLM, page smoke tests
data/processed/ cleaned Parquet + JSON results (committed, about 10 MB)
.github/workflows/ci.yml   Dockerfile   Makefile   requirements*.txt
```

## Testing and CI

`pytest` runs on every push in GitHub Actions (Python 3.12, no PyTorch needed). The tests that matter most:

- **No leakage** (`tests/test_leakage.py`, on the real data): for all six cutoffs the features are rebuilt from a database with every later row deleted and must match; the SQL features are re-derived independently in pandas; the feature SQL may only read `sales` and `returns`; scalers see training rows only; segments are refitted from earlier data only before being validated; labels whose 90-day window is not fully observed are refused instead of cast to integers. A toy version of the first check lives in `tests/test_no_leakage.py`. Deliberately injecting a leak into each of these places makes the matching test fail.
- **SQL guard**: 26 attack cases plus 9 legitimate queries that must pass (DROP, COPY, ATTACH, `read_csv`, two statements, unknown tables, comments, quoted function names, slow queries, a locked connection).
- **SQL vs pandas**: headline numbers, monthly revenue, cohorts and RFM are checked against independent pandas calculations, on a tiny hand-checkable dataset and on the real data.
- **A/B maths** cross-checked against `statsmodels`.
- **Analyst flow** with a fake LLM: repair after an error, prompt-injection, file-read attempt, "unanswerable" replies, failed summary.
- **Every page** renders without an exception; every gold SQL in the evaluation set runs on the real data.

## Limitations (read before quoting a number)

- **One test period.** The model is evaluated on Sep-Dec 2011, which contains the Christmas peak, so absolute repurchase rates are higher than in a quiet quarter. The ranking of models is the point, not the absolute rate.
- **Small gains.** Gradient boosting beats logistic regression by about 0.01 ROC-AUC; most of the signal is recency and frequency. The data has no marketing, price or demographic features.
- **Three preprocessing steps are fitted over all dates, before the split**: the per-product median price used to delete typo prices, the most common description per product, and the product-category clustering. None uses repurchase labels, and their effect is measured, not assumed: the price rule only ever changes guest-checkout rows (which no customer feature reads), and refitting the categories using only products seen before the earliest cutoff changes the test AUC by far less than 0.01 (`tests/test_leakage.py` keeps checking both; the price check needs the raw Excel, so it runs locally but is skipped in CI).
- **Segments**: the dashboard table describes customers as of the end of the data (nothing lies in the future of "today"). The "bought again" validation instead refits the segments at each earlier cutoff using only earlier orders. Champions and Hibernating are ordered the same in every period, but At risk and Promising swap places, so read those two as similar opportunities, not a ranking.
- **Reproducibility**: segmentation is exactly reproducible (rows are read in a fixed order). Model scores reproduce to about 0.001 AUC between runs because multithreaded float sums are not order-stable.
- **Customers with no history are not scored**, and guest checkouts (no customer ID) cannot be linked to a person.
- **The dataset has no category column.** Categories come from embeddings of short descriptions; silhouette scores are low (below 0.06), so they are fuzzy groups, not a catalogue.
- **Cohort retention** treats "first order we can see" as "first order": the Dec 2009 cohort contains every already-active customer.
- **The AI analyst is not perfectly repeatable.** An LLM varies slightly between runs even at temperature 0; accuracy across runs is listed above. It was evaluated on the same 22 questions used to refine its prompt, plus 8 held-out ones; the held-out set is the less biased number.
- The `priority` score in "Who to contact" (spend x chance of not returning) is a simple demo heuristic, not a validated business rule.

## Data and attribution

- **Source:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- **What this repository contains:** only files derived from the dataset, in `data/processed/`. The original Excel file is not
  redistributed; `make data` downloads it from UCI.
- **Changes made** (every step is in `src/rci/pipeline.py`):
  - The two yearly sheets were combined into one table and the columns were renamed. Invoice numbers and stock codes were
    trimmed, stock codes were upper-cased, and descriptions were trimmed with repeated spaces collapsed.
  - Exact duplicate lines were removed.
  - Cancellation lines (invoice numbers starting with "C") were separated into the returns dataset (`returns.parquet`), where
    their negative quantities are kept. Only cancellation lines with a product stock code and a positive unit price are kept
    there; cancellation lines whose stock code does not match the product-code pattern (mostly manual adjustments, postage,
    discounts, samples, bank charges and fees) are dropped.
  - From the remaining sales lines, these were removed: stock codes that do not match the product-code pattern (five digits plus
    up to three letters, or a DCGS code; the rest are mostly postage, fees, manual adjustments, test lines and gift vouchers);
    quantities of zero or less on invoices that are not cancellations (these lines have a zero price and no customer ID); unit
    prices of zero or less; prices more than 20 times the product's median sale price; and lines of more than 50,000 units.
  - Lines without a customer ID were kept.
  - A revenue column (quantity times unit price, rounded to the penny) was added, and `products.parquet` holds one description
    per stock code (the most common spelling).
  - The other files (customer segments, product categories, description embeddings, model scores and analyst evaluation
    results) are computed from the cleaned tables.
- **Licence of the derived files:** CC BY 4.0, with the attribution above.
- The dataset's creator and the UCI Machine Learning Repository are not affiliated with this project and do not endorse it.

## 中文简介

**零售客户智能分析 + AI 数据分析师**：基于 UCI Online Retail II 公开数据集（约 100 万条真实英国网店交易记录），实现数据清洗管道、SQL 分析仪表盘（含月度队列留存）、RFM 客户分群与 A/B 测试样本量规划、90 天复购预测模型、基于句向量的商品分类，以及可以用自然语言提问的 AI 数据分析师（LLM 生成 SQL，经过只读 SQL 防护后执行，并有评测集）。

<!-- RESULTS_ZH:START -->
- **数据**：UCI Online Retail II，原始 1,067,371 行 → 清洗后 1,003,243 行销售记录，5,852 位可识别客户，43 个国家。
- **客户分群**：RFM + KMeans，k = 4；Champions 群占 20% 的客户，贡献 74% 的收入。
- **90 天复购预测**：按时间切分（不随机切分，并有防泄漏测试）。HistGradientBoosting ROC-AUC 0.795，逻辑回归 0.782；按分数排名前 20% 的客户覆盖 39% 的回购客户（随机选只有 20%）。
- **AI 数据分析师**：用自然语言提问，LLM 生成 SQL，经过只读 SQL 防护后在 DuckDB 执行；在 30 道评测题上执行准确率 93%。
- **工程化**：225 个自动化测试，GitHub Actions 持续集成。

<!-- RESULTS_ZH:END -->

技术栈：Python、DuckDB (SQL)、pandas、scikit-learn、PyTorch、MiniLM、Groq、Streamlit、pytest、GitHub Actions、Docker。运行：`make setup && make all && make app`。
