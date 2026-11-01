"""The 90-day repurchase model: honest comparison of four scorers, with uncertainty."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find app/common.py when run on its own

import common
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

common.setup("Repurchase model")
st.title("Will this customer buy again in the next 90 days?")

m = common.load_json("metrics.json")
if m is None:
    st.warning("The model has not been trained yet. Run `make model`.")
    st.stop()

gbm = m["gbm_model"]
models = m["models"]
best = models[gbm]
base = m["base_rate_test"]

st.write(
    "For each customer I pretend it is a certain date (a **cutoff**). Features come only from orders **before** it; "
    "the label is whether they order in the **90 days after**. Scores let a marketing team contact the most promising "
    "customers first."
)

# ── honest split ──────────────────────────────────────────────────────────────
st.subheader("How the data is split: by time, never randomly")
tl = []
for split, info in m["splits"].items():
    for i, (cut, end) in enumerate(zip(info["cutoffs"], info["label_window_ends"]), start=1):
        name = f"{split} {i}" if len(info["cutoffs"]) > 1 else split
        tl.append({"snapshot": name, "start": "2009-12-01", "end": cut, "phase": "Features (history before cutoff)", "split": split})
        tl.append({"snapshot": name, "start": cut, "end": end, "phase": "Label (next 90 days)", "split": split})
tl_df = pd.DataFrame(tl)
fig = px.timeline(tl_df, x_start="start", x_end="end", y="snapshot", color="phase",
                  color_discrete_map={"Features (history before cutoff)": "#CBD5E1", "Label (next 90 days)": common.ACCENT})
fig.update_yaxes(autorange="reversed", title=None)
st.plotly_chart(common.style(fig, height=300))
st.caption(
    f"Train on the four earlier cutoffs, choose settings on the validation cutoff, report on the latest one "
    f"({m['splits']['test']['cutoffs'][0]}). Every label window ends before the next split's cutoff, so nothing from "
    f"the future leaks into training. A unit test enforces this. Test set: {m['test_customers']:,} customers, "
    f"{base:.1%} of whom bought again."
)

# ── results ───────────────────────────────────────────────────────────────────
st.subheader("Results on the held-out test period")


def cell(d: dict, pct: bool = False) -> str:
    if pct:
        return f"{d['value']:.1%} ({d['ci_low']:.1%} to {d['ci_high']:.1%})"
    return f"{d['value']:.3f} ({d['ci_low']:.3f} to {d['ci_high']:.3f})"


results = pd.DataFrame([
    {"Model": name, "ROC-AUC": cell(d["roc_auc"]), "PR-AUC": cell(d["pr_auc"]),
     "Top 20% contains": cell(d["top20_capture"], True), "Top 20% who really return": cell(d["top20_precision"], True)}
    for name, d in models.items()
])
st.dataframe(results, hide_index=True)
st.caption("Numbers in brackets are 95% bootstrap intervals: the test customers were re-drawn 1,000 times to see how much each score would move by luck.")

k1, k2, k3 = st.columns(3)
k1.metric("Overall repurchase rate", f"{base:.0%}")
k2.metric("Of the top 20%, really return", f"{best['top20_precision']['value']:.0%}",
          help=f"Customers ranked highest by the {gbm} score.")
k3.metric("Returners found in top 20%", f"{best['top20_capture']['value']:.0%}",
          help="A random 20% of customers would contain 20%.")
st.info(
    f"**What this means for marketing.** If the team can only contact 20% of customers, contacting the {gbm} top 20% reaches "
    f"{best['top20_capture']['value']:.0%} of everyone who will buy again (a random list reaches 20%), and "
    f"{best['top20_precision']['value']:.0%} of the people on the list really do return, against {base:.0%} overall."
)

diff = m["auc_diff_vs_logreg"]
d_gbm = diff[gbm]
st.write(
    f"**Is the fancier model really better?** {gbm} beats logistic regression by {d_gbm['value']:+.3f} ROC-AUC "
    f"(95% interval {d_gbm['ci_low']:+.3f} to {d_gbm['ci_high']:+.3f}). The interval excludes zero, so the gain is real, "
    "but it is small: most of the signal sits in simple recency and frequency features, which even the one-line "
    "'recency rule' captures well."
)

tab_gain, tab_roc, tab_pr = st.tabs(["Gain curve", "ROC curve", "Precision-recall curve"])
curves = m["curves"]
with tab_gain:
    fig = go.Figure()
    xs = [p[0] for p in next(iter(curves.values()))["gain"]]
    fig.add_scatter(x=xs, y=[min(x / base, 1.0) for x in xs], mode="lines", name="Perfect list",
                    line=dict(color="#CBD5E1", dash="dot"))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random list", line=dict(color="#94A3B8", dash="dash"))
    for name, c in curves.items():
        fig.add_scatter(x=[p[0] for p in c["gain"]], y=[p[1] for p in c["gain"]], mode="lines", name=name,
                        line=dict(color=common.MODEL_COLORS.get(name)))
    fig.update_layout(xaxis_title="Share of customers contacted (best-scored first)",
                      yaxis_title="Share of all returning customers reached")
    st.plotly_chart(common.style(fig, height=420))
with tab_roc:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash"))
    for name, c in curves.items():
        fig.add_scatter(x=[p[0] for p in c["roc"]], y=[p[1] for p in c["roc"]], mode="lines", name=name,
                        line=dict(color=common.MODEL_COLORS.get(name)))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(common.style(fig, height=420))
with tab_pr:
    fig = go.Figure()
    fig.add_hline(y=base, line_dash="dash", line_color="#94A3B8", annotation_text="repurchase rate (random list)")
    for name, c in curves.items():
        fig.add_scatter(x=[p[0] for p in c["pr"]], y=[p[1] for p in c["pr"]], mode="lines", name=name,
                        line=dict(color=common.MODEL_COLORS.get(name)))
    fig.update_layout(xaxis_title="Recall (share of returners found)", yaxis_title="Precision (share of picks who return)")
    st.plotly_chart(common.style(fig, height=420))

# ── importance ────────────────────────────────────────────────────────────────
st.subheader("What drives the prediction")
imp = pd.DataFrame(m["importance"]).head(12).iloc[::-1]
fig = px.bar(imp, x="mean", y="label", orientation="h", error_x="std",
             labels={"mean": "Drop in ROC-AUC when the feature is shuffled", "label": ""})
st.plotly_chart(common.style(fig, height=400, legend_top=False))
st.caption(
    "Permutation importance, measured on the validation period: shuffle one feature's values and see how much the model gets "
    "worse. Recency and order frequency dominate; spend and recent activity come next."
)

# ── contact list ──────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def contact_table() -> pd.DataFrame | None:
    scores = common.load_parquet("current_scores.parquet")
    if scores is None:
        return None
    info = common.get_con().cursor().execute("SELECT customer_id, segment, country, frequency AS orders FROM customers").df()
    return scores.merge(info, on="customer_id", how="left")


st.subheader("Who to contact")
tbl = contact_table()
if tbl is not None:
    mode = st.radio("List type", ["Win-back priority", "Most likely to buy again"], horizontal=True)
    n_rows = st.slider("Customers to show", 10, 100, 25, 5)
    if mode == "Win-back priority":
        tbl = tbl.assign(priority=tbl["monetary"] * (1 - tbl["p_repurchase_90d"])).sort_values("priority", ascending=False)
        note = ("Sorted by lifetime spend x chance of NOT returning: valuable customers who are slipping away. "
                "This priority score is a simple heuristic for the demo, not a validated business rule.")
    else:
        tbl = tbl.sort_values("p_repurchase_90d", ascending=False)
        note = "Sorted by the model's probability of ordering again within 90 days."
    show = tbl.head(n_rows).rename(columns={
        "customer_id": "Customer", "segment": "Segment", "country": "Country", "orders": "Orders",
        "monetary": "Lifetime spend (£)", "recency_days": "Days since last order", "p_repurchase_90d": "P(buys in 90 days)"})
    st.dataframe(show[["Customer", "Segment", "Country", "Orders", "Lifetime spend (£)", "Days since last order", "P(buys in 90 days)"]],
                 hide_index=True,
                 column_config={"P(buys in 90 days)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0f%%"),
                                "Lifetime spend (£)": st.column_config.NumberColumn(format="£%.0f")})
    st.caption(note + " Scores are as of 10 Dec 2011, the day after the data ends. Use them to rank customers: the absolute "
               "percentages depend on the season (the test window includes Christmas).")

gbm_note = "" if m["lightgbm_used"] else (
    " (LightGBM was not available on this machine, so scikit-learn's HistGradientBoosting, "
    "the same algorithm family, was used)")

with st.expander("Interview notes: why these choices?"):
    st.markdown(
        f"""
- **Time split, not random.** A random split would put the same customer's future orders in training and test, so the
  model would look better than it is. Here the test period is strictly later than everything the model learned from.
- **PR-AUC and 'top 20%' alongside ROC-AUC.** Marketing contacts a limited list, so what matters is how good the *top* of
  the ranking is. Plain accuracy would hide that.
- **Baselines first.** A model only matters if it beats something simple: here the one-line recency rule and logistic regression.
- **Why did {gbm} beat the neural network (if it did)?** About 15,000 rows of skewed, mixed-scale table data is where trees
  shine: they need no scaling and find interactions automatically. The MLP needs more tuning and more data to compete,
  and the gap here is small.
- **Gradient boosting library:** this run used **{gbm}**{gbm_note}.
- **Caveats:** one test period; it includes the Christmas peak; brand-new customers (no history) are not scored.
        """
    )
