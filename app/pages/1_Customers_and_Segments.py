"""RFM customer segments (KMeans) and an A/B test planner for a win-back campaign."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find app/common.py when run on its own

import common
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rci import abtest

common.setup("Customers and Segments")
st.title("Customers and segments")

seg = common.load_json("segments.json")
pts = common.load_parquet("segments.parquet")
if seg is None or pts is None:
    st.warning("Segments have not been built yet. Run `make segments`.")
    st.stop()

st.write(
    "Each customer is described by three numbers: **R**ecency (days since the last order), **F**requency "
    "(number of orders) and **M**onetary value (total spend). KMeans groups customers with similar profiles "
    f"into **{seg['chosen_k']} segments**."
)

# ── segment table ─────────────────────────────────────────────────────────────
rows = pd.DataFrame(seg["segments"])
table = pd.DataFrame({
    "Segment": rows["segment"],
    "Customers": rows["customers"],
    "% of customers": rows["customer_share"] * 100,
    "% of revenue": rows["revenue_share"] * 100,
    "Avg days since last order": rows["avg_recency_days"],
    "Avg orders": rows["avg_orders"],
    "Avg spend (£)": rows["avg_spend"],
    "Bought again (90d)": rows["repurchase_rate_next_90d"] * 100 if "repurchase_rate_next_90d" in rows else float("nan"),
})
val = seg.get("validation")
st.dataframe(
    table, hide_index=True,
    column_config={
        "% of customers": st.column_config.NumberColumn(format="%.0f%%"),
        "% of revenue": st.column_config.NumberColumn(format="%.0f%%"),
        "Bought again (90d)": st.column_config.NumberColumn(
            format="%.0f%%",
            help=f"Share who ordered in the 90 days after {seg['validation_cutoff']}, using segments refitted on data before that date only."),
        "Avg spend (£)": st.column_config.NumberColumn(format="£%.0f"),
    },
)
st.caption(
    "The first columns describe customers as of the end of the data; revenue shares are of identified customers' revenue "
    "(guest checkouts excluded). The last column is a validity check that avoids hindsight: the segments were refitted using only "
    f"orders before {seg['validation_cutoff']} (so the future could not shape the boundaries), and it shows who really ordered "
    "in the following 90 days. Segments that predict future behaviour are useful; those that don't are just a picture."
)

left, right = st.columns(2)
with left:
    st.subheader("Few customers, most of the money")
    long = pd.DataFrame({
        "Segment": list(rows["segment"]) * 2,
        "Share": list(rows["customer_share"] * 100) + list(rows["revenue_share"] * 100),
        "Measure": ["% of customers"] * len(rows) + ["% of revenue"] * len(rows),
    })
    fig = px.bar(long, x="Segment", y="Share", color="Measure", barmode="group",
                 color_discrete_map={"% of customers": "#94A3B8", "% of revenue": common.ACCENT},
                 labels={"Share": "%"})
    st.plotly_chart(common.style(fig, height=360))
with right:
    st.subheader("What to do with each segment")
    for r in seg["segments"]:
        st.markdown(f"**{r['segment']}**: {r['action']}")

# ── PCA scatter ───────────────────────────────────────────────────────────────
st.subheader("The segments on a map")
sample = pts.sample(min(2500, len(pts)), random_state=42)
fig = px.scatter(sample, x="pc1", y="pc2", color="segment", color_discrete_map=common.SEGMENT_COLORS,
                 opacity=0.6, hover_data={"recency_days": True, "frequency": True, "monetary": ":.0f", "pc1": False, "pc2": False},
                 labels={"pc1": "Component 1", "pc2": "Component 2", "segment": ""})
fig.update_traces(marker=dict(size=5))
st.plotly_chart(common.style(fig, height=460))
st.caption(
    f"PCA squeezes the three (log-scaled, standardised) R/F/M numbers onto two axes so they can be drawn; the two axes "
    f"keep {sum(seg['pca_explained_variance']):.0%} of the information. A random 2,500 of {len(pts):,} customers are shown."
)

if val:
    with st.expander("Do the segments hold up in every period? (leak-free check)"):
        recs = [{"Cutoff": e["cutoff"], "Segment": s["segment"], "Bought again": s["repurchase_rate"] * 100,
                 "Customers": s["customers"]} for e in val["by_cutoff"] for s in e["segments"]]
        vdf = pd.DataFrame(recs)
        fig = px.line(vdf, x="Cutoff", y="Bought again", color="Segment", markers=True,
                      color_discrete_map=common.SEGMENT_COLORS, category_orders={"Cutoff": [e["cutoff"] for e in val["by_cutoff"]]},
                      labels={"Bought again": "% who ordered in the next 90 days", "Cutoff": "Pretend 'today'"})
        st.plotly_chart(common.style(fig, height=340))
        st.dataframe(vdf.pivot(index="Segment", columns="Cutoff", values="Bought again").round(0).astype("Int64").reset_index(),
                     hide_index=True)
        orders = [[s["segment"] for s in e["segments"]] for e in val["by_cutoff"]]   # best to worst, per cutoff
        tops, bottoms, middles = {o[0] for o in orders}, {o[-1] for o in orders}, {tuple(o[1:-1]) for o in orders}
        reading = []
        if len(tops) == 1 and len(bottoms) == 1:
            reading.append(f"**{tops.pop()}** has the highest repurchase rate and **{bottoms.pop()}** the lowest at every date.")
        else:
            reading.append("The best and worst segments are not the same at every date, so treat the segmentation with caution.")
        if len(middles) > 1:
            reading.append("The middle segments change places between periods, so read them as two similar opportunities, not a strict ranking.")
        st.markdown(" ".join(reading))
        st.caption(
            "At each date the whole recipe (scaling, number of segments, KMeans, names) was refitted using only orders before it, "
            "then customers were followed for 90 days. Overall repurchase rates swing between periods (the autumn windows include "
            "the Christmas peak), so compare segments within a date, not across dates.")

with st.expander("How was the number of segments chosen?"):
    sil = pd.DataFrame({"k": [int(k) for k in seg["silhouette_by_k"]], "silhouette": list(seg["silhouette_by_k"].values())})
    fig = px.line(sil, x="k", y="silhouette", markers=True, labels={"k": "Number of segments (k)", "silhouette": "Silhouette score"})
    fig.add_vline(x=seg["chosen_k"], line_dash="dash", line_color=common.ACCENT, annotation_text="chosen")
    st.plotly_chart(common.style(fig, height=300, legend_top=False))
    st.write(seg["notes"])

st.divider()

# ── A/B planner ───────────────────────────────────────────────────────────────
st.header("A/B test planner: a win-back email")
st.write(
    "**Design.** Unit = one customer. Randomly send the email to half of a segment and nothing to the other half. "
    "Metric = the customer orders again within 90 days. Before sending, work out how many customers are needed."
)
names = [s["segment"] for s in seg["segments"]]
chosen = st.selectbox("Target segment", names, index=names.index("At risk") if "At risk" in names else 0)
row = next(s for s in seg["segments"] if s["segment"] == chosen)
size = int(row["customers"])
has_estimate = row.get("repurchase_rate_next_90d") is not None
observed = float(row["repurchase_rate_next_90d"]) if has_estimate else 0.30

c1, c2, c3 = st.columns(3)
baseline = c1.number_input("Baseline: bought again in 90 days with no email", min_value=0.01, max_value=0.95,
                           value=float(round(observed, 3)), step=0.01, format="%.3f",
                           help=(f"Pre-filled with the rate measured for this segment after {seg['validation_cutoff']}, using segments "
                                 "refitted on earlier data only." if has_estimate else
                                 "There is no leak-free estimate for this segment, so 30% is only a placeholder. Enter your own."))
uplift_pp = c2.slider("Smallest uplift worth detecting (percentage points)", 1.0, 20.0, 5.0, 0.5)
alpha = c3.selectbox("False-alarm rate (significance level)", [0.05, 0.10, 0.01])
power = st.select_slider("Power: chance of catching a real effect of that size", options=[0.7, 0.8, 0.9, 0.95], value=0.8)

uplift = uplift_pp / 100
if baseline + uplift >= 0.99:
    st.warning("Baseline plus uplift would exceed 99%. Lower one of them.")
    st.stop()

n = abtest.sample_size_per_group(baseline, uplift, alpha, power)
per_group_available = size // 2
m1, m2, m3 = st.columns(3)
m1.metric("Needed per group", f"{n:,}")
m2.metric("Needed in total", f"{2 * n:,}")
m3.metric(f"Customers in '{chosen}'", f"{size:,}")

if 2 * n <= size:
    st.success(f"Feasible: {2 * n:,} of the {size:,} customers in this segment are enough.")
else:
    mde = abtest.minimum_detectable_uplift(baseline, per_group_available, alpha, power)
    st.warning(
        f"The segment is too small for this test. Split in half ({per_group_available:,} per group) it could only "
        f"reliably detect an uplift of about **{mde * 100:.1f} percentage points**. Options: accept that larger "
        "effect, combine segments, or run several waves."
    )

grid = [u / 2 for u in range(2, 41) if baseline + (u / 2) / 100 < 0.99]
curve = pd.DataFrame({"uplift_pp": grid,
                      "per_group": [abtest.sample_size_per_group(baseline, g / 100, alpha, power) for g in grid]})
fig = go.Figure()
fig.add_scatter(x=curve["uplift_pp"], y=curve["per_group"], mode="lines", name="Customers needed per group",
                line=dict(color=common.ACCENT))
fig.add_scatter(x=[uplift_pp], y=[n], mode="markers", name="Your choice", marker=dict(size=11, color="#F59E0B"))
fig.add_hline(y=per_group_available, line_dash="dash", line_color="#64748B",
              annotation_text=f"available per group ({per_group_available:,})")
fig.update_layout(xaxis_title="Uplift to detect (percentage points)", yaxis_title="Customers per group", yaxis_type="log")
st.plotly_chart(common.style(fig, height=340))
st.caption("Halving the uplift you want to detect roughly quadruples the customers you need. That is why small effects need big tests.")

with st.expander("Reading the result afterwards: is the difference real?", expanded=False):
    st.write(
        "A **p-value** answers: *if the email did nothing, how often would luck alone produce a gap this big?* "
        "A small p-value (below the false-alarm rate) means 'unlikely to be luck'. It is **not** the probability the email "
        "works, and it does not say whether the gap is big enough to be worth the cost. The confidence interval shows "
        "the plausible size of the true effect."
    )
    a, b = st.columns(2)
    n_a = a.number_input("Control: customers", 10, 200000, max(n, 100), step=50)
    x_a = a.number_input("Control: bought again", 0, int(n_a), int(round(n_a * baseline)))
    n_b = b.number_input("Email group: customers", 10, 200000, max(n, 100), step=50)
    x_b = b.number_input("Email group: bought again", 0, int(n_b), min(int(n_b), int(round(n_b * (baseline + uplift)))))
    r = abtest.two_proportion_ztest(int(x_a), int(n_a), int(x_b), int(n_b), alpha)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Control rate", f"{r['rate_control']:.1%}")
    k2.metric("Email rate", f"{r['rate_treatment']:.1%}")
    k3.metric("Uplift", f"{r['uplift_abs'] * 100:+.1f} pp")
    k4.metric("p-value", f"{r['p_value']:.3f}")
    verdict = ("The gap is unlikely to be luck" if r["significant"] else "The gap could easily be luck")
    st.write(f"**{verdict}.** The true uplift is plausibly between {r['ci_low'] * 100:+.1f} and {r['ci_high'] * 100:+.1f} "
             f"percentage points ({int((1 - alpha) * 100)}% confidence interval).")
