"""Overview page: KPIs, revenue trend, new vs returning customers, cohort retention, data quality."""
import common
import pandas as pd
import plotly.express as px
import streamlit as st

common.setup("Overview")

st.title("Retail Customer Intelligence")
st.caption(
    "Two years of real transactions from a UK online gift shop (UCI Online Retail II, CC BY 4.0). "
    "Cleaned by a tested pipeline, queried with SQL in DuckDB. Use the sidebar for segments, the "
    "repurchase model, product categories and the AI analyst."
)

h = common.run_query("headline").iloc[0]
ret = common.run_query("return_rate").iloc[0]
cols = st.columns(5)
cols[0].metric("Revenue", common.gbp(h["revenue"]))
cols[1].metric("Orders", f"{int(h['orders']):,}")
cols[2].metric("Identified customers", f"{int(h['customers']):,}")
cols[3].metric("Average order value", f"£{h['avg_order_value']:,.0f}")
cols[4].metric("Refunded (share of sales)", f"{ret['return_pct']}%")
st.caption(
    f"{pd.Timestamp(h['first_date']).date()} to {pd.Timestamp(h['last_date']).date()} · "
    f"{int(h['products']):,} products · {int(h['countries'])} countries · "
    f"{h['guest_revenue_pct']}% of revenue comes from guest checkouts without a customer ID."
)

# ── revenue trend ─────────────────────────────────────────────────────────────
st.subheader("Revenue trend")
monthly = common.run_query("monthly_revenue")
tab_rev, tab_orders = st.tabs(["Revenue", "Orders and customers"])
with tab_rev:
    fig = px.bar(monthly, x="month", y="revenue", labels={"month": "", "revenue": "Revenue (£)"})
    st.plotly_chart(common.style(fig, legend_top=False))
with tab_orders:
    long = monthly.melt(id_vars="month", value_vars=["orders", "customers"], var_name="metric", value_name="count")
    fig = px.line(long, x="month", y="count", color="metric", markers=True, labels={"month": "", "count": ""})
    st.plotly_chart(common.style(fig))
st.caption("Revenue climbs every September to November in both years (Christmas stock-up). December 2011 is partial: the data ends on 9 December.")
common.show_sql("monthly_revenue")

# ── new vs returning ──────────────────────────────────────────────────────────
st.subheader("New vs returning customers")
nv = common.run_query("new_vs_returning")
long = nv.melt(id_vars="month", var_name="type", value_name="customers")
long["type"] = long["type"].map({"new_customers": "New", "returning_customers": "Returning"})
fig = px.bar(long, x="month", y="customers", color="type", barmode="stack",
             color_discrete_map={"New": "#3B82F6", "Returning": common.ACCENT}, labels={"month": ""})
st.plotly_chart(common.style(fig))
st.caption("A customer is new in the month of their first order we can see, and returning afterwards. Dec 2009 is all 'new' because the data starts there.")
common.show_sql("new_vs_returning")

# ── cohort retention ──────────────────────────────────────────────────────────
st.subheader("Cohort retention")
coh = common.run_query("cohort_retention")
coh = coh[(coh["months_since_first"] >= 1) & (coh["months_since_first"] <= 12)]
pivot = coh.pivot(index="cohort_month", columns="months_since_first", values="retention_pct")
pivot.index = pd.to_datetime(pivot.index).strftime("%Y-%m")
fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", text_auto=".0f",
                labels={"x": "Months since first order", "y": "Cohort (month of first order)", "color": "% active"})
st.plotly_chart(common.style(fig, height=560, legend_top=False))
st.caption(
    "Each row is the group of customers who first ordered in that month; each cell is the share of them who ordered again "
    "N months later (month 0 is always 100%). Across cohorts a median of 15% of customers still order 12 months after "
    "their first order. The Dec 2009 row looks stickier (38%) because it also contains every customer who was already "
    "active when the data starts: 'first order' really means 'first order we can see'."
)
common.show_sql("cohort_retention")

# ── products and countries ────────────────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("Top products")
    tp = common.run_query("top_products", n=10)
    fig = px.bar(tp.iloc[::-1], x="revenue", y="description", orientation="h", labels={"revenue": "Revenue (£)", "description": ""})
    st.plotly_chart(common.style(fig, height=420, legend_top=False))
    common.show_sql("top_products")
with right:
    st.subheader("Top countries")
    exclude_uk = st.toggle("Exclude United Kingdom", value=True, help="The UK is about 85% of revenue and hides everything else.")
    tc = common.run_query("top_countries", n=15)
    if exclude_uk:
        tc = tc[tc["country"] != "United Kingdom"]
    fig = px.bar(tc.head(10).iloc[::-1], x="revenue", y="country", orientation="h", labels={"revenue": "Revenue (£)", "country": ""})
    st.plotly_chart(common.style(fig, height=420, legend_top=False))
    common.show_sql("top_countries")

# ── data quality ──────────────────────────────────────────────────────────────
st.subheader("Data cleaning and quality")
report = common.load_json("cleaning_report.json")
if report:
    s = report["summary"]
    st.write(
        f"The raw file has **{s['raw_rows']:,}** rows. After the rules below, **{s['sales_rows']:,}** sale lines remain "
        f"and **{s['returns_rows']:,}** return lines are kept separately."
    )
    steps = pd.DataFrame(report["steps"]).rename(
        columns={"rule": "Rule", "removed": "Rows removed", "remaining": "Rows remaining", "why": "Why"})
    st.dataframe(steps, hide_index=True)
    if "checks" in report:
        ok = sum(c["passed"] for c in report["checks"])
        with st.expander(f"Data-quality checks on the final tables: {ok} of {len(report['checks'])} pass"):
            st.dataframe(pd.DataFrame(report["checks"]).rename(
                columns={"check": "Check", "offending_rows": "Offending rows", "passed": "Passed"}), hide_index=True)
    st.caption(
        f"Guest checkouts (no customer ID) are kept: they are real revenue ({s['guest_revenue_share']:.1%}), but they "
        "are excluded from every customer-level analysis."
    )
