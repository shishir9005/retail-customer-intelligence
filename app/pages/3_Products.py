"""Product categories discovered from sentence embeddings, plus a 'similar products' lookup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find app/common.py when run on its own

import common
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

common.setup("Products")
st.title("Product categories")

cats = common.load_json("categories.json")
if cats is None:
    st.warning("Categories have not been built yet. Run `make products`.")
    st.stop()

st.write(
    "The data has no category column, so I created one. A sentence-embedding model (MiniLM) turns each product description "
    f"into 384 numbers, so that descriptions with similar meaning sit close together. KMeans then groups them into "
    f"**{cats['chosen_k']} categories**, and each is named by the words that characterise it most. Colour and size words "
    "were removed first so groups reflect what the item *is*."
)

rev = common.run_query("revenue_by_category")
table = pd.DataFrame([
    {"Category": c["name"], "Products": c["n_products"], "Typical products": ", ".join(p.title() for p in c["typical_products"][:3])}
    for c in cats["categories"]
]).merge(rev.rename(columns={"category": "Category", "revenue": "Revenue (£)"})[["Category", "Revenue (£)"]], on="Category", how="left")
table["% of revenue"] = table["Revenue (£)"] / table["Revenue (£)"].sum() * 100
table = table.sort_values("Revenue (£)", ascending=False)

left, right = st.columns([3, 2])
with left:
    fig = px.bar(table.iloc[::-1], x="Revenue (£)", y="Category", orientation="h", labels={"Category": ""})
    st.plotly_chart(common.style(fig, height=520, legend_top=False))
with right:
    st.dataframe(table[["Category", "Products", "% of revenue", "Typical products"]], hide_index=True, height=520,
                 column_config={"% of revenue": st.column_config.NumberColumn(format="%.1f%%")})
st.caption(
    "Names are the top words of each group, so some read oddly. 'Typical products' are the items closest to each group's centre. "
    + cats["notes"]
)
common.show_sql("revenue_by_category", "monthly_revenue_by_category")

# ── seasonality ───────────────────────────────────────────────────────────────
st.subheader("Seasonality by category")
monthly = common.run_query("monthly_revenue_by_category")
top = list(table["Category"].head(3))
xmas = [c for c in table["Category"] if "Christmas" in c]
chosen = st.multiselect("Categories", list(table["Category"]), default=list(dict.fromkeys(top + xmas)))
if chosen:
    sub = monthly[monthly["category"].isin(chosen)]
    fig = px.line(sub, x="month", y="revenue", color="category", markers=True, labels={"month": "", "revenue": "Revenue (£)"})
    st.plotly_chart(common.style(fig))
    st.caption("Category-specific peaks (e.g. Christmas items from September) are the kind of pattern a merchandising team plans stock around.")

# ── explorer + similar products ───────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def product_table() -> pd.DataFrame:
    # ORDER BY stock_code matches the row order of product_embeddings.npy (see rci/products.py)
    return common.get_con().cursor().execute(
        "SELECT stock_code, description, COALESCE(category, 'Uncategorised') AS category, units_sold, revenue, n_customers "
        "FROM products ORDER BY stock_code").df()


@st.cache_data(show_spinner=False)
def embeddings() -> np.ndarray | None:
    path = common.PROCESSED / "product_embeddings.npy"
    return np.load(path).astype(np.float32) if path.exists() else None


products = product_table()
st.subheader("Product explorer")
f1, f2 = st.columns([2, 1])
query = f1.text_input("Search descriptions", placeholder="e.g. lantern, mug, bunting")
cat_filter = f2.selectbox("Category", ["All"] + list(table["Category"]))
view = products
if query:
    view = view[view["description"].str.contains(query, case=False, na=False, regex=False)]
if cat_filter != "All":
    view = view[view["category"] == cat_filter]
st.dataframe(view.sort_values("revenue", ascending=False).head(200).rename(columns={
    "stock_code": "Code", "description": "Product", "category": "Category", "units_sold": "Units sold",
    "revenue": "Revenue (£)", "n_customers": "Customers"}), hide_index=True,
    column_config={"Revenue (£)": st.column_config.NumberColumn(format="£%.0f")})
st.caption(f"{len(view):,} products match; the 200 with the most revenue are shown.")

E = embeddings()
if E is not None and len(E) == len(products):
    st.subheader("Find similar products")
    st.write(
        "Each product is a vector of 384 numbers. **Cosine similarity** measures how closely two vectors point the same way: "
        "1 means the descriptions mean the same thing, 0 means unrelated. No model runs here: the vectors were computed offline."
    )
    ranked = products.sort_values("revenue", ascending=False)
    label = ranked["description"] + "  (" + ranked["stock_code"] + ")"
    pick = st.selectbox("Pick a product", list(label))
    code = pick.rsplit("(", 1)[1].rstrip(")")
    i = int(np.where(products["stock_code"].to_numpy() == code)[0][0])
    sims = E @ E[i]
    order = np.argsort(-sims)
    order = order[order != i][:8]
    out = products.iloc[order][["description", "category"]].assign(similarity=sims[order].round(3))
    st.dataframe(out.rename(columns={"description": "Product", "category": "Category", "similarity": "Cosine similarity"}),
                 hide_index=True)
