-- The tables the dashboard, the models and the AI analyst all query.
-- db.connect() first exposes the Parquet files as src_sales, src_returns, src_products,
-- src_categories and src_segments, then runs these blocks in order.
-- NOTE: customer_id is NULL for guest checkouts (~23% of lines, ~13% of revenue).

-- name: sales
CREATE TABLE sales AS
SELECT
    s.invoice,
    s.stock_code,
    p.description,
    c.category,
    s.quantity,
    s.invoice_date,
    s.unit_price,
    s.revenue,
    s.customer_id,
    s.country,
    date_trunc('month', s.invoice_date)::DATE AS invoice_month
FROM src_sales s
LEFT JOIN src_products   p USING (stock_code)
LEFT JOIN src_categories c USING (stock_code)
ORDER BY s.invoice_date, s.invoice, s.stock_code;

-- name: returns
CREATE TABLE returns AS
SELECT
    r.invoice,
    r.stock_code,
    p.description,
    c.category,
    r.quantity,                -- negative: units sent back
    r.invoice_date,
    r.unit_price,
    r.revenue,                 -- negative: money refunded
    r.customer_id,
    r.country,
    date_trunc('month', r.invoice_date)::DATE AS invoice_month
FROM src_returns r
LEFT JOIN src_products   p USING (stock_code)
LEFT JOIN src_categories c USING (stock_code)
ORDER BY r.invoice_date, r.invoice, r.stock_code;

-- name: products
CREATE TABLE products AS
SELECT
    p.stock_code,
    p.description,
    c.category,
    COALESCE(s.units_sold, 0)   AS units_sold,
    COALESCE(s.revenue, 0)      AS revenue,
    COALESCE(s.n_orders, 0)     AS n_orders,
    COALESCE(s.n_customers, 0)  AS n_customers,
    s.avg_unit_price,
    s.first_sold,
    s.last_sold
FROM src_products p
LEFT JOIN src_categories c USING (stock_code)
LEFT JOIN (
    SELECT stock_code,
           SUM(quantity)                 AS units_sold,
           ROUND(SUM(revenue), 2)        AS revenue,
           COUNT(DISTINCT invoice)       AS n_orders,
           COUNT(DISTINCT customer_id)   AS n_customers,
           ROUND(AVG(unit_price), 2)     AS avg_unit_price,
           MIN(invoice_date)::DATE       AS first_sold,
           MAX(invoice_date)::DATE       AS last_sold
    FROM sales
    GROUP BY stock_code
) s USING (stock_code);

-- name: customers
-- One row per identified customer: the RFM facts (see rfm.sql) plus their segment.
CREATE TABLE customers AS
SELECT r.*, g.segment
FROM (
    -- the marker below is swapped for the `rfm` block of rfm.sql, so the definition lives in one place
    RFM_QUERY
) r
LEFT JOIN src_segments g USING (customer_id);
