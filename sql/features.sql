-- Repurchase-model features for ONE "pretend today".
-- Parameters: $cutoff = the pretend today, $horizon = days ahead the label looks (90).
--
-- LEAKAGE RULE: every feature below only reads rows with invoice_date < $cutoff.
-- Only the `future` CTE (the label) reads rows on or after $cutoff.
-- tests/test_no_leakage.py proves this by changing future rows and checking features don't move.

-- name: features
WITH hist AS (
    SELECT invoice, stock_code, invoice_date, revenue, country, customer_id
    FROM sales
    WHERE customer_id IS NOT NULL
      AND invoice_date < $cutoff
),
orders AS (                                   -- one row per invoice
    SELECT customer_id, invoice,
           MIN(invoice_date)            AS order_date,
           SUM(revenue)                 AS order_revenue,
           COUNT(*)                     AS n_lines,
           arg_max(country, invoice_date) AS country
    FROM hist
    GROUP BY customer_id, invoice
),
gaps AS (                                     -- days since the customer's previous order
    SELECT customer_id,
           date_diff('day',
                     LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date),
                     order_date) AS gap_days
    FROM orders
),
gap_stats AS (
    SELECT customer_id, AVG(gap_days) AS avg_gap_days FROM gaps GROUP BY customer_id
),
product_stats AS (
    SELECT customer_id, COUNT(DISTINCT stock_code) AS n_products FROM hist GROUP BY customer_id
),
return_stats AS (
    SELECT customer_id, COUNT(DISTINCT invoice) AS n_returns
    FROM returns
    WHERE customer_id IS NOT NULL AND invoice_date < $cutoff
    GROUP BY customer_id
),
order_stats AS (
    SELECT
        customer_id,
        date_diff('day', MAX(order_date)::DATE, $cutoff::DATE)  AS recency_days,
        COUNT(*)                                                AS frequency,
        SUM(order_revenue)                                      AS monetary,
        AVG(order_revenue)                                      AS avg_order_value,
        date_diff('day', MIN(order_date)::DATE, $cutoff::DATE)  AS tenure_days,
        AVG(n_lines)                                            AS avg_lines_per_order,
        COUNT(*) FILTER (WHERE order_date >= $cutoff::DATE - INTERVAL 30 DAY) AS orders_last_30d,
        COUNT(*) FILTER (WHERE order_date >= $cutoff::DATE - INTERVAL 90 DAY) AS orders_last_90d,
        COALESCE(SUM(order_revenue) FILTER (WHERE order_date >= $cutoff::DATE - INTERVAL 90 DAY), 0)
                                                                AS revenue_last_90d,
        (arg_max(country, order_date) = 'United Kingdom')::INTEGER AS is_uk
    FROM orders
    GROUP BY customer_id
),
future AS (                                   -- the ONLY place that looks past the cutoff
    SELECT DISTINCT customer_id
    FROM sales
    WHERE customer_id IS NOT NULL
      AND invoice_date >= $cutoff
      AND invoice_date <  $cutoff::DATE + to_days($horizon)
)
SELECT
    o.customer_id,
    o.recency_days,
    o.frequency,
    ROUND(o.monetary, 2)                                  AS monetary,
    ROUND(o.avg_order_value, 2)                           AS avg_order_value,
    o.tenure_days,
    ROUND(o.avg_lines_per_order, 2)                       AS avg_lines_per_order,
    o.orders_last_30d,
    o.orders_last_90d,
    ROUND(o.revenue_last_90d, 2)                          AS revenue_last_90d,
    p.n_products,
    COALESCE(r.n_returns, 0)                              AS n_returns,
    -- a one-order customer has no gap; use the time since that order (a lower bound)
    ROUND(COALESCE(g.avg_gap_days, o.tenure_days), 1)     AS avg_gap_days,
    o.is_uk,
    (f.customer_id IS NOT NULL)::INTEGER                  AS label
FROM order_stats o
JOIN product_stats p USING (customer_id)
LEFT JOIN gap_stats g USING (customer_id)
LEFT JOIN return_stats r USING (customer_id)
LEFT JOIN future f USING (customer_id)
ORDER BY o.customer_id;

-- name: category_revenue
-- Revenue per customer per product category BEFORE the cutoff (turned into shares in features.py).
SELECT customer_id, category, SUM(revenue) AS revenue
FROM sales
WHERE customer_id IS NOT NULL
  AND invoice_date < $cutoff
  AND category IS NOT NULL
GROUP BY customer_id, category;
