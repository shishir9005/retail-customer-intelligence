-- Dashboard queries. Each block is one query; db.load_queries() reads them by name.
-- "Revenue" counts every clean sale line, including guest checkouts. Customer metrics only
-- use rows where customer_id IS NOT NULL.

-- name: headline
SELECT
    ROUND(SUM(revenue), 0)                                         AS revenue,
    COUNT(DISTINCT invoice)                                        AS orders,
    COUNT(DISTINCT customer_id)                                    AS customers,
    ROUND(SUM(revenue) / COUNT(DISTINCT invoice), 2)               AS avg_order_value,
    COUNT(DISTINCT stock_code)                                     AS products,
    COUNT(DISTINCT country)                                        AS countries,
    MIN(invoice_date)::DATE                                        AS first_date,
    MAX(invoice_date)::DATE                                        AS last_date,
    ROUND(100 * SUM(revenue) FILTER (WHERE customer_id IS NULL) / SUM(revenue), 1) AS guest_revenue_pct
FROM sales;

-- name: return_rate
-- Money refunded as a share of money sold.
SELECT ROUND(100 * ABS((SELECT SUM(revenue) FROM returns)) / (SELECT SUM(revenue) FROM sales), 1) AS return_pct;

-- name: monthly_revenue
SELECT
    invoice_month                       AS month,
    ROUND(SUM(revenue), 0)              AS revenue,
    COUNT(DISTINCT invoice)             AS orders,
    COUNT(DISTINCT customer_id)         AS customers
FROM sales
GROUP BY invoice_month
ORDER BY invoice_month;

-- name: top_products
SELECT
    description,
    category,
    SUM(quantity)                       AS units_sold,
    ROUND(SUM(revenue), 0)              AS revenue
FROM sales
GROUP BY description, category
ORDER BY revenue DESC
LIMIT $n;

-- name: top_countries
SELECT
    country,
    ROUND(SUM(revenue), 0)              AS revenue,
    COUNT(DISTINCT customer_id)         AS customers,
    COUNT(DISTINCT invoice)             AS orders
FROM sales
GROUP BY country
ORDER BY revenue DESC
LIMIT $n;

-- name: new_vs_returning
-- A customer is "new" in the month of their first order and "returning" afterwards.
-- The window function MIN(...) OVER (PARTITION BY customer_id) finds that first month per customer.
WITH monthly AS (
    SELECT DISTINCT customer_id, invoice_month
    FROM sales
    WHERE customer_id IS NOT NULL
),
flagged AS (
    SELECT
        invoice_month,
        customer_id,
        invoice_month = MIN(invoice_month) OVER (PARTITION BY customer_id) AS is_new
    FROM monthly
)
SELECT
    invoice_month                                   AS month,
    COUNT(*) FILTER (WHERE is_new)                  AS new_customers,
    COUNT(*) FILTER (WHERE NOT is_new)              AS returning_customers
FROM flagged
GROUP BY invoice_month
ORDER BY invoice_month;

-- name: revenue_by_category
SELECT
    COALESCE(category, 'Uncategorised')   AS category,
    ROUND(SUM(revenue), 0)                AS revenue,
    COUNT(DISTINCT stock_code)            AS products
FROM sales
GROUP BY 1
ORDER BY revenue DESC;

-- name: monthly_revenue_by_category
SELECT
    invoice_month                         AS month,
    COALESCE(category, 'Uncategorised')   AS category,
    ROUND(SUM(revenue), 0)                AS revenue
FROM sales
GROUP BY 1, 2
ORDER BY 1, 2;

-- name: revenue_by_weekday
SELECT
    dayname(invoice_date)                 AS weekday,
    isodow(invoice_date)                  AS weekday_num,
    COUNT(DISTINCT invoice)               AS orders,
    ROUND(SUM(revenue), 0)                AS revenue
FROM sales
GROUP BY 1, 2
ORDER BY 2;
