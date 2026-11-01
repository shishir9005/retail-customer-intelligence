-- Monthly cohort retention: of the customers who first bought in month X, what share bought
-- again in month X+1, X+2, ...?

-- name: cohort_retention
WITH orders AS (                       -- one row per customer per month they ordered
    SELECT DISTINCT customer_id, invoice_month AS order_month
    FROM sales
    WHERE customer_id IS NOT NULL
),
with_cohort AS (                       -- WINDOW FUNCTION: each customer's first month, on every row
    SELECT
        customer_id,
        order_month,
        MIN(order_month) OVER (PARTITION BY customer_id) AS cohort_month
    FROM orders
),
activity AS (
    SELECT
        cohort_month,
        date_diff('month', cohort_month, order_month) AS months_since_first,
        COUNT(DISTINCT customer_id)                   AS active_customers
    FROM with_cohort
    GROUP BY cohort_month, months_since_first
)
SELECT
    a.cohort_month,
    a.months_since_first,
    a.active_customers,
    z.active_customers                                       AS cohort_size,
    ROUND(100.0 * a.active_customers / z.active_customers, 1) AS retention_pct
FROM activity a
JOIN activity z
  ON z.cohort_month = a.cohort_month AND z.months_since_first = 0   -- month 0 = the whole cohort
ORDER BY a.cohort_month, a.months_since_first;
