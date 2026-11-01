-- RFM = Recency, Frequency, Monetary: the three numbers every retail analyst starts with.
-- Parameter: $as_of = "today". Only orders BEFORE this date are used.

-- name: rfm
SELECT
    customer_id,
    arg_max(country, invoice_date)                                        AS country,
    MIN(invoice_date)::DATE                                               AS first_order_date,
    MAX(invoice_date)::DATE                                               AS last_order_date,
    date_diff('day', MAX(invoice_date)::DATE, $as_of::DATE)               AS recency_days,   -- R
    COUNT(DISTINCT invoice)                                               AS frequency,      -- F
    ROUND(SUM(revenue), 2)                                                AS monetary,       -- M
    ROUND(SUM(revenue) / COUNT(DISTINCT invoice), 2)                      AS avg_order_value
FROM sales
WHERE customer_id IS NOT NULL
  AND invoice_date < $as_of
GROUP BY customer_id
-- A fixed row order matters: KMeans picks its starting centres by row position, and DuckDB's
-- grouping order otherwise changes from run to run, which moved a few customers between segments.
ORDER BY customer_id
