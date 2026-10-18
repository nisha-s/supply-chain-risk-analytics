"""SQL queries on the DuckDB tables for late-delivery analysis."""

from pathlib import Path

import duckdb
import pandas as pd

from src import config

KPI_SQL = """
SELECT
    COUNT(*) AS total_orders,
    CAST(SUM(late_delivery_risk) AS INTEGER) AS late_orders,
    ROUND(AVG(late_delivery_risk), 4) AS late_rate,
    ROUND(AVG(CASE WHEN delivery_status IN ('Shipping on time', 'Advance shipping')
                   THEN 1 ELSE 0 END), 4) AS on_time_rate,
    ROUND(AVG(days_for_shipment_scheduled), 2) AS avg_days_scheduled,
    ROUND(AVG(days_for_shipping_real), 2) AS avg_days_real
FROM orders
"""

BY_SHIPPING_MODE_SQL = """
SELECT
    o.shipping_mode,
    COUNT(*) AS orders,
    ROUND(AVG(o.late_delivery_risk), 4) AS late_rate,
    ROUND(AVG(o.days_for_shipment_scheduled), 2) AS avg_days_scheduled,
    ROUND(AVG(o.days_for_shipping_real), 2) AS avg_days_real
FROM orders AS o
GROUP BY o.shipping_mode
ORDER BY late_rate DESC
"""

BY_REGION_SQL = """
SELECT
    o.market,
    o.order_region,
    COUNT(*) AS orders,
    ROUND(AVG(o.late_delivery_risk), 4) AS late_rate
FROM orders AS o
GROUP BY o.market, o.order_region
ORDER BY late_rate DESC
"""

# An order can contain several categories, so count each order once per category.
BY_CATEGORY_SQL = """
SELECT
    p.category_name,
    p.department_name,
    COUNT(DISTINCT o.order_id) AS orders,
    COUNT(DISTINCT CASE WHEN o.late_delivery_risk = 1 THEN o.order_id END) AS late_orders,
    ROUND(late_orders / orders, 4) AS late_rate
FROM orders AS o
JOIN order_items AS oi ON oi.order_id = o.order_id
JOIN products AS p ON p.product_card_id = oi.product_card_id
GROUP BY p.category_name, p.department_name
ORDER BY late_rate DESC
"""

BY_MONTH_SQL = """
SELECT
    DATE_TRUNC('month', o.order_date_dateorders) AS order_month,
    COUNT(*) AS orders,
    ROUND(AVG(o.late_delivery_risk), 4) AS late_rate
FROM orders AS o
GROUP BY order_month
ORDER BY order_month
"""


def connect(db_path: Path = config.DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB file read-only."""
    return duckdb.connect(str(db_path), read_only=True)


def run_query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Run a SQL string and return the result as a DataFrame."""
    return con.execute(sql).df()


def on_time_kpis(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Overall order counts, late rate, on-time rate and average shipping days."""
    return run_query(con, KPI_SQL)


def late_rate_by_shipping_mode(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Late rate and shipping days per shipping mode."""
    return run_query(con, BY_SHIPPING_MODE_SQL)


def late_rate_by_region(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Late rate per market and order region."""
    return run_query(con, BY_REGION_SQL)


def late_rate_by_category(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Late rate per product category (orders joined to items and products)."""
    return run_query(con, BY_CATEGORY_SQL)


def late_rate_by_month(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Late rate per order month."""
    return run_query(con, BY_MONTH_SQL)


if __name__ == "__main__":
    with connect() as connection:
        print("On-time KPIs\n", on_time_kpis(connection).to_string(index=False), "\n")
        print("By shipping mode\n", late_rate_by_shipping_mode(connection).to_string(index=False), "\n")
        print("By region\n", late_rate_by_region(connection).to_string(index=False), "\n")
        print("By category\n", late_rate_by_category(connection).to_string(index=False), "\n")
        print("By month\n", late_rate_by_month(connection).to_string(index=False))
