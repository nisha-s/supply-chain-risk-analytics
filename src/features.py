"""Build the order-level feature matrix and target for modeling."""

import duckdb
import pandas as pd

from src import config
from src.queries import run_query

# One row per order. Only columns known when the order is placed are selected;
# leakage columns (real shipping days, delivery status, etc.) are never loaded.
MODEL_TABLE_SQL = """
WITH item_totals AS (
    SELECT
        order_id,
        COUNT(*) AS n_items,
        SUM(order_item_quantity) AS total_quantity,
        SUM(sales) AS total_sales,
        AVG(order_item_discount_rate) AS avg_discount_rate
    FROM order_items
    GROUP BY order_id
),
-- The main category is the category of the order line with the highest sales.
main_item AS (
    SELECT
        oi.order_id,
        p.category_name AS main_category,
        ROW_NUMBER() OVER (
            PARTITION BY oi.order_id ORDER BY oi.sales DESC, oi.order_item_id
        ) AS rank
    FROM order_items AS oi
    JOIN products AS p ON p.product_card_id = oi.product_card_id
)
SELECT
    o.order_id,
    o.order_date_dateorders,
    o.shipping_mode,
    o.type AS payment_type,
    o.market,
    o.order_region,
    o.days_for_shipment_scheduled,
    c.customer_segment,
    t.n_items,
    t.total_quantity,
    t.total_sales,
    t.avg_discount_rate,
    m.main_category,
    o.late_delivery_risk
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.order_customer_id
JOIN item_totals AS t ON t.order_id = o.order_id
JOIN main_item AS m ON m.order_id = o.order_id AND m.rank = 1
ORDER BY o.order_id
"""


def load_model_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load one row per order with raw inputs and the target."""
    return run_query(con, MODEL_TABLE_SQL)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, weekday and hour of the order date."""
    df = df.copy()
    order_date = df["order_date_dateorders"]
    df["order_month"] = order_date.dt.month
    df["order_weekday"] = order_date.dt.weekday
    df["order_hour"] = order_date.dt.hour
    return df


def check_no_leakage(columns: list[str]) -> None:
    """Raise an error if any leakage or PII column is among the features."""
    forbidden = set(config.LEAKAGE_COLUMNS) | set(config.PII_COLUMNS)
    found = forbidden.intersection(columns)
    if found:
        raise ValueError(f"Forbidden columns in features: {sorted(found)}")


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the feature columns and one-hot encode the categorical ones."""
    features = df[config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES]
    return pd.get_dummies(features, columns=config.CATEGORICAL_FEATURES, dtype=int)


def build_features(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return the feature matrix X, target y and the matching order IDs."""
    df = add_date_features(load_model_table(con))
    check_no_leakage(config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES)
    X = encode_features(df)
    return X, df[config.TARGET], df["order_id"]


if __name__ == "__main__":
    from src.queries import connect

    with connect() as connection:
        X, y, order_ids = build_features(connection)
    print(f"Orders: {len(X):,}  Features: {X.shape[1]}  Late rate: {y.mean():.4f}")
    print(X.columns.tolist())
