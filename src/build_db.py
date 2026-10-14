"""Load the raw DataCo CSV, clean it, split it into tables and save them to DuckDB."""

import re
from pathlib import Path

import duckdb
import pandas as pd

from src import config

ORDER_COLUMNS = [
    "order_id",
    "order_customer_id",
    "order_date_dateorders",
    "shipping_date_dateorders",
    "type",
    "shipping_mode",
    "days_for_shipment_scheduled",
    "days_for_shipping_real",
    "delivery_status",
    "order_status",
    "market",
    "order_region",
    "order_country",
    "order_state",
    "order_city",
    config.TARGET,
]

ORDER_ITEM_COLUMNS = [
    "order_item_id",
    "order_id",
    "product_card_id",
    "order_item_quantity",
    "order_item_product_price",
    "order_item_discount",
    "order_item_discount_rate",
    "sales",
    "order_item_total",
]

CUSTOMER_COLUMNS = [
    "customer_id",
    "customer_segment",
    "customer_city",
    "customer_state",
    "customer_country",
]

PRODUCT_COLUMNS = [
    "product_card_id",
    "product_name",
    "product_price",
    "product_category_id",
    "category_name",
    "department_id",
    "department_name",
]

DATE_FORMAT = "%m/%d/%Y %H:%M"


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Read the raw CSV with the encoding the file needs."""
    return pd.read_csv(path, encoding=config.CSV_ENCODING)


def clean_column_name(name: str) -> str:
    """Turn e.g. 'Days for shipping (real)' into 'days_for_shipping_real'."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    return name.strip("_")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all columns to snake_case and drop PII columns."""
    df = df.rename(columns=clean_column_name)
    return df.drop(columns=config.PII_COLUMNS, errors="ignore")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the order and shipping date strings to datetimes."""
    df = df.copy()
    for column in ["order_date_dateorders", "shipping_date_dateorders"]:
        df[column] = pd.to_datetime(df[column], format=DATE_FORMAT)
    return df


def unique_rows(df: pd.DataFrame, columns: list[str], key: str) -> pd.DataFrame:
    """Select columns, drop duplicate rows and check the key is unique."""
    table = df[columns].drop_duplicates().reset_index(drop=True)
    # If the key still repeats, the columns are not constant per key.
    if table[key].duplicated().any():
        raise ValueError(f"'{key}' is not unique after dropping duplicate rows.")
    return table


def split_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the order-line data into orders, order_items, customers and products."""
    return {
        "orders": unique_rows(df, ORDER_COLUMNS, "order_id"),
        "order_items": unique_rows(df, ORDER_ITEM_COLUMNS, "order_item_id"),
        "customers": unique_rows(df, CUSTOMER_COLUMNS, "customer_id"),
        "products": unique_rows(df, PRODUCT_COLUMNS, "product_card_id"),
    }


def save_tables(tables: dict[str, pd.DataFrame], db_path: Path) -> None:
    """Write each table to the DuckDB file, replacing old versions."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for name, table in tables.items():
            con.register("table_df", table)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM table_df")
            con.unregister("table_df")


def build_database(csv_path: Path, db_path: Path) -> dict[str, pd.DataFrame]:
    """Run the full load -> clean -> split -> save pipeline."""
    df = parse_dates(clean_columns(load_raw_csv(csv_path)))
    tables = split_tables(df)
    save_tables(tables, db_path)
    return tables


if __name__ == "__main__":
    built = build_database(config.RAW_CSV_PATH, config.DB_PATH)
    for table_name, table_df in built.items():
        print(f"{table_name}: {len(table_df):,} rows")
    print(f"Saved to {config.DB_PATH}")
