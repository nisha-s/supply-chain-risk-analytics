"""Data-quality checks run on the small sample CSV, never the full dataset."""

import pandas as pd
import pytest

from src import config
from src.build_db import (
    CUSTOMER_COLUMNS,
    ORDER_COLUMNS,
    ORDER_ITEM_COLUMNS,
    PRODUCT_COLUMNS,
    build_database,
    clean_columns,
    load_raw_csv,
)


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> dict[str, pd.DataFrame]:
    """Build the DuckDB tables from the sample CSV in a temporary folder."""
    db_path = tmp_path_factory.mktemp("db") / "sample.duckdb"
    return build_database(config.SAMPLE_CSV_PATH, db_path)


def test_required_columns_exist() -> None:
    """Every column the pipeline uses is present after cleaning."""
    columns = set(clean_columns(load_raw_csv(config.SAMPLE_CSV_PATH)).columns)
    required = set(ORDER_COLUMNS + ORDER_ITEM_COLUMNS + CUSTOMER_COLUMNS + PRODUCT_COLUMNS)
    assert required <= columns, f"Missing columns: {sorted(required - columns)}"


def test_no_duplicate_order_ids(tables: dict[str, pd.DataFrame]) -> None:
    """The orders table has exactly one row per order."""
    assert not tables["orders"]["order_id"].duplicated().any()


def test_target_is_binary(tables: dict[str, pd.DataFrame]) -> None:
    """The target only contains 0 and 1, with no missing values."""
    target = tables["orders"][config.TARGET]
    assert target.notna().all()
    assert set(target.unique()) <= {0, 1}
