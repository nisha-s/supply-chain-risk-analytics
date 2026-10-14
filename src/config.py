"""Project paths and constants. Every other module imports from here."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "raw" / "DataCoSupplyChainDataset.csv"
DB_PATH = DATA_DIR / "supply_chain.duckdb"
MODELS_DIR = PROJECT_ROOT / "models"
LOGREG_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.joblib"
TEST_SET_PATH = MODELS_DIR / "test_set.csv"
SCORED_ORDERS_PATH = MODELS_DIR / "scored_orders.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
SAMPLE_CSV_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample.csv"

# Raw data
CSV_ENCODING = "latin-1"

# Modeling
TARGET = "late_delivery_risk"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Model inputs, all known when the order is placed.
CATEGORICAL_FEATURES = [
    "shipping_mode",
    "payment_type",
    "market",
    "order_region",
    "customer_segment",
    "main_category",
]
NUMERIC_FEATURES = [
    "days_for_shipment_scheduled",
    "n_items",
    "total_quantity",
    "total_sales",
    "avg_discount_rate",
    "order_month",
    "order_weekday",
    "order_hour",
]

# Columns known only after delivery (target leakage). Names are after cleaning.
LEAKAGE_COLUMNS = [
    "days_for_shipping_real",
    "delivery_status",
    "shipping_date_dateorders",
    "order_status",
]

# Personal or free-text columns that are never stored or used as features.
PII_COLUMNS = [
    "customer_fname",
    "customer_lname",
    "customer_email",
    "customer_password",
    "customer_street",
    "customer_zipcode",
    "order_zipcode",
    "product_description",
    "product_image",
]

# Dashboard
TOP_RISK_ORDERS = 50
