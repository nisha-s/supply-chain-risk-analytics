"""Train a logistic regression baseline and an XGBoost classifier, then save them."""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config
from src.features import build_features
from src.queries import connect


def split_data(X: pd.DataFrame, y: pd.Series, order_ids: pd.Series) -> tuple:
    """Stratified train/test split with a fixed seed."""
    return train_test_split(
        X, y, order_ids,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit a scaled logistic regression (scaling helps the solver converge)."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=config.LOGREG_MAX_ITER)),
    ])
    return model.fit(X_train, y_train)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Fit an XGBoost classifier."""
    model = XGBClassifier(**config.XGB_PARAMS, random_state=config.RANDOM_SEED)
    return model.fit(X_train, y_train)


def save_test_set(X_test: pd.DataFrame, y_test: pd.Series, ids_test: pd.Series) -> None:
    """Save the held-out rows so evaluate.py scores exactly the same orders."""
    test_set = X_test.copy()
    test_set.insert(0, "order_id", ids_test)
    test_set[config.TARGET] = y_test
    test_set.to_csv(config.TEST_SET_PATH, index=False)


if __name__ == "__main__":
    with connect() as connection:
        X, y, order_ids = build_features(connection)
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_data(X, y, order_ids)
    print(f"Train: {len(X_train):,} orders  Test: {len(X_test):,} orders")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(train_logistic_regression(X_train, y_train), config.LOGREG_MODEL_PATH)
    joblib.dump(train_xgboost(X_train, y_train), config.XGB_MODEL_PATH)
    save_test_set(X_test, y_test, ids_test)
    print(f"Saved models and test set to {config.MODELS_DIR}")
