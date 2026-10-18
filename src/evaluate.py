"""Evaluate both saved models on the held-out test set and save the results."""

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config

THRESHOLD = 0.5


def load_test_set() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the test set saved by train.py as X, y and order IDs."""
    test_set = pd.read_csv(config.TEST_SET_PATH)
    y = test_set.pop(config.TARGET)
    order_ids = test_set.pop("order_id")
    return test_set, y, order_ids


def evaluate_model(name: str, y_true: pd.Series, y_prob: pd.Series) -> dict:
    """Compute ROC-AUC, precision, recall, F1 and the confusion matrix."""
    y_pred = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_scored_orders(order_ids: pd.Series, y_true: pd.Series, y_prob: pd.Series) -> None:
    """Save each test order's predicted late risk for the dashboard."""
    scored = pd.DataFrame({
        "order_id": order_ids,
        config.TARGET: y_true,
        "predicted_risk": y_prob.round(4),
    })
    scored.sort_values("predicted_risk", ascending=False).to_csv(
        config.SCORED_ORDERS_PATH, index=False
    )


if __name__ == "__main__":
    X_test, y_test, ids_test = load_test_set()
    models = {
        "logistic_regression": joblib.load(config.LOGREG_MODEL_PATH),
        "xgboost": joblib.load(config.XGB_MODEL_PATH),
    }

    results = []
    for model_name, model in models.items():
        probabilities = pd.Series(model.predict_proba(X_test)[:, 1])
        results.append(evaluate_model(model_name, y_test, probabilities))
        if model_name == "xgboost":
            save_scored_orders(ids_test, y_test, probabilities)

    metrics = pd.DataFrame(results)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.METRICS_PATH, index=False)

    print(f"Test orders: {len(y_test):,}  Late rate: {y_test.mean():.4f}  Threshold: {THRESHOLD}\n")
    print(metrics.to_string(index=False))
    print(f"\nSaved {config.METRICS_PATH} and {config.SCORED_ORDERS_PATH}")
