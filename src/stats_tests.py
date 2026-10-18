"""Statistical tests of what drives late deliveries: chi-square tests and a logit model."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

from src import config
from src.queries import connect, run_query

ORDERS_WITH_SEGMENT_SQL = """
SELECT
    o.order_id,
    o.shipping_mode,
    o.market,
    o.order_region,
    o.type AS payment_type,
    c.customer_segment,
    o.late_delivery_risk
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.order_customer_id
"""

# Only pre-delivery variables. Scheduled days are left out because they are
# fully determined by shipping mode, which would make the model singular.
LOGIT_FORMULA = (
    f"{config.TARGET} ~ C(shipping_mode, Treatment('Standard Class'))"
    " + C(market) + C(customer_segment) + C(payment_type)"
)


def load_orders() -> pd.DataFrame:
    """Load one row per order with the variables used in the tests."""
    with connect() as con:
        return run_query(con, ORDERS_WITH_SEGMENT_SQL)


def chi_square_test(df: pd.DataFrame, column: str) -> dict:
    """Chi-square test of independence between a column and the target."""
    table = pd.crosstab(df[column], df[config.TARGET])
    chi2, p_value, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    # Cramér's V: effect size from 0 (no association) to 1 (perfect association).
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {
        "variable": column,
        "chi2": round(chi2, 2),
        "dof": dof,
        "p_value": p_value,
        "cramers_v": round(cramers_v, 4),
        "n": n,
    }


def fit_logit(df: pd.DataFrame):
    """Fit a logistic regression of the target on pre-delivery variables."""
    return smf.logit(LOGIT_FORMULA, data=df).fit(disp=False)


def odds_ratio_table(result) -> pd.DataFrame:
    """Turn a fitted logit into a table of odds ratios, 95% CIs and p-values."""
    conf_int = np.exp(result.conf_int())
    table = pd.DataFrame({
        "odds_ratio": np.exp(result.params),
        "ci_low": conf_int[0],
        "ci_high": conf_int[1],
        "p_value": result.pvalues,
    })
    table.index = table.index.str.replace(r"C\((\w+).*?\)\[T\.(.+)\]", r"\1 = \2", regex=True)
    return table.drop(index="Intercept").round(4)


if __name__ == "__main__":
    orders = load_orders()
    print(f"Orders: {len(orders):,}  Late rate: {orders[config.TARGET].mean():.4f}\n")

    chi_results = pd.DataFrame([
        chi_square_test(orders, "shipping_mode"),
        chi_square_test(orders, "order_region"),
    ])
    print("Chi-square tests (late vs. variable)")
    print(chi_results.to_string(index=False), "\n")

    logit = fit_logit(orders)
    print(f"Logistic regression (n={int(logit.nobs):,}, pseudo R2={logit.prsquared:.4f},"
          f" LLR p-value={logit.llr_pvalue:.3g})")
    print("Reference levels: shipping_mode = Standard Class; others = first alphabetically")
    print(odds_ratio_table(logit).to_string())
