"""Streamlit dashboard: on-time KPIs, late rate breakdowns and highest-risk orders."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src import queries

ORDER_DETAILS_SQL = """
SELECT
    order_id,
    order_date_dateorders AS order_date,
    shipping_mode,
    days_for_shipment_scheduled AS days_scheduled,
    market,
    order_region
FROM orders
"""


@st.cache_data
def load_query(name: str) -> pd.DataFrame:
    """Run one of the named analysis queries against the DuckDB file."""
    query_functions = {
        "kpis": queries.on_time_kpis,
        "mode": queries.late_rate_by_shipping_mode,
        "region": queries.late_rate_by_region,
        "category": queries.late_rate_by_category,
        "order_details": lambda con: queries.run_query(con, ORDER_DETAILS_SQL),
    }
    with queries.connect() as con:
        return query_functions[name](con)


@st.cache_data
def load_top_risk_orders() -> pd.DataFrame:
    """Join the model's highest-risk test orders with their order details."""
    scored = pd.read_csv(config.SCORED_ORDERS_PATH).head(config.TOP_RISK_ORDERS)
    table = scored.merge(load_query("order_details"), on="order_id", how="left")
    return table.rename(columns={config.TARGET: "actually_late"})


def late_rate_bar(df: pd.DataFrame, label_column: str, title: str):
    """Horizontal bar chart of late rate, highest at the top."""
    df = df.sort_values("late_rate")
    fig = px.bar(
        df,
        x="late_rate",
        y=label_column,
        orientation="h",
        text="late_rate",
        hover_data={"orders": ":,", "late_rate": ":.1%"},
        title=title,
        color_discrete_sequence=[config.BAR_COLOR],
    )
    fig.update_traces(
        texttemplate="%{text:.1%}", textposition="outside", textfont_size=12, cliponaxis=False
    )
    fig.update_layout(
        xaxis_title="Late rate",
        xaxis_tickformat=".0%",
        yaxis_title=None,
        yaxis_dtick=1,  # show every category label
        yaxis_automargin=True,  # widen the left margin so long labels are not cut off
        barcornerradius=4,
        height=140 + 32 * len(df),
        margin={"r": 60},
    )
    return fig


def show_kpis() -> None:
    """Show the headline on-time KPIs as metric tiles."""
    kpis = load_query("kpis").iloc[0]
    columns = st.columns(4)
    columns[0].metric("Orders", f"{int(kpis['total_orders']):,}")
    columns[1].metric("Late rate", f"{kpis['late_rate']:.1%}")
    columns[2].metric("On-time rate", f"{kpis['on_time_rate']:.1%}")
    columns[3].metric(
        "Avg shipping days (real vs. scheduled)",
        f"{kpis['avg_days_real']:.2f} vs. {kpis['avg_days_scheduled']:.2f}",
    )


def show_breakdowns() -> None:
    """Show late rate by shipping mode, region and product category."""
    st.plotly_chart(late_rate_bar(load_query("mode"), "shipping_mode", "Late rate by shipping mode"))
    st.plotly_chart(late_rate_bar(load_query("region"), "order_region", "Late rate by order region"))
    category = load_query("category")
    # Some category names exist in two departments (e.g. Electronics), so label both.
    category["category"] = category["category_name"] + " (" + category["department_name"] + ")"
    st.plotly_chart(late_rate_bar(category, "category", "Late rate by product category"))


def show_top_risk_orders() -> None:
    """Show the test-set orders the XGBoost model rates most likely to be late."""
    st.subheader(f"Top {config.TOP_RISK_ORDERS} highest-risk orders (XGBoost, test set)")
    st.caption("Orders the model did not see during training. 'actually_late' is the real outcome.")
    st.dataframe(load_top_risk_orders(), hide_index=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Late-Delivery Risk", layout="wide")
    st.title("Supply Chain Late-Delivery Risk")

    if not config.DB_PATH.exists() or not config.SCORED_ORDERS_PATH.exists():
        st.error("Missing data. Run build_db, train and evaluate first (see README).")
        st.stop()

    show_kpis()
    show_breakdowns()
    show_top_risk_orders()
