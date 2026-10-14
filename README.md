# Supply Chain Late-Delivery Risk Analytics

![CI](https://github.com/nisha-s/supply-chain-risk-analytics/actions/workflows/ci.yml/badge.svg)

## Problem

Late deliveries cost money and customer trust. This project asks three questions
about the DataCo supply chain data:

1. **What is happening?** How often are orders late, and where (shipping mode,
   region, product category, month)?
2. **What drives it?** Which factors are statistically linked to late delivery
   (chi-square tests and a logistic regression)?
3. **Can we predict it?** Can a model flag high-risk orders using only
   information known when the order is placed?

Results are shown in a Streamlit dashboard.

## Data

[DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
(Kaggle). About 180k order lines across about 65k orders.

Download `DataCoSupplyChainDataset.csv` and place it in `data/raw/`. The `data/`
folder is git-ignored, so no data is committed.

## Approach

- **Database:** the CSV is cleaned (snake_case columns, PII dropped) and split into
  four DuckDB tables: `orders`, `order_items`, `customers` and `products`.
- **Analysis:** SQL queries with joins give late rates by shipping mode,
  market/region, product category and month.
- **Statistics:** chi-square tests (late vs. shipping mode, late vs. region) with
  Cramér's V as effect size, plus a statsmodels logistic regression reported as
  odds ratios.
- **Modeling:** one row per order. A logistic regression baseline and an XGBoost
  classifier, trained on a stratified 80/20 split with a fixed seed.
- **No target leakage:** anything known only after delivery is excluded from the
  features: real shipping days, delivery status, shipping date and order status.
  A check in `src/features.py` fails if any of them gets in.

## Project structure

```
src/config.py        paths and constants
src/build_db.py      CSV -> cleaned DuckDB tables
src/queries.py       SQL for late rates by mode, region, category, month
src/stats_tests.py   chi-square tests and logistic regression
src/features.py      order-level features, leakage check
src/train.py         train and save both models
src/evaluate.py      metrics CSV and scored test orders
app/dashboard.py     Streamlit dashboard
tests/test_data.py   data-quality tests on a small sample CSV
```

## How to run

Requires Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.build_db             # build data/supply_chain.duckdb
python -m src.queries              # print late-rate breakdowns
python -m src.stats_tests          # print chi-square and regression results
python -m src.train                # train and save models
python -m src.evaluate             # print metrics, save reports/model_metrics.csv

python -m streamlit run app/dashboard.py
python -m pytest -v                # run tests
```

Run commands from the repo root with `python -m` so imports from `src` work.

## Key findings

_To be filled in from actual results._

- Overall late rate: _TBD_
- Shipping mode: _TBD_
- Region: _TBD_
- Chi-square tests: _TBD_
- Logistic regression (strongest drivers): _TBD_
- Model performance (ROC-AUC, precision, recall, F1): _TBD_
