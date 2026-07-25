# Example Routes and Clarifications

## Contents

- [Analysis only](#analysis-only)
- [Classification](#classification)
- [Regression](#regression)
- [Forecasting](#forecasting)
- [Anomaly detection](#anomaly-detection)
- [Model improvement](#model-improvement)

## Analysis only

**Request:** “Help me understand `customers.parquet`. Show useful charts and
things I should worry about.”

Clarify whether any column is a target. Run
analysis-only mode on all permitted data. Produce the EDA artifacts; do not
invent a model, holdout or predictive conclusion.

## Classification

**Request:** “Predict which invoices will be paid late.”

Clarify:

- exact prediction moment;
- when a payment becomes “late” and label maturity;
- customer grouping and invoice time;
- weekly review capacity or false-positive/false-negative costs;
- post-payment fields to exclude.

Prefer grouped-temporal validation. Use PR-AUC/recall-at-capacity when late
payments are rare, select the operational threshold on validation and report
calibration.

## Regression

**Request:** “Predict delivery duration in days.”

Clarify order-time prediction, cancellation/censoring, route/carrier grouping,
error asymmetry and whether intervals are needed. Compare MAE/RMSE with
median/mean baselines; inspect residuals by route, carrier and time.

## Forecasting

**Request:** “Forecast weekly demand for each store for the next eight weeks.”

Clarify horizon, weekly calendar, store/product hierarchy, known promotions,
new stores and stockout-censored demand. Use rolling-origin panel backtests,
seasonal-naive baselines, horizon-level metrics and interval coverage.

## Anomaly detection

**Request:** “Find suspicious journal entries; about 5% were synthetically
injected.”

Clarify whether real labels exist and how many entries can be reviewed. Treat
synthetic anomalies as a limited sanity check. For unlabeled production data,
report rank stability, top-k composition and reviewed precision—not general
accuracy.

## Model improvement

**Request:** “Improve the model in this project.”

Read existing config, metrics, split/data fingerprints and holdout history.
Do not optimize against the historical holdout. Create new validation evidence
or new future/external evaluation data, preserve the old result and record the
new run separately.
