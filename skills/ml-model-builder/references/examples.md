# Example Routes and Clarifications

## Contents

- [Analysis only](#analysis-only)
- [Classification](#classification)
- [Regression](#regression)
- [Forecasting](#forecasting)
- [Anomaly detection](#anomaly-detection)
- [Model improvement](#model-improvement)
- [High-stakes prospective validation](#high-stakes-prospective-validation)

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
calibration. Ask whether the data are a full eligible cohort, a negative sample,
or labels observed only after review; keep representative evaluation and
calibration data and use justified weights. Ask whether a valid incumbent
exists, but proceed with naive and fixed baselines if none does.

If customer/event support requires nested CV, keep global EDA target-blind.
Repeat target-aware decisions inside each outer-training fold or use a
discovery cohort excluded from outer evaluation. Generate calibration from
grouped OOF, disjoint or calibrated-CV predictions and freeze its final-fit
semantics before evaluation.

For a capacity queue, make `score_rows` return real row-aligned probabilities
and make `select_queue` perform deterministic whole-batch eligibility and top-k
selection. Test the saved inference entry point on representative, one-row,
empty, malformed, unseen-category, sub-capacity and tied inputs.

## Regression

**Request:** “Predict delivery duration in days.”

Clarify order-time prediction, cancellation/censoring, route/carrier grouping,
error asymmetry and whether intervals are needed. Compare MAE/RMSE with
median/mean baselines; inspect residuals by route, carrier and time.

## Forecasting

**Request:** “Forecast weekly demand for each store for the next eight weeks.”

Clarify horizon, weekly calendar, store/product hierarchy, known promotions,
new stores and stockout-censored demand. Use rolling-origin panel backtests,
seasonal-naive baselines, horizon-level metrics and interval coverage. For a
large remote panel, preflight size, scan cost, source version, spill space and
compute location; aggregate panel coverage remotely and bound local series
plots rather than downloading or rendering every series.

## Anomaly detection

**Request:** “Find suspicious journal entries; about 5% were synthetically
injected.”

Clarify whether real labels exist and how many entries can be reviewed. Treat
synthetic anomalies as a limited sanity check. For unlabeled production data,
report rank stability, top-k composition and reviewed precision—not general
accuracy. Keep `score_rows` separate from whole-batch `select_queue`; treat
unreviewed and label-pending rows as unknown.

## Model improvement

**Request:** “Improve the model in this project.”

Read existing config, metrics, split/data fingerprints and evaluation history.
Create an immutable child run with parent IDs/hashes and a final-evidence
exposure ledger. Do not optimize against historical final/outer evidence.
Create new development evidence and use untouched future/external data for a
new unbiased claim; otherwise save an explicitly incomplete development-only
child with final evaluation pending. Do not make it pass the completed-run
validator by reusing or fabricating final evidence.

## High-stakes prospective validation

**Request:** “Silently score patients now and validate once outcomes arrive.”

Complete and record the explicit risk assessment; do not default an unassessed
use to standard risk. Freeze the cohort, scorer and maturity rule. Report
scored, matured, pending and lost-to-follow-up counts, and do not treat pending
outcomes as negatives or publish performance before labels mature.
