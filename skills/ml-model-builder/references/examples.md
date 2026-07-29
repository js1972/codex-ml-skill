# Example Routes and Clarifications

## Contents

- [Standalone EDA](#standalone-eda)
- [Classification with track approval](#classification-with-track-approval)
- [Regression](#regression)
- [Forecasting](#forecasting)
- [Anomaly detection](#anomaly-detection)
- [Adding SAP RPT later](#adding-sap-rpt-later)
- [Model improvement](#model-improvement)
- [High-stakes prospective validation](#high-stakes-prospective-validation)

## Standalone EDA

**Request:** “Help me understand `customers.parquet`. Show useful charts and
things I should worry about.”

Use the separate `tabular-eda` skill. Return findings in chat and one
self-contained HTML report. Do not start `ml-model-builder`, create a model
run, reserve splits, or make the EDA output part of future modeling.

## Classification with track approval

**Request:** “Train the best model you can to predict which invoices will be
paid late.”

From the source name and columns, provisionally infer what is reasonably clear
and include it in the single approval gate. Ask a separate blocking question
first only if alternatives materially change label validity, leakage, safety,
or evaluation. Resolve:

- exact prediction moment and label maturity;
- customer grouping and invoice time;
- eligible features and post-payment fields to exclude;
- weekly review capacity or error costs;
- cohort sampling and selective label observation;
- latency and deployment constraints.

Run modeling preflight, then present one approval checkpoint, for example:

```text
Proposed experiment
- Target: paid_late, predicted at invoice issue
- Evaluation: grouped-temporal folds; average precision
- Classical: include; logistic baseline plus XGBoost/LightGBM/CatBoost;
  20-minute Optuna budget
- AutoGluon: include; best_quality; 20-minute build budget
- SAP RPT: include; internal CLI; fixed fold context; at most 20 requests
- Winner: predictive score plus weekly latency/capacity requirements
```

Label inferred row grain, prediction moment, and label meaning as provisional
until the user approves them. Wait for explicit confirmation or changes. Do
not silently omit AutoGluon or SAP RPT merely because the request said
“train.”

Execute approved tracks on the same folds and metrics:

- apply fold-local preprocessing and optional Optuna only to classical models;
- pass eligible raw fold tables to AutoGluon and let it own model building;
- package fold-training labels/features as SAP RPT context and query validation
  rows without training RPT.

For a capacity queue, make `score_rows` return real row-aligned scores and make
`select_queue` apply deterministic whole-batch eligibility and top-k selection.

## Regression

**Request:** “Predict delivery duration in days.”

Clarify order-time prediction, cancellations/censoring, route/carrier grouping,
error asymmetry, intervals, and approved tracks. Compare MAE/RMSE against
median/mean baselines and inspect residuals by route, carrier, and time.

## Forecasting

**Request:** “Forecast weekly demand for each store for the next eight weeks.”

Clarify horizon, weekly calendar, store/product hierarchy, historical vintages
of promotions, new stores, and stockout-censored demand. Use rolling-origin
panel backtests, seasonal-naive baselines, horizon-level metrics, and interval
coverage. For large remote panels, preflight size, scan cost, source snapshot,
spill space, and compute location; do not generate standalone EDA artifacts.

## Anomaly detection

**Request:** “Find suspicious journal entries; about 5% were synthetically
injected.”

Clarify whether real labels exist and how many entries can be reviewed. Treat
synthetic anomalies as a limited sanity check. For unlabeled production data,
report rank stability, top-k composition, and reviewed precision—not general
accuracy. Keep row scoring separate from whole-batch queue selection and treat
unreviewed or label-pending rows as unknown.

## Adding SAP RPT later

**Request:** “Now include SAP RPT in the wine comparison.”

Check whether source fingerprint, target, eligible features, folds, evaluation
rows, weights, and metric code match the existing experiment. If they do:

1. propose and obtain approval for the RPT model/access route, labelled-context
   policy, remote transfer, and request budget;
2. add only `backends/sap_rpt/` and its approved `run.json` backend/result;
3. refresh the root inclusive `report.html` and `results.md`;
4. test `infer.py --backend sap-rpt` on new wine rows.

Do not create a duplicate run containing copied classical models, OOF
predictions, plots, fixtures, or reports. If the contract differs, create a new
experiment and state why.

## Model improvement

**Request:** “Improve the model in this project.”

Read the existing `run.json`, code, report, and final-evidence history. If the
request adds an optional backend to the same experiment, update that run
without copying existing artifacts. Create a new run only for a material change to the data, target,
features, splits, metric, hypothesis, or released winner. Do not optimize
against historical final evidence and still call the same population
unbiased.

## High-stakes prospective validation

**Request:** “Silently score patients now and validate once outcomes arrive.”

Complete the explicit risk assessment and experiment approval. Freeze the
cohort, scorer, maturity rule, and backend access. Report scored, matured,
pending, and lost-to-follow-up counts. Do not convert pending outcomes to
negatives or publish performance before sufficient labels mature.
