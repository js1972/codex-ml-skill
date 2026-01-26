# Defaults and Guidance

Use this file for sensible defaults when the user does not specify choices.

## Minimum Inputs by Task

- Classification:
  - dataset location
  - target column
  - metric (default: f1_macro)
  - split strategy (default: random 80/20 with stratification)
- Regression:
  - dataset location
  - target column
  - metric (default: rmse)
  - split strategy (default: random 80/20)
- Time series forecasting:
  - dataset location
  - time column
  - target column
  - forecast horizon
  - data frequency
  - metric (default: rmse or mape if no zeros)
  - split strategy (default: time-based, no shuffle)
- Anomaly detection:
  - dataset location
  - metric:
    - supervised: f1_macro or roc_auc
    - unsupervised: no label, use anomaly score summary
  - split strategy:
    - supervised: random 80/20
    - unsupervised: time-based if time column is present

## Metrics Defaults

- Classification: f1_macro (also report accuracy; if binary, add roc_auc)
- Regression: rmse (also report mae and r2)
- Time series: rmse; use mape only if target has no zeros
- Anomaly:
  - supervised: f1_macro (also roc_auc)
  - unsupervised: report contamination rate and top-k score stats

## Baseline Models

- Classification: LogisticRegression with standardization and one-hot encoding
- Regression: Ridge regression with standardization and one-hot encoding
- Time series: seasonal naive or last-value baseline
- Anomaly: IsolationForest; if time series, add rolling stats before model

## Iteration Models

- Classification/Regression:
  - RandomForest
  - GradientBoosting or HistGradientBoosting
  - XGBoost, LightGBM, or CatBoost if available or can be installed in the venv
- Time series:
  - SARIMAX or ETS if statsmodels is available
  - ML with lag features + gradient boosting
  - Prophet only if available and approved
- Anomaly:
  - Tune IsolationForest
  - LocalOutlierFactor or OneClassSVM for comparison

## Split Strategy Defaults

- Random split: 80/20; use stratification for classification
- Time series: last 20 percent for validation; no shuffling

## Split and CV Guidance

- Use time-based splits when the target depends on future values or when the
  data is naturally ordered (finance, forecasting, operational logs).
- For time series, prefer walk-forward or expanding-window validation
  (`TimeSeriesSplit`), and avoid shuffling.
- If multiple rows belong to the same entity (customer, account, machine),
  split by entity using `GroupKFold` or `GroupShuffleSplit` to avoid leakage.
- For imbalanced classification, use stratified splits and report class balance.
- For small datasets:
  - If not time-based, prefer K-fold CV (5 or 10 folds).
  - If time-based, use multiple rolling windows rather than random CV.
- Keep a final holdout set when feasible; do not tune on the test set.

## Training Bounds Defaults

- Baseline: 5 minutes or 2 trials, whichever comes first
- Main: 30 minutes or 25 trials, whichever comes first
- Early stop: stop after 6 non-improving trials

## Explainability

- Use SHAP for tree or linear models when requested.
- If SHAP is too heavy for the dataset size, ask the user before running.

## Multiple Dataset Locations

- If multiple paths or URLs are provided and schemas match, default to row-wise
  concat and add a `dataset_source` column.
- If schemas do not align, ask the user whether to:
  - align common columns only, or
  - treat datasets separately.

## Derived Targets

- If the user provides a rule (e.g., threshold or date difference), implement
  the transformation and record it in `artefacts/config.json`.
- For time series with due/clearing dates, suggest a target like
  `delay_days = clearing_date - due_date`, and confirm with the user.

## Default Behaviors

- If a classification threshold is provided, create a binary target and keep
  the original target for reference.
- For anomaly detection with a provided contamination rate, pass it to the
  unsupervised model when no labels exist.
- For time-based splits, use the provided time column and avoid shuffling.

## Reproducibility

- Default random seed: 42 (ask the user first; use default if not provided).
- Apply the seed to Python `random`, NumPy, and model training where supported.

## Leakage Guard

- Flag columns that:
  - exactly match the target,
  - contain the target name (case-insensitive),
  - are computed from the target (if a derivation rule is known).
- If flagged, ask the user to confirm exclusion before training.

## Feature Engineering (Best Practices)

- Date/time columns:
  - Derive: year, quarter, month, week, day_of_week, is_weekend.
  - For hourly data: hour_of_day.
  - Keep time-based splits; do not use future-derived features.
- Time series:
  - Create lag features for target and key covariates (e.g., 1, 2, 3, 7, 14).
  - Add rolling stats on past windows (mean, std, min, max).
  - Align window sizes to data frequency when known.
- Numeric:
  - Log1p transform for heavily skewed positives.
  - Standardize for linear models and distance-based methods.
- Categorical:
  - One-hot encode by default.
  - Avoid target encoding unless explicitly requested.
- Text (if user proceeds with classical ML):
  - Use TF-IDF with a reasonable cap on features (e.g., 5k).
- Always record engineered features in `artefacts/config.json` and summarize
  in `results.md`.

## LLM Suitability Check

## Non-sklearn Model Guidance

- For tabular classification/regression, prefer gradient-boosted trees (XGBoost,
  LightGBM, CatBoost) when available.
- For categorical-heavy datasets, CatBoost often performs well with minimal
  encoding.
- If installing an extra library will take significant time, proceed with
  scikit-learn alternatives and note the limitation in the summary.

Recommend an LLM-based approach (and ask whether to proceed with classical ML)
when most of the signal is unstructured text or the labeled dataset is very
small.

Suggested triggers:
- Text is the primary feature and labeled rows are fewer than 200 total or fewer
  than 20 per class.
- The task is semantic (intent, topic, summarization, extraction) rather than
  numeric prediction.
- Label set is large or frequently changing.

Do not recommend LLMs for:
- Time-series forecasting on numeric signals.
- Numeric regression on structured/tabular data.
- Structured anomaly detection on tabular data.
