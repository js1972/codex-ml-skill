# Supervised Tabular Classification and Regression

## Contents

- [Shared workflow](#shared-workflow)
- [Classification](#classification)
- [Regression](#regression)
- [Candidate families](#candidate-families)
- [Small, wide, and large data](#small-wide-and-large-data)
- [Finalization](#finalization)

## Shared workflow

1. Confirm row grain, target definition, prediction moment and error costs.
2. Reconcile the source-row grain with the decision/action grain. Aggregate,
   deduplicate or rank entities explicitly so frequent entities are not
   accidentally overweighted and one action does not appear multiple times.
3. Select group/time-aware validation before target-aware EDA.
4. Establish a naive reference and fixed simple model.
5. Compare eligible candidates using identical folds and preprocessing
   boundaries.
6. Select thresholds/calibration/transforms on validation only.
7. Refit the frozen pipeline on the permitted development population.
8. Evaluate using the declared holdout, untouched outer folds, external set, or
   prospective cohort and report uncertainty and slices.

Use multiple metrics, but designate one primary selection metric from the
business decision.

## Classification

### Baselines

- Naive prevalence/majority or stratified prediction.
- Regularized logistic regression with fold-fitted preprocessing.

For multiclass tasks, report per-class support and macro metrics. For ordinal
labels, ask whether ordinal error distance matters before treating classes as
nominal.

### Metrics

Choose from:

- log loss or Brier score when probability quality matters;
- PR-AUC for rare positive events;
- ROC-AUC for ranking when both classes have adequate support;
- macro F1 for balanced attention across classes;
- recall/precision at an operational threshold;
- expected cost or utility when error costs are known;
- top-k/recall-at-k when review capacity is fixed.

Accuracy alone is insufficient for imbalanced problems. ROC-AUC can look
strong while precision is unusable in rare-event settings.

If a score determines who receives an intervention, state that high predicted
risk is not the same as high treatment benefit. A churn-risk model may
prioritize customers who would churn regardless of a call. Recommend an
experiment and uplift/causal modeling when the customer needs incremental
impact, and record feedback-loop risk when prior actions affect labels.

### Thresholds and calibration

- Keep probability-model training separate from decision-threshold selection.
- Select the threshold on validation using cost, capacity, or required
  precision/recall.
- Evaluate calibration with reliability plots, Brier/log loss and expected
  calibration error where useful.
- Fit Platt/isotonic calibration inside nested or held-out validation. Do not
  calibrate on holdout.
- Record the chosen threshold, rationale and fallback behavior.

For fixed-capacity outreach or review, include precision/recall/lift at the
actual capacity and simulate the scheduling unit (for example, per day), not
only a global threshold.

### Uncertainty and slices

Report bootstrap or fold-based uncertainty for the primary metric. Preserve
groups/time blocks when resampling. Report confusion matrices and performance
by meaningful segments with support counts; label statistically unstable
groups clearly.

## Regression

### Baselines

- Training median for absolute-error objectives.
- Training mean for squared-error objectives.
- Regularized Ridge/ElasticNet model with fold-fitted preprocessing.

### Metrics

Use:

- MAE for typical absolute error and robustness;
- RMSE when large errors are disproportionately costly;
- median absolute error for heavy tails;
- R² only as a secondary variance-explained measure;
- RMSLE or target transforms only for non-negative, multiplicative problems;
- MAPE only when targets are safely away from zero and percentage error is
  meaningful;
- pinball loss and coverage for quantile/interval forecasts.

Always report metric units and compare error magnitude with the target scale and
business tolerance.

### Diagnostics

- Plot residuals against predictions, time and key segments.
- Check heteroscedasticity, systematic under/over-prediction and tail errors.
- Evaluate target transformation bias when converting predictions back to the
  original scale.
- Produce prediction intervals when decisions require uncertainty. Validate
  empirical coverage and width on untouched evaluation data.

For zero-heavy, non-negative amounts, consider a Tweedie objective or a
two-part occurrence/severity model and evaluate the combined prediction in
original units. Do not treat paid-to-date, open duration, or another incomplete
outcome as the final target. Establish mature labels or a defensible
censoring/development model with domain review. Closed-only cohorts can be
selection-biased; report sensitivity to the maturity rule.

## Candidate families

Select families from data properties rather than enforcing artificial
diversity:

- regularized linear/logistic models;
- HistGradientBoosting;
- RandomForest/ExtraTrees when memory permits;
- XGBoost, LightGBM or CatBoost when available;
- native categorical models for high-cardinality categoricals;
- KNN only for suitably scaled, low/moderate-dimensional data;
- Naive Bayes only when its distribution/input assumptions are plausible.

ElasticNet does not create interactions unless interaction features are
explicitly supplied. GaussianNB may require dense input and can be unsafe for
large one-hot matrices.

## Small, wide, and large data

- **Small data:** use repeated/nested CV, simple models, strong regularization
  and uncertainty. Avoid spending a large search budget on noisy differences.
- **Wide sparse data:** prefer regularized sparse linear models; control
  vocabulary/cardinality; do not densify silently.
- **Large data:** use task-aware subsampling for search, histogram/tree methods,
  controlled threads and final full-data refit. Verify that the sample retains
  time/group/class structure.

## Finalization

Freeze:

- feature list and preprocessing;
- model family and hyperparameters;
- probability calibration and threshold, if any;
- target inverse transform;
- expected input schema;
- random seed and dependency versions.

Then refit on the permitted development data and evaluate according to the
declared design. If the final refit changes model behavior materially, report
the risk and retain the validation-fitted candidate for comparison.
