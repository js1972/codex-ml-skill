---
name: ml-model-builder
description: Analyze and explain local or URL datasets and build production-minded classical machine learning models in Codex or Claude Code. Use when users ask to explore, profile, understand, visualize, train, evaluate, compare, or improve tabular classification, regression, time-series forecasting, or anomaly-detection solutions. Covers leakage-aware data analysis, task-appropriate validation, Optuna optimization, optional AutoGluon benchmarking, explainability, and deployable train/infer artifacts.
---

# ML Model Builder

## Purpose

Help a user understand a dataset, decide whether machine learning is suitable,
and—when requested—produce an honestly evaluated, reproducible model with
deployment-ready artifacts. Prefer a defensible simple solution over an
elaborate model whose evaluation or serving assumptions are weak.

## Operating modes

Choose one mode during intake and record it in `artefacts/config.json`.

1. **Analysis only** — explain the dataset and produce an EDA report without
   training a predictive model.
2. **Model building** — analyze the dataset, build task-appropriate candidates,
   evaluate them, and save training/inference artifacts.
3. **Model improvement** — inspect an existing run and its artifacts, preserve
   the original holdout result, then improve the workflow using new validation
   evidence. Do not repeatedly tune against the old holdout.

## Non-negotiable safeguards

- Define the prediction or scoring moment before choosing features.
- Derive split assignments before target-aware analysis.
- Fit every learned preprocessing step on training folds only.
- Keep holdout targets sealed until the final candidate is fixed.
- Respect time, group, and repeated-entity structure in every split and
  permutation.
- Treat unlabeled anomaly detection as prioritization for review, not measured
  predictive accuracy.
- Report uncertainty, limitations, and operational constraints alongside point
  estimates.
- Never claim that a search heuristic proves a dataset's theoretical ceiling.
- Treat the supplied dataset as available for analysis. Show observed labels and
  values when they make the report more useful; do not add automatic redaction.
- Never load an untrusted `joblib`/pickle artifact; deserialization can execute
  code.

## Reference routing

Read only the references needed for the selected route:

| Need | Read |
|---|---|
| User decisions, progress, managed processes | `references/governance.md` |
| EDA, charts, sampling, reporting | `references/data-analysis.md` |
| Data contracts, splitting, preprocessing, leakage | `references/data-and-leakage.md` |
| Classification or regression | `references/supervised-tabular.md` |
| Forecasting or time-dependent prediction | `references/time-series.md` |
| Supervised or unsupervised anomaly detection | `references/anomaly-detection.md` |
| Optuna, signal checks, model search, stacking | `references/optimization-and-ensembling.md` |
| Metrics, uncertainty, explainability, deployment | `references/evaluation-and-production.md` |
| Optional AutoGluon comparison | `references/automl.md` |
| Output files and versioned schemas | `references/artifacts.md` |
| Example requests and clarification patterns | `references/examples.md` |

Always read `governance.md`, `data-analysis.md`, `data-and-leakage.md`, and
`artifacts.md`. In model-building or improvement mode, also read the selected
task reference, `optimization-and-ensembling.md`, and
`evaluation-and-production.md`. Read `automl.md` only after an explicit opt-in.
For classification/regression with future outcomes or delayed labels, read
`time-series.md` as well for temporal validation and censoring safeguards.

## Workflow

Maintain a user-visible task list when the host supports one. Send a concise
progress update at every major boundary. Follow `references/governance.md`.

### 1. Frame the problem

Ask only for missing information:

- operating mode;
- dataset location(s);
- business question and decision the output will support;
- task type and target, if known;
- unit of observation and entity/group identifier;
- unit of decision/action when it differs from a source row;
- prediction/scoring moment and features available then;
- time column, horizon, frequency, and known-future covariates when relevant;
- cost of false positives, false negatives, over- and under-prediction;
- deployment/batch context, latency or resource constraints;
- data-governance constraints and required subgroup checks;
- compute/time budget.

Use documented defaults only for low-risk reversible choices. Record all
assumptions and unresolved domain questions.

### 2. Prepare the environment

Create `.venv` in the user's project unless a suitable environment already
exists. Install only required packages into that environment. Ask before large
optional installations. Use the host's managed-process/session mechanism for
long-running work; do not shorten an approved ML budget to fit a tool timeout.

### 3. Load and establish the data contract

Support local or HTTP(S) CSV/Parquet inputs. Validate formats, schema, target
derivation, row grain, join keys, units, time coverage, label timing, and
feature availability. For multiple datasets, confirm join/concat semantics;
never assume row-wise concatenation merely because schemas align.

Identify duplicate records, repeated entities, source-system columns,
identifier-like fields, high-cardinality categoricals, free text, sensitive
attributes, and suspected post-outcome fields. Do not remove anything before
understanding its business meaning.

### 4. Create evaluation partitions

Choose random stratified, grouped, temporal, or grouped-temporal validation
from the data-generating process—not from a fixed row-count rule. Persist row
assignments or deterministic split rules. Check minimum class/event counts and
coverage once when establishing partitions, then seal holdout targets before
EDA or model decisions.

In analysis-only mode, do not invent a holdout. Analyze the full permitted
dataset and label the report as descriptive rather than predictive.

### 5. Analyze and explain the dataset

Run `scripts/profile_dataset.py` when its supported inputs match the task;
otherwise follow `references/data-analysis.md` directly.

- Use full data only for structural facts that do not inspect held-out targets.
- Use the training partition for target-aware statistics and plots.
- Calculate statistics on all permitted rows; sample only expensive charts and
  record the sample method.
- Generate `data_report.html`, `data_summary.md`, `data_profile.json`, and
  `figures/`.
- Present findings as **blocker**, **warning**, or **information**.
- Explain what each important chart means and what decision it may affect.

If the mode is analysis only, validate the report, summarize the main findings
and limitations, write the configured artifacts, and stop here.

### 6. Build a fixed baseline and sanity checks

Follow the selected task reference. Use a baseline appropriate to the decision:

- classification: simple probabilistic linear baseline plus a naive prevalence
  reference;
- regression: median/mean naive reference plus regularized linear baseline;
- forecasting: last-value and seasonal-naive references;
- supervised anomaly detection: classification baselines suited to rare events;
- unsupervised anomaly detection: stable ranking diagnostics and domain review.

Run task-appropriate signal or learnability checks from
`optimization-and-ensembling.md`. Treat them as diagnostics. Do not halt a
nonlinear search solely because a linear probe missed plausible nonlinear
signal.

### 7. Improve candidates within the budget

Use task-aware cross-validation and bounded model-family searches. Give each
eligible family minimum coverage before adaptive optimization. Keep
preprocessing inside folds, record failed trials, control parallelism and
memory, and stop on budget or defensible convergence.

Treat stacking as optional. Attempt it only when diverse, competitive
candidates and sufficient out-of-fold data exist. Select it only on validation
evidence; never require diversity models that are unsuitable for the data.

### 8. Select, calibrate, and evaluate

Finalize the model, threshold, calibration, or anomaly-review budget using
training/validation evidence. Refit on training plus validation when valid,
then evaluate once on holdout. Report:

- primary and secondary metrics;
- uncertainty intervals or repeated-fold variation;
- calibration/threshold behavior where relevant;
- error slices and requested subgroup checks;
- comparison with naive and fixed baselines;
- practical significance, not only statistical significance;
- known limitations and likely failure modes.

For unlabeled anomaly ranking, do not invent a labeled holdout or predictive
score. Freeze the scorer on historical/reference windows, assess stability and
queue behavior on future scoring windows, and record reviewed outcomes when
available.

If holdout results influence a choice between candidates, label the holdout a
benchmark-selection set and require new future/external data for an unbiased
estimate.

### 9. Explain and test production behavior

Generate explainability only when it answers a user question. State the
limitations of SHAP or other importance methods, especially with correlated
features. Validate inference on representative rows, missing optional values,
unseen categories, extra columns, wrong dtypes, and empty inputs. Measure
artifact size and latency only with a documented benchmark context.

### 10. Save and verify artifacts

Follow `references/artifacts.md`. Always include `schema_version`. Save the data
contract, feature manifest, data fingerprint, dependency versions, intended-use
limitations, and exact inference command. Run
`scripts/validate_run.py <project-directory>` before declaring completion.

## Completion checklist

- [ ] Mode, business decision, prediction moment, row grain, and error costs
      are recorded.
- [ ] Split strategy matches time/group/entity structure and holdout targets
      remained sealed.
- [ ] EDA artifacts exist and target-aware analysis used training data only.
- [ ] Feature availability and target derivation were audited for leakage.
- [ ] Preprocessing, resampling, and feature selection occurred inside folds.
- [ ] Baselines, model search, and metrics match the selected task reference.
- [ ] Thresholds/calibration/forecast horizon/anomaly review budget are recorded
      when applicable.
- [ ] Final evaluation includes uncertainty, error analysis, and limitations.
- [ ] Inference schema and round-trip behavior were tested.
- [ ] Required artifacts pass `scripts/validate_run.py`.
- [ ] Optional AutoGluon or explainability ran only when requested.
- [ ] `results.md` and `data_summary.md` agree with machine-readable artifacts.

If a checklist item cannot apply, record the reason rather than omitting it
silently.
