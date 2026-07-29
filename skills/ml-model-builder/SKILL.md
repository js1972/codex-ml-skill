---
name: ml-model-builder
description: Build, evaluate, compare, improve, and package production-minded tabular machine-learning solutions in Codex or Claude Code. Use when users ask to train or improve classification, regression, time-series forecasting, or anomaly-detection models; compare classical ML with optional AutoGluon or SAP RPT; or create reproducible training and inference artifacts. Covers leakage-safe validation, explicit execution approval, track-appropriate optimization, uncertainty, high-stakes safeguards, and deployable inference. Use the separate tabular-eda skill for exploratory data analysis without model building.
---

# ML Model Builder

## Purpose

Build an honestly evaluated, reproducible model and leave only the artifacts
needed to understand, rebuild, validate, and use it. Prefer a defensible simple
solution over complexity whose evaluation or serving assumptions are weak.

## Scope boundary

Do not perform standalone exploratory data analysis. When the request is only
to understand, profile, visualize, or explore tabular data, use the sibling
`tabular-eda` skill and stop this workflow.

Treat EDA and modeling as independent:

- Start modeling from the declared source data.
- Never discover, read, copy, link, or react to an EDA report or its files.
- Never make an EDA run a prerequisite or ancestor of a model run.
- Do not create `README.md`, `data_summary.md`, `data_profile.json`, `figures/`,
  or an EDA report during model building.

Perform only the narrow data checks required to define a valid modeling
experiment: target validity, prediction moment, row grain, source cohort,
label observation, schema, duplicates/groups, leakage, feature availability,
class/event support, and split feasibility.

## Non-negotiable safeguards

- Define the prediction or scoring moment before choosing features.
- Establish how rows entered the dataset and why each label is observed.
- Keep holdout, external, and active outer-fold targets outside every model,
  feature, threshold, calibration, and backend-selection decision.
- Fit every learned classical preprocessing step on training folds only.
- Cross-fit target-derived encodings while preserving group and time
  boundaries.
- Respect time, group, repeated-entity, and source-event structure in every
  split, permutation, resample, and uncertainty estimate.
- Choose candidates from the task, data, and deployment constraints rather
  than installed packages.
- Treat unlabeled anomaly detection as review prioritization, not measured
  predictive accuracy.
- Report uncertainty, limitations, and operational constraints with point
  estimates.
- Never claim that a search plateau proves a dataset's theoretical ceiling.
- Never load an untrusted pickle/joblib artifact or execute untrusted project
  training or inference code.

## Reference routing

Read only the references needed for the selected route:

| Need | Read |
|---|---|
| Approval, decisions, budgets, managed execution | `references/governance.md` |
| Data contract, splitting, preprocessing, leakage | `references/data-and-leakage.md` |
| Classification or regression | `references/supervised-tabular.md` |
| Forecasting or time-dependent prediction | `references/time-series.md` |
| Supervised or unsupervised anomaly detection | `references/anomaly-detection.md` |
| Classical Optuna search, diagnostics, stacking | `references/optimization-and-ensembling.md` |
| Metrics, uncertainty, explainability, deployment | `references/evaluation-and-production.md` |
| Healthcare, finance, employment, insurance, or other high-stakes use | `references/high-stakes.md` |
| AutoGluon track | `references/automl.md` |
| SAP RPT track | `references/sap-rpt.md` |
| Minimal run files and backend layout | `references/artifacts.md` |
| Large or remote data | `references/large-data.md` |
| Example requests and clarifications | `references/examples.md` |

Always read `governance.md`, `data-and-leakage.md`, `artifacts.md`, the selected
task reference, and `evaluation-and-production.md`. Read
`optimization-and-ensembling.md` only when the classical track is approved.
Read `automl.md` or `sap-rpt.md` only when its track is approved. Read
`time-series.md` for forecasting and for classification/regression with future
outcomes or delayed labels. Read `large-data.md` before scanning data that may
exceed local resources. Read `high-stakes.md` whenever predictions could
materially affect a person's rights, health, safety, or access to essential
services.

## Workflow

Maintain a user-visible task list when supported and send concise progress
updates at major boundaries. Follow `references/governance.md`.

### 1. Frame the modeling problem

Ask only for missing information:

- dataset location;
- business question, target, task type, intended use, and prohibited uses;
- row, entity/group, and decision/action grain;
- cohort construction, sampling, label-observation mechanism, and weights;
- prediction/scoring moment and features available then;
- time column, horizon, frequency, and known-future inputs when relevant;
- error costs, review capacity, or asymmetric loss;
- deployment, batch, latency, resource, privacy, and governance constraints;
- incumbent model, rule, or measurable manual process;
- risk classification;
- compute/time budget.

When the source and request make one row grain, prediction moment, target, and
label meaning reasonably likely, state them as provisional and place them
inside the single approval gate instead of asking separate preliminary
questions. Ask a blocking question before that gate only when plausible
alternatives would materially change leakage, label validity, split design,
safety, or the business decision. Never record a provisional inference as
confirmed.

Use documented defaults only for low-risk reversible mechanics. Record
assumptions and unresolved domain questions.

### 2. Run modeling preflight

Inspect only enough data to validate the experiment. Confirm:

- source identity, format, size, row grain, keys, schema, and join semantics;
- target derivation, label timing, maturity, observability, and support;
- exact/key duplicates, repeated entities, groups, time ordering, and source
  boundaries;
- prediction-time feature availability, identifiers, sensitive fields, and
  suspected post-outcome or target-derived columns;
- split strategy and whether each development/evaluation fold has adequate
  independent support;
- environment and resource feasibility for the proposed tracks.

Do not generate exploratory charts or descriptive reports. Do not consume an
existing EDA report. Use source data and user/domain input only.

For a supported local CSV, TSV, or Parquet file, use the artifact-free helper
with an existing interpreter that already imports pandas and NumPy:

```text
<python-with-pandas-and-numpy> scripts/inspect_model_data.py <dataset> \
  --target <target> --task <auto|classification|regression> \
  --row-grain "<confirmed-or-provisional grain>" \
  --prediction-moment "<confirmed-or-provisional moment>" \
  [--group-column <group>] [--time-column <time>] \
  [--exclude <column> ...]
```

The helper prints a compact assessment to stdout and must create no files. Do
not install pandas, NumPy, or another dependency before experiment approval.
If the current interpreter lacks them, report the missing dependency cleanly
and use another existing suitable interpreter; if none exists, include the
installation in the approval plan.

### 3. Propose one experiment and obtain approval

Before fitting, building, querying, installing a large optional dependency, or
starting any backend, present a concise execution plan containing:

- target and prediction moment;
- included/excluded feature contract;
- evaluation population, split strategy, primary metric, and uncertainty plan;
- shared per-track CPU, parallel-job, memory, and GPU controls;
- classical track: include/decline, candidate families, minimum coverage,
  wall-time, and Optuna-trial budget;
- AutoGluon track: include/decline, preset, wall-time, and disk budget;
- SAP RPT track: include/decline, model/access route, context-row,
  total-request-row, query-batch-row, column, request/retry/timeout budget;
- operational constraints used to select a winner.

Require explicit user confirmation. “Train the best model” does not authorize
silently omitting AutoGluon or SAP RPT; recommend a choice for each track and
ask the user to approve or change it. Record the four confirmed experiment
semantics as true booleans under `approval.scope` and the choice/budget for
all three tracks under `approval.tracks`. Initialize structured
`approval.amendments` and `approval.remote_transfers` lists. Record later
approved plan changes and remote-data permissions there, not only in narrative
notes. Do not execute any track until approval is received.

Obtain any additional remote-data-transfer confirmation required by the SAP
RPT route before its first request.

### 4. Freeze evaluation boundaries

Choose stratified, grouped, temporal, grouped-temporal, rolling-origin,
nested-CV, external, or prospective validation from the data-generating
process. Persist assignments or a deterministic rule and its fingerprint in
`run.json`. Audit group/duplicate overlap, temporal order, purge gaps, label
maturity, and per-fold support.

Do not force a separate holdout when it would leave too few independent groups
or rare events for meaningful evaluation. Predeclare nested/repeated outer CV,
external validation, or prospective validation and state the independence
limits.

When no natural entity or source-event identifier exists, derive an exact
feature signature from canonical eligible prediction-time feature values. If
signatures repeat, keep every identical signature in one split and one
uncertainty cluster. Exclude the target, identifiers, fold metadata, and
post-outcome fields from the signature and record its columns and fingerprint.

### 5. Execute approved tracks

Use the same target, eligible features, evaluation rows, folds, weights, and
metric implementation across approved tracks. Share information boundaries,
not implementation mechanics.

#### Classical

Build naive and fixed simple baselines. Apply fold-local preprocessing and
bounded, task-aware model-family search. Use Optuna only here and only when it
adds value. Give eligible families fair initial coverage, record failures, and
control compute and memory.

#### AutoGluon

Pass each eligible raw training table, target, approved metric, fold
boundaries, preset, and time/resource budget to AutoGluon. Let AutoGluon own
its preprocessing, model construction, tuning, and ensembling. Do not put it
inside external Optuna or feed it the classical transformed matrix.

When `parallel_jobs=1`, set AutoGluon's fold fitting strategy to
`sequential_local` and record the reason. Capture the native leaderboard and a
structured internal-failure ledger; an internal model failure does not make a
successfully completed AutoGluon track a failure.

#### SAP RPT

Package a fold-valid labelled context from training rows and query the
pretrained RPT model for the corresponding evaluation rows. Do not describe
this as training, fitting, hyperparameter tuning, or Optuna. Manage context
selection, schema, batching, request budget, response validation, and access
metadata.

Treat SAP RPT as a production-capable model. Distinguish the model from the
access route: the internal CLI is a convenient internal managed access route;
SAP AI Core is the paying-customer production route to the same model. Permit
SAP RPT to be the predictive or operational winner when evidence and
deployment constraints support it.

### 6. Select and evaluate

Select candidates, thresholds, calibration, forecast strategy, or review
budget using only permitted development evidence. Evaluate the frozen
procedure once on the declared holdout/external set or aggregate untouched
outer folds.

Report every approved backend's status and failure/unavailability reason or
same-population score. Include uncertainty, error and subgroup slices,
calibration/threshold behavior when relevant, practical value,
latency/resource trade-offs, intended/prohibited uses, known limitations, and
monitoring. Whenever the corresponding backend was approved, include the
classical baseline/leaderboard, AutoGluon preset, or SAP RPT
context/access/latency section even if it only records an unavailable or failed
status. Separate “best predictive result” from “recommended operational
choice” when they differ.

### 7. Package and verify

Follow `references/artifacts.md`. Create one compact run directory with:

- `run.json`;
- self-contained `report.html`;
- `results.md`;
- root `infer.py` and, when classical or AutoGluon build reproducibility needs
  it, root `train.py`;
- `requirements.lock`;
- only the backend artifacts required to rebuild or infer;
- `validation.json`.

The compact run is self-contained for reading the report and running
inference. Do not copy the raw training dataset into it by default. Rebuilding
a classical or AutoGluon backend may depend on the original source recorded in
`run.json.data.source`; `train.py` must verify that source against the recorded
fingerprint, fail clearly when it is missing or changed, and write to a new or
explicitly empty output run rather than overwrite the validated run in place.

For a retained AutoGluon backend, capture metrics, leaderboard, failures, and
training diagnostics before creating `clone_for_deployment(model="best")`.
Compare original and clone predictions on a temporary fixture, then retain the
validated deployment clone at `backends/autogluon/predictor`. Remove the full
training predictor after validation unless continued AutoGluon analysis is
explicitly required. Record final predictor size and peak packaging disk use.

Omit `train.py` for an RPT-only run. Make `infer.py` select the approved
operational recommendation by default and support
`--backend classical|autogluon|sap-rpt` for every retained backend. Give SAP
RPT a tested new-row inference path using the frozen context policy and access
configuration. The RPT adapter must accept arbitrary input sizes, split them
into ordered `max_query_batch_rows` chunks, enforce the request and column
limits, preserve row IDs and input order, and fail before transfer when the
approved request budget is insufficient. Its backend directory must not
contain `train.py`.

Record exact required/optional inputs, dtypes, missing/extra policy, excluded
target, identifier/feature order, output columns, finite/probability bounds,
and backend commands under `run.json.inference`. For every retained backend,
declare representative, single-row, empty-input, and missing-required-column
cases in `validation.json`. Dispatch the real backend, preserve row IDs, and
repeat representative/single-row cases to verify deterministic output.
Initialize validation with `status: "pending"` and `validated_at: null`; never
pre-write a passing status.

Run every inference case in a fresh subprocess. Before importing NumPy,
pandas, Torch, AutoGluon, or a backend that imports them, set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` to `1` in `infer.py`.

Run the artifact validator and real inference round trips:

```text
python <ml-model-builder-skill>/scripts/validate_run.py <project-directory> \
  --artifacts-dir <run-directory> --run-inference-test
```

Use temporary fixtures and prediction outputs during validation; do not retain
them in the final run. Reconcile code, folds, metrics, report, and inference
behavior before handoff.

### 8. Extend or improve without duplication

Keep all approved backends for the same data, target, feature contract, splits,
and metric in one experiment. When the user later adds AutoGluon or SAP RPT to
that experiment, add only its `backends` entry and directory, update the
structured approval amendment/transfer records, inference, selection,
validation, and `lineage.notes` contracts, then refresh the inclusive root
report and results. Do not copy the parent model, folds, predictions, plots,
fixtures, or other unchanged files.

Create a separate run only when the data, target, split, metric, modeling
hypothesis, or released winner changes materially. Do not tune descendants
against previously opened final evidence while claiming a new unbiased result.

## Completion checklist

- [ ] Record the decision, target, prediction moment, row grain, cohort, labels,
      feature contract, error costs, risk, and deployment constraints.
- [ ] Obtain and record explicit approval for the evaluation plan, every track,
      and each track's budget before execution.
- [ ] Verify splits, leakage controls, label maturity, support, and fold-local
      classical preprocessing.
- [ ] Keep AutoGluon autonomous and outside external Optuna/classical
      preprocessing; package a prediction-equivalent deployment clone and test
      cold-start inference.
- [ ] Treat SAP RPT as pretrained; package context/query data without training
      artifacts or training terminology.
- [ ] Compare approved tracks on shared evaluation boundaries and metric code.
- [ ] Report baselines, uncertainty, errors, limitations, predictive winner,
      operational recommendation, and exact inference commands.
- [ ] Test unified inference for every retained backend.
- [ ] Keep the run minimal, inference-ready, and free of EDA or duplicated
      parent artifacts; record external rebuild prerequisites explicitly.
- [ ] Pass `<ml-model-builder-skill>/scripts/validate_run.py`.

If an item does not apply, record why rather than omitting it silently.
