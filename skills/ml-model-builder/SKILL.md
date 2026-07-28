---
name: ml-model-builder
description: Analyze and explain local, URL, or remote warehouse/lake datasets and build production-minded classical machine learning models in Codex or Claude Code. Use when users ask to explore, profile, understand, visualize, train, evaluate, compare, or improve tabular classification, regression, time-series forecasting, or anomaly-detection solutions. Covers memory- and leakage-aware data analysis, task-appropriate validation, high-stakes safeguards, Optuna optimization, optional AutoGluon and SAP RPT benchmarking, explainability, and deployable train/infer artifacts.
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
   the meaning of its historical evaluation, then improve the workflow using
   fresh development evidence. Do not repeatedly tune against an old holdout,
   external test, or outer-fold result. Write every improvement to a new
   versioned run, link it to its parent, and record which evaluation results
   have already been exposed.

## Non-negotiable safeguards

- Define the prediction or scoring moment before choosing features.
- Establish how rows entered the dataset and why each label is observed.
- Derive split assignments before target-aware analysis.
- Audit earlier analysis-only target exposure before reusing a population for
  modeling; never retroactively call overlapping rows sealed final evidence.
- Keep global EDA target-blind under nested CV unless target-aware decisions are
  repeated inside every outer training fold or use a separate discovery cohort
  excluded from the outer estimate.
- Fit every learned preprocessing step on training folds only.
- Cross-fit target-derived encodings; ordinary fold placement alone is not
  sufficient for training-row encodings.
- Keep holdout/external targets sealed until the final candidate is fixed.
- Respect time, group, and repeated-entity structure in every split and
  permutation.
- Choose model candidates from the task, data and constraints before checking
  which libraries happen to be installed.
- Treat unlabeled anomaly detection as prioritization for review, not measured
  predictive accuracy.
- Report uncertainty, limitations, and operational constraints alongside point
  estimates.
- Never claim that a search heuristic proves a dataset's theoretical ceiling.
- Treat the supplied dataset as available for analysis. Show observed labels and
  values when they make the report more useful; do not add automatic redaction.
- Never load an untrusted `joblib`/pickle artifact or execute untrusted
  training/inference code; both can execute arbitrary code.

## Reference routing

Read only the references needed for the selected route:

| Need | Read |
|---|---|
| User decisions, progress, managed processes | `references/governance.md` |
| EDA, charts, sampling, reporting | `references/data-analysis.md` |
| Data larger than local memory/disk | `references/large-data.md` |
| Data contracts, splitting, preprocessing, leakage | `references/data-and-leakage.md` |
| Classification or regression | `references/supervised-tabular.md` |
| Forecasting or time-dependent prediction | `references/time-series.md` |
| Supervised or unsupervised anomaly detection | `references/anomaly-detection.md` |
| Optuna, signal checks, model search, stacking | `references/optimization-and-ensembling.md` |
| Metrics, uncertainty, explainability, deployment | `references/evaluation-and-production.md` |
| Healthcare, finance, employment, insurance, or other high-stakes use | `references/high-stakes.md` |
| Optional AutoGluon comparison | `references/automl.md` |
| Optional SAP RPT comparison | `references/sap-rpt.md` |
| Output files and versioned schemas | `references/artifacts.md` |
| Example requests and clarification patterns | `references/examples.md` |

Always read `governance.md`, `data-analysis.md`, `data-and-leakage.md`, and
`artifacts.md`. In model-building or improvement mode, also read the selected
task reference, `optimization-and-ensembling.md`, and
`evaluation-and-production.md`. Read `automl.md` only after an explicit opt-in.
Read `sap-rpt.md` only after offering SAP RPT for a compatible supervised
tabular task and receiving an explicit opt-in.
For classification/regression with future outcomes or delayed labels, read
`time-series.md` as well for temporal validation and censoring safeguards.
Read `large-data.md` before loading data that may exceed local memory or disk.
Read `high-stakes.md` whenever predictions could materially affect a person's
health, safety, liberty, employment, credit, insurance, education, housing, or
access to essential services.

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
- cohort inclusion/sampling rule, label-observation mechanism and sample
  weights when applicable;
- prediction/scoring moment and features available then;
- time column, horizon, frequency, and known-future covariates when relevant;
- cost of false positives, false negatives, over- and under-prediction;
- deployment/batch context, latency or resource constraints;
- existing model, rule or measurable manual process, if one exists;
- data-governance constraints and required subgroup checks;
- standard or high-stakes risk classification;
- compute/time budget.

Use documented defaults only for low-risk reversible choices. Record all
assumptions and unresolved domain questions.

### 2. Prepare the environment

Create `.venv` in the user's project unless a suitable environment already
exists. Install only required packages into that environment. Ask before large
optional installations. Use the host's managed-process/session mechanism for
long-running work; do not shorten an approved ML budget to fit a tool timeout.

An EDA-only environment does not define the later model search space. At model
building time, freeze the scientifically eligible roster from task, data and
deployment criteria even if package state is already known. Treat XGBoost,
LightGBM and CatBoost as normal modeling dependencies unless a concrete
incompatibility or resource constraint applies, then install missing selected
dependencies into the project environment. Keep heavyweight optional systems
such as AutoGluon opt-in.

Treat SAP RPT as a separate opt-in remote benchmark for tabular classification
or regression, not as a normal dependency or candidate-roster member. Do not
clone its private repository or configure authentication. If the user opts in
and `sap-rpt` is unavailable or unconfigured, give the repository URL, ask the
user to perform the clone, installation, and interactive playground
configuration, and pause the RPT branch until they confirm completion. Never
request credentials or tokens. Continue the ordinary local workflow while RPT
is declined or unavailable.

Before loading data, estimate memory, disk, scan, and compute requirements.
Use the DuckDB profiler for data larger than safe in-memory limits. If data
cannot fit local disk or compute, execute aggregations/training where the data
already lives rather than downloading it.

Fail closed before scanning a remote source whose size/cost or declared version
is unknown. Require `--expected-source-bytes` and `--remote-source-version`;
use `--allow-unknown-remote-preflight` only after an explicitly recorded
decision. The generic URL profiler cannot verify version binding, so mark it
limited until a native source client verifies the snapshot/version or the
content is hashed.

### 3. Load and establish the data contract

Support local/object-store CSV/Parquet inputs and warehouse/lake tables through
their native query engines. Validate formats, schema, target derivation, row
grain, join keys, units, time coverage, label timing, and feature availability.
Record the source population, inclusion/exclusion rule, positive/negative
sampling, selective labeling, inclusion probabilities and weights. Do not treat
unlabeled rows as negatives without evidence.

For multiple datasets, confirm join/concat semantics; never assume row-wise
concatenation merely because schemas align.

Identify duplicate records, repeated entities, source-system columns,
identifier-like fields, high-cardinality categoricals, free text, sensitive
attributes, and suspected post-outcome fields. Do not remove anything before
understanding its business meaning.

### 4. Create evaluation partitions

Choose random stratified, grouped, temporal, or grouped-temporal validation
from the data-generating process—not from a fixed row-count rule. Persist row
assignments or deterministic split rules in `split_manifest.json`. Audit group
and duplicate overlap, temporal order, purge gaps and per-fold support. Check
minimum class/event counts and coverage once when establishing partitions,
then seal holdout targets before EDA or model decisions.

In analysis-only mode, do not invent a holdout. Analyze the full permitted
dataset and label the report as descriptive rather than predictive.

Before later modeling the same population, audit whether analysis-only work
inspected its targets. Treat overlapping rows as target-exposed
discovery/development data; a later split cannot make them sealed. Use fresh
external/prospective evidence for an unbiased final estimate, or report only
development/previously exposed evidence and its limitation.

Do not force a separate holdout when it would leave too few independent groups
or rare events for meaningful evaluation. Predeclare nested/repeated outer CV,
external validation, or prospective validation instead and state exactly what
independence the estimate does and does not provide.

### 5. Analyze and explain the dataset

Run `scripts/profile_dataset.py` when its supported inputs match the task;
otherwise follow `references/data-analysis.md` directly.

- Use full data only for structural facts that do not inspect held-out targets.
- Use the training partition for target-aware statistics and plots.
- Under nested CV, keep global EDA target-blind. Nest target-aware choices
  inside each outer training fold or exclude a dedicated discovery cohort from
  the reported outer estimate.
- Calculate statistics on all permitted rows; sample only expensive charts and
  record the sample method.
- Generate `data_report.html`, `data_summary.md`, `data_profile.json`, and
  `figures/`.
- Present findings as **blocker**, **warning**, or **information**.
- Explain what each important chart means and what decision it may affect.
- For panel time series, bound detailed example series with
  `--max-panel-series` and report panel coverage separately; do not collapse
  the panel into one unexplained mean.

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

Ask whether an existing operational model, rule or measurable manual process
exists. Compare it on the same eligible evaluation population when available;
otherwise record that no incumbent baseline exists and continue with the naive
and fixed baselines.

### 7. Improve candidates within the budget

Freeze candidate membership and suitability from task, data and deployment
criteria even if dependency state is already known, and before any dependency
probe or final evaluation. Use task-aware cross-validation and bounded
model-family searches. For supervised tabular work, write exactly one candidate
ledger row for each of XGBoost, LightGBM and CatBoost using the status vocabulary
in `references/governance.md`. Attempt each scientifically suitable family or
record a concrete environment-independent exclusion, installation failure,
user decline or quantified budget deferral. Give each eligible family minimum
coverage before adaptive optimization. Keep preprocessing inside folds, record
failed trials, control parallelism and memory, and stop on budget or defensible
convergence.

Treat stacking as optional. Attempt it only when diverse, competitive
candidates and sufficient out-of-fold data exist. Select it only on validation
evidence; never require diversity models that are unsuitable for the data.

For classification or regression, offer AutoGluon and SAP RPT once as optional
comparisons regardless of whether EDA ran earlier. Run either only after
explicit opt-in and follow its dedicated reference. Keep SAP RPT outside
`search.candidates`; compare it on the same permitted splits and metric, and
do not present the internal playground CLI as a deployable production model.

### 8. Select, calibrate, and evaluate

Finalize the model, threshold, calibration, or anomaly-review budget using only
the inner/development evidence permitted by the declared evaluation design.
Build calibration from group/time-valid out-of-fold predictions, a permanently
disjoint calibration set, or a declared calibrated-CV ensemble; never refit a
calibrator on in-sample final-model predictions.

Then evaluate once on the predeclared holdout/external set or aggregate
untouched outer folds. Report:

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

When a supervised decision uses a fixed daily/weekly capacity, separate
row-level scoring from whole-batch queue selection and test eligibility,
cutoffs, ties, duplicate handling, empty and sub-capacity batches.

### 10. Save and verify artifacts

Follow `references/artifacts.md`. Always include `schema_version`. Save the data
contract, feature manifest, data fingerprint, split manifest, run lineage,
candidate ledger, dependency versions, intended-use limitations, and exact
inference commands. Run the artifact-contract validator and real declared
inference cases:

```text
python scripts/validate_run.py <project-directory> \
  --artifacts-dir <run-directory> --run-inference-test
```

Contract validation cannot prove that the scientific design was followed, but
it must verify that declared inference cases produced real, schema-compatible
outputs. Reconcile scripts, logs, folds, metrics and reports before declaring
completion.

## Completion checklist

- [ ] Mode, business decision, prediction moment, row grain, and error costs
      are recorded.
- [ ] Cohort construction, label observation, sampling and weights are recorded.
- [ ] Split/evaluation strategy matches time/group/entity structure and every
      holdout/external/outer-fold target boundary remained sealed; the split
      manifest passes its structural audits.
- [ ] EDA artifacts exist and target-aware analysis used training data only.
- [ ] Feature availability and target derivation were audited for leakage.
- [ ] Preprocessing, resampling, and feature selection occurred inside folds.
- [ ] Baselines, optional incumbent comparison, candidate ledger, model search,
      and metrics match the selected task reference.
- [ ] Thresholds/calibration/forecast horizon/anomaly review budget are recorded
      when applicable.
- [ ] Final evaluation includes uncertainty, error analysis, and limitations.
- [ ] Inference schema and round-trip behavior were tested.
- [ ] Improvement lineage and evaluation-exposure history are preserved.
- [ ] Required artifacts pass `scripts/validate_run.py`.
- [ ] Optional AutoGluon, SAP RPT, or explainability ran only when requested;
      any RPT installation/configuration remained user-managed.
- [ ] `results.md` and `data_summary.md` agree with machine-readable artifacts.

If a checklist item cannot apply, record the reason rather than omitting it
silently.
