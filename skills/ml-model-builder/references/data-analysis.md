# Dataset Analysis and Visualization

## Contents

- [Analysis modes](#analysis-modes)
- [Transitioning analysis to modeling](#transitioning-analysis-to-modeling)
- [Reporting and holdout boundaries](#reporting-and-holdout-boundaries)
- [Core analysis](#core-analysis)
- [Task-specific charts](#task-specific-charts)
- [Chart selection and scale](#chart-selection-and-scale)
- [Findings and interpretation](#findings-and-interpretation)
- [Bundled report generator](#bundled-report-generator)
- [Required outputs](#required-outputs)

## Analysis modes

Support:

- **Analysis only** — describe all permitted data. Do not imply predictive
  validity or create a holdout unless the user asks for modeling.
- **Model building/improvement** — show target-blind structural facts on all
  permitted rows. For holdout/external/prospective designs, restrict
  target-aware work to development data. For nested CV, repeat target-aware
  analysis inside each outer-training partition or use a fingerprinted
  discovery cohort excluded from every reported outer-fold estimate.

EDA answers “what is in this dataset and what should we investigate?” Data
profiling answers “is this data safe and suitable for the proposed model?”
Include both, but do not let charts silently determine transformations.

## Transitioning analysis to modeling

Analysis-only mode may inspect targets across the full permitted population.
Before reusing that population for modeling, audit the prior report and
decisions for target exposure. Do not retroactively carve an untouched holdout
or outer-fold estimate from exposed rows; rerunning model-mode EDA with correct
boundaries does not erase the earlier exposure.

Treat overlapping rows as discovery/development data. Use a fresh external or
prospective population for an unbiased final estimate. If none is available,
label model-selection evidence as development or previously exposed benchmark
evidence and state the limitation. Record the exposure source, values viewed,
decisions influenced and final-population overlap under
`config.analysis.pre_partition_target_exposure`.

## Reporting and holdout boundaries

- Treat supplied data as available for analysis and show observed category,
  class and feature values whenever they improve interpretation.
- Do not add automatic redaction or authorization gates.
- Keep charts readable: summarize high-cardinality fields and large tables
  because exhaustive rendering is unhelpful, not because values are hidden.
- In modeling mode, require a persisted development, discovery or current
  outer-training boundary before target-aware analysis.
- Under nested CV, keep global EDA target-blind. Do not use all-row
  feature-target relationships, target distributions or label-informed data
  cleaning to make a shared choice across outer folds.
- Do not plot holdout, external, prospective, or active outer-fold targets,
  target rates, residuals, or feature-target relationships before their
  permitted evaluation step.

## Core analysis

Calculate feature-only structural facts on all permitted rows. Calculate target
values, target frequencies/quantiles and feature-target relationships only on
the permitted development, discovery or current outer-training population:

- row/column counts, dtypes, inferred semantic types, memory footprint;
- missing counts/patterns and suspected sentinel values;
- exact and key-based duplicates;
- unique counts, cardinality ratios, constant and identifier-like columns;
- numeric quantiles, robust spread, skew, zeros, sign and impossible ranges;
- categorical frequency/support and rare levels;
- date ranges, timezone consistency, gaps, duplicates and event density;
- entity counts, rows per entity and entities spanning candidate partitions;
- target support on the permitted development or outer-training population;
- pairwise numeric correlations and redundant/proxy candidates on the
  permitted development or outer-training population;
- feature availability at prediction time;
- source-system and partition-distribution differences.

## Task-specific charts

| Route | Charts |
|---|---|
| All | Missingness bars, numeric histograms/ECDFs, robust boxplots, category-frequency bars, cardinality, time coverage |
| Classification | Training class counts, numeric distributions by class, supported category response rates; after modeling: PR/ROC, calibration, threshold trade-offs |
| Regression | Training target distribution, feature-target scatter/hexbin; after modeling: predicted-vs-actual, residual distribution and residuals by segment |
| Forecasting | Training history, gaps, seasonal views, rolling summaries, lag/ACF diagnostics, series length/coverage; after modeling: backtest traces and interval coverage |
| Anomaly | Robust feature distributions, entity/time volumes, optional PCA projection labelled exploratory; after scoring: score distribution and reviewed top-k composition |

Do not use t-SNE/UMAP clusters as evidence of class separability or anomalies.
If included for exploration, label them stochastic projections and record
parameters.

## Chart selection and scale

- Prefer ECDF/quantile views for heavy tails; avoid hiding them with arbitrary
  histogram bins.
- Limit default charts to the most informative 12 numeric and 12 categorical
  columns. Put the complete statistics in JSON.
- Limit correlation heatmaps to 20 variables chosen by variance/support and
  task relevance.
- Avoid pair plots on wide or large datasets.
- Sample plots deterministically (default 10,000 rows, seed 42) but compute
  summary statistics on full permitted data.
- Record population size, plotted sample size, sampling method and seed.
- Use accessible palettes, descriptive titles, units, legends and alt-text-like
  captions in the HTML report.

## Findings and interpretation

Classify each finding:

- **Blocker** — invalid target, unresolved row grain, leakage, unusable split,
  insufficient events, corrupt schema.
- **Warning** — drift, severe missingness, unstable categories, imbalance,
  weak coverage, outliers, potential proxy leakage.
- **Information** — descriptive patterns that affect interpretation but do not
  require action.

For each material finding state:

1. evidence;
2. why it matters;
3. whether it is model-dependent;
4. recommended next check or decision;
5. what not to conclude.

## Bundled report generator

Use:

```text
python scripts/profile_dataset.py \
  --input <csv-or-parquet> \
  --output-dir artefacts \
  --mode analysis-only|model \
  [--run-kind initial|improvement] \
  --task analysis|classification|regression|time-series|anomaly \
  [--target <column>] [--time-column <column>] \
  [--group-column <column>] [--partition-column <column>] \
  [--train-label <development-label>] \
  [--split-strategy random|stratified_random|grouped|temporal|grouped_temporal] \
  [--group-overlap-policy disallow|known_series_temporal|known_entity_temporal] \
  [--evaluation-design holdout|nested_cv|external_test|prospective_validation] \
  [--expected-source-bytes <bytes>] [--expected-source-rows <rows>] \
  [--remote-source-version <etag-version-or-snapshot>] \
  [--risk-tier not_assessed|standard|high] \
  [--max-panel-series 12]
```

The script estimates local in-memory footprint before loading a local file.
Auto mode routes beyond-memory data to `profile_large_dataset.py` when DuckDB
is installed. Configure `--duckdb-memory-limit`, `--duckdb-temp-directory` and
`--threads`; read `large-data.md` when data may exceed local disk or practical
local scan time. Remote profiling requires expected bytes and a declared source
version unless the user explicitly accepts the recorded
`--allow-unknown-remote-preflight` override. The generic URL profiler records
that declaration as unverified and reproducibility as limited.

For panel time series, `--max-panel-series` bounds detailed local series
diagnostics (default 12). Treat aggregate coverage as the population view and
record how displayed series were selected; the profiler does not replace
rolling-origin evaluation.

In model mode, the script must fail closed when target-aware analysis is
requested without a valid development partition. Declare `--split-strategy`;
the profiler does not infer evaluation mechanics merely from the presence of a
time or group column. Group overlap is disallowed unless a known-series
forecasting contract selects `known_series_temporal` or a future-event
classification/regression contract selects `known_entity_temporal`. Both
require temporal ordering, a group key and as-of feature checks. Use
`--run-kind improvement` when profiling a new child run; an output directory
containing `run_manifest.json` is immutable. The selected evaluation design is
written to `config.json`; external and prospective designs still
require their immutable cohort fingerprint before validation. The generic
profiler deliberately keeps a global nested-CV report target-blind. Run
target-aware analysis separately inside each outer-training fold, or use a
custom analysis on a fingerprinted discovery cohort that the split manifest
excludes from outer evaluation. Preserve the same reporting boundaries when
inputs or task semantics exceed the script.

## Required outputs

- `<output-dir>/data_report.html`
- `<output-dir>/data_summary.md`
- `<output-dir>/data_profile.json`
- `<output-dir>/figures/*.png`

In model mode, link these from the run's `results.md`. In analysis-only mode,
`data_summary.md` is the primary user-facing result.
