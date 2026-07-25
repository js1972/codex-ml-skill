# Dataset Analysis and Visualization

## Contents

- [Analysis modes](#analysis-modes)
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
- **Model building/improvement** — show full-data structural facts but restrict
  target-aware statistics and charts to training data.

EDA answers “what is in this dataset and what should we investigate?” Data
profiling answers “is this data safe and suitable for the proposed model?”
Include both, but do not let charts silently determine transformations.

## Reporting and holdout boundaries

- Treat supplied data as available for analysis and show observed category,
  class and feature values whenever they improve interpretation.
- Do not add automatic redaction or authorization gates.
- Keep charts readable: summarize high-cardinality fields and large tables
  because exhaustive rendering is unhelpful, not because values are hidden.
- In modeling mode, require a persisted partition column before target-aware
  analysis. Use only rows marked `train`.
- Do not plot holdout targets, target rates, residuals, or feature-target
  relationships before final evaluation.

## Core analysis

Calculate on all permitted rows unless noted:

- row/column counts, dtypes, inferred semantic types, memory footprint;
- missing counts/patterns and suspected sentinel values;
- exact and key-based duplicates;
- unique counts, cardinality ratios, constant and identifier-like columns;
- numeric quantiles, robust spread, skew, zeros, sign and impossible ranges;
- categorical frequency/support and rare levels;
- date ranges, timezone consistency, gaps, duplicates and event density;
- entity counts, rows per entity and entities spanning candidate partitions;
- target support on training data;
- pairwise numeric correlations and redundant/proxy candidates on training
  data;
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
  --task analysis|classification|regression|time-series|anomaly \
  [--target <column>] [--time-column <column>] \
  [--group-column <column>] [--partition-column <column>]
```

In model mode, the script must fail closed when target-aware analysis is
requested without a valid training partition. Use custom analysis when inputs
or task semantics exceed the script; preserve the same reporting and holdout
rules.

## Required outputs

- `artefacts/data_report.html`
- `artefacts/data_summary.md`
- `artefacts/data_profile.json`
- `artefacts/figures/*.png`

Link these from `results.md`. In analysis-only mode, `data_summary.md` is the
primary user-facing result.
