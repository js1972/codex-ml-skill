---
name: tabular-eda
description: Explore and explain local tabular CSV or Parquet datasets, including data quality, distributions, relationships, optional target behavior, time coverage, and group structure. Use when Codex is asked for exploratory data analysis, dataset profiling, data understanding, or an EDA report without model training.
---

# Tabular EDA

Analyze the source dataset without changing it or preparing a modeling run.

## Workflow

1. Confirm the local source file and any user-supplied target, time, or group columns. Do not infer that a likely column has one of these meanings without saying so.
2. Choose the requested report path, normally `<project>/eda_report.html`. Keep it outside model artifact or run directories.
3. Run the bundled profiler:

   ```bash
   /path/to/python-with-pandas-and-numpy scripts/analyze_tabular.py \
     --input /absolute/path/data.csv \
     --output /absolute/path/eda_report.html \
     [--target target] [--time-column timestamp] [--group-column entity_id]
   ```

   The interpreter must already provide `pandas` and `numpy`; Parquet additionally
   requires `pyarrow` or `fastparquet`. Use the project environment, bundled workspace
   runtime, or another interpreter where these packages are installed. Do not install
   packages automatically. Use `--delimiter ';'` when automatic CSV delimiter detection
   is unsuitable. Use `--max-plot-rows` and `--seed` only to adjust deterministic chart
   sampling.
4. Use the concise Markdown findings printed to stdout as the basis for the chat response. Prioritize implications and limitations over a list of raw statistics.
5. Link the generated `eda_report.html`. Verify that the requested report is the only persistent file created by this skill.

## Output contract

- Return important findings in chat.
- Create exactly one persistent artifact: a self-contained HTML report with inline CSS and SVG charts.
- Do not create JSON files, Markdown files, README files, figure folders, split files, model artifacts, or run directories.
- Do not feed this report into later model training. A modeling skill must independently read the source data and perform its own modeling preflight.
- Do not expose raw record samples in the report. Summaries and charts may use a deterministic row sample, disclosed in the report, while data-quality counts use the complete loaded dataset.

## Boundaries

- Treat target analysis as descriptive only. Do not choose metrics, splits, features, or candidate models.
- Do not perform model ablations. You may identify feature/source groups that
  merit a later ablation, but only `ml-model-builder` may test their effect on
  model performance after an approved modeling plan.
- Report possible identifiers, pipeline metadata, high cardinality, repeated
  observations, missingness, imbalance, outliers, and suspicious target copies as
  observations requiring domain review, not automatic deletion instructions.
- Assess repeated observations after excluding likely row identifiers and pipeline
  metadata such as split, partition, fold, and duplicate-group fields. State which
  columns were excluded so the result is auditable.
- Stop with a clear error for remote URLs, missing hint columns, unreadable formats, empty datasets, or unavailable Parquet support.
