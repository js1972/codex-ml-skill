---
name: ml-model-builder
description: Build classical machine learning models from local or URL datasets, including requirement gathering, baseline training, iterative improvement with bounds, and saving train/infer scripts and artifacts in artefacts/. Use when users ask to create, train, or improve classification, regression, time-series forecasting, or anomaly detection models.
---

# ML Model Builder

## Overview

Guide Codex to gather ML requirements, build a baseline, iterate to a stronger
model, and save code and artifacts under `artefacts/`.

## Workflow

1. Intake and clarify requirements.
2. Set up Python environment.
3. Load and validate dataset.
4. Train and report baseline model.
5. Iterate to better model within bounds.
6. Save artifacts and summarize results.

## 0) Environment setup (required)

- Before running any Python code or installing dependencies, create a venv in
  the current working directory:
  - `python3 -m venv .venv`
- Always run Python and pip from the venv:
  - `.venv/bin/python`, `.venv/bin/pip`
- Assume implicit approval to install dependencies into the venv. Do not
  install system-wide packages.
- Keep dependencies minimal and report what was installed.

## 1) Intake and clarification

- Ask the minimum required inputs before training:
  - dataset location(s) (local path(s) or URL(s))
  - task type
  - evaluation metric (or accept default)
  - split strategy
- Ask for time column and any entity/group identifier to choose an appropriate
  split and CV strategy (see `references/defaults.md`).
- Ask for a random seed (default in `references/defaults.md`).
- Ask task-specific requirements (see `references/defaults.md`).
- Ask for any domain-specific feature ideas and confirm whether to apply
  standard feature engineering (date parts, lags, transforms).
- Ask whether to run explainability (SHAP) and whether to change training bounds.
- If multiple dataset locations are provided, ask how to combine them and
  whether to add a source column.
- Run a quick LLM suitability check (see `references/defaults.md`). If it
  triggers, recommend an LLM-based approach and ask whether to proceed with
  classical ML anyway.
- Confirm defaults when the user does not specify values.

## 2) Dataset handling

- Support local CSV/Parquet or HTTP(S) URL to CSV/Parquet only.
- Support multiple files; default to row-wise concat if schemas align.
- Validate target column exists (if applicable) and identify feature types.
- If target is derived (threshold or date difference), record the rule in
  `artefacts/config.json`.
- Apply default feature engineering where appropriate (see
  `references/defaults.md`) and allow the user to override.
- Run a quick leakage guard: flag features that are identical to the target,
  contain the target name, or are derived directly from the target.
- Do not attempt authenticated cloud buckets.

## 3) Baseline model

- Use a simple, fast pipeline with minimal tuning.
- Report baseline metrics and store them in `artefacts/metrics.json`.
- Use default models and metrics if the user did not specify them.

## 4) Iteration

- Improve preprocessing and model selection.
- Include non-sklearn models when appropriate (e.g., XGBoost, LightGBM,
  CatBoost). Install them into the venv if needed.
- Expand feature engineering for the iteration stage if it improves the metric
  and does not introduce leakage.
- Use the agreed metric to pick the best model.
- Respect bounds and stop early after repeated non-improving trials.
- Keep a clear audit trail in `artefacts/config.json`.

## 5) Outputs

- Create `artefacts/` if it does not exist.
- Save:
  - `train.py`
  - `infer.py`
  - model artifact
  - `metrics.json`
  - `config.json`
  - optional SHAP output if requested
- Ensure `infer.py` uses the same preprocessing as `train.py`.
- Provide a concise final summary to the user:
  - best model, key metric(s), baseline vs final score
  - split strategy and key preprocessing choices
  - training bounds used and whether early stopping occurred
  - artifact paths
- Also save the same summary to `results.md` in the project root.
- Include a short, non-expert explanation of each reported metric in the final
  summary and in `results.md`.
- Include a data profile summary in `results.md` (row/column counts,
  missingness, target distribution or summary).
- Include a brief feature engineering summary in `results.md`.

## References

- Defaults, task requirements, baseline and iteration guidance:
  `references/defaults.md`
- Artifact naming and JSON structure: `references/artifacts.md`
- Example prompts and expected clarifications:
  `references/examples.md`
