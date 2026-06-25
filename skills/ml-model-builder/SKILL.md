---
name: ml-model-builder
description: Build classical machine learning models from local or URL datasets, including requirement gathering, data profiling, baseline training, convergence-driven iterative improvement with Optuna, and saving train/infer scripts and artifacts in artefacts/. Use when users ask to create, train, or improve classification, regression, time-series forecasting, or anomaly detection models.
---

# ML Model Builder

## Overview

Guide Codex to gather ML requirements, profile the dataset for data quality
issues, build a baseline, iterate to a stronger model using Optuna with
convergence-based stopping, and save code and artifacts under `artefacts/`.

## Workflow

1. Intake and clarify requirements.
2. Set up Python environment.
3. Load and validate dataset.
4. Profile dataset and resolve data quality issues with user confirmation.
5. Split into train/validation/holdout sets and train baseline model.
6. Run a signal check: compare baseline against a label-shuffled baseline.
   If no detectable signal, halt and recommend alternatives.
7. Iterate with Optuna/TPESampler until convergence.
8. Build a stacking ensemble from the top diverse trial models
   (classification and regression only).
9. Evaluate final model on holdout test set and save artifacts.

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
- Run a quick leakage guard: flag features that are identical to the target,
  contain the target name, or are derived directly from the target.
- Do not attempt authenticated cloud buckets.

## 2.5) Pre-training data profiling

Run this analysis after loading the dataset and before any splitting or
training. The goal is to configure the training run correctly, not just report
findings.

**Communication rule for all confirmations:** Always explain in plain,
jargon-free language — what was found, why it matters in one sentence, and a
clear recommendation. State the default. Never present raw statistical terms
without explaining them. Example format:
> "2 columns have almost no variety — over 95% of rows contain the same value.
> Columns like this can't help the model learn anything and slow down training.
> I recommend removing them. OK to drop them? [Y/n]"

**Shape and duplicates**
- Report row/column count.
- Detect duplicate rows; if found: explain that identical rows can unfairly
  skew the model's learning, recommend dropping, confirm before doing so
  (default: yes).

**Missing values**
- Report missingness per column (count + %).
- Columns > 50% missing: explain that more than half the data is absent so
  filling it in would mean mostly guessing; ask whether to drop the column or
  fill it anyway (default: drop).
- Columns 20–50% missing: record for median/mode imputation inside the training
  pipeline — do NOT fill now, before splitting, as that would let test data
  influence training.
- Record final imputation strategy per column in `artefacts/config.json`.

**Near-zero variance**
- Flag columns where > 95% of values are identical.
- Explain that a column with almost no variety cannot teach the model anything;
  recommend dropping; confirm before doing so (default: yes).
- Record dropped columns in `artefacts/config.json`.

**Highly correlated features**
- Flag numeric column pairs with |r| > 0.95.
- Explain in plain terms: "These two columns contain almost identical
  information. Keeping both is redundant and can confuse the model. I recommend
  keeping [X] and removing [Y] because X has a stronger relationship with what
  we're trying to predict. (this is called multicollinearity)"
- Confirm before dropping (default: yes); record in `artefacts/config.json`.

**Numeric skew**
- Flag columns with |skew| > 1.0.
- Explain: "This column has a few very large values that could pull the model
  in the wrong direction. I'll apply a standard adjustment to balance it out
  (log transformation)." Record which columns need log1p transforms in
  `artefacts/config.json` — but do NOT apply them yet. Transforms are applied
  only inside the training pipeline after splitting, to avoid test data
  influencing training.
- Record transforms in `artefacts/config.json`.

**Outliers**
- Flag columns with rows beyond 3 IQRs from the median; report count and %.
- Explain: "X rows in column Y have extreme values that are far outside the
  normal range. These can throw off the model. Options: (1) cap them at a
  sensible limit [recommended] (Winsorization), (2) remove those rows,
  (3) leave them as-is."
- Confirm choice before acting (default: cap); record decision in
  `artefacts/config.json`.
- Capping/removal is applied inside the training pipeline after splitting —
  not on the raw dataset now, to avoid test data influencing training.

**Target leakage signals**
- Flag features with |correlation to target| > 0.9.
- Explain: "The column '[name]' is almost perfectly linked to what we're trying
  to predict. This usually means it was calculated using the answer, which would
  make the model look accurate in testing but fail in real use. I recommend
  removing it. (this is called target leakage)"
- Confirm before dropping (default: yes); record in `artefacts/config.json`.

**Class imbalance (classification and supervised anomaly detection)**
- Report class distribution in plain terms: e.g. "87% of rows are class A,
  13% are class B".
- If minority class < 20%: explain the imbalance will bias the model toward
  the majority class; apply `class_weight='balanced'` automatically and note
  it in the summary.
- If minority class < 5%: additionally explain the imbalance is severe; ask:
  "Would you like me to also artificially generate extra examples of the rare
  class to help the model learn it better? (this technique is called SMOTE)
  [Y/n]"

**Temporal integrity (time series only)**
- Check for gaps or irregular frequency in the time column.
- Report in plain terms: e.g. "The data runs from Jan 2020 to Dec 2023 but
  there are 14 missing weeks."
- Ask: "Missing time periods can disrupt forecasting. Should I fill them in
  using the surrounding values, or leave the gaps? [fill in / leave gaps]"

**Profiling summary**
- Print a concise plain-language summary of all findings before proceeding.
- List what was fixed automatically and what decisions the user made.
- Only proceed to baseline once all confirmations are resolved.

## 3) Baseline model

- Split the dataset first (see `references/defaults.md`) into train, validation,
  and holdout test sets before any fitting:
  - Train: used to fit the baseline model
  - Validation: used to report baseline metrics
  - Holdout test: set aside now and not touched until final evaluation in step 5
- Use a simple, fixed-configuration pipeline — no hyperparameter search.
- All preprocessing transforms (imputation, log1p, encoding, scaling) must be
  fit on the training fold only and applied to validation and holdout.
- Report baseline metrics from the validation set and store them in
  `artefacts/metrics.json`.
- Use default models and metrics if the user did not specify them (see
  `references/defaults.md`).

## 3.5) Signal check (required before iteration)

After the baseline trains, verify that the features actually contain predictive
signal before spending compute on hyperparameter search. This prevents the skill
from confidently producing a polished model on a dataset that has nothing to
learn from.

**How to run it**

- Permute the target column 5 times using different seeds derived from the
  agreed random seed (e.g. `random_seed + i`).
- For each permutation, train the same baseline pipeline on the (shuffled-label)
  training fold and score it on the validation set.
- Compute the mean and standard deviation of the 5 shuffled scores.
- Compare the real baseline score to the shuffled distribution.

**Decision rule**

- Higher-is-better metrics (f1, auc, r2, accuracy):
  - Real baseline ≤ shuffled mean + 2·shuffled std → **no detectable signal**
  - Otherwise → signal present, proceed to iteration
- Lower-is-better metrics (rmse, mae, mape):
  - Real baseline ≥ shuffled mean − 2·shuffled std → **no detectable signal**
  - Otherwise → signal present, proceed to iteration

**Skip when**

- Unsupervised anomaly detection (no labels to shuffle).
- Time series forecasting where shuffling destroys the temporal structure —
  use a naive-forecast baseline (last-value or seasonal naive) as the
  signal floor instead, and only proceed if the real baseline beats it.

**If no signal is detected**

- Halt iteration. Do **not** run Optuna.
- Report the finding in plain language. Example:
  > "I ran 5 sanity checks where the answers were randomly shuffled. Our
  > baseline scored 0.51 on the real data and 0.49 ± 0.02 on shuffled data.
  > This means the model can barely tell the real labels apart from random
  > ones — the features in this dataset don't contain enough information to
  > predict the target. Continuing would produce a model that looks plausible
  > but isn't actually useful."
- Suggest alternatives: collect more or different features, reframe the target,
  consider an LLM-based approach if the signal might be in free text, or
  confirm the target derivation is correct.
- Ask the user whether to (a) stop here, or (b) proceed with Optuna anyway
  knowing the result is unlikely to be useful.
- Record the signal-check result in `artefacts/metrics.json` regardless of
  the user's choice.

## 4) Iteration

- Use **Optuna with TPESampler** as the hyperparameter optimizer. Install into
  venv if needed (`optuna`).
- Seed TPESampler with the agreed random seed:
  `optuna.samplers.TPESampler(seed=random_seed)` — required for reproducibility.
- Define a search space per model and let Optuna suggest parameters each trial.
- Each Optuna trial is evaluated on the validation set only — the holdout test
  set is never used during optimization.
- **CV strategy per trial:**
  - If dataset has ≥ 5k rows: use a single validation split per trial
  - If dataset has < 5k rows: use 5-fold cross-validation per trial; average
    the fold scores as the trial objective
- All preprocessing transforms (imputation, log1p, encoding, scaling) must be
  fit inside each trial's training fold — never fit on the full dataset before
  splitting.
- Include non-sklearn models when appropriate (XGBoost, LightGBM, CatBoost).
  Install into venv if needed.
- Expand feature engineering if it improves the metric and does not introduce
  leakage.
- Use the agreed metric to pick the best model.
- Stop when convergence criteria are met (see `references/defaults.md`) — not
  on a fixed time limit.
- Log progress every 10 trials: trial number, best score so far, current score.
- Keep a clear audit trail in `artefacts/config.json`.

## 4.5) Stacking ensemble (classification and regression only)

After Optuna converges, build a stacking ensemble from the top diverse trial
models. On tabular problems this typically adds 1–3% on the chosen metric over
the single best model, and rarely loses.

**When to skip**

- Time-series forecasting (stacking interacts badly with temporal CV; keep
  the single best model).
- Anomaly detection (no clean way to stack scores across heterogeneous methods).
- Fewer than 3 distinct model families converged with reasonable scores —
  stacking 3 copies of the same model family adds nothing.
- User opts out, or compute budget is exhausted.

**How to build it**

- Select the top model from each distinct model family that finished within
  10% of the best validation score (e.g. LightGBM best + XGBoost best +
  RandomForest best). Cap at 5 base learners.
- Generate out-of-fold predictions for each base learner using 5-fold CV on
  the training fold only (or 5 walk-forward folds for time-aware data).
- Train a simple meta-learner on the stacked out-of-fold predictions:
  - Classification: `LogisticRegression` with `class_weight='balanced'`
  - Regression: `Ridge`
- Fit each base learner on the full training fold (no CV) for the final
  pipeline. The meta-learner uses their predictions on validation/holdout.
- All preprocessing for each base learner stays inside its own pipeline —
  fit on training fold only.

**Acceptance rule**

- Score the stacked ensemble on the validation set.
- Adopt the ensemble only if it beats the single best model by ≥ 0.5%
  relative on the chosen metric. Otherwise keep the single best model and
  record that stacking was tried but rejected.

**Record in `artefacts/config.json` and `metrics.json`**

- Whether stacking was attempted, the base learners selected, the meta-learner,
  the ensemble validation score, and the adoption decision.

## 5) Outputs

- Create `artefacts/` if it does not exist.
- Save all artifacts as described in `references/artifacts.md`.
- Ensure `infer.py` uses the same preprocessing pipeline as `train.py`.
- **Final metrics must come from the holdout test set** — not the validation
  set used during Optuna optimization. Report both in `metrics.json`.
- Provide a concise final summary to the user (see `references/artifacts.md`
  for the full structure of `results.md`).

## References

- Defaults, task requirements, baseline and iteration guidance:
  `references/defaults.md`
- Artifact naming and JSON structure: `references/artifacts.md`
- Example prompts and expected clarifications:
  `references/examples.md`
