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

This workflow has 9 steps. **Every step must run** unless its own "skip when"
clause applies. Steps 4 (signal check) and 6 (stacking) are mandatory gates,
not optional enrichments — skipping them silently is a defect.

1. Intake and clarify requirements.
2. Set up Python environment.
3. Load, validate, and profile the dataset; resolve data quality issues with
   user confirmation.
4. Split into train/validation/holdout sets and train baseline model.
5. **Signal check** — compare baseline against label-shuffled baselines.
   If no signal is detected, halt and ask the user before proceeding.
6. Iterate with Optuna/TPESampler until convergence.
7. **Stacking ensemble** — for classification/regression, attempt a stacking
   ensemble of top diverse trial models. Adopt if it beats the best single
   model on validation; otherwise record the attempt and keep the single best.
8. Evaluate the chosen model **once** on the holdout test set.
9. Save artifacts and produce `results.md`.

Before declaring the run complete, work through the compliance checklist in
step 9. Any unchecked item means the run is not finished.

## Progress reporting (required)

The user cannot see what step you are on unless you tell them. The skill runs
across many tool calls and can otherwise feel like a long silence punctuated
by output. **Both of the following are required on every run** — missing
either is a defect:

**1. Maintain a TodoWrite list for the whole run.**

Before any other tool call in step 1, call `TodoWrite` with one todo per
workflow step (1–9), using the step names from the workflow overview above.
Mark each as `in_progress` when you start it and `completed` when you finish
it. If a step is replaced or skipped per its own "skip when" rules, mark it
`completed` with a brief note in the activeForm explaining what happened
(e.g. "skipped — time-series forecasting").

**2. Print a plain-text header at each major step boundary.**

This is a backup channel in case the todo UI is not visible (headless mode,
piped output, IDE without checkbox rendering). At the start of each major
step, print a single header line in this exact format:

```
▶ Step N/9 — <step name>
```

At the end of each step, print a one-line result summary:

```
✓ Step N/9 done — <one-line result, e.g. "baseline AUC 0.81 on validation">
```

For the two gates specifically, the closing line must include the verdict:

```
✓ Step 5/9 done — signal detected (real 0.81 vs shuffled 0.50 ± 0.02)
✓ Step 7/9 done — stacking rejected (4 families, +0.48% < 0.5% threshold)
```

These headers go to normal terminal output — not inside a tool call, not in
a code block. They give the user a heartbeat they can scan even if the
todo list is hidden.

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

## 3.5) Signal check — REQUIRED before iteration

**You MUST complete this step before running any Optuna trials.** Skipping it
silently is a defect. If this step is missing from `metrics.json` at the end
of the run, the run is incomplete.

The purpose is to verify the features actually contain predictive signal before
spending compute on hyperparameter search. Without this check, the skill will
confidently produce a polished model on a dataset that has nothing to learn from.

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

**When this step is replaced (not skipped)**

- Unsupervised anomaly detection: replace with a comparison against random
  scoring (no labels to shuffle).
- Time-series forecasting: replace with a comparison against a naive-forecast
  baseline (last-value or seasonal naive). Only proceed if the real baseline
  beats the naive forecast by a non-trivial margin.

In both cases the result of the replacement check **must still be recorded in
`metrics.json`** under the `signal_check` key.

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

**Always record the result**

Whether signal is detected or not, populate `signal_check` in
`artefacts/metrics.json` with: `ran` (bool), `permutations` (int),
`real_baseline_score` (float), `shuffled_mean` (float), `shuffled_std` (float),
`signal_detected` (bool), `user_overrode_no_signal` (bool, default false).

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

## 4.5) Stacking ensemble — REQUIRED to attempt for classification and regression

**For classification and regression, you MUST attempt this step after Optuna
converges.** Skipping it silently — when none of the explicit "skip when"
conditions apply — is a defect. If `stacking` is missing from `metrics.json`
at the end of a classification/regression run, the run is incomplete.

On tabular problems, stacking typically adds 1–3% on the chosen metric over
the single best model and rarely loses. The acceptance rule below ensures
you only adopt the ensemble when it actually helps.

**Explicit skip conditions (record the reason in `metrics.json` if any apply)**

- Time-series forecasting (stacking interacts badly with temporal CV; keep
  the single best model).
- Anomaly detection (no clean way to stack scores across heterogeneous methods).
- Fewer than 3 distinct model families converged within 10% of the best
  validation score — stacking 3 copies of the same model family adds nothing.
- User explicitly opted out, or compute budget is exhausted.

For any of these, record `stacking.attempted = false` and
`stacking.reason = "<which skip condition>"` in `metrics.json`. Do not
silently omit the key.

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

**Always record the result**

Whether adopted or rejected, populate `stacking` in `artefacts/metrics.json`
with: `attempted` (bool), `base_learners` (list of family names),
`meta_learner` (string), `best_single_score` (float), `ensemble_score` (float
or null if skipped), `adopted` (bool), `reason` (string explaining adopt /
reject / skip).

## 5) Outputs

- Create `artefacts/` if it does not exist.
- Save all artifacts as described in `references/artifacts.md`.
- Ensure `infer.py` uses the same preprocessing pipeline as `train.py`.
- **Final metrics must come from the holdout test set** — not the validation
  set used during Optuna optimization. Report both in `metrics.json`.
- Provide a concise final summary to the user (see `references/artifacts.md`
  for the full structure of `results.md`).

### Compliance checklist (run before declaring done)

Before telling the user the run is complete, verify every item below. If any
item is unchecked, the run is incomplete — finish the missing step, do not
hand off a partial run.

- [ ] Progress reporting: `TodoWrite` list was used, and `▶ / ✓` step headers
      were printed for every step.
- [ ] Profiling decisions recorded in `config.json` (dropped columns,
      imputation strategy, transforms, outlier handling).
- [ ] **Signal check** ran and the result is recorded in `metrics.json`
      under `signal_check` (real score, shuffled mean and std, verdict).
      For unsupervised or time-series tasks, the replacement check ran and
      its result is recorded.
- [ ] If signal was not detected, the user was asked whether to proceed and
      their choice is recorded in `signal_check.user_overrode_no_signal`.
- [ ] Optuna ran (or was explicitly skipped due to no-signal halt) and the
      trial count plus convergence reason is recorded.
- [ ] **Stacking** is recorded in `metrics.json` under `stacking`. For
      classification/regression, `attempted` is `true` unless an explicit
      skip condition applied (recorded in `stacking.reason`). For time-series
      and anomaly detection, `attempted` is `false` with the appropriate
      reason.
- [ ] Final score comes from the holdout test set, not validation, and
      `final.eval_set == "holdout_test"` in `metrics.json`.
- [ ] `model.joblib`, `train.py`, `infer.py`, `metrics.json`, `config.json`,
      and `results.md` all exist (unless the run halted at the no-signal gate,
      in which case `metrics.json` and `config.json` still exist).
- [ ] Plain-language `results.md` written, including the signal-check verdict
      and the stacking outcome.

If any item failed, fix it and re-check before reporting completion.

## References

- Defaults, task requirements, baseline and iteration guidance:
  `references/defaults.md`
- Artifact naming and JSON structure: `references/artifacts.md`
- Example prompts and expected clarifications:
  `references/examples.md`
