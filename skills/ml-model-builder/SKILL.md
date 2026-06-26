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

This workflow has 11 steps. **Every step must run** unless its own "skip when"
clause applies. Steps 5 (signal check) and 7 (stacking) are mandatory gates,
not optional enrichments — skipping them silently is a defect. Step 9
(AutoGluon comparison) runs only if the user opted in at intake.

1. Intake and clarify requirements.
2. Set up Python environment.
3. Load, validate, and profile the dataset; resolve data quality issues with
   user confirmation.
4. Split into train/validation/holdout sets and train baseline model.
5. **Signal check** — compare baseline against label-shuffled baselines.
   If no signal is detected, halt and ask the user before proceeding.
6. Iterate with Optuna/TPESampler until convergence. The pool must include
   at least one non-tree model family (see §4 and `references/defaults.md`).
7. **Stacking ensemble** — for classification/regression, attempt a stacking
   ensemble of top diverse trial models. Adopt if it beats the best single
   model on validation; otherwise record the attempt and keep the single best.
8. **Ceiling check** — decide whether the dataset is near its predictive
   ceiling and record the verdict.
9. **AutoGluon comparison** (opt-in only) — if the user opted in at intake,
   run AutoGluon on the same split as a head-to-head reference.
10. Evaluate the chosen model **once** on the holdout test set.
11. Save artifacts and produce `results.md`.

Before declaring the run complete, work through the compliance checklist in
step 9. Any unchecked item means the run is not finished.

## Progress reporting (required)

The user cannot see what step you are on unless you tell them. The skill runs
across many tool calls and can otherwise feel like a long silence punctuated
by output. **Both of the following are required on every run** — missing
either is a defect:

**1. Maintain a TodoWrite list for the whole run.**

Before any other tool call in step 1, call `TodoWrite` with one todo per
workflow step (1–11), using the step names from the workflow overview above.
Mark each as `in_progress` when you start it and `completed` when you finish
it. If a step is replaced or skipped per its own "skip when" rules, mark it
`completed` with a brief note in the activeForm explaining what happened
(e.g. "skipped — time-series forecasting"; "skipped — user did not opt in
to AutoGluon").

**2. Print a plain-text header at each major step boundary.**

This is a backup channel in case the todo UI is not visible (headless mode,
piped output, IDE without checkbox rendering). At the start of each major
step, print a single header line in this exact format:

```
▶ Step N/11 — <step name>
```

At the end of each step, print a one-line result summary:

```
✓ Step N/11 done — <one-line result, e.g. "baseline AUC 0.81 on validation">
```

For the mandatory gates and AutoGluon comparison, the closing line must
include the verdict:

```
✓ Step 5/11 done — signal detected (real 0.81 vs shuffled 0.50 ± 0.02)
✓ Step 7/11 done — stacking rejected (4 families, +0.48% < 0.5% threshold)
✓ Step 8/11 done — near ceiling (top-3 spread 0.8%, stacking failed)
✓ Step 9/11 done — AutoGluon 0.881 vs pipeline 0.875 (+0.7% gap)
```

These headers go to normal terminal output — not inside a tool call, not in
a code block. They give the user a heartbeat they can scan even if the
todo list is hidden.

## Spec discipline (read before any step)

Your job on every decision in this skill is **either to follow the spec or
to propose a deviation to the user and wait for approval** — never to deviate
silently. Each decision in the steps below is tagged with one of three
categories. **The tag determines what you are allowed to do**:

### `[ASK]` — pause and ask the user, every time

The user makes the call. You explain the situation in plain language, state
your recommendation if you have one, and **wait for a Y/N (or specific
choice) response before proceeding**.

You must NOT:
- Apply the recommended default silently and report what you did.
- Batch multiple `[ASK]` items into a single "defaults I'll apply unless you
  object" block. Each `[ASK]` item gets its own prompt and its own answer.
- Treat "default: yes" wording as permission to skip the question. The default
  is what you **recommend**, not what you apply unilaterally.

### `[DEFAULT]` — apply the documented value silently if the user didn't specify

These are inexpensive, easily-reversed, low-impact choices (random seed,
metric default when unspecified, file naming). You may apply the documented
value without asking. You may mention it in a brief defaults summary. No
confirmation needed.

### `[SPEC]` — use the documented value unless the data warrants a deviation

These are documented parameters (timeouts, split ratios, baseline algorithm,
convergence rules) chosen because they are sensible defaults across most
datasets.

**Default behaviour: use the exact documented value.**

**Permitted exception: dataset-grounded deviation, proposed to the user.**
If the EDA you just performed reveals a **specific concrete property of this
dataset** (size, imbalance, distribution, structure, leakage signal) that
makes the documented value clearly suboptimal, you may **propose** a
deviation to the user. The proposal must include:
- The exact property of this dataset that triggered the proposal
- The documented value and the proposed alternative
- The trade-off in one sentence
- A Y/N choice between spec and proposal

You must wait for the user's response. You must NOT deviate silently — even
a well-reasoned deviation applied without asking is a defect.

You must NOT propose a deviation based on **general preferences** ("X usually
works better than Y on tabular data"). Those generic preferences are already
encoded in the spec; the spec is what we want by default. A proposal is only
valid when grounded in this dataset's specific profile.

### Examples — the right shape of behaviour

**Right (`[ASK]`):**
> "I found 1,177 exact duplicate rows (18% of the dataset). Duplicates can
> unfairly skew training. I recommend dropping them. OK? [Y/n]"

**Wrong (`[ASK]` applied as default):**
> "I'll apply these defaults unless you object: drop 1,177 duplicates, apply
> log1p to 6 skewed columns, cap outliers in 3 columns, ..."

**Right (`[SPEC]` with dataset-grounded proposal):**
> "The skill default split is 80/10/10. This dataset has 5,320 rows with only
> 19% positive class (~1,011 positive examples). 80/10/10 leaves ~101
> positives in the holdout, which gives noisy AUC estimates (95% CI ≈ ±0.04).
> I'd suggest 70/15/15 to halve that noise, at the cost of ~530 fewer
> training rows. Stick with 80/10/10 (spec), or use 70/15/15? [80/70]"

**Wrong (`[SPEC]` deviation, silent):**
> "Split: 70/15/15."

**Wrong (`[SPEC]` deviation, generic justification):**
> "I'll use 70/15/15 because gradient-boosted trees usually prefer larger
> validation sets."  *(Generic preference, not dataset-grounded.)*

### One rule above all

**"Improving" on the spec without telling the user is a defect, not a
courtesy.** A spec deviation applied silently — even one that produces a
better model — is wrong, because it removes the user's ability to know what
trade-offs the skill made on their behalf. Transparency is what makes this
skill distinct from a black-box AutoML system; silent deviations erode it.

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

- Ask the minimum required inputs before training (each item is `[ASK]`):
  - dataset location(s) (local path(s) or URL(s))
  - task type
  - evaluation metric (or accept default — `[DEFAULT]` if user doesn't specify)
  - split strategy (or accept default — `[SPEC]`, see §3)
- `[ASK]` — Ask for time column and any entity/group identifier to choose an
  appropriate split and CV strategy (see `references/defaults.md`).
- `[DEFAULT]` — Random seed: use 42 unless the user specifies otherwise.
- `[ASK]` — Task-specific requirements (see `references/defaults.md`).
- `[ASK]` — Domain-specific feature ideas; confirm whether to apply standard
  feature engineering (date parts, lags, transforms).
- `[ASK]` — Whether to run explainability (SHAP). Discrete question.
- `[ASK]` — **Time budget for hyperparameter search.** Phrase as a discrete
  question with the spec default presented:
  > "The default is to let Optuna run until convergence (25 non-improving
  > trials with <0.1% relative gain) with an 8-hour failsafe. Would you like
  > to set a shorter time budget? Press Enter to keep the default, or enter
  > minutes (e.g. 30)."
  Record the chosen value in `config.json` under `bounds.main_minutes` or
  leave the 8-hour failsafe. Do not invent a shorter timeout on your own.
- `[ASK]` — **Time budget for stacking ensemble.** Same pattern:
  > "The default is no time limit for stacking. Would you like to set one?
  > Press Enter to keep the default, or enter minutes."
  Record under `bounds.stacking_minutes` if set; otherwise leave unlimited.
- `[ASK]` — Whether to run an **AutoGluon comparison** alongside the main
  pipeline. **This MUST be a discrete question; do NOT batch it into a
  "defaults" list.** Phrase in plain language and default to **No**:
  > "Would you like me to also run AutoGluon (an AutoML system) for a
  > head-to-head comparison against the transparent pipeline? It adds
  > ~2 GB of install and 5–15 minutes of extra runtime. This is useful if
  > you want to know whether an off-the-shelf AutoML system would produce
  > a better model on this dataset. [y/N]"
  Wait for a Y/N answer before proceeding to step 2. Record the user's
  choice in `artefacts/config.json` under `comparison.autogluon` (bool).
- `[ASK]` — If multiple dataset locations are provided, ask how to combine
  them and whether to add a source column.
- Run a quick LLM suitability check (see `references/defaults.md`). If it
  triggers, **`[ASK]`** — recommend an LLM-based approach and ask whether to
  proceed with classical ML anyway.
- For `[DEFAULT]` items: confirm by stating them briefly in one summary
  ("I'll use random seed 42, metric f1_macro since you didn't specify").
  Do not include `[ASK]` items in this summary.

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

**Every decision in this section is `[ASK]`.** Each finding gets its own
plain-language question and waits for the user's response. The "default: yes"
or "default: cap" wording below is what to **recommend** to the user — not
what to apply unilaterally. Do NOT batch these decisions into a "defaults
I'll apply unless you object" block.

**Communication rule for all confirmations:** Always explain in plain,
jargon-free language — what was found, why it matters in one sentence, and a
clear recommendation. State the default. Never present raw statistical terms
without explaining them. Example format:
> "2 columns have almost no variety — over 95% of rows contain the same value.
> Columns like this can't help the model learn anything and slow down training.
> I recommend removing them. OK to drop them? [Y/n]"

**Shape and duplicates** `[ASK]`
- Report row/column count.
- Detect duplicate rows; if found: explain that identical rows can unfairly
  skew the model's learning, recommend dropping, **ask the user before doing
  so** (default recommendation: yes).

**Missing values** `[ASK]`
- Report missingness per column (count + %).
- Columns > 50% missing: explain that more than half the data is absent so
  filling it in would mean mostly guessing; **ask** whether to drop the column
  or fill it anyway (default recommendation: drop).
- Columns 20–50% missing: record for median/mode imputation inside the training
  pipeline — do NOT fill now, before splitting, as that would let test data
  influence training.
- Record final imputation strategy per column in `artefacts/config.json`.

**Near-zero variance** `[ASK]`
- Flag columns where > 95% of values are identical.
- Explain that a column with almost no variety cannot teach the model anything;
  recommend dropping; **ask the user before doing so** (default
  recommendation: yes).
- Record dropped columns in `artefacts/config.json`.

**Highly correlated features** `[ASK]`
- Flag numeric column pairs with |r| > 0.95.
- Explain in plain terms: "These two columns contain almost identical
  information. Keeping both is redundant and can confuse the model. I recommend
  keeping [X] and removing [Y] because X has a stronger relationship with what
  we're trying to predict. (this is called multicollinearity)"
- **Ask the user before dropping** (default recommendation: yes); record in
  `artefacts/config.json`.

**Numeric skew** `[ASK]`
- Flag columns with |skew| > 1.0.
- Explain: "This column has a few very large values that could pull the model
  in the wrong direction. I'll apply a standard adjustment to balance it out
  (log transformation)." **Ask the user before recording** which columns get
  log1p transforms in `artefacts/config.json`. Transforms are applied
  only inside the training pipeline after splitting, to avoid test data
  influencing training.
- Record approved transforms in `artefacts/config.json`.

**Outliers** `[ASK]`
- Flag columns with rows beyond 3 IQRs from the median; report count and %.
- Explain: "X rows in column Y have extreme values that are far outside the
  normal range. These can throw off the model. Options: (1) cap them at a
  sensible limit [recommended] (Winsorization), (2) remove those rows,
  (3) leave them as-is."
- **Ask the user for a choice** (default recommendation: cap); record decision
  in `artefacts/config.json`.
- Capping/removal is applied inside the training pipeline after splitting —
  not on the raw dataset now, to avoid test data influencing training.

**Target leakage signals** `[ASK]`
- Flag features with |correlation to target| > 0.9.
- Explain: "The column '[name]' is almost perfectly linked to what we're trying
  to predict. This usually means it was calculated using the answer, which would
  make the model look accurate in testing but fail in real use. I recommend
  removing it. (this is called target leakage)"
- **Ask the user before dropping** (default recommendation: yes); record in
  `artefacts/config.json`.

**Class imbalance (classification and supervised anomaly detection)** `[ASK]`
- Report class distribution in plain terms: e.g. "87% of rows are class A,
  13% are class B".
- If minority class < 20%: explain the imbalance will bias the model toward
  the majority class. The `class_weight='balanced'` mitigation is `[DEFAULT]`
  (apply automatically and note it in the summary).
- If minority class < 5%: additionally explain the imbalance is severe;
  **`[ASK]`**:
  "Would you like me to also artificially generate extra examples of the rare
  class to help the model learn it better? (this technique is called SMOTE)
  [Y/n]"

**Temporal integrity (time series only)** `[ASK]`
- Check for gaps or irregular frequency in the time column.
- Report in plain terms: e.g. "The data runs from Jan 2020 to Dec 2023 but
  there are 14 missing weeks."
- **Ask**: "Missing time periods can disrupt forecasting. Should I fill them
  in using the surrounding values, or leave the gaps? [fill in / leave gaps]"

**Profiling summary**
- After all `[ASK]` questions have been answered, print a concise
  plain-language summary of all findings and decisions.
- List what was decided (dropped, imputed, transformed, capped, etc.) and
  what the user chose where they overrode the default recommendation.
- Only proceed to baseline once all confirmations are resolved.

## 3) Baseline model (fixed by task — do not substitute)

The baseline is intentionally a **simple, deliberately under-powered model**.
Its purpose is to give a stable reference point so the rest of the run can
report meaningful gains. It is **not** "the first reasonable model I tried."

**The baseline algorithm is `[SPEC]`.** Use the algorithm specified for the
task type in `references/defaults.md`. A dataset-grounded deviation
(e.g. "this dataset has 30,000 categorical levels and LogReg would not fit
in memory") may be proposed to the user, but **never silently substituted**.
"RandomForest usually performs better than LogReg" is a generic preference,
not a dataset-grounded reason, and is not a valid basis for deviation.

The fixed defaults are:
- Classification: `LogisticRegression` (standardised, one-hot encoded)
- Regression: `Ridge` (standardised, one-hot encoded)
- Time series: seasonal naive or last-value
- Anomaly: `IsolationForest`

**The split ratio is `[SPEC]`.** Use 80/10/10 (stratified for classification,
chronological for time series). A dataset-grounded deviation may be proposed
to the user with the trade-off stated (e.g. extreme class imbalance, very
small dataset). Do not pick a different ratio silently.

Other rules:

- Split the dataset first (see `references/defaults.md`) into train, validation,
  and holdout test sets before any fitting:
  - Train: used to fit the baseline model
  - Validation: used to report baseline metrics
  - Holdout test: set aside now and not touched until final evaluation in step 5
- Use a simple, fixed-configuration pipeline — no hyperparameter search.
- All preprocessing transforms (imputation, log1p, encoding, scaling) must be
  fit on the training fold only and applied to validation and holdout.
- Report baseline metrics from the validation set and store them in
  `artefacts/metrics.json`, naming the baseline algorithm explicitly under
  `baseline.details.model`.
- Use the default metric if the user did not specify one (`[DEFAULT]`, see
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
- **Time budget** `[SPEC]`: pass `timeout=28800` (8 hours) to
  `study.optimize(...)` unless the user supplied a different value at intake
  (`config.json.bounds.main_minutes`). **Do NOT invent a shorter timeout on
  your own** — that is a defect. The convergence rule below is the intended
  primary stopping mechanism; the 8-hour timeout is a failsafe only.
- **Convergence rule** `[SPEC]`: stop early when 25 consecutive non-improving
  trials produce a relative gain of less than 0.1% over those 25 trials.
  Implement this as an Optuna callback. Do not substitute a shorter
  non-improving threshold or a higher gain threshold.
- Include non-sklearn models when appropriate (XGBoost, LightGBM, CatBoost).
  Install into venv if needed.
- **You MUST include at least one non-tree model family in the iteration
  pool** (see `references/defaults.md`). Tree-only pools make stacking
  ineffective because tree models produce correlated errors. Concretely:
  for classification, include `LogisticRegression(penalty='elasticnet')`
  *and* at least one of `KNeighborsClassifier` / `GaussianNB`; for
  regression, include `ElasticNet` and `KNeighborsRegressor`. Use small
  search spaces (1–3 hyperparameters each) so these don't dominate the
  Optuna budget — their purpose is stacking diversity, not winning solo.
- Expand feature engineering if it improves the metric and does not introduce
  leakage.
- Use the agreed metric to pick the best model.
- Log progress every 10 trials: trial number, best score so far, current score.
- Keep a clear audit trail in `artefacts/config.json`. Record the actual
  timeout used in `config.json.bounds.main_seconds_used` and whether it was
  the spec default or a user override.

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

**Time budget** `[SPEC]`

- Default: **no time limit** for the stacking step. It runs to completion.
- If the user supplied a stacking time budget at intake
  (`config.json.bounds.stacking_minutes`), respect that budget.
- **Do NOT invent a stacking timeout on your own.** If you believe the
  stacking step is risky on a particular dataset (e.g. very slow base
  learners), surface that to the user before starting it — do not silently
  cap it.

**Always record the result**

Whether adopted or rejected, populate `stacking` in `artefacts/metrics.json`
with: `attempted` (bool), `base_learners` (list of family names),
`meta_learner` (string), `best_single_score` (float), `ensemble_score` (float
or null if skipped), `adopted` (bool), `reason` (string explaining adopt /
reject / skip), `time_limit_seconds_used` (int or null if unlimited),
`time_limit_source` (one of "spec_unlimited", "user_override").

## 4.6) Ceiling check — diagnose whether more compute would help

After stacking is resolved, decide whether the dataset is **near its predictive
ceiling**. This is what tells the user "more trials won't help — you need
different data" instead of leaving them to guess.

**How to decide**

Compute the validation-score spread across distinct model families that
finished in Optuna (best single per family). Then apply:

- **Near ceiling** if **all** of the following hold:
  - The top-3 model families' validation scores are within 1.5% relative
    of each other.
  - Stacking was attempted and rejected (or skipped due to too few diverse
    families).
  - The Optuna best score is within 2% relative of the baseline.
- **Headroom remains** otherwise.

**Record in `metrics.json`**

Populate `ceiling_check` with:
- `near_ceiling` (bool)
- `family_score_spread_pct` (float, relative spread across top-3 families)
- `baseline_to_best_gain_pct` (float, relative gain from baseline to best
  single model)
- `note` (one-sentence explanation)

**Reflect in `results.md`**

When `near_ceiling` is true, the "Best model" section must include a
plain-language note like:
> "Across LightGBM, XGBoost, and RandomForest the validation AUC sits
> between 0.843 and 0.851 — a spread of less than 1%. Stacking also failed
> to improve the result. This dataset appears to be near its predictive
> ceiling for these features; running more trials or trying more models is
> unlikely to help. Better gains will come from new features or a different
> target definition (see 'What to try next' below)."

## 4.7) AutoGluon comparison (opt-in)

Run this step **only if the user opted in** during intake
(`comparison.autogluon == true`). The purpose is a fair head-to-head: how
does the transparent pipeline compare to an off-the-shelf AutoML system on
this exact dataset?

This is co-existence, not competition. AutoGluon produces its own model and
its own score, on the same data, with the same split. Both models are saved.
The user sees both. The skill does not pick a "winner" — it reports the gap
and lets the user decide.

**Skip without asking when**

- Time-series forecasting (AutoGluon's tabular API doesn't handle this
  directly; the comparison would be unfair).
- Unsupervised anomaly detection (no labels for AutoGluon to learn from).
- Dataset is very small (< 200 rows) — AutoGluon needs a meaningful amount
  of data to work well; report the skip in `results.md`.

In these cases record `autogluon.attempted = false` and `autogluon.reason`
in `metrics.json`. Do not silently omit the section.

**Install**

- Install `autogluon.tabular` into the venv only if not already present.
  This is large (~2 GB) — the user has already opted in at intake, but if
  install fails (disk space, network), record `autogluon.attempted = false`
  and `autogluon.reason = "install failed: <error>"` and continue with the
  main pipeline result. Do not halt the run.

**How to run it**

- Use the **same `train + validation` data** that the main pipeline uses for
  Optuna optimization. Use the **same untouched holdout set** for final
  evaluation. This guarantees a fair comparison.
- Construct `TabularPredictor(label=<target>, eval_metric=<metric>, path=...)`
  using the user's chosen metric.
- Call `predictor.fit(train_data, time_limit=<budget_seconds>, presets="medium_quality")`.
- Default budget: **5 minutes** (300 s) for datasets < 100k rows; 15 minutes
  (900 s) for larger datasets. Pass these as `time_limit`. The presets
  `"medium_quality"` is the right default — `"best_quality"` runs much longer
  and uses more compute than the comparison warrants.
- Score the resulting predictor on the holdout set with the same metric the
  main pipeline uses. Record both predictions and the score.

**Record in `metrics.json`**

Populate `autogluon` with: `attempted` (bool), `reason` (string if not
attempted), `time_limit_seconds` (int), `preset` (string),
`holdout_score` (float), `holdout_score_vs_main_pct` (float, relative
difference: positive means AutoGluon better).

**Reflect in `results.md`**

Add an "AutoML comparison" section under "Best model":
> "**AutoML comparison.** AutoGluon (medium_quality preset, 5 min budget)
> scored 0.881 AUC on the same holdout set, compared to 0.875 from our
> transparent pipeline — a gap of +0.7% in AutoGluon's favour.
>
> If you want the strongest possible model and don't need transparency,
> consider using AutoGluon directly. If you want a reproducible, inspectable
> pipeline with comparable performance, the transparent model is a good
> choice."

If AutoGluon scored within 1% of the main pipeline, say so plainly:
> "Within margin of error — the transparent pipeline is competitive with
> off-the-shelf AutoML on this dataset."

If AutoGluon was meaningfully better (>3%), say so equally plainly:
> "AutoGluon outperformed the transparent pipeline by 3.4% on this dataset.
> The most likely reason is its diverse model zoo (neural nets, fastai
> tabular) which our pipeline does not include."

**Save the AutoGluon model**

Save AutoGluon's predictor directory as `artefacts/autogluon_predictor/`
(it's a directory, not a single file). Document its inference command in
`results.md` alongside the main `infer.py` command. Do not replace
`model.joblib` — the main pipeline's model remains the primary artefact.

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
      trial count plus convergence reason is recorded. The iteration pool
      included **at least one non-tree model family** (LogReg-elasticnet,
      KNN, GaussianNB, or equivalent). The Optuna timeout used matches
      `config.json.bounds.main_seconds_used` (either the spec default of
      28800 or the user-supplied override — never a silently-shortened
      value).
- [ ] **Stacking** is recorded in `metrics.json` under `stacking`. For
      classification/regression, `attempted` is `true` unless an explicit
      skip condition applied (recorded in `stacking.reason`). For time-series
      and anomaly detection, `attempted` is `false` with the appropriate
      reason. The stacking timeout used matches
      `config.json.bounds.stacking_seconds_used` (null = unlimited, the spec
      default, or the user-supplied override — never a silently-invented
      timeout).
- [ ] **Ceiling check** ran and is recorded in `metrics.json` under
      `ceiling_check` (`near_ceiling`, `family_score_spread_pct`,
      `baseline_to_best_gain_pct`, `note`). If `near_ceiling` is `true`,
      the "Best model" section of `results.md` includes the plain-language
      explanation.
- [ ] **AutoGluon comparison**: if the user opted in at intake
      (`config.json.comparison.autogluon == true`), the comparison ran and
      `autogluon` is populated in `metrics.json` with `attempted` (bool),
      `holdout_score` (float or null if skipped),
      `holdout_score_vs_main_pct` (float or null), and `reason` (string
      explaining any skip). The `autogluon_predictor/` directory exists if
      a model was produced. If the user did not opt in, `autogluon` is
      either omitted or `attempted: false, reason: "user did not opt in"`.
- [ ] Final score comes from the holdout test set, not validation, and
      `final.eval_set == "holdout_test"` in `metrics.json`.
- [ ] `model.joblib`, `train.py`, `infer.py`, `metrics.json`, `config.json`,
      and `results.md` all exist (unless the run halted at the no-signal gate,
      in which case `metrics.json` and `config.json` still exist).
- [ ] Plain-language `results.md` written, including the signal-check verdict,
      stacking outcome, ceiling-check verdict, AutoML comparison (only if the
      user opted in), a "What to try next" section tailored to the run, and a
      reproducibility footer (seed, trials, package versions, timestamp).

If any item failed, fix it and re-check before reporting completion.

## References

- Defaults, task requirements, baseline and iteration guidance:
  `references/defaults.md`
- Artifact naming and JSON structure: `references/artifacts.md`
- Example prompts and expected clarifications:
  `references/examples.md`
