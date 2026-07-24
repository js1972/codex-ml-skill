# ml-model-builder

A reusable Codex and Claude Code skill for building classical machine learning
models from local files or HTTP(S) datasets. It emphasises **transparent,
repeatable workflows**: decisions are explained in plain language, defaults are
documented, and material deviations require user approval.

The aim is a skill that produces a model you can deploy and a `results.md`
you can hand to a non-technical stakeholder.

## What the skill can do

| Area | Capability |
|---|---|
| ML tasks | Binary/multiclass classification, regression, time-series forecasting, and supervised/unsupervised anomaly detection |
| Data input | Local or HTTP(S) CSV/Parquet files; multiple schema-aligned datasets |
| Data quality | Finds duplicates, missingness, low-variance features, correlated features, skew, outliers, imbalance, and temporal gaps |
| Leakage control | Creates the holdout split before target-aware profiling; checks direct target leakage, post-event fields, entity overlap, and look-ahead features |
| Baseline and sanity checks | Uses a fixed baseline per task; runs a permutation test for supervised tabular tasks, a naive-forecast comparison for time series, or stability review for unlabeled anomalies |
| Model improvement | Runs seeded Optuna search with convergence stopping across tree and non-tree model families |
| Ensembling | Tests a diverse stacking ensemble and keeps it only when validation improves |
| Benchmarking | Optionally compares the transparent pipeline with AutoGluon on the same data boundaries |
| Explainability | Optionally produces a SHAP summary |
| Deliverables | Saves a fitted pipeline, reproducible training and inference scripts, metrics/config JSON, and a stakeholder-friendly `results.md` |
| Platforms | Uses the same skill source in Codex and Claude Code through platform-specific symlinks |

## Workflow at a glance

The skill runs an 11-step workflow with two mandatory gates and one opt-in
comparison:

1. Intake and clarify requirements (structured user questions)
2. Set up Python virtual environment in the project directory
3. Load and validate data, create train/validation/holdout assignments, then
   profile the training partition (with per-finding user confirmation)
4. Train the fixed baseline on the prepared split
5. **Signal check** — run the task-appropriate supervised or diagnostic gate;
   halt or ask before continuing when no signal is detected
6. Iterate with Optuna/TPESampler until convergence (tree + non-tree
   model families for stacking diversity)
7. **Stacking ensemble** — attempt and adopt only if it beats the single
   best model by ≥0.5% relative
8. **Ceiling check** — diagnose whether the dataset is near its predictive
   ceiling so the user knows whether more trials will help
9. **AutoGluon comparison** (opt-in only) — head-to-head against an
   off-the-shelf AutoML system
10. Evaluate the chosen model once on the holdout test set
11. Save artefacts and produce `results.md`

Each step prints a `▶ Step N/11 — <name>` header at start and a
`✓ Step N/11 done — <result>` summary at end. It also uses the host's task-list
tool when available (`update_plan` in Codex or `TodoWrite` in Claude Code).

## Methodology highlights

### Data profiling with user confirmation

After resolving exact duplicates and creating persistent split assignments,
the skill profiles the training partition and presents each finding in plain
language:

- Duplicate rows
- Missing-value patterns (>50% drop recommendation; retained columns imputed
  in-pipeline)
- Near-zero-variance columns
- Multicollinearity (|r| > 0.95)
- Skewed numeric distributions (|skew| > 1.0 → valid signed/non-negative
  transform)
- Outliers (>3 IQRs, Winsorise / remove / keep choice)
- Target leakage signals (using task-appropriate association checks)
- Post-event features unavailable at inference time (training-serving skew)
- Class imbalance (auto `class_weight='balanced'` <20%; SMOTE prompt <5%)
- Temporal gaps (time-series tasks)

Each finding is asked as a discrete question through the host's structured
question UI when available, or as a concise plain-text question otherwise. The
skill records the decision in `config.json`, and only user-approved transforms
are fitted on training folds.

### Fixed, documented baseline

The baseline is a deliberately simple algorithm fixed by task type
(`LogisticRegression` for classification, `Ridge` for regression, etc.).
The model is not allowed to "improve" the baseline by silently swapping
in a stronger algorithm, because that destroys cross-run comparability.
Any deviation must be proposed to the user with a dataset-grounded reason.

### Signal check (anti-noise gate)

After the baseline trains, the skill permutes the training target 20 times and
scores each shuffled-label baseline against unchanged validation labels. It
uses a one-sided empirical permutation p-value (α = 0.05), reports effect size,
and halts or asks before continuing when signal is not detected. Time-series
tasks instead compare a fixed autoregressive probe with a naive forecast;
unlabeled anomaly tasks report stability diagnostics and request domain review
rather than claiming a test that is impossible without labels.

### Convergence-driven Optuna search

- TPESampler with seeded reproducibility
- Trial budget effectively uncapped (500 trials, 8-hour failsafe)
- **Primary stopping rule**: 25 consecutive non-improving trials with
  <0.1% relative gain
- Single train/val split per trial for ≥5k rows; 5-fold CV per trial
  for <5k rows
- Pool **must include at least one non-tree model family**
  (`LogisticRegression(elasticnet)`, `KNeighbors`, `GaussianNB`) so the
  stacking step has diverse base learners
- All preprocessing fit inside each trial's training fold — no leakage
  to validation

### Stacking ensemble

Picks the top model per distinct family within 10% of the best validation
score (min 3 families, max 5 base learners). Out-of-fold predictions
stacked via `LogisticRegression`/`Ridge` meta-learner. Adopted only if
the ensemble beats the single best by ≥0.5% relative — otherwise rejected
and the rejection recorded.

### Search-plateau check

After stacking, uses a documented heuristic to flag when the explored model
families appear to have plateaued:

- Top-3 family validation scores within 1.5% relative of each other
- Stacking attempted and rejected (or skipped for too few families)
- Baseline→best gain within 2% relative

When `near_ceiling=true`, `results.md` recommends focusing on new data or
features rather than more trials. It does not claim the dataset's theoretical
predictive ceiling has been proven.

### AutoGluon comparison (opt-in)

If the user opts in at intake, the skill runs AutoGluon on the same
train/validation/holdout boundaries using the `medium_quality` preset and a
5-minute budget (15 minutes for >100k rows). It compares candidates on
validation, finalizes them, and then reports their one-time holdout scores. Gap
bands are descriptive (small / moderate / large), not confidence intervals.

**This is not just a score comparison — it's a deployment-cost comparison.**
See "Choosing which model to deploy" below for the trade-offs that go
beyond holdout AUC.

### Honest holdout

The holdout test set is set aside at the very start and not touched
until step 10. Final metrics are reported from holdout — not from the
validation set used during search.

### Reproducibility

- Default random seed: 42 (overridable)
- Seed applied where supported: Python `random`, NumPy,
  every sklearn model constructor, every split function, XGBoost,
  LightGBM, CatBoost, Optuna TPESampler
- A `Reproducibility` footer in `results.md` records seed, trial count,
  baseline algorithm, package versions, source commit, and timestamp
  so the artefact on disk is the source of truth
- Repeatability is scoped to a recorded software/hardware environment;
  bit-for-bit identity is not promised for threaded or GPU-backed libraries

## Artefacts produced

In the project's `artefacts/` directory:

| File | Description |
|---|---|
| `model.joblib` | Full preprocessor + model pipeline as a single joblib object |
| `train.py` | Reproducible training script |
| `infer.py` | Inference CLI (`python infer.py --input X.csv --output preds.csv`) |
| `metrics.json` | Baseline, signal check, Optuna details, stacking, ceiling check, AutoGluon (if run), final holdout score |
| `config.json` | Full run configuration: dataset, splits, profiling decisions, feature engineering, training bounds, package versions |
| `shap_summary.html` | (Optional) Beeswarm plot of top 20 features by mean \|SHAP\| value |
| `autogluon_predictor/` | (Opt-in only) Full AutoGluon predictor directory |

Plus `results.md` in the project root with a human-readable summary:
data profile, profiling decisions, signal check, best model, AutoML
comparison (if run), ceiling check, training process, "What to try next"
(state-dependent recommendations), inference command, and reproducibility
footer.

## Choosing which model to deploy

When the skill is run with the AutoGluon opt-in enabled, you end up with
two candidate models on disk:

- `model.joblib` — the **transparent pipeline** trained by the skill
- `autogluon_predictor/` — the **AutoGluon predictor**

The score can favour either candidate, and serving cost depends on the dataset,
selected models, runtime, and batch size. The skill therefore compares both
quality and measured deployment properties:

| Dimension | Transparent pipeline (`model.joblib`) | AutoGluon predictor |
|---|---|---|
| Packaging | One fitted pipeline plus its selected estimator dependencies | Predictor directory plus the dependencies required by its selected ensemble |
| Inspectability | Direct preprocessing and estimator graph | Leaderboard and model graph, but typically more components |
| Size, latency, memory | Measured for the produced artifact | Measured with the same host, samples, warm-up, and batch sizes |
| Serverless / edge fit | Depends on package limits, native libraries, and possibly model conversion | Usually requires more packaging work; verify against the target runtime |
| Repeatability | Seeded and version-recorded; some libraries remain nondeterministic | Seeded and version-recorded; ensembles and threaded libraries may vary |

The generated comparison reports artifact size and, where practical, median
and p95 inference latency, cold start, and peak memory. Unmeasured dimensions
are labelled as such; the skill does not invent universal performance
multipliers.

### When the transparent pipeline is the right choice

- Its validation quality is competitive and its measured serving profile fits
  the target
- The model needs to be **inspectable** by stakeholders, auditors, or
  another engineer
- A smaller, simpler dependency graph is operationally valuable
- The target has strict cold-start, memory, or native-library constraints

### When AutoGluon is the right choice

- It shows a repeatable validation gain that matters for the use case
- Its measured inference profile fits the production service-level objectives
- A larger ensemble and dependency set are acceptable operationally
- The team values automated model breadth more than a single-pipeline handoff

### When both can coexist

- Use the transparent pipeline in production and retain AutoGluon as a periodic
  benchmark.
- Use AutoGluon for batch scoring while keeping the transparent model for a
  latency-sensitive path.
- Re-test both when data drift, dependency versions, or serving infrastructure
  change.

Both candidates are compared on validation before the one-time holdout
evaluation. If the holdout result is then used to choose between them, it is a
benchmark-selection set; use new future or external data for an unbiased
estimate of the selected model.

## How decisions are made — spec discipline

Every decision the skill makes is tagged with one of three categories
so the model knows what it's allowed to do:

- **`[ASK]`** — pause and ask the user every time, using the host's structured
  question UI when available. No batching or "I'll apply these defaults unless
  you object" lists.
- **`[DEFAULT]`** — apply the documented value silently for inexpensive,
  easily-reversed choices (random seed, file naming, default metric).
- **`[SPEC]`** — use the documented value (timeouts, split ratios,
  baseline algorithm, convergence rules). A **dataset-grounded** deviation
  may be proposed to the user, but never applied silently and never
  justified by a generic preference like "X usually works better than Y."

"Improving" on the spec without telling the user is treated as a defect.
Transparency is the skill's distinctive value.

## Long-running steps use managed processes

The skill writes long-running Python entry points to disk and launches them
through the host's non-blocking process/session mechanism. Codex retains and
polls the command session; Claude Code uses background Bash and polls the task.
This lets Optuna, stacking, and AutoGluon honour user-approved budgets without
depending on a foreground tool-call timeout.

## Installation

Both hosts can point to the same source directory:

| Host | User-level skill path | Project-level skill path |
|---|---|---|
| Codex | `~/.agents/skills/ml-model-builder` | `.agents/skills/ml-model-builder` |
| Claude Code | `~/.claude/skills/ml-model-builder` | `.claude/skills/ml-model-builder` |

For a user-level installation, create both symlinks:

```sh
mkdir -p "$HOME/.agents/skills" "$HOME/.claude/skills"

ln -s /absolute/path/to/codex-ml-skill/skills/ml-model-builder \
      "$HOME/.agents/skills/ml-model-builder"

ln -s /absolute/path/to/codex-ml-skill/skills/ml-model-builder \
      "$HOME/.claude/skills/ml-model-builder"
```

Use only the link for the host you need, or both for a shared installation.
Start a new Codex task or Claude Code session after the first install. Source
edits are then available to subsequent sessions through the symlink.

The paths follow the current
[Codex skill discovery](https://learn.chatgpt.com/docs/build-skills#where-to-save-skills)
and
[Claude Code skills](https://code.claude.com/docs/en/slash-commands)
documentation.

For a repository-scoped installation, link from the corresponding project
directory instead:

```sh
mkdir -p .agents/skills .claude/skills

ln -s ../../skills/ml-model-builder \
      .agents/skills/ml-model-builder

ln -s ../../skills/ml-model-builder \
      .claude/skills/ml-model-builder
```

The packaged snapshot at `dist/ml-model-builder.skill` is useful for
distribution; the source under `skills/ml-model-builder/` remains
authoritative for development.

## Repository layout

```
skills/ml-model-builder/
├── SKILL.md                  # Workflow, spec discipline, all 11 steps
└── references/
    ├── defaults.md           # Defaults, models, splits, training bounds
    ├── artifacts.md          # Output file formats, JSON schemas
    └── examples.md           # Example prompts per task type
dist/
└── ml-model-builder.skill    # Packaged distribution snapshot
```

## Notes

- Local artefacts and virtualenvs stay untracked (see `.gitignore`).
- Codex and Claude Code share the same `SKILL.md` and reference files; only
  their discovery paths and process tools differ.
