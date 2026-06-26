# ml-model-builder

A Claude Code (and originally Codex) skill that builds classical machine
learning models from local files or HTTP(S) datasets. The skill emphasises
**transparent, reproducible workflows** — every decision is explained in
plain language, every default is documented, and every deviation from the
spec must be either user-approved or surfaced explicitly in the run summary.

The aim is a skill that produces a model you can deploy and a `results.md`
you can hand to a non-technical stakeholder.

## Supported tasks

- Binary and multiclass **classification**
- **Regression** (numeric prediction)
- **Time-series forecasting** (chronological splits, lag features, naive
  forecast floor)
- **Anomaly detection** (supervised and unsupervised)

## Workflow at a glance

The skill runs an 11-step workflow with two mandatory gates and one opt-in
comparison:

1. Intake and clarify requirements (structured user questions)
2. Set up Python virtual environment in the project directory
3. Load, validate, and profile the dataset (with per-finding user confirmation)
4. Split into train/validation/holdout and train baseline
5. **Signal check** — compare baseline against label-shuffled baselines;
   halt if no signal is detected
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
`✓ Step N/11 done — <result>` summary at end, plus the model maintains a
TodoWrite checklist throughout for live progress.

## Methodology highlights

### Data profiling with user confirmation

Before any training, the skill profiles the dataset and presents each
finding in plain language:

- Duplicate rows
- Missing-value patterns (>50% drop, 20-50% impute in-pipeline)
- Near-zero-variance columns
- Multicollinearity (|r| > 0.95)
- Skewed numeric distributions (|skew| > 1.0 → log1p)
- Outliers (>3 IQRs, Winsorise / remove / keep choice)
- Target leakage signals (|corr to target| > 0.9)
- Class imbalance (auto `class_weight='balanced'` <20%; SMOTE prompt <5%)
- Temporal gaps (time-series tasks)

Each finding is asked as a discrete structured question via Claude Code's
`AskUserQuestion` tool — the user chooses, the skill records the decision
in `config.json`, and only the user-approved transforms are applied
(inside the training fold, never on the full dataset).

### Fixed, documented baseline

The baseline is a deliberately simple algorithm fixed by task type
(`LogisticRegression` for classification, `Ridge` for regression, etc.).
The model is not allowed to "improve" the baseline by silently swapping
in a stronger algorithm, because that destroys cross-run comparability.
Any deviation must be proposed to the user with a dataset-grounded reason.

### Signal check (anti-noise gate)

After the baseline trains, the skill permutes the target column 5 times
and trains the same baseline on each shuffled dataset. If the real
baseline's score is within 2 standard deviations of the shuffled
distribution, the skill halts and reports in plain language that the
dataset has no detectable signal, suggesting alternatives (more features,
target reframing, LLM approach). Prevents the skill from confidently
producing a polished model on noise.

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

### Ceiling check

After stacking, diagnoses whether the dataset is **near its predictive
ceiling**:

- Top-3 family validation scores within 1.5% relative of each other
- Stacking attempted and rejected (or skipped for too few families)
- Baseline→best gain within 2% relative

When `near_ceiling=true`, `results.md` says so plainly so the user knows
that "run more trials" is not the right next step.

### AutoGluon comparison (opt-in)

If the user opts in at intake, the skill runs AutoGluon on the same
train/val/holdout split using `medium_quality` preset and a 5-minute
budget (15 minutes for >100k rows). Reports both scores plus the gap
under a plain-language verdict band (within 1% / 1-3% / >3%), so the
user can answer "would I do better with off-the-shelf AutoML on this
dataset?" without guessing.

**This is not just a score comparison — it's a deployment-cost comparison.**
See "Choosing which model to deploy" below for the trade-offs that go
beyond holdout AUC.

### Honest holdout

The holdout test set is set aside at the very start and not touched
until step 10. Final metrics are reported from holdout — not from the
validation set used during search.

### Reproducibility

- Default random seed: 42 (overridable)
- Seed applied to every stochastic component: Python `random`, NumPy,
  every sklearn model constructor, every split function, XGBoost,
  LightGBM, CatBoost, Optuna TPESampler
- A `Reproducibility` footer in `results.md` records seed, trial count,
  baseline algorithm, package versions, source commit, and timestamp
  so the artefact on disk is the source of truth

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

AutoGluon will often win on holdout score by 1–3% on tabular data — but
that is rarely the only thing that matters. The two artefacts have very
different costs at inference time:

| Dimension | Transparent pipeline (`model.joblib`) | AutoGluon predictor |
|---|---|---|
| Dependencies | scikit-learn + (lightgbm/xgboost/catboost) | AutoGluon + PyTorch + FastAI + ~30 deps |
| Disk footprint (inference) | ~150–300 MB | ~1.5–3 GB |
| Cold start | < 500 ms | 5–15 s |
| Per-row latency | 1–10 ms | 50–500 ms |
| Resident memory | ~100 MB | ~500 MB – 2 GB |
| Deployable to Lambda / edge / Workers | ✅ Yes | ❌ No (too large, PyTorch required) |
| Reproducibility at fixed seed | Identical runs | Mostly reproducible; ensemble can vary |
| Inspectable / debuggable | ✅ Single sklearn pipeline | ❌ Opaque ensemble runtime |

The numbers vary slightly by dataset, but the order-of-magnitude gaps
above are real and structural — **AutoGluon is roughly 10–50× heavier on
every inference-time dimension.** A 2% AUC lift can easily cost 30× more
memory, 40× more latency, and exclude entire deployment targets.

> ⚠ **If you're building a real-time API, AutoGluon is probably too slow.**
> A per-row prediction latency of 50–500 ms (AutoGluon) vs 1–10 ms
> (transparent pipeline) is the difference between an API that feels
> instantaneous and one that introduces a visible delay on every call.
> For any latency-sensitive serving path — user-facing predictions,
> high-volume scoring, fraud detection, real-time recommendations —
> the transparent pipeline is the safer default even when AutoGluon
> wins on score.

### When the transparent pipeline is the right choice

- Deployment target is **AWS Lambda, Cloud Functions, Cloudflare Workers**,
  or any environment with size/memory limits
- Inference latency matters (real-time APIs, user-facing predictions,
  high-throughput batch scoring)
- The model needs to be **inspectable** by stakeholders, auditors, or
  another engineer
- Exact reproducibility at fixed seed matters (research, regulated industries)
- The model will be served on a small/cheap instance class
- The AutoML comparison gap is **within 1%** ("competitive") — almost
  always pick the transparent pipeline here; the AutoGluon score lift is
  inside the noise floor and the deployment cost is huge

### When AutoGluon is the right choice

- Maximum holdout score is the only thing that matters and you have
  generous infrastructure for inference
- The model will be served from a long-running container with ≥4 GB RAM
  allocated and no cold-start sensitivity
- A 2–3% lift on the chosen metric is worth meaningful operational cost
  (i.e. the model is high-stakes and "good enough" isn't)
- The AutoML comparison gap is **>3%** ("meaningfully better") and the
  deployment can absorb the inference footprint

### When both can coexist

- Use the transparent pipeline as the **production model** (Lambda,
  edge, anywhere lean)
- Keep AutoGluon as the **occasional benchmark** — re-run it every quarter
  to confirm your pipeline still tracks the AutoML state of the art on
  your dataset. If the gap grows beyond your tolerance, that's a signal
  to invest in better features or expand the transparent pipeline's
  model pool.

### The honest summary

The skill's distinctive value is **a transparent, reproducible, lightweight
pipeline you can deploy anywhere**. AutoGluon's distinctive value is
**raw model quality**. They are not interchangeable, and the AutoML
comparison feature exists specifically so you can see exactly what
you're trading off before you ship. Don't pick blind.

## How decisions are made — spec discipline

Every decision the skill makes is tagged with one of three categories
so the model knows what it's allowed to do:

- **`[ASK]`** — pause and ask the user every time via the
  `AskUserQuestion` tool. No batching, no "I'll apply these defaults
  unless you object" lists.
- **`[DEFAULT]`** — apply the documented value silently for inexpensive,
  easily-reversed choices (random seed, file naming, default metric).
- **`[SPEC]`** — use the documented value (timeouts, split ratios,
  baseline algorithm, convergence rules). A **dataset-grounded** deviation
  may be proposed to the user, but never applied silently and never
  justified by a generic preference like "X usually works better than Y."

"Improving" on the spec without telling the user is treated as a defect.
Transparency is the skill's distinctive value.

## Long-running steps run in background mode

The Claude Code Bash tool imposes a 10-minute foreground ceiling.
Optuna, stacking, and AutoGluon are therefore invoked via
`Bash(run_in_background=true)` and polled with `TaskOutput`, with the
script written to disk first (e.g. `artefacts/_optuna_search.py`).
This is how the skill honours user-supplied time budgets longer than 10
minutes; foreground mode would silently kill them at the ceiling.

## Installation

The skill is consumed by Claude Code via a symlink from
`~/.claude/skills/ml-model-builder/` to this repo's
`skills/ml-model-builder/` directory:

```sh
ln -s /path/to/this/repo/skills/ml-model-builder \
      ~/.claude/skills/ml-model-builder
```

After symlinking, restart Claude Code so the skill is loaded. Updates
to this repo are picked up automatically by new sessions.

Alternative (legacy): a packaged `.skill` archive lives at
`dist/ml-model-builder.skill` but this is **not kept in sync** with the
current source. For active use, prefer the symlink.

## Repository layout

```
skills/ml-model-builder/
├── SKILL.md                  # Workflow, spec discipline, all 11 steps
└── references/
    ├── defaults.md           # Defaults, models, splits, training bounds
    ├── artifacts.md          # Output file formats, JSON schemas
    └── examples.md           # Example prompts per task type
dist/
└── ml-model-builder.skill    # Legacy packaged archive (likely stale)
```

## Notes

- Local artefacts and virtualenvs stay untracked (see `.gitignore`).
- The `.codex/skills/` directory is no longer used; the skill is now
  loaded from `~/.claude/skills/` via the symlink above.
