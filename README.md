# ml-model-builder

A shared Codex and Claude Code skill for understanding datasets and building
reliable classical machine-learning solutions. It supports analysis-only work
as well as reproducible model development, honest evaluation, and deployable
training/inference artifacts.

## What it can do

| Capability | What the skill does |
|---|---|
| Dataset analysis | Profiles structure, types, missingness, duplicates, cardinality, imbalance, identifiers, outliers, correlations, and temporal coverage |
| Visual EDA | Produces labeled distribution, frequency, missingness, correlation, target, feature–target, and time-coverage charts with plain-language findings |
| Problem framing | Defines the business decision, row grain, target, prediction moment, error costs, horizon, review capacity, and deployment constraints |
| Leakage prevention | Creates split assignments before target-aware EDA; audits post-outcome fields, target derivation, entity overlap, temporal look-ahead, and train/serve availability |
| Classification | Handles binary, multiclass, imbalanced, probability, ranking, calibration, and decision-threshold use cases |
| Regression | Supports robust losses, skewed targets, prediction intervals, asymmetric costs, and segment error analysis |
| Forecasting | Uses rolling-origin evaluation, naive/seasonal baselines, safe lag features, horizon-specific metrics, intervals, and panel/intermittent-series guidance |
| Anomaly detection | Separates supervised rare-event prediction from unlabeled anomaly ranking; evaluates review yield, stability, contamination, and domain feedback |
| Model improvement | Runs task-aware cross-validation and bounded Optuna searches across suitable model families; tries stacking only when evidence supports it |
| Honest evaluation | Keeps holdout targets sealed, reports uncertainty and subgroup/error slices, and distinguishes validation selection from final evaluation |
| Production handoff | Saves versioned data/schema/feature contracts, train/infer scripts, a fitted pipeline, pinned dependencies, model card, inference test, metrics, and reports |
| Optional comparisons | Runs AutoGluon or explainability only when requested and compares operational cost as well as predictive quality |

## Operating modes

| Mode | Outcome |
|---|---|
| Analysis only | A deterministic EDA report, charts, data profile, schema, fingerprint, prioritized findings, and recommended next actions—no placeholder model |
| Model building | EDA plus baselines, model selection, one-time holdout evaluation, production artifacts, and stakeholder-ready results |
| Model improvement | Audits an existing run, preserves the meaning of its historical holdout, and uses fresh validation evidence to improve it |

## Workflow

1. Frame the customer decision and prediction/scoring moment.
2. Establish the data, target, governance, and inference contracts.
3. Persist random, grouped, temporal, or grouped-temporal partitions as
   appropriate.
4. Analyze the permitted population; target-aware EDA uses training rows only.
5. Establish naive and fixed baselines plus task-appropriate sanity checks.
6. Improve suitable candidates within an explicit compute budget.
7. Select thresholds, calibration, intervals, horizons, or review capacity on
   validation evidence.
8. Evaluate the fixed candidate once on holdout with uncertainty and error
   analysis.
9. Test inference behavior and document operational limitations.
10. Validate and save the complete artifact set.

The core [SKILL.md](skills/ml-model-builder/SKILL.md) is deliberately a concise
router. Detailed methodology lives in focused references, so Codex and Claude
Code load only the guidance needed for the active task.

## EDA output

Run the bundled profiler directly when useful:

```sh
python skills/ml-model-builder/scripts/profile_dataset.py \
  --input data.csv \
  --output-dir artefacts \
  --mode analysis-only
```

For model-building EDA, create the split first and identify the persisted
training partition:

```sh
python skills/ml-model-builder/scripts/profile_dataset.py \
  --input prepared.csv \
  --output-dir artefacts \
  --mode model \
  --task classification \
  --target churned \
  --partition-column _ml_partition \
  --train-label train
```

The profiler intentionally:

- calculates statistics on all permitted rows and samples only expensive plots;
- uses observed category and class labels directly in charts;
- refuses model mode without a persisted partition;
- reports blockers, warnings, information, sampling, and interpretation limits;
- exits with status `2` when it detects a modeling blocker such as a
  single-class training target.

## Outputs

| Artifact | Purpose |
|---|---|
| `data_report.html`, `data_summary.md`, `figures/` | Human-readable EDA and charts |
| `data_profile.json`, `schema.json`, `data_fingerprint.json` | Versioned machine-readable data contract and provenance |
| `config.json`, `feature_manifest.json` | Problem, split, feature, and selection decisions |
| `train.py`, `infer.py`, `model.joblib` or `model/` | Reproducible training and trusted local inference |
| `metrics.json`, `model_card.md`, `results.md` | Evaluation, intended use, limitations, and stakeholder handoff |
| `requirements.lock`, `inference_test.json` | Pinned environment and tested inference contract |

Never load an untrusted `joblib`/pickle file: deserialization can execute code.

Validate a completed run without loading its model:

```sh
python skills/ml-model-builder/scripts/validate_run.py /path/to/project
```

## Installation for Codex and Claude Code

Both hosts should link to the same authoritative source:

| Host | User-level discovery path | Project-level discovery path |
|---|---|---|
| Codex | `~/.agents/skills/ml-model-builder` | `.agents/skills/ml-model-builder` |
| Claude Code | `~/.claude/skills/ml-model-builder` | `.claude/skills/ml-model-builder` |

For a shared user-level installation:

```sh
mkdir -p "$HOME/.agents/skills" "$HOME/.claude/skills"

ln -s /absolute/path/to/codex-ml-skill/skills/ml-model-builder \
  "$HOME/.agents/skills/ml-model-builder"

ln -s /absolute/path/to/codex-ml-skill/skills/ml-model-builder \
  "$HOME/.claude/skills/ml-model-builder"
```

If either destination already exists, inspect it first and remove only the
obsolete skill or link you intend to replace. Start a new Codex task or Claude
Code session after the first installation. Later source edits are immediately
available through the symlinks.

`skills/ml-model-builder/` is the development source of truth.
`dist/ml-model-builder.skill` is a portable snapshot, not Codex's live
installation. Rebuild it with:

```sh
python scripts/package_skill.py
```

## Repository layout

```text
skills/ml-model-builder/
├── SKILL.md
├── agents/openai.yaml
├── scripts/
│   ├── profile_dataset.py
│   └── validate_run.py
└── references/
    ├── governance.md
    ├── data-analysis.md
    ├── data-and-leakage.md
    ├── supervised-tabular.md
    ├── time-series.md
    ├── anomaly-detection.md
    ├── optimization-and-ensembling.md
    ├── evaluation-and-production.md
    ├── automl.md
    ├── artifacts.md
    └── examples.md
scripts/package_skill.py
tests/
dist/ml-model-builder.skill
```

## Design principles

- Prefer a defensible simple model over complexity with weak evaluation.
- Fit learned preprocessing and resampling inside training folds.
- Match splits, metrics, permutations, and uncertainty to the data-generating
  process.
- Treat automated EDA and explainability as evidence for investigation, not
  causal proof.
- Treat unlabeled anomaly detection as review prioritization, not known
  accuracy.
- Record assumptions, limitations, failures, resource bounds, and stop reasons.
- Never call a search plateau the dataset's theoretical predictive ceiling.
