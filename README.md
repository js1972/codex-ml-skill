# ml-model-builder

A shared Codex and Claude Code skill for understanding datasets and building
reliable classical machine-learning solutions. It supports analysis-only work
as well as reproducible model development, honest evaluation, and deployable
training/inference artifacts.

## What it can do

| Capability | What the skill does |
|---|---|
| Dataset analysis | Profiles structure, types, missingness, duplicates, cardinality, imbalance, identifiers, outliers, correlations, and temporal coverage |
| Cohort and label audit | Records how rows were sampled, why labels are observed, whether unlabeled rows are truly negative, and when representative weights or evaluation data are required |
| Visual EDA | Produces labeled distribution, frequency, missingness, correlation, target, feature–target, and time-coverage charts with plain-language findings |
| Large data | Routes beyond-memory local files through disk-backed DuckDB and guides warehouse, lakehouse, cluster, cloud-VM, and out-of-core execution |
| Problem framing | Defines the business decision, row grain, target, prediction moment, error costs, horizon, review capacity, and deployment constraints |
| Leakage prevention | Creates split assignments before target-aware EDA; audits post-outcome fields, target derivation, entity overlap, temporal look-ahead, and train/serve availability |
| Classification | Handles binary, multiclass, imbalanced, probability, ranking, calibration, and decision-threshold use cases |
| Regression | Supports robust losses, skewed targets, prediction intervals, asymmetric costs, and segment error analysis |
| Forecasting | Uses rolling-origin evaluation, naive/seasonal baselines, safe lag features, horizon-specific metrics, intervals, and panel/intermittent-series guidance |
| Anomaly detection | Separates supervised rare-event prediction from unlabeled anomaly ranking; evaluates review yield, stability, contamination, and domain feedback |
| Model search | Chooses candidates from the problem before inspecting installed packages, considers XGBoost, LightGBM and CatBoost for supervised tabular work, and records every attempt or exclusion |
| Model improvement | Preserves immutable parent runs and evaluation-exposure history while testing further improvements on valid development evidence |
| Honest evaluation | Uses holdout, nested CV, external, or prospective validation as scientifically appropriate; reports uncertainty and subgroup/error slices |
| High-stakes safeguards | Requires domain ownership, oversight, harm-specific evidence, approval status, and prospective/external validation before deployment claims |
| Production handoff | Saves versioned data/schema/feature/split/run contracts, train/infer scripts, a fitted pipeline, pinned dependencies, model card, metrics, and reports |
| Executable verification | Runs declared inference cases and verifies real prediction rows, columns, order, finite values, and checksums or golden tolerances |
| Optional comparisons | Runs AutoGluon or explainability only when requested and compares operational cost as well as predictive quality |

## Operating modes

| Mode | Outcome |
|---|---|
| Analysis only | A deterministic EDA report, charts, data profile, schema, fingerprint, prioritized findings, and recommended next actions—no placeholder model |
| Model building | EDA plus baselines, model selection, predeclared honest evaluation, production artifacts, and stakeholder-ready results |
| Model improvement | Audits an existing run, preserves the meaning of its historical evaluation, and uses fresh development evidence to improve it |

Analysis-only EDA may inspect the target across the full dataset. If that
population is later reused for modeling, a new split does not make overlapping
rows unexposed. Treat them as discovery/development data and use fresh
external/prospective evidence for an unbiased final estimate; otherwise report
the result as previously exposed evidence.

## Workflow

1. Frame the customer decision and prediction/scoring moment.
2. Establish the population, sampling, label, target, governance, and inference
   contracts.
3. Persist and audit random, grouped, temporal, or grouped-temporal partitions
   in a split manifest.
4. Analyze the permitted population; target-aware EDA uses training rows only.
5. Establish naive and fixed baselines, plus an incumbent comparison only when
   an existing process is available and measurable.
6. Freeze suitable candidates before dependency inspection, install selected
   normal modeling dependencies, and search within an explicit compute budget.
7. Select thresholds, calibration, intervals, horizons, or review capacity on
   validation evidence.
8. Evaluate using the predeclared holdout, untouched outer folds, external set,
   or prospective cohort with uncertainty and error analysis.
9. Test real inference outputs and document operational limitations.
10. Save versioned run lineage and validate the complete artifact set.

The core [SKILL.md](skills/ml-model-builder/SKILL.md) is deliberately a concise
router. Detailed methodology lives in focused references, so Codex and Claude
Code load only the guidance needed for the active task.

## Model dependencies

EDA and modeling perform separate dependency preflights. Packages installed for
EDA never define the model search space. For supervised tabular work, the skill
freezes candidates from task, data and deployment criteria even when package
state is already known. XGBoost, LightGBM and CatBoost are normal modeling
dependencies unless a concrete incompatibility or resource constraint applies,
so missing selected libraries are installed into the project `.venv`. Every
family receives one candidate-ledger row with the exact status vocabulary
defined in the
[governance reference](skills/ml-model-builder/references/governance.md#budget-and-dependency-control);
exclusions are environment-independent and budget deferrals quantify the
approved budget and shortfall. AutoGluon remains an explicit heavyweight
opt-in.

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
  --output-dir artefacts/runs/example-run \
  --mode model \
  --task classification \
  --target churned \
  --partition-column _ml_partition \
  --train-label train \
  --split-strategy stratified_random
```

Run it from the project root so generated artifact pointers remain
project-relative.

The profiler intentionally:

- calculates statistics on all permitted rows and samples only expensive plots;
- uses observed category and class labels directly in charts;
- refuses model mode without a persisted partition;
- keeps global EDA target-blind for nested-CV designs;
- records partition support plus group, temporal, and duplicate-overlap audit
  status without inspecting sealed targets;
- requires the model split mechanics to be declared explicitly rather than
  inferring them merely because a date or group column exists;
- reports blockers, warnings, information, sampling, and interpretation limits;
- exits with status `2` when it detects a modeling blocker such as a
  single-class training target.

For a local file that is too large for safe in-memory pandas analysis, auto
mode routes to DuckDB when installed:

```sh
python -m pip install duckdb

python skills/ml-model-builder/scripts/profile_dataset.py \
  --input very-large.csv \
  --output-dir artefacts \
  --engine auto \
  --expected-source-bytes 50000000000 \
  --duckdb-memory-limit 4GB \
  --duckdb-temp-directory /path/to/fast-working-disk
```

DuckDB keeps memory bounded and spills to disk. If the data also exceeds local
disk or practical local scan time, run profiling and feature preparation in
the warehouse/lakehouse/cluster where the data already lives. See
[large-data.md](skills/ml-model-builder/references/large-data.md).

## Outputs

| Artifact | Purpose |
|---|---|
| `data_report.html`, `data_summary.md`, `figures/` | Human-readable EDA and charts |
| `data_profile.json`, `schema.json`, `data_fingerprint.json` | Versioned machine-readable data contract and provenance |
| `config.json`, `feature_manifest.json`, `split_manifest.json` | Problem, split, feature, and selection decisions |
| `run_manifest.json` | Immutable run identity, parent lineage, changes, and final-evaluation exposure |
| `train.py`, `infer.py`, `model.joblib` or `model/` | Reproducible training and trusted local inference |
| `metrics.json`, `model_card.md`, `results.md` | Evaluation, intended use, limitations, and stakeholder handoff |
| `requirements.lock`, `inference_test.json` | Pinned environment and tested inference contract |

Never load an untrusted `joblib`/pickle file: deserialization can execute code.

Validate the artifact contract and execute the declared inference round trip:

```sh
python skills/ml-model-builder/scripts/validate_run.py \
  /path/to/project \
  --artifacts-dir artefacts/runs/example-run \
  --run-inference-test
```

For new v2.1 runs this checks artifact consistency, split/run manifests,
candidate outcomes, metric declarations, high-stakes governance, dependency
locks, model hashes, and actual inference output. A command that exits
successfully without predictions does not pass. It cannot prove that training
code actually respected folds or that the scientific design was correct; the
Skill also reconciles training code, logs, fold assignments, metrics, and
reports.

New model and model-improvement runs use separate directories such as
`artefacts/runs/<run_id>/`; stable artifact names live inside each directory.
Analysis-only reports may continue to use `artefacts/`. Never overwrite a
parent run when trying an improvement.

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

`skills/ml-model-builder/` is the single source of truth used by both symlinks.

## Repository layout

```text
skills/ml-model-builder/
├── SKILL.md
├── agents/openai.yaml
├── scripts/
│   ├── profile_dataset.py
│   ├── profile_large_dataset.py
│   └── validate_run.py
└── references/
    ├── governance.md
    ├── data-analysis.md
    ├── large-data.md
    ├── data-and-leakage.md
    ├── supervised-tabular.md
    ├── time-series.md
    ├── anomaly-detection.md
    ├── optimization-and-ensembling.md
    ├── evaluation-and-production.md
    ├── high-stakes.md
    ├── automl.md
    ├── artifacts.md
    └── examples.md
tests/
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
- Preserve every improvement run and record when final evaluation evidence has
  been viewed.
- Never call a search plateau the dataset's theoretical predictive ceiling.
