# Tabular data and ML skills

This repository contains two independent skills:

| Skill | Use it for | Persistent user-facing output |
|---|---|---|
| [`tabular-eda`](skills/tabular-eda/SKILL.md) | Explore, visualize, and explain a CSV or Parquet dataset without model building | Chat findings plus one self-contained `eda_report.html` |
| [`ml-model-builder`](skills/ml-model-builder/SKILL.md) | Build, evaluate, compare, improve, and package a tabular ML solution | One compact model-run directory with an inclusive report and tested inference |

Install either skill by itself or install both. They are siblings, not stages
of one workflow.

## EDA is independent

`tabular-eda`:

- reads the source dataset;
- explains important findings in chat;
- writes one portable HTML report with inline styling and charts;
- creates no README, Markdown summary, JSON profile, figure directory, split,
  model, or run artifacts.

Run EDA with an existing interpreter that already has pandas and NumPy:

```sh
<python-with-pandas-and-numpy> \
  skills/tabular-eda/scripts/analyze_tabular.py \
  --input /absolute/path/to/data.csv \
  --output /absolute/path/to/eda_report.html \
  --target optional_target
```

The script prints chat-ready Markdown findings and persists only the requested
HTML report. Parquet additionally requires `pyarrow` or `fastparquet`.

`ml-model-builder` never discovers, reads, copies, links, or reacts to that
report. It starts from the declared source data and performs only the narrow
preflight required to validate a modeling experiment: target, prediction
moment, row grain, cohort/labels, schema, groups/duplicates, leakage, feature
availability, support, and split feasibility.

For supported local files, run that modeling-only helper with an existing
Python interpreter that already has pandas and NumPy:

```sh
<python-with-pandas-and-numpy> \
  skills/ml-model-builder/scripts/inspect_model_data.py data.csv \
  --target quality \
  --task classification \
  --row-grain "one tested wine sample" \
  --prediction-moment "after laboratory measurements are available"
```

It prints a compact modeling preflight to stdout and creates no files. It does
not produce EDA. If the interpreter lacks a dependency, report that cleanly and
use another existing suitable interpreter. Do not auto-install dependencies
before experiment approval.

## Model-building approval

Before any model is fitted, AutoGluon build starts, SAP RPT request is sent, or
large optional dependency is installed, `ml-model-builder` presents one
experiment plan for explicit approval:

- target/label meaning, prediction moment, intended use, and prohibited uses;
- eligible/excluded features;
- evaluation rows, splits, primary metric, and uncertainty;
- per-track CPU, memory, parallel-job, and GPU controls;
- classical candidate families, minimum coverage, time, and Optuna trials;
- AutoGluon choice, preset, estimated runtime, run-to-completion or
  time-limited mode, optional time limit, and disk/resources;
- SAP RPT choice, context rows, context-plus-query request rows, query rows per
  call, columns, request/retry/timeout budget, access route, named destination,
  and transferred data scope;
- operational constraints used to recommend a winner.

“Train the best model” does not silently mean “run a few classical models and
ignore the optional systems.” The skill recommends a choice for each track and
waits for the user to confirm or change it. It uses the host's structured
question tool when available and never requires a typed approval sentence. If
SAP RPT is selected, this one approval includes its disclosed remote transfer;
there is no second RPT confirmation unless the destination or data scope later
expands materially.

The structured approval contains every foreseeable question before execution,
including planned dependency installs and backend readiness. Best-model
requests default AutoGluon to run-to-completion with no arbitrary cutoff. If
that may take many hours, the question shows the estimate and lets the user
approve completion or select a time limit. Once approved, the skill runs the
classical, AutoGluon, and RPT tracks through managed processes without routine
follow-up prompts.

## Three execution tracks

| Track | Execution |
|---|---|
| Classical ML | Build naive/fixed baselines, fit preprocessing inside folds, and run bounded task-aware family search with optional Optuna |
| AutoGluon | Supply eligible raw fold tables, target, metric, preset, folds, and time/resources; AutoGluon owns preprocessing, model building, tuning, and ensembling |
| SAP RPT | Package labelled training-fold context and query rows, then query the pretrained model; no training, fitting, Optuna, or model hyperparameters |

All approved tracks use the same target, prediction-time feature contract,
evaluation row IDs, folds, weights, and metric code. Their implementation
mechanics intentionally differ.

The classical approval names the candidate families and minimum coverage. Its
ledger uses unique candidate names and records why each family was considered,
then marks it completed, failed, or excluded. The skill does not install or
force every theoretically available library into every experiment.

After an AutoGluon build, the skill records its native leaderboard and
internal model failures, creates
`clone_for_deployment(model="best")`, verifies original/clone prediction
equivalence, and retains the smaller prediction-only clone. It tests inference
in a fresh bounded-thread subprocess so a warm training-process prediction
cannot hide cold-start failures. Single-job builds use
`fold_fitting_strategy="sequential_local"`.

### SAP RPT

SAP RPT is a production-capable tabular foundation model. The internal CLI is
a convenient internal managed access route: users do not need to provision a
BTP subaccount, SAP AI Core instance, service keys, or an RPT deployment for
that route. Paying customers can use the same RPT model through SAP AI Core and
manage their production route there.

The internal-use status of the CLI describes the access channel, not the
model's production capability. RPT is a first-class selectable backend and can
be the best predictive candidate, the operational recommendation, or the
default inference backend.

The skill records the model separately from the access route. It manages and
fingerprints labelled context, validates query/response alignment, records
request/latency limits, and leaves a tested new-row command:

```sh
python infer.py --backend sap-rpt \
  --input new_rows.csv --output predictions.csv
```

Credentials and interactive authentication remain user-managed and are never
stored in artifacts.

## One experiment, one inclusive report

Classical, AutoGluon, and SAP RPT belong to one experiment when they share the
same source fingerprint, target, eligible features, splits, evaluation rows,
weights, and metric implementation.

Adding an approved optional backend later updates that experiment with only:

- its required backend artifacts;
- its approval, budget, and result in `run.json`;
- refreshed inclusive `report.html` and `results.md`.

It does not create a duplicate full run or copy the existing model, folds, OOF
predictions, reports, plots, fixtures, or search outputs. Create a separate run
only when the data, target, feature contract, split, metric, hypothesis, or
released winner changes materially.

## Model-run output

```text
artefacts/runs/<run-id>/
├── run.json
├── report.html
├── results.md
├── train.py                 # only with classical/AutoGluon rebuilds
├── infer.py
├── requirements.lock
├── validation.json
└── backends/
    ├── classical/           # only when approved and attempted
    ├── autogluon/           # only when approved and attempted
    └── sap_rpt/             # only when approved and attempted
```

- `run.json` consolidates the problem/data contract, modeling preflight,
  evaluation, approval and budgets, structured amendments and remote-transfer
  permissions, backend evidence, selection, inference, and lineage.
- `report.html` is self-contained and includes preflight, baselines, all
  approved backend statuses/results, same-fold comparisons, uncertainty,
  errors, intended/prohibited uses, limitations, monitoring, predictive winner,
  operational recommendation, and commands. Classical baseline/leaderboard,
  AutoGluon preset, and RPT context/access/latency sections appear whenever
  those backends were approved and clearly state failed, unavailable, or
  unmeasured dimensions.
- `results.md` is the concise text handoff and carries the same required
  statuses/scores, metric, selection, use constraints, uncertainty, monitoring,
  backend-specific sections, and `infer.py` command.
- `train.py` is required only when classical or AutoGluon build reproducibility
  needs it. It verifies the external source fingerprint and writes to a new or
  explicitly empty run; it never overwrites the validated run. An RPT-only run
  must not contain it.
- `infer.py` defaults to the approved operational backend and supports every
  retained backend explicitly. SAP RPT inference chunks arbitrary input into
  approved query batches while preserving row IDs and input order.
- `run.json.inference` defines required/optional inputs, dtypes, missing/extra
  policies, target exclusion, identifiers/feature order, output prediction and
  probability columns, finite-value/bounds requirements, and one real dispatch
  command per retained backend.
- Backend directories contain only model/predictor/context material required
  for rebuilding, inference, or a material audit.
- The run is self-contained for its report and inference. Raw source data stays
  external by default, with its location and fingerprint recorded as an
  explicit rebuild prerequisite.
- `validation.json` starts pending. The validator runs structural checks and
  real temporary inference round trips, then atomically records a pass and
  timestamp only after executable success. Every retained backend covers
  representative, single-row, empty-input, and missing-required-column cases;
  repeated success cases must be deterministic and preserve row IDs. Test
  fixtures and outputs are removed after validation.

Never load an untrusted pickle/joblib file: deserialization can execute code.

Validate a model run:

```sh
python skills/ml-model-builder/scripts/validate_run.py \
  /absolute/path/to/project \
  --artifacts-dir artefacts/runs/<run-id> \
  --run-inference-test
```

## Methodological safeguards

The ML skill:

- defines the prediction moment and label-observation process before features;
- matches splits to time, groups, repeated entities, and source events;
- groups repeated exact eligible feature signatures when no natural entity ID
  exists;
- keeps final evidence outside selection;
- fits classical preprocessing, target encoding, resampling, calibration, and
  feature selection within valid fold boundaries;
- uses AutoGluon as an autonomous builder without external Optuna;
- uses SAP RPT as a pretrained context/query model without training artifacts;
- reports uncertainty, practical value, error/subgroup slices, and
  operational limitations;
- treats unlabeled anomaly detection as review prioritization;
- applies stronger governance to high-stakes decisions;
- never calls a search plateau the dataset's theoretical ceiling.

## Installation

Each skill is independently installable. Link only the skill or skills you
want.

| Host | User-level path | Project-level path |
|---|---|---|
| Codex | `~/.agents/skills/<skill-name>` | `.agents/skills/<skill-name>` |
| Claude Code | `~/.claude/skills/<skill-name>` | `.claude/skills/<skill-name>` |

Example user-level installation for both:

```sh
mkdir -p ~/.agents/skills ~/.claude/skills

ln -s /absolute/path/to/codex-ml-skill/skills/tabular-eda \
  ~/.agents/skills/tabular-eda
ln -s /absolute/path/to/codex-ml-skill/skills/ml-model-builder \
  ~/.agents/skills/ml-model-builder

ln -s /absolute/path/to/codex-ml-skill/skills/tabular-eda \
  ~/.claude/skills/tabular-eda
ln -s /absolute/path/to/codex-ml-skill/skills/ml-model-builder \
  ~/.claude/skills/ml-model-builder
```

Inspect an existing destination before replacing it. Start a new Codex task or
Claude Code session after first installation.

## Repository layout

```text
skills/
├── tabular-eda/
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   └── scripts/analyze_tabular.py
└── ml-model-builder/
    ├── SKILL.md
    ├── agents/openai.yaml
    ├── scripts/
    │   ├── inspect_model_data.py
    │   ├── render_report.py
    │   └── validate_run.py
    └── references/
        ├── governance.md
        ├── large-data.md
        ├── data-and-leakage.md
        ├── supervised-tabular.md
        ├── time-series.md
        ├── anomaly-detection.md
        ├── optimization-and-ensembling.md
        ├── evaluation-and-production.md
        ├── high-stakes.md
        ├── automl.md
        ├── sap-rpt.md
        ├── artifacts.md
        └── examples.md
tests/
```
