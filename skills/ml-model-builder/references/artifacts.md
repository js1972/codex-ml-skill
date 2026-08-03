# Minimal Model-Run Artifacts

## Contents

- [Principles](#principles)
- [Exact run layout](#exact-run-layout)
- [run.json](#runjson)
- [Optional analyses](#optional-analyses)
- [Backend artifacts](#backend-artifacts)
- [Entry points](#entry-points)
- [Reports](#reports)
- [Adding a backend to the same experiment](#adding-a-backend-to-the-same-experiment)
- [validation.json](#validationjson)

## Principles

Create the smallest artifact set that can explain the approved experiment,
reproduce its evaluation within the declared evidence limits, rebuild
build-based backends, infer with every retained backend, and validate the
handoff.

Use one consolidated `run.json`. Do not split the contract across config,
schema, feature, split, metric, model-card, and run-manifest files.

Do not include EDA files, `README.md`, external figures, cache directories,
retained inference outputs, copied parent artifacts, or redundant manifests.

## Exact run layout

The run root may contain only:

```text
artefacts/runs/<run-id>/
├── run.json
├── report.html
├── results.md
├── train.py                 # only with retained classical/AutoGluon
├── infer.py
├── requirements.lock
├── validation.json
└── backends/                # selected backends only
    ├── classical/
    ├── autogluon/
    └── sap_rpt/
```

Require `run.json`, `report.html`, `results.md`, `infer.py`,
`requirements.lock`, and `validation.json`. Require `train.py` when a retained
classical or AutoGluon backend needs rebuilding. Forbid `train.py` in an
SAP-RPT-only run.

Create a backend directory only for an approved track. Do not create
`extensions/`, `diagnostics/`, `figures/`, `inference_outputs/`, or
`__pycache__/` directories.

## run.json

Use exactly these top-level contracts:

- `run_id`;
- `created_at`;
- `problem`;
- `data`;
- `modeling_preflight`;
- `evaluation`;
- `approval`;
- `backends`;
- `selection`;
- `inference`;
- `lineage`.

Add `analyses` only when an optional analysis was approved. It contains
development-only evidence and never adds a directory or raw prediction files.

The following example shows the required shape for an experiment retaining all
three backends:

```json
{
  "run_id": "20260729T120000Z-wine-quality",
  "created_at": "2026-07-29T12:00:00+08:00",
  "problem": {
    "task": "classification",
    "target": "quality_class",
    "prediction_moment": "after laboratory measurements are available",
    "row_grain": "one row per wine sample",
    "intended_use": "decision support for comparing wine-quality classifiers",
    "prohibited_uses": [
      "not for autonomous safety or health decisions"
    ],
    "feature_contract": {
      "included": ["fixed_acidity", "alcohol"],
      "excluded": ["quality_class", "sample_id"]
    }
  },
  "data": {
    "source": "winequality-red.csv",
    "fingerprint": "sha256:...",
    "row_count": 1599
  },
  "modeling_preflight": {
    "status": "passed",
    "target_validated": true,
    "row_grain_validated": true,
    "prediction_moment_validated": true,
    "leakage_reviewed": true,
    "feature_availability_reviewed": true,
    "split_suitable": true,
    "findings": [
      "Stratification preserves every observed quality class."
    ]
  },
  "evaluation": {
    "design": "repeated_stratified_cross_validation",
    "split_fingerprint": "sha256:...",
    "evaluation_rows_fingerprint": "sha256:...",
    "primary_metric": {
      "name": "macro_f1",
      "direction": "maximize"
    }
  },
  "approval": {
    "approved_at": "2026-07-29T12:05:00+08:00",
    "scope": {
      "target": true,
      "feature_contract": true,
      "split_design": true,
      "primary_metric": true
    },
    "tracks": {
      "classical": {
        "selected": true,
        "status": "approved",
        "budget": {
          "cpu_count": 4,
          "parallel_jobs": 2,
          "memory_gb": 8,
          "gpu_enabled": false,
          "candidate_families": ["gradient_boosting"],
          "time_limit_seconds": 1200,
          "optuna_trials": 30,
          "minimum_family_coverage": 1
        }
      },
      "autogluon": {
        "selected": true,
        "status": "approved",
        "budget": {
          "cpu_count": 4,
          "parallel_jobs": 1,
          "memory_gb": 8,
          "gpu_enabled": false,
          "preset": "best_quality",
          "run_mode": "run_to_completion",
          "time_limit_seconds": null,
          "runtime_estimate": {
            "lower_seconds": 900,
            "upper_seconds": 7200,
            "basis": "1599 rows, 11 features, best_quality, CPU-only"
          },
          "disk_gb": 20
        }
      },
      "sap_rpt": {
        "selected": true,
        "status": "approved",
        "budget": {
          "cpu_count": 2,
          "parallel_jobs": 1,
          "memory_gb": 4,
          "gpu_enabled": false,
          "max_requests": 20,
          "max_context_rows": 5200,
          "max_request_rows": 5700,
          "max_query_batch_rows": 500,
          "max_columns": 16,
          "max_retries": 2,
          "timeout_seconds": 120
        },
        "plan": {
          "model_ids": ["rpt-standard", "rpt-large"],
          "full_context_fits": true,
          "use_full_context_when_supported": true,
          "context_size_candidates": [2048, 5197],
          "retrieval_strategies": ["full", "random", "vectorsearch"],
          "context_seed": 0,
          "input_format": "parquet",
          "retrieval_extra_status": "installed",
          "estimated_configurations": 3
        }
      }
    },
    "amendments": [],
    "remote_transfers": [
      {
        "id": "rpt-transfer-1",
        "approved_at": "2026-07-29T12:06:00+08:00",
        "backend": "sap_rpt",
        "destination": "SAP internal managed RPT endpoint",
        "purpose": "shared model evaluation and retained inference",
        "data_scope": {
          "features": ["fixed_acidity", "alcohol"],
          "labels": true,
          "query_rows": true,
          "identifiers": ["row_id"]
        }
      }
    ]
  },
  "backends": {
    "classical": {
      "status": "completed",
      "retained": true,
      "evaluation": {
        "split_fingerprint": "sha256:...",
        "evaluation_rows_fingerprint": "sha256:...",
        "primary_metric": "macro_f1",
        "score": 0.72
      },
      "evidence": {
        "result_source": "shared_evaluation_rows",
        "rows_scored": 320
      },
      "preprocessing": {
        "scope": "fold_local",
        "steps": ["median_imputation", "standardization"]
      },
      "search": {
        "method": "optuna",
        "trials_budget": 30,
        "trials_completed": 30
      },
      "candidates": [
        {
          "name": "xgboost",
          "family": "gradient_boosting",
          "consideration_basis": "nonlinear interactions within the approved CPU budget",
          "status": "completed",
          "score": 0.72
        }
      ],
      "artifacts": {
        "model": "backends/classical/model.joblib"
      }
    },
    "autogluon": {
      "status": "completed",
      "retained": true,
      "evaluation": {
        "split_fingerprint": "sha256:...",
        "evaluation_rows_fingerprint": "sha256:...",
        "primary_metric": "macro_f1",
        "score": 0.79
      },
      "evidence": {
        "result_source": "shared_evaluation_rows",
        "rows_scored": 320
      },
      "build": {
        "preset": "best_quality",
        "run_mode": "run_to_completion",
        "time_limit_seconds": null,
        "predictor_path": "backends/autogluon/predictor",
        "fold_fitting_strategy": "sequential_local",
        "fold_fitting_strategy_reason": "parallel_jobs=1 and bounded local execution",
        "training_diagnostics": {
          "fit_summary_captured": true,
          "elapsed_seconds": 1184.6,
          "completion_status": "completed_configuration",
          "stop_reason": "configured model roster completed"
        },
        "packaging": {
          "method": "clone_for_deployment",
          "model": "best",
          "diagnostics_captured_before_clone": true,
          "prediction_equivalence": {
            "validated": true,
            "rows": 16,
            "absolute_tolerance": 1e-12
          },
          "training_predictor_retained": false,
          "training_predictor_path": null,
          "retention_reason": null,
          "deployment_predictor_bytes": 155712325,
          "peak_packaging_disk_bytes": 1650236416
        }
      },
      "data_handling": {
        "raw_tabular": true,
        "external_preprocessing": false,
        "external_optuna": false
      },
      "runtime": {
        "cold_start_subprocess": true,
        "limits_set_before_imports": true,
        "native_thread_limits": {
          "OMP_NUM_THREADS": 1,
          "MKL_NUM_THREADS": 1,
          "OPENBLAS_NUM_THREADS": 1,
          "VECLIB_MAXIMUM_THREADS": 1
        }
      },
      "native_leaderboard": [
        {
          "model": "WeightedEnsemble_L2",
          "score_val": 0.79
        }
      ],
      "internal_failures": []
    },
    "sap_rpt": {
      "status": "completed",
      "retained": true,
      "evaluation": {
        "split_fingerprint": "sha256:...",
        "evaluation_rows_fingerprint": "sha256:...",
        "primary_metric": "macro_f1",
        "score": 0.83
      },
      "evidence": {
        "result_source": "shared_evaluation_rows",
        "rows_scored": 320
      },
      "model": {
        "name": "SAP RPT",
        "id": "rpt-large",
        "version": "recorded model version",
        "production_capable": true
      },
      "access": {
        "route": "internal_managed_cli",
        "client": "sap-rpt",
        "customer_production_route": "sap_ai_core"
      },
      "context": {
        "manifest": "backends/sap_rpt/context_manifest.json",
        "fingerprint": "sha256:...",
        "policy": "frozen labelled context reconstructed for inference",
        "selected_configuration_id": "rpt-large-full"
      },
      "configurations": [
        {
          "id": "rpt-standard-random-2048",
          "status": "completed",
          "model_id": "rpt-standard",
          "context_candidate_rows": 5197,
          "context_rows_planned": 2048,
          "context_rows_sent": 2048,
          "context_strategy": "random",
          "cli_strategy": "random::2048",
          "context_seed": 0,
          "input_format": "parquet",
          "fold_eligibility_policy": "fold-training rows only",
          "score": 0.80,
          "latency_ms": {"median": 3300, "p95": 4100},
          "throughput_queries_per_second": 118.4,
          "request_count": 10,
          "retrieval_extra_used": false,
          "selected": false,
          "failure_reason": null
        },
        {
          "id": "rpt-standard-vectorsearch-2048",
          "status": "completed",
          "model_id": "rpt-standard",
          "context_candidate_rows": 5197,
          "context_rows_planned": 2048,
          "context_rows_sent": 2048,
          "context_strategy": "vectorsearch",
          "cli_strategy": "vectorsearch::2048",
          "context_seed": 0,
          "input_format": "parquet",
          "fold_eligibility_policy": "fold-training retrieval corpus only",
          "score": 0.81,
          "latency_ms": {"median": 3400, "p95": 4200},
          "throughput_queries_per_second": 113.2,
          "request_count": 10,
          "retrieval_extra_used": true,
          "selected": false,
          "failure_reason": null
        },
        {
          "id": "rpt-large-full",
          "status": "completed",
          "model_id": "rpt-large",
          "context_candidate_rows": 5197,
          "context_rows_planned": 5197,
          "context_rows_sent": 5197,
          "context_strategy": "full",
          "cli_strategy": null,
          "context_seed": null,
          "input_format": "parquet",
          "fold_eligibility_policy": "fold-training rows only",
          "score": 0.83,
          "latency_ms": {"median": 3750, "p95": 4600},
          "throughput_queries_per_second": 124.7,
          "request_count": 10,
          "retrieval_extra_used": false,
          "selected": true,
          "failure_reason": null
        }
      ],
      "evaluation_coverage": {
        "summary": "evaluated under the approved configurations",
        "context_scale_tested": true,
        "retrieval_comparison_tested": true,
        "model_variants_tested": true,
        "full_context_tested": true,
        "coverage_gaps": []
      },
      "transfer_confirmation": {
        "approval_id": "rpt-transfer-1",
        "schema_validated": true,
        "labels_validated": true,
        "query_rows_excluded_from_context": true
      }
    }
  },
  "selection": {
    "predictive_winner": "sap_rpt",
    "operational_recommendation": "sap_rpt",
    "rationale": "Highest macro F1 on the shared evaluation rows and acceptable operations.",
    "primary_metric": "macro_f1"
  },
  "inference": {
    "entrypoint": "infer.py",
    "default_backend": "sap_rpt",
    "input": {
      "format": "csv",
      "required_columns": ["fixed_acidity", "alcohol"],
      "optional_columns": ["row_id"],
      "dtypes": {
        "fixed_acidity": "float64",
        "alcohol": "float64",
        "row_id": "string"
      },
      "missing_value_policy": {
        "required": "reject missing columns; allow values handled by the saved backend",
        "optional": "allow"
      },
      "extra_column_policy": "reject",
      "target_column": "quality_class",
      "identifier_columns": ["row_id"],
      "feature_order": ["fixed_acidity", "alcohol"]
    },
    "output": {
      "format": "csv",
      "prediction_column": "prediction",
      "probability_columns": ["probability"],
      "row_id_column": "row_id",
      "finite_values": true,
      "probability_bounds": [0, 1]
    },
    "backends": {
      "classical": "python infer.py --backend classical --input new.csv --output predictions.csv",
      "autogluon": "python infer.py --backend autogluon --input new.csv --output predictions.csv",
      "sap_rpt": "python infer.py --backend sap_rpt --input new.csv --output predictions.csv"
    }
  },
  "lineage": {
    "source_data_fingerprint": "sha256:...",
    "parent_run_id": null,
    "notes": []
  }
}
```

Use real SHA-256 digests, not the abbreviated placeholders shown above.

`approval.scope` must contain exactly `target`, `feature_contract`,
`split_design`, and `primary_metric`, each set to true only after the user
confirms the current values already stored in `problem` and `evaluation`. Do
not duplicate those values inside `approval`.

`approval.tracks` must contain exactly `classical`, `autogluon`, and `sap_rpt`.
For an approved track, use `selected: true`, `status: "approved"`, and a
non-empty track-appropriate budget. An approved SAP RPT track also requires a
non-empty `plan` fixed by the upfront approval. For a declined track, use
`selected: false`, `status: "declined"`, `budget: null`, and, for SAP RPT,
`plan: null`.

`approval.amendments` and `approval.remote_transfers` must always be lists.
Record each later plan change as a unique amendment with `approved_at`,
`reason`, and non-empty `path`/`before`/`after` changes. Record each permitted
RPT transfer with a unique ID, approval time, backend, destination, purpose,
and structured feature/label/query/identifier scope. Reference the transfer ID
from `backends.sap_rpt.transfer_confirmation.approval_id`.

Populate the RPT transfer record from the single consolidated experiment
approval. Do not require a standalone second confirmation before the first RPT
request. `transfer_confirmation` records execution checks against that approved
scope; it does not imply another user prompt.

When an ablation is approved, add `approval.analyses.ablations` with
`selected: true`, positive `max_variants` and `time_limit_seconds`, and one or
more approved `feature_groups`. Each group has a unique `id`, a non-empty list
of source `columns` from `problem.feature_contract.included`, and a non-empty
`hypothesis`. When ablations are declined, set `selected: false`, use null
budget values, and leave `feature_groups` empty. Do not add this optional
approval object when no ablation decision was needed.

Every approved budget records positive `cpu_count`, `parallel_jobs`, and
`memory_gb`, plus boolean `gpu_enabled`. Additionally require:

- classical: a non-empty unique `candidate_families` list,
  `time_limit_seconds`, `optuna_trials`, and `minimum_family_coverage`;
- AutoGluon: `preset`, `run_mode`, `runtime_estimate`,
  `time_limit_seconds`, and `disk_gb`. Use null time for
  `run_to_completion` and a positive time for `time_limited`;
- SAP RPT: `max_requests`, `max_context_rows`, `max_request_rows`,
  `max_query_batch_rows`, `max_columns`, `max_retries`, and
  `timeout_seconds`.

The approved RPT `plan` records non-empty accessible `model_ids`,
the discovered `full_context_fits` decision,
`use_full_context_when_supported: true`, positive unique
`context_size_candidates`, unique `retrieval_strategies` drawn from `full`,
`random`, and `vectorsearch`, a non-negative `context_seed`, column-oriented
`input_format`, retrieval-extra status, and a positive estimated configuration
count. Include `vectorsearch` only when the retrieval extra is installed or
its installation was approved. Use full context as the primary configuration
whenever it fits; context-size candidates are diagnostic comparisons, not a
fixed cap.

Make the classical ledger's family set equal
`approval.tracks.classical.budget.candidate_families`. Make classical
`search.trials_budget` equal the approved `optuna_trials`. Make AutoGluon
`build.preset`, `build.run_mode`, and `build.time_limit_seconds` match its
approved budget.

Record one `backends` entry for every approved track, even when it failed or
was unavailable. Create its backend directory when status is `completed` or
`failed`; omit the directory when status is `unavailable`. Retain only
completed usable backends. Do not create a backend entry or directory for a
declined track.

Every completed backend's evaluation must repeat the shared
`split_fingerprint`, `evaluation_rows_fingerprint`, and primary metric so the
validator can prove that the comparison uses the same evidence.

## Optional analyses

For an approved ablation, record `analyses.ablations` with exactly one entry
per approved feature group. Each entry records a unique `id`,
`approved_group_id`, completed `backend`, `status`,
`procedure: "full_pipeline_retrain"`,
`evidence_scope: "development_only"`, non-empty `conclusion`, and null
`failure_reason` when completed. A completed entry contains:

```json
{
  "development_evaluation": {
    "split_fingerprint": "sha256:...",
    "rows_fingerprint": "sha256:...",
    "primary_metric": "macro_f1",
    "reference_score": 0.72,
    "ablated_score": 0.69,
    "delta": -0.03,
    "uncertainty": "paired fold-level interval"
  }
}
```

The development fingerprints must differ from the sealed final evaluation
fingerprints. The primary metric must match the experiment metric, and `delta`
equals `ablated_score - reference_score`. Failed or skipped entries instead use
a non-empty `failure_reason` and no development score. The validator rejects
unapproved, post-fit masking, final-evidence, or partially recorded ablations.

## Backend artifacts

### Classical

Keep the complete fitted pipeline at a path under `backends/classical/`, for
example:

```text
backends/classical/model.joblib
```

Include preprocessing, calibration, and post-processing in that fitted object.
Record fold-local preprocessing, classical candidates, search method/budget,
result, evidence, and artifact path in `run.json`. Use `search.method: "none"`
with a reason when Optuna is unnecessary.

### AutoGluon

Keep the validated prediction-only deployment clone:

```text
backends/autogluon/predictor/
```

Before cloning, capture metrics, native leaderboard, internal failures, and
training diagnostics. Create the clone with
`clone_for_deployment(model="best")`, compare original and clone predictions,
and run cold-start root inference in a fresh subprocess. Point
`build.predictor_path` at `backends/autogluon/predictor`.

Record its preset, run mode, optional time limit, runtime estimate,
completion status, fold-fitting strategy/reason, result, native leaderboard,
dependency-preflight result and approved compatibility resolutions, structured
internal-failure list, runtime thread limits,
prediction-equivalence evidence, final predictor bytes, and peak packaging
disk bytes. Do not describe a `time_limit_reached` result as best without
qualifying that it is the best found within the approved limit.

Store the FastAI check under `build.dependency_preflight` with its status,
interpreter, detected AutoGluon/FastAI/Fastcore/Torch versions, and any approved
resolution. A completed build whose proposed roster included FastAI must record
the preflight as passed.

Require:

```json
{
  "raw_tabular": true,
  "external_preprocessing": false,
  "external_optuna": false
}
```

Remove the full training predictor after clone validation unless continued
AutoGluon analysis was explicitly requested. If retained, use
`backends/autogluon/training_predictor` and explain why. Do not copy a
classical transformed matrix, Optuna study, or classical model.

### SAP RPT

Keep the context manifest needed to reconstruct or locate the frozen labelled
inference context:

```text
backends/sap_rpt/context_manifest.json
```

The manifest may reference a securely managed context source; if the context
must travel with the run, keep it inside `backends/sap_rpt/` and reference it
from the manifest. Record context fingerprint/policy, RPT model identity,
access route, customer production route, transfer checks, result, and evidence
in `run.json`.

For every completed RPT backend, retain a non-empty
`backends.sap_rpt.configurations` ledger with one row per attempted
model/context/retrieval combination. Record a unique ID, status, exact model
ID, candidate/planned/sent context rows, `full`, `random`, or `vectorsearch`
strategy, exact CLI strategy, seed, column-oriented input format, fold/time
eligibility policy, comparable score, median/p95 latency, throughput, request
count, retrieval-extra use, selection flag, and failure reason. Exactly one
completed row is selected; its score matches the backend evaluation, its model
ID matches `backends.sap_rpt.model.id`, and
`context.selected_configuration_id` references it.

Also retain `backends.sap_rpt.evaluation_coverage` with the fixed summary
“evaluated under the approved configurations,” booleans for context-scale,
retrieval, model-variant, and full-context coverage, plus a `coverage_gaps`
list. Derive the booleans from completed ledger rows. Explain why retrieval was
not needed when full context fit; when truncation was required, name any
available but untested strategy, size range, or model variant.

Do not create an RPT model pickle, `train.py`, training/fit fields,
hyperparameters, Optuna/search/trial fields, or copied artifacts from another
backend. Never store credentials.

## Entry points

### train.py

Use root `train.py` as the single rebuild entry point for retained build-based
backends:

```text
python train.py --backend classical
python train.py --backend autogluon
python train.py --backend all
```

Keep classical and AutoGluon mechanics separate. AutoGluon receives raw
eligible tabular data and owns its build. Its rebuild path must also capture
diagnostics, create and verify the deployment clone, and leave
`backends/autogluon/predictor` inference-ready. It must pass
`time_limit=None` for `run_to_completion` or the positive approved limit for
`time_limited`. Do not expose an RPT training option. Omit `train.py` entirely
from an RPT-only run.

The run is self-contained for report reading and inference, not necessarily
for rebuilding from raw data. Do not duplicate the source dataset in the run
by default. Resolve the rebuild source through `run.json.data.source`, verify
its recorded fingerprint before building, and fail with an actionable message
when it is missing or changed. Require a new or explicitly empty output run;
never overwrite a validated run in place.

### infer.py

Use one inference interface:

```text
python infer.py --input new.csv --output predictions.csv
python infer.py --backend classical --input new.csv --output predictions.csv
python infer.py --backend autogluon --input new.csv --output predictions.csv
python infer.py --backend sap_rpt --input new.csv --output predictions.csv
```

Default to `inference.default_backend`. Support every retained backend and only
retained backends. Validate input columns, dtypes, missingness, target
exclusion, extra-column behavior, and output alignment.

In `inference.input` record:

- `format`;
- disjoint `required_columns` and `optional_columns`;
- a non-empty dtype for every declared input;
- `missing_value_policy`;
- `extra_column_policy`: `reject` or `ignore`;
- `target_column`, which must not be an input;
- `identifier_columns`;
- `feature_order`, containing every non-identifier input exactly once.

In `inference.output` record:

- `format`;
- one `prediction_column`;
- zero or more distinct `probability_columns`;
- `row_id_column`, or null when row IDs genuinely do not apply;
- `finite_values: true`;
- `probability_bounds: [0, 1]` when probability columns exist, otherwise null
  or an empty list.

Make `inference.default_backend` equal
`selection.operational_recommendation`. Make `inference.backends` contain
exactly one real root-`infer.py` command for every retained backend.

For SAP RPT, load or reconstruct the frozen labelled context from
`context_manifest.json`, package query rows, invoke the configured route,
validate responses, and emit the shared task-level output contract. Fail
actionably when access is unavailable.

## Reports

Make `report.html` self-contained with inline CSS, SVG, tables, and charts. Do
not use JavaScript or external assets. Include:

- problem, prediction moment, feature and evaluation contracts;
- modeling-preflight findings;
- approved and declined tracks with budgets, plus each approved backend's
  completed/failed/unavailable status, score when completed, and reason when it
  did not complete;
- approved plan amendments and remote-transfer scopes;
- a baseline and classical candidate leaderboard section whenever classical
  was approved, showing failures/unavailability when no score exists;
- the AutoGluon preset/time/resource settings, fold strategy, deployment-clone
  evidence, native leaderboard, internal failures, runtime safeguards, and
  result/status whenever AutoGluon was approved;
- SAP RPT context policy/coverage, model/access distinction, and one row per
  attempted configuration with model ID, context size, retrieval strategy,
  score, request count, latency, throughput, and failure; state that RPT was
  evaluated under the approved configurations and label unavailable,
  untested, or unmeasured dimensions explicitly;
- same-row/same-fold comparison;
- approved ablation hypotheses and completed development-only ablation results,
  including the score delta, uncertainty, and correlation limitation;
- uncertainty, calibration/threshold behavior, errors, subgroup analysis, and
  evidence limitations;
- predictive winner and operational recommendation;
- intended and prohibited uses, failure modes, monitoring/rebuild triggers, and
  exact inference commands.

Make `results.md` the concise text counterpart. It must include every approved
backend's status/score, shared metric, predictive winner, operational
recommendation, unified `infer.py` command, intended/prohibited uses,
limitations, uncertainty, and monitoring. When applicable, it must also name
the classical baseline/leaderboard, AutoGluon preset/deployment clone/internal
failure ledger, and SAP RPT model ID/context/retrieval/latency/configuration
coverage. Include approved
amendments/transfers and any completed ablation's feature group, development
score delta, uncertainty, and interpretation. Do not create a separate model
card; include governance in the report, results, and `run.json.problem` as
applicable.

## Adding a backend to the same experiment

When the user adds AutoGluon or SAP RPT later, keep it in the same run only if
the source fingerprint, target, feature contract, split fingerprint,
evaluation-row fingerprint, weights, and metric implementation are unchanged.

Obtain approval, append structured `approval.amendments` and any required
`approval.remote_transfers`, add only the backend entry/directory, and update
`inference.backends`, selection when justified, `lineage.notes`,
`validation.json`, `report.html`, and `results.md`. Do not create an extension
directory or copy any existing artifact.

If already-opened final evidence influences backend selection, record that
limitation in `lineage.notes` and require untouched future/external evidence
for a new unbiased winner claim.

## validation.json

Record:

```json
{
  "status": "pending",
  "validated_at": null,
  "inference_cases": [
    {
      "name": "sap-rpt-representative-new-rows",
      "backend": "sap_rpt",
      "kind": "representative",
      "argv": [
        "{python}",
        "infer.py",
        "--backend",
        "sap_rpt",
        "--input",
        "{input}",
        "--output",
        "{output}"
      ],
      "input": {
        "format": "csv",
        "columns": ["row_id", "fixed_acidity", "alcohol"],
        "rows": [
          {"row_id": "wine-1", "fixed_acidity": 7.1, "alcohol": 10.2},
          {"row_id": "wine-2", "fixed_acidity": 6.8, "alcohol": 11.1}
        ]
      },
      "expect": {
        "exit_code": 0,
        "output": {
          "format": "csv",
          "required_columns": ["row_id", "prediction", "probability"],
          "min_rows": 2,
          "max_rows": 2
        }
      },
      "repeat_runs": 2
    },
    {
      "name": "sap-rpt-single-row",
      "backend": "sap_rpt",
      "kind": "single_row",
      "argv": [
        "{python}",
        "infer.py",
        "--backend",
        "sap_rpt",
        "--input",
        "{input}",
        "--output",
        "{output}"
      ],
      "input": {
        "format": "csv",
        "columns": ["row_id", "fixed_acidity", "alcohol"],
        "rows": [
          {"row_id": "wine-1", "fixed_acidity": 7.1, "alcohol": 10.2}
        ]
      },
      "expect": {
        "exit_code": 0,
        "output": {
          "format": "csv",
          "required_columns": ["row_id", "prediction", "probability"],
          "min_rows": 1,
          "max_rows": 1
        }
      },
      "repeat_runs": 2
    },
    {
      "name": "sap-rpt-empty-input",
      "backend": "sap_rpt",
      "kind": "empty_input",
      "argv": [
        "{python}",
        "infer.py",
        "--backend",
        "sap_rpt",
        "--input",
        "{input}",
        "--output",
        "{output}"
      ],
      "input": {
        "format": "csv",
        "columns": ["row_id", "fixed_acidity", "alcohol"],
        "rows": []
      },
      "expect": {
        "exit_code": 0,
        "output": {
          "format": "csv",
          "required_columns": ["row_id", "prediction", "probability"],
          "min_rows": 0,
          "max_rows": 0
        }
      },
      "repeat_runs": 1
    },
    {
      "name": "sap-rpt-missing-required-column",
      "backend": "sap_rpt",
      "kind": "missing_required_column",
      "argv": [
        "{python}",
        "infer.py",
        "--backend",
        "sap_rpt",
        "--input",
        "{input}",
        "--output",
        "{output}"
      ],
      "input": {
        "format": "csv",
        "columns": ["row_id", "fixed_acidity"],
        "rows": [
          {"row_id": "wine-1", "fixed_acidity": 7.1}
        ]
      },
      "expect": {
        "exit_code": 2,
        "stderr_contains": "alcohol"
      },
      "repeat_runs": 1
    }
  ]
}
```

Create this file with `status: "pending"` and `validated_at: null`. Never
pre-write `passed`. With `--run-inference-test`, the validator resets the file
to pending, runs structural and executable checks, and atomically writes
`status: "passed"` plus a timezone-aware `validated_at` only after every check
succeeds. A failed validation remains pending.

Use inline inputs and `{input}`/`{output}` placeholders so the validator creates
temporary files. For every retained backend, use unique case names and cover
the four required `kind` values:

- `representative`: at least two rows and `repeat_runs >= 2`;
- `single_row`: exactly one row and `repeat_runs >= 2`;
- `empty_input`: zero rows with a schema-bearing input and output;
- `missing_required_column`: omit at least one required input and require a
  non-zero exit plus an actionable stderr fragment.

Each case must dispatch the declared backend with adjacent literal arguments
`--backend <backend>`, use the global input/output format, and keep successful
output row bounds equal to input rows. Successful expectations must include
the prediction, probability, and row-ID columns declared in `run.json`.

The executable check must preserve identifier order, emit finite prediction
values, enforce `[0, 1]` probability bounds, and produce byte-identical outputs
across repeated representative and single-row calls. Exercise optional-column
omission and the declared extra-column policy in additional temporary checks
when applicable. Execute every repeat as a fresh subprocess and apply bounded
native thread environment variables before interpreter startup. Remove all
temporary inputs and outputs after execution.

Run:

```text
python <ml-model-builder-skill>/scripts/validate_run.py <project-directory> \
  --artifacts-dir artefacts/runs/<run-id> --run-inference-test
```

The validator checks the exact root layout, approval/backend correspondence,
shared evaluation fingerprints and metric, backend-specific semantics,
self-contained report, pinned requirements, conditional `train.py`, and real
inference outputs. It cannot prove scientific correctness; reconcile source
code, folds, metrics, and reports before handoff.
