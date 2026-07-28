# Artifact Contract and Schemas

## Contents

- [Versioning](#versioning)
- [Analysis-only outputs](#analysis-only-outputs)
- [Model outputs](#model-outputs)
- [Run and split manifests](#run-and-split-manifests)
- [config.json](#configjson)
- [metrics.json](#metricsjson)
- [Task-specific evaluation contracts](#task-specific-evaluation-contracts)
- [Data and inference contracts](#data-and-inference-contracts)
- [Human-readable reports](#human-readable-reports)
- [Validation](#validation)
- [Legacy compatibility](#legacy-compatibility)

## Versioning

Use `schema_version: "2.1"` in `config.json`, `metrics.json`,
`data_profile.json`, `schema.json`, `feature_manifest.json`, and
`data_fingerprint.json`, `split_manifest.json` and `run_manifest.json`.

Keep artifact filenames stable. Add fields compatibly; do not silently change
the meaning of an existing field. Record deprecations in
`config.json.compatibility`. Read v2.0 and legacy v1 artifacts, but write new
runs as v2.1.

## Analysis-only outputs

Required:

- `artefacts/config.json`
- `artefacts/data_profile.json`
- `artefacts/data_report.html`
- `artefacts/data_summary.md`
- `artefacts/data_fingerprint.json`
- `artefacts/schema.json`
- `artefacts/figures/` with selected PNG charts

Do not create placeholder model files.

## Model outputs

For every new v2.1 model run, choose an immutable project-relative run
directory such as `artefacts/runs/<run_id>/`. Pass that directory to profilers
with `--output-dir` and to the validator with `--artifacts-dir`. Keep stable
filenames inside it. The lists below use `<run-dir>` for that selected
directory; place the required analysis outputs there as well.

Required in addition to analysis outputs:

- `<run-dir>/train.py`
- `<run-dir>/infer.py`
- `<run-dir>/model.joblib` or a versioned `<run-dir>/model/` directory when the
  deployable solution contains multiple fitted components
- `<run-dir>/metrics.json`
- `<run-dir>/feature_manifest.json`
- `<run-dir>/split_manifest.json`
- `<run-dir>/run_manifest.json`
- `<run-dir>/model_card.md`
- `<run-dir>/requirements.lock` or an equivalent fully pinned inference
  environment
- `<run-dir>/inference_test.json`
- `<run-dir>/results.md`

Optional:

- `<run-dir>/shap_summary.html`
- `<run-dir>/autogluon_predictor/`
- `<run-dir>/search.db` for resumable Optuna storage
- `predictions.csv`, `review_queue.csv`, or another requested operational
  scoring output with passthrough identifiers, as-of timestamp, model/data
  version, scores/ranks and eligibility/exclusion reasons

Never load an untrusted `model.joblib` or execute untrusted `train.py`,
`infer.py` or inference-test commands.

## Run and split manifests

Write every improvement into a new artifact directory. Do not overwrite its
parent. Use `run_manifest.json`:

```json
{
  "schema_version": "2.1",
  "run_id": "20260728T143000Z-lightgbm-v2",
  "run_kind": "improvement",
  "artifact_directory": "artefacts/runs/20260728T143000Z-lightgbm-v2",
  "parent_run_id": "20260727T110000Z-baseline",
  "parent_artifact_hashes": {
    "config.json": "sha256:...",
    "metrics.json": "sha256:...",
    "run_manifest.json": "sha256:..."
  },
  "code_revision": "git:abcdef123456",
  "data_fingerprint": "sha256:...",
  "split_fingerprint": "sha256:...",
  "created_at": "2026-07-28T14:30:00Z",
  "changes": ["added native categorical candidates"],
  "roster_frozen_at": "2026-07-28T15:00:00Z",
  "prior_evidence": [
    {
      "source_run_id": "20260727T110000Z-baseline",
      "run_manifest_sha256": "sha256:...",
      "metrics_sha256": "sha256:...",
      "status": "benchmark_selection",
      "final_set": "holdout_test",
      "population_fingerprint": "sha256:...",
      "opened_at": "2026-07-27T13:00:00Z",
      "opened_for": "parent final report",
      "values_viewed": ["primary metric"],
      "decisions_influenced": ["v2 candidate hypothesis"]
    }
  ],
  "evaluation_exposure": {
    "status": "sealed",
    "final_set": "holdout_test",
    "population_fingerprint": "sha256:...",
    "opened_at": null,
    "opened_for": null,
    "values_viewed": [],
    "decisions_influenced": []
  }
}
```

For an initial run, use `run_kind: "initial"`, a null `parent_run_id`, an empty
parent-hash object and `prior_evidence: []`. Every improvement carries forward
append-only parent/ancestor evidence references and must include its direct
parent. Bind each reference to the source `run_manifest.json` and
`metrics.json` SHA-256 values; bind the direct parent config, metrics and run
manifest in `parent_artifact_hashes`. Do not rewrite the parent when its
evidence later influences a child. If final
evidence has influenced a decision, use `benchmark_selection` and stop
describing that set as an unbiased test for later improvements.

Use exposure states consistently:

- `sealed` before any final value is viewed;
- `opened` after the predeclared final evaluation is reported;
- `benchmark_selection` if those values influence another choice;
- `pending_labels` while a prospective cohort has not matured.

Always identify the final set and its population fingerprint. For `sealed` and
`pending_labels`, keep `opened_at`/`opened_for` null and both audit arrays empty.
For `opened`, record the timestamp, purpose and non-empty `values_viewed`, while
`decisions_influenced` remains empty. Use `benchmark_selection` once a
subsequent decision is influenced and list that decision explicitly.

Use `split_manifest.json` to make the evaluation population reproducible:

```json
{
  "schema_version": "2.1",
  "strategy": "grouped_temporal",
  "assignment": {
    "column": "_ml_partition",
    "source": "persisted_column",
    "fingerprint": {
      "algorithm": "sha256",
      "value": "...",
      "basis": "invoice_id plus partition label"
    }
  },
  "partitions": [
    {"name": "train", "role": "development", "rows": 80000},
    {"name": "holdout", "role": "final_evaluation", "rows": 20000}
  ],
  "audits": {
    "group_overlap": {
      "checked": true,
      "groups_spanning_partitions": 0,
      "null_group_rows": 0,
      "allowed": false
    },
    "temporal_order": {
      "checked": true,
      "valid": true,
      "invalid_timestamp_rows": 0,
      "purge_gap": "90 days",
      "ranges": [
        {
          "name": "train",
          "rows": 80000,
          "start": "2023-01-01T00:00:00+00:00",
          "end": "2024-09-30T23:59:59+00:00"
        },
        {
          "name": "holdout",
          "rows": 20000,
          "start": "2025-01-01T00:00:00+00:00",
          "end": "2025-06-30T23:59:59+00:00"
        }
      ]
    },
    "duplicate_overlap": {
      "checked": true,
      "rows_crossing_partitions": 0
    }
  }
}
```

For a non-applicable audit, set `checked: false` and record why. For nested CV,
set `config.split.development_label` to null: each outer-training population is
the complement of its outer-evaluation fold, not a separate disjoint
partition. Partition every non-discovery row into an `outer_evaluation` fold
and record matching fold IDs/support; a seed alone is not a split manifest.
This generic complement contract is only valid for non-temporal nested CV.
For future prediction, use rolling-origin development folds inside a holdout,
external or prospective final design; do not let future outer folds enter a
training complement.

If nested-CV modeling uses a separate target-aware discovery cohort, give its
partition `role: "discovery_excluded"`, record its fingerprint and set
`analysis.target_aware_partition` to that partition with
`analysis.discovery_excluded_from_outer: true`. No discovery row may appear in
an outer fold. Without such a cohort, set the global target-aware partition to
null.

## config.json

Use `mode: "analysis-only"`, `"model-building"` or `"model-improvement"`.
Use this model-building shape:

```json
{
  "schema_version": "2.1",
  "mode": "model-building",
  "problem": {
    "task": "classification",
    "business_decision": "prioritize invoices for manual review",
    "target": "late_payment",
    "target_derivation": null,
    "prediction_moment": "invoice issue time",
    "row_grain": "one row per invoice",
    "cohort": {
      "source_population": "all issued invoices eligible for review",
      "inclusion_rule": "issued in the scoring region and period",
      "label_observation": "payment outcome observed after 90-day maturity",
      "sampling_design": "complete cohort",
      "inclusion_probability": 1.0,
      "sample_weight": null,
      "evaluation_representative": true,
      "calibration_representative": true,
      "selective_labels": false,
      "label_acquisition": null
    },
    "group_column": "customer_id",
    "time_column": "invoice_date",
    "error_costs": {
      "false_positive": "review time",
      "false_negative": "missed late payment"
    }
  },
  "data": {
    "locations": ["data.csv"],
    "fingerprint_file": "artefacts/runs/20260728T143000Z-lightgbm-initial/data_fingerprint.json",
    "schema_file": "artefacts/runs/20260728T143000Z-lightgbm-initial/schema.json",
    "split_manifest": "artefacts/runs/20260728T143000Z-lightgbm-initial/split_manifest.json"
  },
  "split": {
    "strategy": "grouped_temporal",
    "group_overlap_policy": "disallow",
    "assignment_column": "_ml_partition",
    "development_label": "train",
    "holdout_target_sealed": true,
    "seed": 42
  },
  "evaluation": {
    "design": "holdout",
    "status": "complete",
    "final_eval_set": "holdout_test",
    "independent_test": true,
    "selection_nested": false
  },
  "analysis": {
    "report": "artefacts/runs/20260728T143000Z-lightgbm-initial/data_report.html",
    "target_aware_partition": "train",
    "pre_partition_target_exposure": {
      "status": "none",
      "source": null,
      "final_population_overlap": false,
      "values_viewed": [],
      "decisions_influenced": []
    },
    "plot_sample_size": 10000,
    "plot_sample_seed": 42,
    "reported_columns": ["customer_id", "email"]
  },
  "feature_contract": {
    "manifest": "artefacts/runs/20260728T143000Z-lightgbm-initial/feature_manifest.json",
    "inference_unavailable": ["payment_date", "final_status"],
    "sensitive_attributes": ["region"],
    "target_sources_excluded": []
  },
  "selection": {
    "primary_metric": "average_precision",
    "secondary_metrics": ["log_loss", "recall_at_review_capacity"],
    "validation": "grouped_temporal_cv",
    "threshold_rule": "top 200 invoices per week",
    "calibration": {
      "method": "sigmoid",
      "protocol": "grouped-temporal out-of-fold",
      "final_fit": "calibrated-CV ensemble retained for inference"
    },
    "capacity": {
      "enabled": true,
      "unit": "week",
      "limit": 200,
      "timezone": "Australia/Perth",
      "cutoff": "Monday 09:00",
      "eligibility_rule": "open invoices only",
      "tie_breaker": "invoice_id ascending",
      "sub_capacity_behavior": "return every eligible invoice"
    }
  },
  "baselines": {
    "incumbent": {
      "available": false,
      "reason": "no existing scored process"
    }
  },
  "search": {
    "sampler": "TPESampler",
    "budget_seconds": 1800,
    "stop_reason": "budget",
    "execution_mode": "managed_process",
    "task_or_session_id": "host-process-123",
    "roster_frozen_at": "2026-07-28T15:00:00Z",
    "candidates": [
      {
        "family": "xgboost",
        "consideration_basis": "nonlinear tabular interactions are plausible",
        "suitability_status": "eligible",
        "dependency_status": "installed_for_run",
        "execution_status": "attempted",
        "reason": null
      },
      {
        "family": "lightgbm",
        "consideration_basis": "efficient histogram boosting fits the data size",
        "suitability_status": "eligible",
        "dependency_status": "installed",
        "execution_status": "attempted",
        "reason": null
      },
      {
        "family": "catboost",
        "consideration_basis": "native categoricals considered for high cardinality",
        "suitability_status": "excluded",
        "dependency_status": "not_required",
        "execution_status": "excluded",
        "reason": "deployment image does not support its native runtime"
      }
    ]
  },
  "run": {
    "manifest": "artefacts/runs/20260728T143000Z-lightgbm-initial/run_manifest.json",
    "run_id": "20260728T143000Z-lightgbm-initial",
    "run_kind": "initial"
  },
  "comparison": {
    "autogluon": false
  },
  "governance": {
    "risk_tier": "standard",
    "risk_assessed": true,
    "risk_assessment_rationale": "does not affect an individual's essential rights",
    "unresolved_hazards": [],
    "domain_owner": "accounts-receivable operations",
    "deployment_decision": "decision_support",
    "approval_status": "not_required"
  },
  "environment": {
    "python": "3.x.y",
    "platform": "recorded at runtime",
    "requirements": "artefacts/runs/20260728T143000Z-lightgbm-initial/requirements.lock"
  },
  "compatibility": {
    "reads_legacy_v1": true,
    "reads_schema_v2_0": true,
    "deprecated_fields": ["ceiling_check"]
  }
}
```

In `problem.cohort`, use `inclusion_probability: 1.0` for a complete cohort.
For a sampled design, use either a numeric probability in `(0, 1]` or an
object containing exactly one of `column` or `formula`, plus optional `scope`.
Use `sample_weight: null` when no weight applies; otherwise use an object
containing exactly one of `column` or `formula`, plus `scope` (`training`,
`evaluation`, `calibration` or a declared combination). State separately
whether evaluation and calibration populations represent deployment.

Set `selective_labels: false` and `label_acquisition: null` only when label
observation is not controlled by review, treatment, response or another
selection policy. Otherwise set `selective_labels: true` and record
`label_acquisition.development` (mechanism, positivity and optional
selection-probability column/formula) plus `label_acquisition.evaluation`
(identified design, positivity, selection unit, claim scope, probability
support and whether population performance/calibration are supported). For
fixed-capacity off-policy evaluation, row propensities alone are insufficient;
the probability scope must be the queue or slate.

## metrics.json

Keep every metric name, direction, dataset and uncertainty explicit:

```json
{
  "schema_version": "2.1",
  "task": "classification",
  "primary_metric": {
    "name": "average_precision",
    "direction": "maximize"
  },
  "baselines": {
    "naive": {
      "validation_mean": 0.12
    },
    "fixed": {
      "model": "LogisticRegression",
      "validation_mean": 0.41,
      "validation_std": 0.03
    }
  },
  "signal_diagnostics": {
    "kind": "group_preserving_permutation",
    "permutations": 99,
    "empirical_p_value": 0.01,
    "effect_size": 0.29,
    "verdict": "learnable signal detected"
  },
  "search": {
    "completed_trials": 84,
    "failed_trials": 2,
    "best_family": "LightGBM",
    "validation_mean": 0.53,
    "validation_std": 0.02,
    "stop_reason": "budget",
    "plateau_detected": false,
    "family_results": [
      {
        "family": "xgboost",
        "status": "attempted",
        "completed_trials": 20,
        "best_validation": 0.51
      },
      {
        "family": "lightgbm",
        "status": "attempted",
        "completed_trials": 24,
        "best_validation": 0.53
      }
    ]
  },
  "stacking": {
    "attempted": true,
    "adopted": false,
    "reason": "gain was not repeatable across folds"
  },
  "selection": {
    "model": "LightGBM",
    "threshold_rule": "top 200 invoices per week",
    "calibration": "sigmoid"
  },
  "final": {
    "eval_set": "holdout_test",
    "score": 0.51,
    "metric": "average_precision",
    "confidence_interval": [0.47, 0.55],
    "uncertainty": {
      "method": "customer-week block bootstrap with queue reselection",
      "confidence_level": 0.95,
      "resampling_unit": "customer_week",
      "repetitions": 2000,
      "seed": 42,
      "effective_sample_size": 4120,
      "capacity_unit": "week",
      "selection_population": "full_eligible_queue",
      "policy_recomputed_per_resample": true
    },
    "secondary": {
      "log_loss": 0.31,
      "recall_at_review_capacity": 0.68
    }
  },
  "subgroups": {
    "reported": true,
    "minimum_support": 50,
    "summary": "see results.md"
  },
  "autogluon": {
    "attempted": false,
    "reason": "user did not opt in"
  }
}
```

Retain the top-level `final` object for existing consumers.

Keep candidate suitability, dependency and execution states separate.
`not installed` is not a scientific exclusion. Every attempted family in
`config.json.search.candidates` must have a corresponding family result; every
non-attempted family must have a concrete reason. Every candidate also records
an environment-independent `consideration_basis`.

Record whether an incumbent operational process exists in
`config.json.baselines.incumbent`. Require its metric only when it exists and
can be measured fairly on the same evaluation population. Store an available
result in `metrics.json.baselines.incumbent` with a finite `score`, the
`metric`, `eval_set` and `population_fingerprint`; the metric must match the
declared primary metric.

Record the uncertainty method, confidence level, resampling unit and effective
sample size/support. Add repetitions and seed when the method uses them. A bare
interval is not enough for new v2.1 runs.

For `evaluation.design: "nested_cv"`, use
`final.eval_set: "outer_cv"`, set `selection_nested: true` and
`independent_test: false`, and store outer-fold scores, an explicit
mean/median aggregation, and uncertainty. Do not combine this generic contract
with temporal split mechanics, because a fold complement would contain future
rows. Do not call it an independent holdout result. `external_test` and
`prospective_validation` designs must fingerprint and describe the external or
future cohort.

For a prospective cohort whose outcomes have not matured, use
`evaluation.status: "pending_labels"` and:

```json
{
  "final": {
    "eval_set": "prospective_validation",
    "score": null,
    "metric": "average_precision",
    "outcomes_mature": false,
    "validated_performance_available": false,
    "maturity_rule": "90 days after prediction",
    "cohort_counts": {
      "scored": 12000,
      "matured": 0,
      "pending": 12000,
      "lost_to_follow_up": 0
    }
  }
}
```

Do not fabricate a score to make a pending run look complete.

For `governance.risk_tier: "high"`, record the domain owner, meaningful human
oversight, deployment decision, approval status, prohibited uses, critical
harms, label provenance, external/prospective validation plan, appeal path,
incident owner, monitoring cadence and rollback plan. New model runs may not
leave risk classification unassessed. Every tier must include the assessment
rationale and unresolved hazards; do not infer `standard` merely because the
profiler was run with defaults.

## Task-specific evaluation contracts

### Unlabeled anomaly ranking

Do not fabricate a holdout target or predictive score. Set
`problem.labels_available: false`, define the reference/scoring windows and
daily queue contract in `config.json`, and use:

```json
{
  "schema_version": "2.1",
  "task": "anomaly",
  "primary_metric": null,
  "anomaly_evaluation": {
    "scoring_unit": "UTC day",
    "review_capacity": 200,
    "same_population_rank_stability": {
      "spearman_mean": 0.91,
      "top_k_overlap_mean": 0.82
    },
    "queue_concentration": {
      "maximum_per_account": 3,
      "largest_merchant_fraction": 0.08
    },
    "reviewed_precision_at_k": null,
    "reviewed_rows": 0,
    "unreviewed_rows_treated_as_negative": false
  },
  "final": {
    "eval_set": "future_scoring_window",
    "score": null,
    "queue_size": 200,
    "predictive_performance_available": false
  }
}
```

The inference contract must distinguish row scoring from whole-batch queue
selection and define timezone/cutoff, eligibility, sub-capacity behavior,
stable tie-breaking, identifier passthrough, ranks, flags and reason codes.

### Forecasting

Record forecast origin, issuance/retraining cadence, horizon, direct/recursive/
multi-output strategy, historical covariate-vintage rule, target meaning
(observed sales versus latent demand), quantiles, interval coverage semantics
and cumulative lead-time outputs. Inference inputs must separate historical
observations from future-known covariates and outputs must include entity,
forecast date, horizon and quantile/point columns.

## Data and inference contracts

### data_fingerprint.json

Record cryptographic file hashes, sizes, row/column counts, source commit when
available, and generation timestamp. Hashes identify inputs; they do not
anonymize them. For warehouse, lake or remote objects that cannot be
content-hashed locally, record an immutable snapshot/version identifier and the
exact bounded query. Do not claim reproducibility from a mutable URL.
If the user explicitly accepts an unknown remote preflight, retain the recorded
override and set `reproducibility_status: "limited_remote_source"`. Use the
same status when the generic URL profiler records
`version_verification: "declared_not_verified"`. Validation may warn, but the
report must not present either run as exactly reproducible.

### schema.json

Separate `observational_completeness` from the inference contract. Observing no
missing values does not make a field semantically required. Record the profiled
population, row/missing counts and status under that object. Populate
`inference.required_inputs` and `inference.optional_inputs` only after the final
pipeline is known, together with dtypes, semantic types, units, timezone,
allowed missingness, target, partition column and compatibility behavior.

### feature_manifest.json

Record:

- raw input features;
- engineered features and formulas;
- excluded identifiers/sensitive/post-event/target-source columns;
- prediction-time availability;
- preprocessing per feature;
- expected category/unseen-category behavior.

The target, partition field, identifiers used only for alignment, sensitive
attributes excluded by policy and post-event/target-source fields must not
silently appear in `raw_input_features`.

### model/manifest.json

For a multi-file model directory, list every relative component path and its
SHA-256 hash. The validator must verify each component, not only hash the
manifest itself.

### requirements.lock and environment.lock

Pin every package used by the deployable inference environment. Pin VCS
dependencies to immutable commits and URL artifacts to content hashes; do not
use floating include/constraint files as a lock.

When using `environment.lock` instead, write structured JSON with
`schema_version`, exact Python version, platform and an exact package
name/version list. Arbitrary non-empty text is not an environment lock.

### inference_test.json

Record executable cases rather than only a claimed pass:

```json
{
  "schema_version": "2.1",
  "trusted_model_sha256": "...",
  "prediction_constraints": {
    "probability": {
      "semantic": "probability",
      "minimum": 0.0,
      "maximum": 1.0
    }
  },
  "cases": [
    {
      "name": "representative_batch",
      "argv": [
        "{python}",
        "artefacts/runs/20260728T143000Z-lightgbm-initial/infer.py",
        "--input",
        "test.csv",
        "--output",
        "artefacts/runs/20260728T143000Z-lightgbm-initial/inference_outputs/test_predictions.csv"
      ],
      "expected_exit_code": 0,
      "output": {
        "path": "artefacts/runs/20260728T143000Z-lightgbm-initial/inference_outputs/test_predictions.csv",
        "format": "csv",
        "row_count": 3,
        "required_columns": ["invoice_id", "probability", "review_flag"],
        "prediction_columns": ["probability"],
        "row_id_column": "invoice_id",
        "expected_row_ids": ["A", "B", "C"],
        "golden_predictions": {"probability": [0.1, 0.8, 0.4]},
        "absolute_tolerance": 1e-8
      }
    }
  ]
}
```

Include cases for a representative batch, one row, empty input, missing
required fields, extra fields, wrong dtypes and unseen categories. Add
all-missing optional fields when applicable. `representative_batch` and
`one_row` must succeed with parsed outputs; `missing_required` and
`wrong_dtypes` must fail with actionable messages. Other edge cases may either
succeed with a verified output or use a controlled non-zero exit and actionable
error substring. Use a SHA-256 output checksum instead of golden values only
for byte-stable outputs. For a zero-row JSON success, use a schema-bearing
object such as `{"columns": ["id", "score"], "rows": []}` rather than a bare
array.

When supervised decisions use fixed capacity, add separate successful cases for
the exact names `score_rows`, `select_queue`, `capacity_ties`,
`capacity_duplicates`, `capacity_empty` and `capacity_sub_capacity`. Queue-case
outputs include `selection_rank` and `selected`, with their actual values in
`golden_values`, plus `eligible_count`, `selected_count`, `capacity_limit`,
timezone, cutoff and tie-breaker metadata. This verifies the executed queue,
not merely a claimed success flag.

For schema 2.1 the validator runs the selected run's own `infer.py` with the
current Python interpreter; it does not accept shell snippets or arbitrary
executables. Every declared output must use the selected run's dedicated
`inference_outputs/` directory; duplicate paths and artifact collisions are
rejected. It completes static/hash validation first, replaces each declared
test output, parses successful output, verifies row count, columns, order and
finite predictions plus declared semantic bounds, then compares the checksum
or golden values. A command that
exits zero but emits no fresh predictions must fail. Execute only trusted local
code.

## Human-readable reports

### data_summary.md

Include:

- dataset purpose and scope;
- structural profile;
- important charts and plain-language interpretation;
- blockers/warnings/information;
- reporting/sampling/holdout boundaries;
- recommended next actions.

### results.md

Include:

- business question, prediction moment and intended use;
- cohort construction, label observation, sampling and weighting;
- dataset/split summary linked to EDA;
- leakage and data-quality decisions;
- naive/fixed baselines, conditional incumbent comparison and signal diagnostics;
- candidate roster, dependency outcomes, model-selection method and compute budget;
- development and declared final/outer metrics with uncertainty;
- threshold/calibration/forecast horizon/anomaly review budget;
- subgroup/error analysis;
- explainability with limitations;
- production schema, inference command and trusted-artifact warning;
- limitations, prohibited uses, monitoring and retraining guidance;
- run/parent lineage, evaluation-exposure state and reproducibility footer.

### model_card.md

Summarize intended users, intended/out-of-scope uses, training data, evaluation,
ethical/fairness limitations, operational constraints and ownership.

## Validation

Run the structural/semantic artifact checks and the real declared inference
round trip:

```text
python scripts/validate_run.py <project-directory> \
  --artifacts-dir artefacts/runs/<run_id> --run-inference-test
```

The validator verifies the declared artifact structure, cross-contract
consistency, hashes and executable inference behavior. It does not prove that
split construction, fold-local preprocessing, model selection or uncertainty
estimation were scientifically correct. Reconcile the training code, logs,
fold assignments and reports. Treat validator warnings as explicit handoff
limitations; fix errors before completion.

## Legacy compatibility

Treat artifacts without `schema_version` as version 1:

- accept legacy `final.eval_set == "holdout_test"`;
- accept `ceiling_check` on read but write `search.plateau_detected`;
- accept legacy `inference_trigger` but write
  `problem.prediction_moment`;
- accept existing filenames;
- warn that v1 lacks the full data/schema/model-card contract.

Accept v2.0 using its historical contract and warn that it lacks the v2.1 split
manifest, run lineage, candidate ledger and executable case contract. Do not
retroactively fail or rewrite a historical v2.0 run.

Do not rewrite a historical artifact silently. Migrate into a new run directory
or preserve a backup.
