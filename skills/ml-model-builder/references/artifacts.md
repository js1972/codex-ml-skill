# Artifact Contract and Schemas

## Contents

- [Versioning](#versioning)
- [Analysis-only outputs](#analysis-only-outputs)
- [Model outputs](#model-outputs)
- [config.json](#configjson)
- [metrics.json](#metricsjson)
- [Task-specific evaluation contracts](#task-specific-evaluation-contracts)
- [Data and inference contracts](#data-and-inference-contracts)
- [Human-readable reports](#human-readable-reports)
- [Validation](#validation)
- [Legacy compatibility](#legacy-compatibility)

## Versioning

Use `schema_version: "2.0"` in `config.json`, `metrics.json`,
`data_profile.json`, `schema.json`, `feature_manifest.json`, and
`data_fingerprint.json`.

Keep artifact filenames stable. Add fields compatibly; do not silently change
the meaning of an existing field. Record deprecations in
`config.json.compatibility`.

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

Required in addition to analysis outputs:

- `artefacts/train.py`
- `artefacts/infer.py`
- `artefacts/model.joblib` or a versioned `artefacts/model/` directory when the
  deployable solution contains multiple fitted components
- `artefacts/metrics.json`
- `artefacts/feature_manifest.json`
- `artefacts/model_card.md`
- `artefacts/requirements.lock` or an equivalent fully pinned inference
  environment
- `artefacts/inference_test.json`
- `results.md` in the project root

Optional:

- `artefacts/shap_summary.html`
- `artefacts/autogluon_predictor/`
- `artefacts/search.db` for resumable Optuna storage
- `predictions.csv`, `review_queue.csv`, or another requested operational
  scoring output with passthrough identifiers, as-of timestamp, model/data
  version, scores/ranks and eligibility/exclusion reasons

Never load an untrusted `model.joblib`.

## config.json

Use this shape:

```json
{
  "schema_version": "2.0",
  "mode": "model-building",
  "problem": {
    "task": "classification",
    "business_decision": "prioritize invoices for manual review",
    "target": "late_payment",
    "target_derivation": null,
    "prediction_moment": "invoice issue time",
    "row_grain": "one row per invoice",
    "group_column": "customer_id",
    "time_column": "invoice_date",
    "error_costs": {
      "false_positive": "review time",
      "false_negative": "missed late payment"
    }
  },
  "data": {
    "locations": ["data.csv"],
    "fingerprint_file": "artefacts/data_fingerprint.json",
    "schema_file": "artefacts/schema.json"
  },
  "split": {
    "strategy": "grouped_temporal",
    "assignment_column": "_ml_partition",
    "development_label": "train",
    "holdout_target_sealed": true,
    "seed": 42
  },
  "evaluation": {
    "design": "holdout",
    "final_eval_set": "holdout_test",
    "independent_test": true,
    "selection_nested": false
  },
  "analysis": {
    "report": "artefacts/data_report.html",
    "target_aware_partition": "train",
    "plot_sample_size": 10000,
    "plot_sample_seed": 42,
    "reported_columns": ["customer_id", "email"]
  },
  "feature_contract": {
    "manifest": "artefacts/feature_manifest.json",
    "inference_unavailable": ["payment_date", "final_status"],
    "sensitive_attributes": ["region"],
    "target_sources_excluded": []
  },
  "selection": {
    "primary_metric": "average_precision",
    "secondary_metrics": ["log_loss", "recall_at_review_capacity"],
    "validation": "grouped_temporal_cv",
    "threshold_rule": "top 200 invoices per week",
    "calibration": "sigmoid"
  },
  "search": {
    "sampler": "TPESampler",
    "budget_seconds": 1800,
    "stop_reason": "budget",
    "execution_mode": "managed_process",
    "task_or_session_id": "host-process-123"
  },
  "comparison": {
    "autogluon": false
  },
  "governance": {
    "risk_tier": "standard",
    "domain_owner": "accounts-receivable operations",
    "deployment_decision": "decision_support",
    "approval_status": "not_required"
  },
  "environment": {
    "python": "3.x.y",
    "platform": "recorded at runtime",
    "requirements": "artefacts/requirements.lock"
  },
  "compatibility": {
    "reads_legacy_v1": true,
    "deprecated_fields": ["ceiling_check"]
  }
}
```

## metrics.json

Keep every metric name, direction, dataset and uncertainty explicit:

```json
{
  "schema_version": "2.0",
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
    "plateau_detected": false
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

For `evaluation.design: "nested_cv"`, use
`final.eval_set: "outer_cv"`, set `selection_nested: true` and
`independent_test: false`, and store outer-fold scores, an explicit
mean/median aggregation, and uncertainty. Do not call this an independent
holdout result. `external_test` and
`prospective_validation` designs must fingerprint and describe the external or
future cohort.

## Task-specific evaluation contracts

### Unlabeled anomaly ranking

Do not fabricate a holdout target or predictive score. Set
`problem.labels_available: false`, define the reference/scoring windows and
daily queue contract in `config.json`, and use:

```json
{
  "schema_version": "2.0",
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
anonymize them.

### schema.json

Record required/optional columns, dtypes, semantic types, units, timezone,
allowed missingness, target, partition column and compatibility behavior.

### feature_manifest.json

Record:

- raw input features;
- engineered features and formulas;
- excluded identifiers/sensitive/post-event/target-source columns;
- prediction-time availability;
- preprocessing per feature;
- expected category/unseen-category behavior.

### inference_test.json

Record the display command, a non-shell `argv` array, trusted model hash, test
input schema, expected output schema, row count, prediction
checksum/tolerance, and tested edge cases. Use `"{python}"` in `argv` to request
the interpreter running the validator.

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
- dataset/split summary linked to EDA;
- leakage and data-quality decisions;
- baselines and signal diagnostics;
- model-selection method and compute budget;
- development and declared final/outer metrics with uncertainty;
- threshold/calibration/forecast horizon/anomaly review budget;
- subgroup/error analysis;
- explainability with limitations;
- production schema, inference command and trusted-artifact warning;
- limitations, prohibited uses, monitoring and retraining guidance;
- reproducibility footer.

### model_card.md

Summarize intended users, intended/out-of-scope uses, training data, evaluation,
ethical/fairness limitations, operational constraints and ownership.

## Validation

Run the structural/semantic artifact checks and the real declared inference
round trip:

```text
python scripts/validate_run.py <project-directory> --run-inference-test
```

The validator does not prove that split construction, fold-local preprocessing,
model selection, or uncertainty estimation were scientifically correct.
Reconcile the training code, logs, fold assignments and reports. Treat
validator warnings as explicit handoff limitations; fix errors before
completion.

## Legacy compatibility

Treat artifacts without `schema_version` as version 1:

- accept legacy `final.eval_set == "holdout_test"`;
- accept `ceiling_check` on read but write `search.plateau_detected`;
- accept legacy `inference_trigger` but write
  `problem.prediction_moment`;
- accept existing filenames;
- warn that v1 lacks the full data/schema/model-card contract.

Do not rewrite a historical artifact silently. Migrate into a new run directory
or preserve a backup.
