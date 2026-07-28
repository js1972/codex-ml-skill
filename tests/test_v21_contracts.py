from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
VALIDATE = REPOSITORY / "skills/ml-model-builder/scripts/validate_run.py"
RUN_ID = "test-run"
ARTIFACTS_RELATIVE = Path("artefacts/runs") / RUN_ID


def run(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


def validate(project: Path, *arguments: str) -> subprocess.CompletedProcess:
    return run(
        VALIDATE,
        project,
        "--artifacts-dir",
        ARTIFACTS_RELATIVE.as_posix(),
        *arguments,
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def refresh_run_hash(artifacts: Path, filename: str, field: str) -> None:
    manifest_path = artifacts / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = (
        "sha256:" + hashlib.sha256((artifacts / filename).read_bytes()).hexdigest()
    )
    write_json(manifest_path, manifest)


def build_v21_project(
    project: Path,
    *,
    mode: str = "model-building",
    design: str = "holdout",
    evaluation_status: str = "complete",
    risk_tier: str = "standard",
    capacity: bool = False,
) -> Path:
    artifacts = project / ARTIFACTS_RELATIVE
    artifact_prefix = ARTIFACTS_RELATIVE.as_posix()
    (artifacts / "figures").mkdir(parents=True)
    (artifacts / "figures/chart.png").write_bytes(b"png")
    (artifacts / "inference_outputs").mkdir()

    model_bytes = b"trusted-test-model"
    (artifacts / "model.joblib").write_bytes(model_bytes)
    model_hash = hashlib.sha256(model_bytes).hexdigest()

    infer_source = """\
import argparse
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--case", required=True)
parser.add_argument("--output")
args = parser.parse_args()

if args.case in {"missing_required", "wrong_dtypes"}:
    print("actionable input contract error", file=sys.stderr)
    raise SystemExit(2)

rows = {
    "representative_batch": [
        {"row_id": "A", "probability": 0.1},
        {"row_id": "B", "probability": 0.8},
        {"row_id": "C", "probability": 0.4},
    ],
    "one_row": [{"row_id": "A", "probability": 0.1}],
    "empty_input": [],
    "extra_columns": [{"row_id": "A", "probability": 0.1}],
    "unseen_categories": [{"row_id": "A", "probability": 0.2}],
    "all_missing_optional": [{"row_id": "A", "probability": 0.3}],
    "score_rows": [
        {"row_id": "A", "probability": 0.1},
        {"row_id": "B", "probability": 0.8},
        {"row_id": "C", "probability": 0.4},
    ],
    "select_queue": [
        {"row_id": "B", "probability": 0.8, "selection_rank": 1, "selected": True},
        {"row_id": "C", "probability": 0.4, "selection_rank": 2, "selected": True},
    ],
    "capacity_ties": [
        {"row_id": "A", "probability": 0.8, "selection_rank": 1, "selected": True},
        {"row_id": "B", "probability": 0.8, "selection_rank": 2, "selected": True},
    ],
    "capacity_duplicates": [
        {"row_id": "A", "probability": 0.8, "selection_rank": 1, "selected": True},
        {"row_id": "B", "probability": 0.7, "selection_rank": 2, "selected": True},
    ],
    "capacity_empty": [],
    "capacity_sub_capacity": [
        {"row_id": "A", "probability": 0.8, "selection_rank": 1, "selected": True},
    ],
}[args.case]

with open(args.output, "w", newline="", encoding="utf-8") as handle:
    queue_cases = {
        "select_queue",
        "capacity_ties",
        "capacity_duplicates",
        "capacity_empty",
        "capacity_sub_capacity",
    }
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            ["row_id", "probability", "selection_rank", "selected"]
            if args.case in queue_cases
            else ["row_id", "probability"]
        ),
    )
    writer.writeheader()
    writer.writerows(rows)
"""
    (artifacts / "infer.py").write_text(infer_source, encoding="utf-8")
    (artifacts / "train.py").write_text(
        "# Refit the frozen final pipeline.\n", encoding="utf-8"
    )

    final_set = {
        "holdout": "holdout_test",
        "nested_cv": "outer_cv",
        "external_test": "external_test",
        "prospective_validation": "prospective_validation",
    }[design]
    independent_test = design != "nested_cv"
    pending = evaluation_status == "pending_labels"

    candidates = [
        {
            "family": "xgboost",
            "consideration_basis": "strong nonlinear tabular candidate",
            "suitability_status": "eligible",
            "dependency_status": "installed_for_run",
            "execution_status": "attempted",
            "reason": None,
        },
        {
            "family": "lightgbm",
            "consideration_basis": "efficient histogram boosting candidate",
            "suitability_status": "eligible",
            "dependency_status": "installed",
            "execution_status": "attempted",
            "reason": None,
        },
        {
            "family": "catboost",
            "consideration_basis": "native categorical handling considered",
            "suitability_status": "excluded",
            "dependency_status": "not_required",
            "execution_status": "excluded",
            "reason": "not suitable for the deployment runtime",
        },
    ]
    run_kind = "improvement" if mode == "model-improvement" else "initial"
    parent_run_id = "parent-run" if run_kind == "improvement" else None
    parent_hashes = (
        {
            "config.json": "sha256:" + "b" * 64,
            "metrics.json": "sha256:" + "c" * 64,
            "run_manifest.json": "sha256:" + "d" * 64,
        }
        if parent_run_id
        else {}
    )

    config = {
        "schema_version": "2.1",
        "mode": mode,
        "problem": {
            "task": "classification",
            "target": "outcome",
            "prediction_moment": "application time",
            "row_grain": "one row per application",
            "cohort": {
                "source_population": "all eligible applications",
                "inclusion_rule": "submitted during the study period",
                "label_observation": "outcome observed after maturity",
                "sampling_design": "complete cohort",
                "inclusion_probability": 1.0,
                "sample_weight": None,
                "evaluation_representative": True,
                "calibration_representative": True,
                "selective_labels": False,
                "label_acquisition": None,
            },
        },
        "data": {
            "fingerprint_file": f"{artifact_prefix}/data_fingerprint.json",
            "schema_file": f"{artifact_prefix}/schema.json",
            "split_manifest": f"{artifact_prefix}/split_manifest.json",
        },
        "split": {
            "strategy": "stratified_random",
            "group_overlap_policy": "disallow",
            "assignment_column": "_ml_partition",
            "development_label": None if design == "nested_cv" else "train",
            "holdout_target_sealed": design == "holdout",
        },
        "analysis": {
            "target_aware_partition": None if design == "nested_cv" else "train",
            "target_aware": design != "nested_cv",
            "pre_partition_target_exposure": {
                "status": "none",
                "source": None,
                "final_population_overlap": False,
                "values_viewed": [],
                "decisions_influenced": [],
            },
        },
        "feature_contract": {
            "manifest": f"{artifact_prefix}/feature_manifest.json",
            "inference_unavailable": [],
            "sensitive_attributes": [],
            "target_sources_excluded": [],
        },
        "evaluation": {
            "design": design,
            "status": evaluation_status,
            "final_eval_set": final_set,
            "independent_test": independent_test,
            "selection_nested": design == "nested_cv",
        },
        "selection": {
            "primary_metric": "average_precision",
            "capacity": {"enabled": False},
        },
        "baselines": {
            "incumbent": {
                "available": False,
                "reason": "no existing scored process",
            }
        },
        "search": {
            "roster_frozen_at": "2026-07-28T01:00:00Z",
            "candidates": candidates,
        },
        "run": {
            "manifest": f"{artifact_prefix}/run_manifest.json",
            "run_id": RUN_ID,
            "run_kind": run_kind,
        },
        "governance": {
            "risk_tier": risk_tier,
            "risk_assessed": risk_tier in {"standard", "high"},
            "risk_assessment_rationale": (
                "ordinary operational prioritization"
                if risk_tier == "standard"
                else "assessment is pending"
            ),
            "unresolved_hazards": [],
        },
        "environment": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": sys.platform,
            "requirements": f"{artifact_prefix}/requirements.lock",
        },
    }
    if design in {"external_test", "prospective_validation"}:
        config["evaluation"]["cohort_fingerprint"] = "sha256:" + "c" * 64
    if capacity:
        config["selection"]["capacity"] = {
            "enabled": True,
            "unit": "day",
            "limit": 2,
            "timezone": "Australia/Perth",
            "cutoff": "09:00",
            "eligibility_rule": "open cases",
            "tie_breaker": "row_id ascending",
            "sub_capacity_behavior": "select every eligible row",
        }

    metrics = {
        "schema_version": "2.1",
        "task": "classification",
        "primary_metric": {
            "name": "average_precision",
            "direction": "maximize",
        },
        "baselines": {
            "naive": {"validation_mean": 0.1},
            "fixed": {
                "model": "LogisticRegression",
                "validation_mean": 0.4,
            },
        },
        "search": {
            "family_results": [
                {
                    "family": "xgboost",
                    "status": "attempted",
                    "completed_trials": 2,
                    "best_validation": 0.5,
                },
                {
                    "family": "lightgbm",
                    "status": "attempted",
                    "completed_trials": 2,
                    "best_validation": 0.52,
                },
            ]
        },
        "final": {
            "eval_set": final_set,
            "score": None if pending else 0.5,
            "metric": "average_precision",
            "outcomes_mature": not pending,
            "validated_performance_available": not pending,
        },
    }
    if pending:
        metrics["final"].update(
            {
                "maturity_rule": "90 days after prediction",
                "cohort_counts": {
                    "scored": 100,
                    "matured": 0,
                    "pending": 100,
                    "lost_to_follow_up": 0,
                },
            }
        )
    if not pending:
        metrics["final"].update(
            {
                "confidence_interval": [0.4, 0.6],
                "uncertainty": {
                    "method": "row bootstrap",
                    "confidence_level": 0.95,
                    "resampling_unit": "row",
                    "repetitions": 1000,
                    "seed": 42,
                    "effective_sample_size": 100,
                },
            }
        )
        if capacity:
            metrics["final"]["uncertainty"].update(
                {
                    "method": "deployment-day block bootstrap",
                    "resampling_unit": "deployment_day",
                    "capacity_unit": "day",
                    "selection_population": "full_eligible_queue",
                    "policy_recomputed_per_resample": True,
                }
            )
    if design == "nested_cv":
        metrics["final"].update(
            {
                "score": 0.5,
                "fold_scores": [0.4, 0.6],
                "aggregation": "mean",
            }
        )

    split_manifest = {
        "schema_version": "2.1",
        "strategy": "stratified_random",
        "assignment": {
            "column": "_ml_partition",
            "source": "persisted_column",
            "fingerprint": {
                "algorithm": "sha256",
                "value": "d" * 64,
                "basis": "row_id plus partition",
            },
        },
        "partitions": [
            {"name": "train", "role": "development", "rows": 80},
            {"name": "holdout", "role": "final_evaluation", "rows": 20},
        ],
        "audits": {
            "group_overlap": {
                "checked": False,
                "groups_spanning_partitions": 0,
                "allowed": False,
                "reason": "no repeated group key",
            },
            "temporal_order": {
                "checked": False,
                "valid": True,
                "reason": "non-temporal deployment",
            },
            "duplicate_overlap": {
                "checked": True,
                "rows_crossing_partitions": 0,
            },
        },
    }
    if design == "nested_cv":
        split_manifest["partitions"] = [
            {"name": "outer-0", "role": "outer_evaluation", "rows": 50},
            {"name": "outer-1", "role": "outer_evaluation", "rows": 50},
        ]
        split_manifest["folds"] = [
            {"id": "outer-0", "role": "outer_evaluation", "rows": 50},
            {"id": "outer-1", "role": "outer_evaluation", "rows": 50},
        ]

    documents = {
        "config.json": config,
        "data_profile.json": {
            "schema_version": "2.1",
            "mode": "model",
            "task": "classification",
        },
        "data_fingerprint.json": {
            "schema_version": "2.1",
            "rows": 100,
            "columns": 3,
            "input": {
                "kind": "file_sha256",
                "sha256": "a" * 64,
                "bytes": 1000,
            },
        },
        "schema.json": {
            "schema_version": "2.1",
            "columns": {
                "age": {
                    "dtype": "int64",
                    "observational_completeness": {
                        "population": "train",
                        "population_rows": 80,
                        "missing_count": 0,
                        "missing_fraction": 0.0,
                        "non_missing_count": 80,
                        "status": "observed",
                    },
                },
                "outcome": {
                    "dtype": "int64",
                    "observational_completeness": {
                        "population": "train",
                        "population_rows": 80,
                        "missing_count": 0,
                        "missing_fraction": 0.0,
                        "non_missing_count": 80,
                        "status": "observed",
                    },
                },
                "_ml_partition": {
                    "dtype": "object",
                    "observational_completeness": {
                        "population": "train",
                        "population_rows": 80,
                        "missing_count": 0,
                        "missing_fraction": 0.0,
                        "non_missing_count": 80,
                        "status": "observed",
                    },
                },
            },
            "target": "outcome",
            "partition_column": "_ml_partition",
            "inference": {
                "required_inputs": ["age"],
                "optional_inputs": [],
            },
        },
        "feature_manifest.json": {
            "schema_version": "2.1",
            "raw_input_features": ["age"],
            "engineered_features": [],
            "excluded_features": [
                {"name": "outcome", "reason": "target"},
                {"name": "_ml_partition", "reason": "partition"},
            ],
        },
        "metrics.json": metrics,
        "split_manifest.json": split_manifest,
    }
    for name, document in documents.items():
        write_json(artifacts / name, document)

    data_fingerprint_hash = hashlib.sha256(
        (artifacts / "data_fingerprint.json").read_bytes()
    ).hexdigest()
    split_manifest_hash = hashlib.sha256(
        (artifacts / "split_manifest.json").read_bytes()
    ).hexdigest()
    population_fingerprint = config["evaluation"].get(
        "cohort_fingerprint", "sha256:" + "e" * 64
    )
    run_manifest = {
        "schema_version": "2.1",
        "run_id": RUN_ID,
        "run_kind": run_kind,
        "artifact_directory": artifact_prefix,
        "parent_run_id": parent_run_id,
        "parent_artifact_hashes": parent_hashes,
        "code_revision": "git:abcdef123456",
        "data_fingerprint": "sha256:" + data_fingerprint_hash,
        "split_fingerprint": "sha256:" + split_manifest_hash,
        "created_at": "2026-07-28T00:00:00Z",
        "changes": ["initial model"] if run_kind == "initial" else ["new candidates"],
        "roster_frozen_at": "2026-07-28T01:00:00Z",
        "prior_evidence": (
            []
            if run_kind == "initial"
            else [
                {
                    "source_run_id": "parent-run",
                    "run_manifest_sha256": "sha256:" + "f" * 64,
                    "metrics_sha256": "sha256:" + "8" * 64,
                    "status": "benchmark_selection",
                    "final_set": "holdout_test",
                    "population_fingerprint": "sha256:" + "9" * 64,
                    "opened_at": "2026-07-27T00:00:00Z",
                    "opened_for": "parent final evaluation",
                    "values_viewed": ["primary metric"],
                    "decisions_influenced": ["child candidate hypothesis"],
                }
            ]
        ),
        "evaluation_exposure": {
            "status": "pending_labels" if pending else "opened",
            "final_set": final_set,
            "population_fingerprint": population_fingerprint,
            "opened_at": None if pending else "2026-07-28T02:00:00Z",
            "opened_for": None if pending else "declared final evaluation",
            "values_viewed": [] if pending else ["primary metric"],
            "decisions_influenced": [],
        },
    }
    write_json(artifacts / "run_manifest.json", run_manifest)

    cases = [
        {
            "name": "representative_batch",
            "argv": [
                "{python}",
                f"{artifact_prefix}/infer.py",
                "--case",
                "representative_batch",
                "--output",
                f"{artifact_prefix}/inference_outputs/representative.csv",
            ],
            "expected_exit_code": 0,
            "output": {
                "path": f"{artifact_prefix}/inference_outputs/representative.csv",
                "format": "csv",
                "row_count": 3,
                "required_columns": ["row_id", "probability"],
                "prediction_columns": ["probability"],
                "row_id_column": "row_id",
                "expected_row_ids": ["A", "B", "C"],
                "golden_predictions": {"probability": [0.1, 0.8, 0.4]},
                "absolute_tolerance": 1e-12,
            },
        },
        {
            "name": "one_row",
            "argv": [
                "{python}",
                f"{artifact_prefix}/infer.py",
                "--case",
                "one_row",
                "--output",
                f"{artifact_prefix}/inference_outputs/one.csv",
            ],
            "expected_exit_code": 0,
            "output": {
                "path": f"{artifact_prefix}/inference_outputs/one.csv",
                "format": "csv",
                "row_count": 1,
                "required_columns": ["row_id", "probability"],
                "prediction_columns": ["probability"],
                "golden_predictions": {"probability": [0.1]},
                "absolute_tolerance": 1e-12,
            },
        },
        {
            "name": "empty_input",
            "argv": [
                "{python}",
                f"{artifact_prefix}/infer.py",
                "--case",
                "empty_input",
                "--output",
                f"{artifact_prefix}/inference_outputs/empty.csv",
            ],
            "expected_exit_code": 0,
            "output": {
                "path": f"{artifact_prefix}/inference_outputs/empty.csv",
                "format": "csv",
                "row_count": 0,
                "required_columns": ["row_id", "probability"],
                "prediction_columns": ["probability"],
                "golden_predictions": {"probability": []},
                "absolute_tolerance": 1e-12,
            },
        },
        {
            "name": "missing_required",
            "argv": [
                "{python}",
                f"{artifact_prefix}/infer.py",
                "--case",
                "missing_required",
            ],
            "expected_exit_code": 2,
            "stderr_contains": "actionable input contract error",
        },
        {
            "name": "extra_columns",
            "argv": [
                "{python}",
                f"{artifact_prefix}/infer.py",
                "--case",
                "extra_columns",
                "--output",
                f"{artifact_prefix}/inference_outputs/extra.csv",
            ],
            "expected_exit_code": 0,
            "output": {
                "path": f"{artifact_prefix}/inference_outputs/extra.csv",
                "format": "csv",
                "row_count": 1,
                "required_columns": ["row_id", "probability"],
                "prediction_columns": ["probability"],
                "golden_predictions": {"probability": [0.1]},
                "absolute_tolerance": 1e-12,
            },
        },
        {
            "name": "wrong_dtypes",
            "argv": [
                "{python}",
                f"{artifact_prefix}/infer.py",
                "--case",
                "wrong_dtypes",
            ],
            "expected_exit_code": 2,
            "stderr_contains": "actionable input contract error",
        },
        {
            "name": "unseen_categories",
            "argv": [
                "{python}",
                f"{artifact_prefix}/infer.py",
                "--case",
                "unseen_categories",
                "--output",
                f"{artifact_prefix}/inference_outputs/unseen.csv",
            ],
            "expected_exit_code": 0,
            "output": {
                "path": f"{artifact_prefix}/inference_outputs/unseen.csv",
                "format": "csv",
                "row_count": 1,
                "required_columns": ["row_id", "probability"],
                "prediction_columns": ["probability"],
                "golden_predictions": {"probability": [0.2]},
                "absolute_tolerance": 1e-12,
            },
        },
        {
            "name": "all_missing_optional",
            "argv": [
                "{python}",
                f"{artifact_prefix}/infer.py",
                "--case",
                "all_missing_optional",
                "--output",
                f"{artifact_prefix}/inference_outputs/missing-optional.csv",
            ],
            "expected_exit_code": 0,
            "output": {
                "path": f"{artifact_prefix}/inference_outputs/missing-optional.csv",
                "format": "csv",
                "row_count": 1,
                "required_columns": ["row_id", "probability"],
                "prediction_columns": ["probability"],
                "golden_predictions": {"probability": [0.3]},
                "absolute_tolerance": 1e-12,
            },
        },
    ]
    if capacity:
        for case_name, row_ids, predictions, eligible_count in [
            ("score_rows", ["A", "B", "C"], [0.1, 0.8, 0.4], 3),
            ("select_queue", ["B", "C"], [0.8, 0.4], 3),
            ("capacity_ties", ["A", "B"], [0.8, 0.8], 3),
            ("capacity_duplicates", ["A", "B"], [0.8, 0.7], 3),
            ("capacity_empty", [], [], 0),
            ("capacity_sub_capacity", ["A"], [0.8], 1),
        ]:
            output_path = f"{artifact_prefix}/inference_outputs/{case_name}.csv"
            queue_case = case_name != "score_rows"
            output_contract = {
                "path": output_path,
                "format": "csv",
                "row_count": len(row_ids),
                "required_columns": (
                    ["row_id", "probability", "selection_rank", "selected"]
                    if queue_case
                    else ["row_id", "probability"]
                ),
                "prediction_columns": ["probability"],
                "row_id_column": "row_id",
                "expected_row_ids": row_ids,
                "golden_predictions": {"probability": predictions},
                "absolute_tolerance": 1e-12,
            }
            if queue_case:
                output_contract.update(
                    {
                        "eligible_count": eligible_count,
                        "selected_count": len(row_ids),
                        "capacity_limit": 2,
                        "timezone": "Australia/Perth",
                        "cutoff": "09:00",
                        "tie_breaker": "row_id ascending",
                        "golden_values": {
                            "selection_rank": list(range(1, len(row_ids) + 1)),
                            "selected": [True] * len(row_ids),
                        },
                    }
                )
            cases.append(
                {
                    "name": case_name,
                    "argv": [
                        "{python}",
                        f"{artifact_prefix}/infer.py",
                        "--case",
                        case_name,
                        "--output",
                        output_path,
                    ],
                    "expected_exit_code": 0,
                    "output": output_contract,
                }
            )
    write_json(
        artifacts / "inference_test.json",
        {
            "schema_version": "2.1",
            "trusted_model_sha256": model_hash,
            "prediction_constraints": {
                "probability": {
                    "semantic": "probability",
                    "minimum": 0.0,
                    "maximum": 1.0,
                }
            },
            "cases": cases,
        },
    )
    for name, contents in {
        "data_report.html": "<html></html>",
        "data_summary.md": "# Summary",
        "model_card.md": "# Model card",
        "requirements.lock": "scikit-learn==1.7.1\n",
    }.items():
        (artifacts / name).write_text(contents, encoding="utf-8")
    (artifacts / "results.md").write_text(
        "# Results\n\nPrediction moment: application time.\n",
        encoding="utf-8",
    )
    return artifacts


class Version21ContractTests(unittest.TestCase):
    def test_complete_v21_model_and_real_inference_cases_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_v21_project(project)
            completed = validate(project, "--run-inference-test")
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_no_op_inference_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            (artifacts / "infer.py").write_text("pass\n", encoding="utf-8")
            completed = validate(project, "--run-inference-test")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("output", completed.stdout.lower())

    def test_supervised_candidate_family_cannot_disappear(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["search"]["candidates"] = [
                item
                for item in config["search"]["candidates"]
                if item["family"] != "catboost"
            ]
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("catboost", completed.stdout.lower())

    def test_pending_prospective_labels_do_not_require_fake_score(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_v21_project(
                project,
                design="prospective_validation",
                evaluation_status="pending_labels",
            )
            completed = validate(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_nested_cv_target_blind_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_v21_project(project, design="nested_cv")
            completed = validate(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_nested_cv_fold_ids_must_match_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project, design="nested_cv")
            split = json.loads((artifacts / "split_manifest.json").read_text())
            split["folds"][0]["id"] = "wrong-fold"
            write_json(artifacts / "split_manifest.json", split)
            refresh_run_hash(
                artifacts,
                "split_manifest.json",
                "split_fingerprint",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("fold", completed.stdout.lower())

    def test_nested_cv_fingerprinted_discovery_partition_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project, design="nested_cv")
            config = json.loads((artifacts / "config.json").read_text())
            config["analysis"].update(
                {
                    "target_aware": True,
                    "target_aware_partition": "discovery",
                    "discovery_excluded_from_outer": True,
                }
            )
            write_json(artifacts / "config.json", config)
            split = json.loads((artifacts / "split_manifest.json").read_text())
            for partition in split["partitions"]:
                partition["rows"] = 45
            for fold in split["folds"]:
                fold["rows"] = 45
            split["partitions"].append(
                {
                    "name": "discovery",
                    "role": "discovery_excluded",
                    "rows": 10,
                    "fingerprint": "sha256:" + "7" * 64,
                }
            )
            write_json(artifacts / "split_manifest.json", split)
            refresh_run_hash(
                artifacts,
                "split_manifest.json",
                "split_fingerprint",
            )
            completed = validate(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_improvement_requires_parent_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project, mode="model-improvement")
            manifest = json.loads((artifacts / "run_manifest.json").read_text())
            manifest["parent_run_id"] = None
            manifest["parent_artifact_hashes"] = {}
            write_json(artifacts / "run_manifest.json", manifest)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("parent", completed.stdout.lower())

    def test_improvement_prior_evidence_must_reference_direct_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project, mode="model-improvement")
            manifest = json.loads((artifacts / "run_manifest.json").read_text())
            manifest["prior_evidence"][0]["source_run_id"] = "unrelated-run"
            write_json(artifacts / "run_manifest.json", manifest)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("prior_evidence", completed.stdout)

    def test_improvement_cannot_reseal_prior_opened_population(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project, mode="model-improvement")
            manifest = json.loads((artifacts / "run_manifest.json").read_text())
            manifest["prior_evidence"][0]["population_fingerprint"] = manifest[
                "evaluation_exposure"
            ]["population_fingerprint"]
            write_json(artifacts / "run_manifest.json", manifest)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("previously opened", completed.stdout)

    def test_unassessed_risk_cannot_complete_model_run(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_v21_project(project, risk_tier="not_assessed")
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("risk", completed.stdout.lower())

    def test_floating_url_dependency_is_not_a_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            (artifacts / "requirements.lock").write_text(
                "https://example.test/package.whl\n",
                encoding="utf-8",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requirements.lock", completed.stdout)

    def test_symbolic_dependency_version_is_not_a_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            (artifacts / "requirements.lock").write_text(
                "numpy==latest\n",
                encoding="utf-8",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requirements.lock", completed.stdout)

    def test_unverified_remote_version_requires_limited_disclosure(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            fingerprint = json.loads((artifacts / "data_fingerprint.json").read_text())
            fingerprint["input"] = {
                "kind": "remote_declared_version",
                "sha256": None,
                "bytes": 1000,
                "source": "https://example.test/data.parquet",
                "remote_source_version": "declared-etag",
                "immutable_source_id": (
                    "https://example.test/data.parquet@declared-etag"
                ),
                "version_verification": "declared_not_verified",
                "reproducibility_status": "limited_remote_source",
                "remote_preflight": {
                    "applies": True,
                    "status": "passed",
                    "override_used": False,
                    "unknown_fields": [],
                },
            }
            write_json(artifacts / "data_fingerprint.json", fingerprint)
            refresh_run_hash(
                artifacts,
                "data_fingerprint.json",
                "data_fingerprint",
            )
            config = json.loads((artifacts / "config.json").read_text())
            config["data"]["reproducibility_status"] = "limited_remote_source"
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("limited_remote_source", completed.stdout)
            with (artifacts / "results.md").open("a", encoding="utf-8") as handle:
                handle.write("\nReproducibility: limited_remote_source.\n")
            completed = validate(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_evaluation_population_fingerprint_must_be_cryptographic(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project, design="external_test")
            config = json.loads((artifacts / "config.json").read_text())
            config["evaluation"]["cohort_fingerprint"] = "banana"
            write_json(artifacts / "config.json", config)
            manifest = json.loads((artifacts / "run_manifest.json").read_text())
            manifest["evaluation_exposure"]["population_fingerprint"] = "banana"
            write_json(artifacts / "run_manifest.json", manifest)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("sha256", completed.stdout.lower())

    def test_arbitrary_inference_executable_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            contract = json.loads((artifacts / "inference_test.json").read_text())
            contract["cases"][0]["argv"] = ["/bin/sh", "-c", "exit 0"]
            write_json(artifacts / "inference_test.json", contract)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("infer.py", completed.stdout)

    def test_inference_output_cannot_overwrite_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            contract = json.loads((artifacts / "inference_test.json").read_text())
            representative = contract["cases"][0]
            replacement = f"{ARTIFACTS_RELATIVE.as_posix()}/config.json"
            representative["argv"][-1] = replacement
            representative["output"]["path"] = replacement
            write_json(artifacts / "inference_test.json", contract)
            original = (artifacts / "config.json").read_bytes()
            completed = validate(project, "--run-inference-test")
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual((artifacts / "config.json").read_bytes(), original)
            self.assertIn("inference_outputs", completed.stdout)

    def test_artifact_directory_symlink_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside_project = root / "outside"
            outside_project.mkdir()
            outside_artifacts = build_v21_project(outside_project)
            project = root / "project"
            run_parent = project / "artefacts/runs"
            run_parent.mkdir(parents=True)
            (run_parent / RUN_ID).symlink_to(
                outside_artifacts, target_is_directory=True
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("project", completed.stdout.lower())

    def test_model_artifact_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            outside = project / "outside-model.joblib"
            outside.write_bytes((artifacts / "model.joblib").read_bytes())
            (artifacts / "model.joblib").unlink()
            (artifacts / "model.joblib").symlink_to(outside)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("symlink", completed.stdout.lower())

    def test_static_contract_errors_prevent_inference_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            marker = project / "executed.marker"
            (artifacts / "infer.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n",
                encoding="utf-8",
            )
            config = json.loads((artifacts / "config.json").read_text())
            config["governance"]["risk_tier"] = "banana"
            write_json(artifacts / "config.json", config)
            completed = validate(project, "--run-inference-test")
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())

    def test_touching_stale_predictions_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            first = validate(project, "--run-inference-test")
            self.assertEqual(first.returncode, 0, first.stdout)
            (artifacts / "infer.py").write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--case', required=True)\n"
                "parser.add_argument('--output')\n"
                "args = parser.parse_args()\n"
                "if args.output:\n"
                "    Path(args.output).touch()\n",
                encoding="utf-8",
            )
            completed = validate(project, "--run-inference-test")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("output", completed.stdout.lower())

    def test_reported_final_score_cannot_remain_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            manifest = json.loads((artifacts / "run_manifest.json").read_text())
            manifest["evaluation_exposure"].update(
                {
                    "status": "sealed",
                    "opened_at": None,
                    "opened_for": None,
                    "values_viewed": [],
                    "decisions_influenced": [],
                }
            )
            write_json(artifacts / "run_manifest.json", manifest)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exposure", completed.stdout.lower())

    def test_unlabeled_anomaly_cannot_report_predictive_score(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["problem"].update(
                {
                    "task": "anomaly",
                    "target": None,
                    "labels_available": False,
                }
            )
            config["analysis"]["population_partition"] = "train"
            config["evaluation"].update(
                {
                    "design": "future_review",
                    "status": "opened",
                    "final_eval_set": "future_scoring_window",
                    "independent_test": False,
                    "selection_nested": False,
                }
            )
            write_json(artifacts / "config.json", config)
            schema = json.loads((artifacts / "schema.json").read_text())
            schema["target"] = None
            write_json(artifacts / "schema.json", schema)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics.update(
                {
                    "task": "anomaly",
                    "primary_metric": None,
                    "anomaly_evaluation": {
                        "review_capacity": 10,
                        "unreviewed_rows_treated_as_negative": False,
                    },
                }
            )
            metrics["final"].update(
                {
                    "eval_set": "future_scoring_window",
                    "score": 0.99,
                    "metric": None,
                    "predictive_performance_available": False,
                }
            )
            write_json(artifacts / "metrics.json", metrics)
            manifest = json.loads((artifacts / "run_manifest.json").read_text())
            manifest["evaluation_exposure"]["final_set"] = "future_scoring_window"
            write_json(artifacts / "run_manifest.json", manifest)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("score", completed.stdout.lower())
            self.assertIn("null", completed.stdout.lower())

    def test_split_partitions_must_reconcile_to_data(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            manifest = json.loads((artifacts / "split_manifest.json").read_text())
            manifest["partitions"] = [
                {"name": "foo", "role": "nonsense", "rows": 1},
                {"name": "bar", "role": "nonsense", "rows": 1},
            ]
            write_json(artifacts / "split_manifest.json", manifest)
            refresh_run_hash(
                artifacts,
                "split_manifest.json",
                "split_fingerprint",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("partition", completed.stdout.lower())

    def test_invalid_mode_or_task_cannot_bypass_route_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["mode"] = "banana"
            config["problem"]["task"] = "banana"
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("mode", completed.stdout.lower())
            self.assertIn("problem.task", completed.stdout)

    def test_malformed_contract_fails_closed_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["evaluation"]["design"] = {"not": "a string"}
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn("Traceback", completed.stdout + completed.stderr)
            self.assertIn("ERROR:", completed.stdout)

    def test_evaluation_status_is_enumerated(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["evaluation"]["status"] = "totally_fake"
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("evaluation.status", completed.stdout)

    def test_feature_cannot_be_both_available_and_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["feature_contract"]["inference_unavailable"] = ["age"]
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("inference_unavailable", completed.stdout)

    def test_schema_observational_completeness_is_reconciled(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            schema = json.loads((artifacts / "schema.json").read_text())
            schema["columns"]["age"]["observational_completeness"]["missing_count"] = 3
            write_json(artifacts / "schema.json", schema)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("observational_completeness", completed.stdout)

    def test_split_profiler_blockers_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            split = json.loads((artifacts / "split_manifest.json").read_text())
            split["blockers"] = [
                {
                    "code": "missing_partition_assignments",
                    "message": "one row has no assignment",
                }
            ]
            split["partitions"][1]["name"] = "<missing>"
            write_json(artifacts / "split_manifest.json", split)
            refresh_run_hash(
                artifacts,
                "split_manifest.json",
                "split_fingerprint",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("blockers", completed.stdout.lower())

    def test_completed_run_requires_duplicate_overlap_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            split = json.loads((artifacts / "split_manifest.json").read_text())
            split["audits"]["duplicate_overlap"] = {
                "checked": False,
                "rows_crossing_partitions": None,
                "reason": "not checked",
            }
            write_json(artifacts / "split_manifest.json", split)
            refresh_run_hash(
                artifacts,
                "split_manifest.json",
                "split_fingerprint",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate_overlap", completed.stdout)

    def test_temporal_ranges_are_recomputed_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["problem"]["time_column"] = "age"
            config["split"]["strategy"] = "temporal"
            write_json(artifacts / "config.json", config)
            schema = json.loads((artifacts / "schema.json").read_text())
            schema["columns"]["age"]["semantic_type"] = "datetime"
            write_json(artifacts / "schema.json", schema)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["final"]["uncertainty"].update(
                {
                    "method": "time-block bootstrap",
                    "resampling_unit": "time_block",
                }
            )
            write_json(artifacts / "metrics.json", metrics)
            split = json.loads((artifacts / "split_manifest.json").read_text())
            split["strategy"] = "temporal"
            split["audits"]["temporal_order"] = {
                "checked": True,
                "valid": True,
                "invalid_timestamp_rows": 0,
                "purge_gap": None,
                "ranges": [
                    {
                        "name": "train",
                        "role": "development",
                        "rows": 80,
                        "start": "2025-01-01T00:00:00+00:00",
                        "end": "2025-12-31T00:00:00+00:00",
                    },
                    {
                        "name": "holdout",
                        "role": "final_evaluation",
                        "rows": 20,
                        "start": "2025-06-01T00:00:00+00:00",
                        "end": "2026-01-31T00:00:00+00:00",
                    },
                ],
            }
            write_json(artifacts / "split_manifest.json", split)
            refresh_run_hash(
                artifacts,
                "split_manifest.json",
                "split_fingerprint",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("strictly before", completed.stdout)

    def test_unknown_split_strategy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["split"]["strategy"] = "my_magic_split"
            write_json(artifacts / "config.json", config)
            split = json.loads((artifacts / "split_manifest.json").read_text())
            split["strategy"] = "my_magic_split"
            write_json(artifacts / "split_manifest.json", split)
            refresh_run_hash(
                artifacts,
                "split_manifest.json",
                "split_fingerprint",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("strategy", completed.stdout.lower())

    def test_unknown_completed_risk_tier_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["governance"]["risk_tier"] = "banana"
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("risk_tier", completed.stdout.lower())

    def test_core_input_contract_cases_need_real_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            contract = json.loads((artifacts / "inference_test.json").read_text())
            case = next(
                item for item in contract["cases"] if item["name"] == "missing_required"
            )
            case["expected_exit_code"] = 0
            case.pop("stderr_contains")
            write_json(artifacts / "inference_test.json", contract)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing_required", completed.stdout)

    def test_capacity_workflow_requires_separate_inference_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["selection"]["capacity"] = {
                "enabled": True,
                "unit": "day",
                "limit": 10,
                "timezone": "Australia/Perth",
                "cutoff": "09:00",
                "eligibility_rule": "open cases",
                "tie_breaker": "row_id ascending",
                "sub_capacity_behavior": "select every eligible row",
            }
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("score_rows", completed.stdout)
            self.assertIn("select_queue", completed.stdout)

    def test_capacity_workflow_with_separate_cases_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_v21_project(project, capacity=True)
            completed = validate(project, "--run-inference-test")
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_capacity_cases_must_verify_executed_ranks_and_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project, capacity=True)
            contract = json.loads((artifacts / "inference_test.json").read_text())
            queue_case = next(
                item for item in contract["cases"] if item["name"] == "select_queue"
            )
            queue_case["output"].pop("golden_values")
            write_json(artifacts / "inference_test.json", contract)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("selection_rank", completed.stdout)

    def test_pending_labels_require_maturity_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(
                project,
                design="prospective_validation",
                evaluation_status="pending_labels",
            )
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["final"].pop("maturity_rule")
            metrics["final"].pop("cohort_counts")
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("maturity", completed.stdout.lower())

    def test_bare_interval_is_not_an_uncertainty_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["final"].pop("uncertainty")
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("uncertainty", completed.stdout.lower())

    def test_uncertainty_details_must_be_interpretable(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["final"]["uncertainty"]["confidence_level"] = 2.0
            metrics["final"]["uncertainty"]["resampling_unit"] = ""
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("confidence_level", completed.stdout)
            self.assertIn("resampling_unit", completed.stdout)

    def test_uncertainty_metadata_needs_a_quantitative_result(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["final"].pop("confidence_interval")
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("quantitative", completed.stdout.lower())

    def test_grouped_uncertainty_cannot_bootstrap_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["problem"]["group_column"] = "age"
            config["split"]["strategy"] = "grouped"
            write_json(artifacts / "config.json", config)
            split = json.loads((artifacts / "split_manifest.json").read_text())
            split["strategy"] = "grouped"
            split["audits"]["group_overlap"] = {
                "checked": True,
                "group_column": "age",
                "groups_spanning_partitions": 0,
                "null_group_rows": 0,
                "allowed": False,
                "reason": "no groups span partitions",
            }
            write_json(artifacts / "split_manifest.json", split)
            refresh_run_hash(
                artifacts,
                "split_manifest.json",
                "split_fingerprint",
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("group/entity", completed.stdout)

    def test_primary_metric_must_match_selection_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["selection"]["primary_metric"] = "roc_auc"
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("primary_metric", completed.stdout)

    def test_known_metric_direction_and_fold_bounds_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project, design="nested_cv")
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["primary_metric"]["direction"] = "minimize"
            metrics["final"]["fold_scores"] = [-1.0, 2.0]
            metrics["final"]["score"] = 0.5
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("direction must be maximize", completed.stdout)
            self.assertIn("fold_scores", completed.stdout)

    def test_attempted_family_result_must_contain_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            result = metrics["search"]["family_results"][0]
            result["status"] = "whatever"
            result.pop("completed_trials")
            result.pop("best_validation")
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("family_results", completed.stdout)

    def test_best_family_must_come_from_successful_roster_result(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["search"]["best_family"] = "mystery_booster"
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("best_family", completed.stdout)

    def test_candidate_requires_environment_independent_consideration_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["search"]["candidates"][0].pop("consideration_basis")
            write_json(artifacts / "config.json", config)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("consideration_basis", completed.stdout)

    def test_inference_probability_bounds_are_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            source = (artifacts / "infer.py").read_text()
            (artifacts / "infer.py").write_text(
                source.replace(
                    '{"row_id": "B", "probability": 0.8}',
                    '{"row_id": "B", "probability": 5.0}',
                ),
                encoding="utf-8",
            )
            contract = json.loads((artifacts / "inference_test.json").read_text())
            representative = next(
                item
                for item in contract["cases"]
                if item["name"] == "representative_batch"
            )
            representative["output"]["golden_predictions"]["probability"][1] = 5.0
            write_json(artifacts / "inference_test.json", contract)
            completed = validate(project, "--run-inference-test")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("above 1.0", completed.stdout)

    def test_task_identity_must_match_all_machine_readable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["task"] = "time-series"
            write_json(artifacts / "metrics.json", metrics)
            profile = json.loads((artifacts / "data_profile.json").read_text())
            profile["task"] = "anomaly"
            write_json(artifacts / "data_profile.json", profile)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("metrics.json: task", completed.stdout)
            self.assertIn("data_profile.json: task", completed.stdout)

    def test_available_incumbent_needs_comparable_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["baselines"]["incumbent"] = {
                "available": True,
                "name": "current rules",
            }
            write_json(artifacts / "config.json", config)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["baselines"]["incumbent"] = {"note": "exists"}
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("incumbent", completed.stdout.lower())

    def test_available_incumbent_with_comparable_metric_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            config = json.loads((artifacts / "config.json").read_text())
            config["baselines"]["incumbent"] = {
                "available": True,
                "name": "current rules",
            }
            write_json(artifacts / "config.json", config)
            metrics = json.loads((artifacts / "metrics.json").read_text())
            metrics["baselines"]["incumbent"] = {
                "score": 0.3,
                "metric": "average_precision",
                "eval_set": "holdout_test",
                "population_fingerprint": "sha256:" + "e" * 64,
            }
            write_json(artifacts / "metrics.json", metrics)
            completed = validate(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_deployable_model_forms_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = build_v21_project(project)
            model_dir = artifacts / "model"
            model_dir.mkdir()
            model_file = model_dir / "weights.bin"
            model_file.write_bytes(b"weights")
            write_json(
                model_dir / "manifest.json",
                {
                    "files": [
                        {
                            "path": "weights.bin",
                            "sha256": hashlib.sha256(b"weights").hexdigest(),
                        }
                    ]
                },
            )
            completed = validate(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("both", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
