#!/usr/bin/env python3
"""Validate artifact contracts and optionally run declared inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ANALYSIS_FILES = {
    "config.json",
    "data_profile.json",
    "data_report.html",
    "data_summary.md",
    "data_fingerprint.json",
    "schema.json",
}
MODEL_FILES = ANALYSIS_FILES | {
    "train.py",
    "infer.py",
    "metrics.json",
    "feature_manifest.json",
    "model_card.md",
    "inference_test.json",
}
LEGACY_MODEL_FILES = {
    "config.json",
    "train.py",
    "infer.py",
    "model.joblib",
    "metrics.json",
}
VERSIONED_JSON = {
    "config.json",
    "data_profile.json",
    "data_fingerprint.json",
    "schema.json",
    "metrics.json",
    "feature_manifest.json",
}
JSON_FILES = VERSIONED_JSON | {"inference_test.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", default=".")
    parser.add_argument("--artifacts-dir", default="artefacts")
    parser.add_argument(
        "--run-inference-test",
        action="store_true",
        help="Execute inference_test.json argv without a shell",
    )
    parser.add_argument("--inference-timeout-seconds", type=int, default=120)
    return parser.parse_args()


def read_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: invalid JSON ({exc})")
        return None


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_metric_contract(metrics, errors: list[str]):
    primary = metrics.get("primary_metric")
    if not isinstance(primary, dict):
        errors.append("metrics.json: primary_metric object is required")
        return
    name = primary.get("name")
    direction = primary.get("direction")
    if not isinstance(name, str) or not name.strip():
        errors.append("metrics.json: primary_metric.name is required")
    if direction not in {"maximize", "minimize"}:
        errors.append(
            "metrics.json: primary_metric.direction must be maximize or minimize"
        )

    final = metrics.get("final")
    if not isinstance(final, dict):
        errors.append("metrics.json: final object is required")
        return
    score = final.get("score")
    if not is_finite_number(score):
        errors.append("metrics.json: final.score must be a finite number")
    if name and final.get("metric") != name:
        errors.append("metrics.json: final.metric must match primary_metric.name")

    interval = final.get("confidence_interval")
    fold_scores = final.get("fold_scores")
    uncertainty = final.get("uncertainty")
    if interval is not None:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(is_finite_number(value) for value in interval)
            or interval[0] > interval[1]
        ):
            errors.append(
                "metrics.json: final.confidence_interval must be two ordered "
                "finite numbers"
            )
        elif is_finite_number(score) and not interval[0] <= score <= interval[1]:
            errors.append(
                "metrics.json: final.confidence_interval must contain final.score"
            )
    elif (
        not isinstance(fold_scores, list)
        or len(fold_scores) < 2
        or not all(is_finite_number(value) for value in fold_scores)
    ) and not isinstance(uncertainty, dict):
        errors.append(
            "metrics.json: final requires confidence_interval, at least two "
            "fold_scores, or an uncertainty object"
        )

    bounded = {
        "accuracy",
        "average_precision",
        "balanced_accuracy",
        "brier_score",
        "f1",
        "macro_f1",
        "precision",
        "recall",
        "roc_auc",
    }
    non_negative = {
        "log_loss",
        "mae",
        "mape",
        "mean_absolute_error",
        "mean_squared_error",
        "pinball_loss",
        "rmse",
    }
    if name in bounded and is_finite_number(score) and not 0 <= score <= 1:
        errors.append(f"metrics.json: {name} must be between 0 and 1")
    if name in non_negative and is_finite_number(score) and score < 0:
        errors.append(f"metrics.json: {name} must be non-negative")


def validate_evaluation_contract(config, metrics, errors: list[str]):
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        errors.append("config.json: evaluation contract is required")
        return
    design = evaluation.get("design")
    allowed = {"holdout", "nested_cv", "external_test", "prospective_validation"}
    if design not in allowed:
        errors.append(
            "config.json: evaluation.design must be holdout, nested_cv, "
            "external_test, or prospective_validation"
        )
        return
    expected = {
        "holdout": "holdout_test",
        "nested_cv": "outer_cv",
        "external_test": "external_test",
        "prospective_validation": "prospective_validation",
    }[design]
    if evaluation.get("final_eval_set") != expected:
        errors.append(
            f"config.json: evaluation.final_eval_set must be '{expected}' for {design}"
        )
    if nested(metrics, "final", "eval_set") != expected:
        errors.append(f"metrics.json: final.eval_set must be '{expected}' for {design}")
    if design == "holdout":
        if nested(config, "split", "holdout_target_sealed") is not True:
            errors.append("config.json: holdout_target_sealed must be true")
        if evaluation.get("independent_test") is not True:
            errors.append("config.json: holdout must declare independent_test true")
    elif design == "nested_cv":
        if evaluation.get("selection_nested") is not True:
            errors.append("config.json: nested_cv must declare selection_nested true")
        if evaluation.get("independent_test") is not False:
            errors.append("config.json: nested_cv must declare independent_test false")
        fold_scores = nested(metrics, "final", "fold_scores")
        if not isinstance(fold_scores, list) or len(fold_scores) < 2:
            errors.append("metrics.json: nested_cv requires outer fold_scores")
        elif all(is_finite_number(value) for value in fold_scores):
            aggregation = nested(metrics, "final", "aggregation")
            if aggregation not in {"mean", "median"}:
                errors.append(
                    "metrics.json: nested_cv final.aggregation must be mean or median"
                )
            else:
                ordered = sorted(float(value) for value in fold_scores)
                midpoint = len(ordered) // 2
                expected_score = (
                    sum(ordered) / len(ordered)
                    if aggregation == "mean"
                    else (
                        ordered[midpoint]
                        if len(ordered) % 2
                        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
                    )
                )
                score = nested(metrics, "final", "score")
                if is_finite_number(score) and not math.isclose(
                    float(score), expected_score, rel_tol=1e-9, abs_tol=1e-12
                ):
                    errors.append(
                        "metrics.json: final.score does not match the declared "
                        "outer-fold aggregation"
                    )
    else:
        if evaluation.get("independent_test") is not True:
            errors.append(f"config.json: {design} must declare independent_test true")
        if not evaluation.get("cohort_fingerprint"):
            errors.append(
                f"config.json: {design} requires evaluation.cohort_fingerprint"
            )


def validate_high_stakes(config, errors: list[str]):
    governance = config.get("governance")
    if not isinstance(governance, dict):
        errors.append("config.json: governance object is required")
        return
    if governance.get("risk_tier") != "high":
        return
    for field in [
        "domain_owner",
        "human_oversight",
        "deployment_decision",
        "approval_status",
        "prohibited_uses",
    ]:
        if not governance.get(field):
            errors.append(f"config.json: high-stakes governance.{field} is required")
    if governance.get("deployment_decision") == "autonomous":
        if governance.get("approval_status") != "approved":
            errors.append(
                "config.json: autonomous high-stakes deployment requires recorded "
                "approval"
            )
        if governance.get("human_oversight") in {None, "none", False}:
            errors.append(
                "config.json: autonomous high-stakes deployment requires an "
                "explicit oversight/escalation design"
            )


def execute_inference_test(
    project: Path,
    inference_test,
    timeout_seconds: int,
    errors: list[str],
):
    argv = inference_test.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) and value for value in argv)
    ):
        errors.append(
            "inference_test.json: argv must be a non-empty array of strings when "
            "--run-inference-test is used"
        )
        return
    command = [sys.executable if value == "{python}" else value for value in argv]
    try:
        completed = subprocess.run(
            command,
            cwd=project,
            text=True,
            capture_output=True,
            timeout=max(timeout_seconds, 1),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"inference round trip could not run: {exc}")
        return
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-500:]
        errors.append(f"inference round trip exited {completed.returncode}: {detail}")


def validate_pinned_requirements(path: Path, errors: list[str], warnings: list[str]):
    if not path.exists():
        errors.append("missing pinned inference environment: requirements.lock")
        return
    unpinned = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith(("#", "-", "git+", "http://", "https://")):
            continue
        if "==" not in item or item.startswith(("-e ", ".")):
            unpinned.append(item)
    if unpinned:
        errors.append(
            "requirements.lock contains unpinned entries: " + ", ".join(unpinned[:5])
        )
    if not path.read_text(encoding="utf-8").strip():
        warnings.append("requirements.lock is empty")


def validate_model(
    project: Path,
    artifacts: Path,
    documents,
    errors,
    warnings,
    args,
    legacy=False,
):
    config = documents.get("config.json") or {}
    metrics = documents.get("metrics.json") or {}
    schema = documents.get("schema.json") or {}
    manifest = documents.get("feature_manifest.json") or {}
    inference_test = documents.get("inference_test.json") or {}
    if legacy:
        if nested(metrics, "final", "eval_set") != "holdout_test":
            errors.append("legacy metrics.json: final.eval_set must be 'holdout_test'")
        if nested(metrics, "final", "score") is None:
            errors.append("legacy metrics.json: final.score is required")
        if not (project / "results.md").exists():
            warnings.append("legacy run has no project-level results.md")
        warnings.append(
            "legacy v1 validation cannot verify the v2 data, feature, "
            "environment, model-card, or inference contracts"
        )
        return
    task = nested(config, "problem", "task")
    unlabeled_anomaly = (
        task == "anomaly" and nested(config, "problem", "labels_available") is False
    )

    if not nested(config, "problem", "prediction_moment"):
        errors.append("config.json: problem.prediction_moment is required")
    if not task:
        errors.append("config.json: problem.task is required")
    if not nested(config, "split", "assignment_column"):
        errors.append("config.json: split.assignment_column is required")
    if unlabeled_anomaly:
        if nested(config, "analysis", "population_partition") not in {
            "train",
            "reference",
        }:
            errors.append(
                "config.json: unlabeled anomaly analysis requires a historical "
                "population_partition"
            )
        if nested(metrics, "final", "eval_set") not in {
            "future_scoring_window",
            "prospective_review_window",
        }:
            errors.append(
                "metrics.json: unlabeled anomaly final.eval_set must be a future "
                "scoring/review window"
            )
        if nested(metrics, "final", "predictive_performance_available") is not False:
            errors.append(
                "metrics.json: unlabeled anomaly final must state that predictive "
                "performance is unavailable"
            )
        anomaly_evaluation = metrics.get("anomaly_evaluation")
        if not isinstance(anomaly_evaluation, dict):
            errors.append("metrics.json: anomaly_evaluation object is required")
        else:
            if anomaly_evaluation.get("review_capacity", 0) < 1:
                errors.append(
                    "metrics.json: anomaly_evaluation.review_capacity must be positive"
                )
            if (
                anomaly_evaluation.get("unreviewed_rows_treated_as_negative")
                is not False
            ):
                errors.append(
                    "metrics.json: unreviewed anomaly rows must remain unlabeled"
                )
    else:
        development_label = nested(config, "split", "development_label") or "train"
        if nested(config, "analysis", "target_aware_partition") != development_label:
            errors.append(
                "config.json: analysis.target_aware_partition must match "
                "split.development_label"
            )
        validate_evaluation_contract(config, metrics, errors)
        validate_metric_contract(metrics, errors)
    validate_high_stakes(config, errors)
    if not isinstance(manifest.get("raw_input_features"), list):
        errors.append("feature_manifest.json: raw_input_features must be a list")
    if schema.get("partition_column") is None:
        errors.append("schema.json: partition_column is required in model mode")

    if not inference_test.get("command"):
        errors.append("inference_test.json: command is required")
    if inference_test.get("status") not in {"passed", "pass", True}:
        errors.append("inference_test.json: status must record a passed test")
    if inference_test.get("row_count", 0) < 1:
        errors.append("inference_test.json: row_count must be positive")
    if args.run_inference_test:
        execute_inference_test(
            project,
            inference_test,
            args.inference_timeout_seconds,
            errors,
        )
    else:
        warnings.append(
            "inference was not executed; rerun with --run-inference-test before handoff"
        )

    model_path = artifacts / "model.joblib"
    model_manifest = artifacts / "model/manifest.json"
    trusted_path = model_path if model_path.exists() else model_manifest
    if not trusted_path.exists():
        errors.append("missing artefacts/model.joblib or artefacts/model/manifest.json")
    else:
        digest = sha256_file(trusted_path)
        recorded = inference_test.get("trusted_model_sha256")
        if not recorded:
            warnings.append("inference_test.json: trusted_model_sha256 is not recorded")
        elif recorded != digest:
            errors.append(
                "inference_test.json: trusted_model_sha256 does not match the "
                "deployable model artifact"
            )

    if task == "time-series":
        forecast = config.get("forecast")
        if not isinstance(forecast, dict):
            errors.append("config.json: forecast contract is required")
        else:
            for field in ["origin", "horizon", "strategy"]:
                if not forecast.get(field):
                    errors.append(f"config.json: forecast.{field} is required")

    requirements = artifacts / "requirements.lock"
    environment_lock = artifacts / "environment.lock"
    if requirements.exists():
        validate_pinned_requirements(requirements, errors, warnings)
    elif (
        environment_lock.exists()
        and not environment_lock.read_text(encoding="utf-8").strip()
    ):
        errors.append("environment.lock is empty")
    results_path = project / "results.md"
    if not results_path.exists():
        errors.append("missing project-level results.md")
    elif not re.search(
        r"prediction moment", results_path.read_text(encoding="utf-8"), re.IGNORECASE
    ):
        warnings.append("results.md does not mention the prediction moment")


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    artifacts = project / args.artifacts_dir
    errors: list[str] = []
    warnings: list[str] = []

    if not artifacts.is_dir():
        print(f"ERROR: artifact directory does not exist: {artifacts}")
        return 1

    config_path = artifacts / "config.json"
    if not config_path.exists():
        print("ERROR: missing artefacts/config.json")
        return 1
    config = read_json(config_path, errors)
    mode = (config or {}).get("mode", "model-building")
    analysis_only = mode == "analysis-only"
    legacy_run = isinstance(config, dict) and config.get("schema_version") is None
    if legacy_run and not analysis_only:
        required = LEGACY_MODEL_FILES
    else:
        required = ANALYSIS_FILES if analysis_only else MODEL_FILES
    for filename in sorted(required):
        path = artifacts / filename
        if not path.exists():
            errors.append(f"missing artefacts/{filename}")
    if (
        not legacy_run
        and not analysis_only
        and not (
            (artifacts / "requirements.lock").exists()
            or (artifacts / "environment.lock").exists()
        )
    ):
        errors.append("missing pinned inference environment")

    if not legacy_run:
        figures = artifacts / "figures"
        if not figures.is_dir() or not any(figures.glob("*.png")):
            errors.append("artefacts/figures must contain at least one PNG chart")

    documents = {"config.json": config}
    for filename in sorted(required & JSON_FILES):
        path = artifacts / filename
        if path.exists() and filename != "config.json":
            documents[filename] = read_json(path, errors)

    versions = {
        filename: document.get("schema_version")
        for filename, document in documents.items()
        if filename in VERSIONED_JSON and isinstance(document, dict)
    }
    legacy_files = [
        filename for filename, version in versions.items() if version is None
    ]
    unsupported = [
        f"{filename}={version}"
        for filename, version in versions.items()
        if version not in {None, "2.0"}
    ]
    if legacy_files:
        warnings.append(
            "legacy v1 artifacts lack schema_version: " + ", ".join(legacy_files)
        )
    if unsupported:
        errors.append("unsupported schema versions: " + ", ".join(unsupported))

    profile = documents.get("data_profile.json") or {}
    if not legacy_files and profile.get("mode") != (
        "analysis-only" if analysis_only else "model"
    ):
        errors.append("data_profile.json: mode does not match config.json")
    if not analysis_only:
        validate_model(
            project,
            artifacts,
            documents,
            errors,
            warnings,
            args,
            legacy=legacy_run,
        )

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        print(f"Validation failed with {len(errors)} error(s).")
        return 1
    print(
        f"Artifact contract valid for "
        f"{'analysis-only' if analysis_only else 'model'} artifacts "
        f"at {artifacts} ({len(warnings)} warning(s))."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
