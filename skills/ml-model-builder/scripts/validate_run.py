#!/usr/bin/env python3
"""Validate artifact contracts and optionally run declared inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CURRENT_SCHEMA_VERSION = "2.1"
COMPATIBLE_SCHEMA_VERSIONS = {None, "2.0", CURRENT_SCHEMA_VERSION}
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
STRICT_MODEL_FILES = {"split_manifest.json", "run_manifest.json"}
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
    "split_manifest.json",
    "run_manifest.json",
}
JSON_FILES = VERSIONED_JSON | {"inference_test.json"}
CORE_INFERENCE_CASES = {
    "representative_batch",
    "one_row",
    "empty_input",
    "missing_required",
    "extra_columns",
    "wrong_dtypes",
    "unseen_categories",
}
CAPACITY_INFERENCE_CASES = {
    "score_rows",
    "select_queue",
    "capacity_ties",
    "capacity_duplicates",
    "capacity_empty",
    "capacity_sub_capacity",
}
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
PINNED_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?"
    r"==(?P<version>[^;\s]+)(?:\s*;\s*.+)?$"
)
DIRECT_REQUIREMENT_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9._,-]+\])?$"
)
EXACT_VERSION_RE = re.compile(
    r"^(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
    r"(?:(?:[._-]?)(?:a|b|rc|post|dev)[0-9]+)*"
    r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?$",
    re.IGNORECASE,
)


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


def is_nonempty_string(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_nonempty_collection(value) -> bool:
    if isinstance(value, str):
        return is_nonempty_string(value)
    return isinstance(value, (list, dict)) and bool(value)


def is_string_array(value, *, nonempty: bool = False, unique: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (not nonempty or bool(value))
        and all(is_nonempty_string(item) for item in value)
        and (not unique or len(value) == len(set(value)))
    )


def is_nonnegative_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_positive_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_sha256(value) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value.strip()))


def require(condition: bool, message: str, errors: list[str]) -> bool:
    if not condition:
        errors.append(message)
    return condition


def parse_timestamp(value, field: str, errors: list[str]):
    if not is_nonempty_string(value):
        errors.append(f"{field} must be a timezone-aware ISO-8601 timestamp")
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        errors.append(f"{field} must be a timezone-aware ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(f"{field} must include a timezone")
        return None
    return parsed


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


def resolve_project_path(
    project: Path,
    value,
    field: str,
    errors: list[str],
):
    project = project.resolve()
    if not is_nonempty_string(value):
        errors.append(f"{field} must be a non-empty project-relative path")
        return None
    relative = Path(value)
    if relative.is_absolute():
        errors.append(f"{field} must be project-relative")
        return None
    candidate = (project / relative).resolve()
    if candidate != project and project not in candidate.parents:
        errors.append(f"{field} must stay within the project directory")
        return None
    return candidate


def path_is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def path_has_symlink_component(path: Path, directory: Path) -> bool:
    try:
        relative = path.relative_to(directory)
    except ValueError:
        return False
    current = directory
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def resolve_artifact_path(
    project: Path,
    artifacts: Path,
    value,
    field: str,
    errors: list[str],
):
    if not is_nonempty_string(value):
        errors.append(f"{field} must be a non-empty project-relative path")
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{field} must be a project-relative path without '..'")
        return None
    lexical = project / relative
    try:
        candidate = lexical.resolve()
    except (OSError, RuntimeError) as exc:
        errors.append(f"{field} could not be resolved safely ({exc})")
        return None
    if candidate == artifacts or not path_is_within(candidate, artifacts):
        errors.append(f"{field} must stay within the selected artifact directory")
        return None
    if path_has_symlink_component(lexical, project):
        errors.append(f"{field} must not traverse or identify a symlink")
        return None
    return candidate


def validate_metric_contract(
    metrics,
    errors: list[str],
    pending_labels: bool = False,
):
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
    if pending_labels:
        if score is not None:
            errors.append(
                "metrics.json: pending-label prospective validation requires "
                "final.score null"
            )
        if final.get("outcomes_mature") is not False:
            errors.append(
                "metrics.json: pending-label prospective validation requires "
                "final.outcomes_mature false"
            )
        if final.get("validated_performance_available") is not False:
            errors.append(
                "metrics.json: pending-label prospective validation requires "
                "final.validated_performance_available false"
            )
    elif not is_finite_number(score):
        errors.append("metrics.json: final.score must be a finite number")
    if name and final.get("metric") not in ({name, None} if pending_labels else {name}):
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
        not pending_labels
        and (
            not isinstance(fold_scores, list)
            or len(fold_scores) < 2
            or not all(is_finite_number(value) for value in fold_scores)
        )
        and not isinstance(uncertainty, dict)
    ):
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
    maximize_metrics = bounded - {"brier_score"}
    minimize_metrics = non_negative | {"brier_score"}
    if name in maximize_metrics and direction != "maximize":
        errors.append(f"metrics.json: {name} direction must be maximize")
    if name in minimize_metrics and direction != "minimize":
        errors.append(f"metrics.json: {name} direction must be minimize")
    evidence_values = []
    if is_finite_number(score):
        evidence_values.append(("final.score", score))
    if isinstance(interval, list):
        evidence_values.extend(
            (f"final.confidence_interval[{index}]", value)
            for index, value in enumerate(interval)
            if is_finite_number(value)
        )
    if isinstance(fold_scores, list):
        evidence_values.extend(
            (f"final.fold_scores[{index}]", value)
            for index, value in enumerate(fold_scores)
            if is_finite_number(value)
        )
    for field, value in evidence_values:
        if name in bounded and not 0 <= value <= 1:
            errors.append(f"metrics.json: {field} for {name} must be between 0 and 1")
        if name in non_negative and value < 0:
            errors.append(f"metrics.json: {field} for {name} must be non-negative")


def validate_strict_metric_evidence(
    metrics,
    errors: list[str],
    pending_labels: bool,
    config=None,
):
    final = metrics.get("final")
    if not isinstance(final, dict):
        return
    if pending_labels:
        require(
            is_nonempty_string(final.get("maturity_rule")),
            "metrics.json: pending-label prospective validation requires "
            "final.maturity_rule",
            errors,
        )
        counts = final.get("cohort_counts")
        required_counts = {
            "scored",
            "matured",
            "pending",
            "lost_to_follow_up",
        }
        valid_counts = (
            isinstance(counts, dict)
            and required_counts.issubset(counts)
            and all(
                is_nonnegative_integer(counts.get(name)) for name in required_counts
            )
        )
        require(
            valid_counts,
            "metrics.json: pending-label final.cohort_counts requires "
            "non-negative scored, matured, pending, and lost_to_follow_up counts",
            errors,
        )
        if valid_counts:
            require(
                counts["scored"]
                == counts["matured"] + counts["pending"] + counts["lost_to_follow_up"],
                "metrics.json: final.cohort_counts components must sum to scored",
                errors,
            )
        return

    if not is_finite_number(final.get("score")):
        return
    uncertainty = final.get("uncertainty")
    if not isinstance(uncertainty, dict):
        errors.append(
            "metrics.json: completed final score requires an uncertainty object"
        )
        return
    require(
        is_nonempty_string(uncertainty.get("method")),
        "metrics.json: final.uncertainty.method is required",
        errors,
    )
    confidence_level = uncertainty.get("confidence_level")
    require(
        is_finite_number(confidence_level) and 0 < confidence_level < 1,
        "metrics.json: final.uncertainty.confidence_level must be finite and "
        "strictly between 0 and 1",
        errors,
    )
    require(
        is_nonempty_string(uncertainty.get("resampling_unit")),
        "metrics.json: final.uncertainty.resampling_unit is required",
        errors,
    )
    support = uncertainty.get(
        "effective_sample_size",
        uncertainty.get("support"),
    )
    require(
        is_finite_number(support) and support > 0,
        "metrics.json: final.uncertainty requires positive finite support "
        "(effective_sample_size or support)",
        errors,
    )
    interval = final.get("confidence_interval")
    valid_interval = (
        isinstance(interval, list)
        and len(interval) == 2
        and all(is_finite_number(value) for value in interval)
        and interval[0] <= interval[1]
    )
    standard_error = uncertainty.get("standard_error")
    valid_standard_error = is_finite_number(standard_error) and standard_error >= 0
    fold_distribution = final.get("fold_scores")
    valid_fold_distribution = (
        isinstance(fold_distribution, list)
        and len(fold_distribution) >= 2
        and all(is_finite_number(value) for value in fold_distribution)
    )
    require(
        valid_interval or valid_standard_error or valid_fold_distribution,
        "metrics.json: completed final uncertainty requires a quantitative "
        "confidence interval, standard error, or finite fold-score distribution",
        errors,
    )
    if "repetitions" in uncertainty:
        require(
            is_positive_integer(uncertainty.get("repetitions")),
            "metrics.json: final.uncertainty.repetitions must be a positive "
            "integer when present",
            errors,
        )
    if "seed" in uncertainty:
        seed = uncertainty.get("seed")
        require(
            isinstance(seed, int) and not isinstance(seed, bool),
            "metrics.json: final.uncertainty.seed must be an integer when present",
            errors,
        )
    split_strategy = nested(config or {}, "split", "strategy")
    uncertainty_text = " ".join(
        [
            str(uncertainty.get("method", "")),
            str(uncertainty.get("resampling_unit", "")),
        ]
    ).lower()
    if split_strategy in {"grouped", "grouped_temporal"}:
        require(
            any(
                token in uncertainty_text
                for token in (
                    "group",
                    "cluster",
                    "entity",
                    "customer",
                    "account",
                    "patient",
                    "series",
                    "multiway",
                )
            ),
            "metrics.json: grouped evaluation uncertainty must preserve the "
            "group/entity dependence unit",
            errors,
        )
    if split_strategy in {"temporal", "grouped_temporal"}:
        require(
            any(
                token in uncertainty_text
                for token in (
                    "time",
                    "block",
                    "window",
                    "origin",
                    "week",
                    "day",
                    "month",
                    "period",
                    "batch",
                )
            ),
            "metrics.json: temporal evaluation uncertainty must preserve "
            "time/block dependence",
            errors,
        )
    if nested(config or {}, "selection", "capacity", "enabled") is True:
        require(
            uncertainty.get("policy_recomputed_per_resample") is True,
            "metrics.json: fixed-capacity uncertainty must rerun the full queue "
            "selection policy inside every resample",
            errors,
        )
        require(
            uncertainty.get("capacity_unit")
            == nested(config, "selection", "capacity", "unit"),
            "metrics.json: final.uncertainty.capacity_unit must match the "
            "configured selection capacity",
            errors,
        )
        require(
            uncertainty.get("selection_population") == "full_eligible_queue",
            "metrics.json: fixed-capacity uncertainty must resample the full "
            "eligible queue rather than selected rows",
            errors,
        )
        normalized_unit = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(uncertainty.get("resampling_unit", "")).lower(),
        ).strip("_")
        require(
            normalized_unit
            not in {
                "row",
                "rows",
                "record",
                "records",
                "selected_row",
                "selected_rows",
            },
            "metrics.json: fixed-capacity uncertainty must preserve deployment "
            "batch/group/time dependence rather than bootstrap selected rows",
            errors,
        )


def validate_evaluation_contract(
    config,
    metrics,
    errors: list[str],
    strict_21: bool = False,
):
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
    preexposed_final = (
        nested(
            config,
            "analysis",
            "pre_partition_target_exposure",
            "status",
        )
        == "full_population"
        and nested(
            config,
            "analysis",
            "pre_partition_target_exposure",
            "final_population_overlap",
        )
        is True
    )
    if design == "holdout":
        expected_sealed = not preexposed_final
        if nested(config, "split", "holdout_target_sealed") is not expected_sealed:
            errors.append(
                "config.json: holdout_target_sealed must reflect pre-partition "
                "target exposure"
            )
        if evaluation.get("independent_test") is not expected_sealed:
            errors.append(
                "config.json: holdout independent_test must reflect whether its "
                "targets were exposed before partitioning"
            )
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
        expected_independent = not preexposed_final
        if evaluation.get("independent_test") is not expected_independent:
            errors.append(
                f"config.json: {design} independent_test must reflect prior "
                "target exposure"
            )
        if not is_sha256(evaluation.get("cohort_fingerprint")):
            errors.append(
                f"config.json: {design} requires a sha256 evaluation.cohort_fingerprint"
            )
    status = evaluation.get("status")
    if strict_21:
        allowed_statuses = (
            {"pending_labels", "complete", "validated"}
            if design == "prospective_validation"
            else {"complete", "validated"}
        )
        require(
            status in allowed_statuses,
            f"config.json: {design} evaluation.status must be one of "
            + ", ".join(sorted(allowed_statuses)),
            errors,
        )
    pending_labels = design == "prospective_validation" and status == "pending_labels"
    return pending_labels


def validate_high_stakes(
    config,
    errors: list[str],
    strict_21: bool = False,
):
    governance = config.get("governance")
    if not isinstance(governance, dict):
        errors.append("config.json: governance object is required")
        return
    risk_tier = governance.get("risk_tier")
    if strict_21:
        require(
            governance.get("risk_assessed") is True,
            "config.json: governance.risk_assessed must be true",
            errors,
        )
        require(
            is_nonempty_string(governance.get("risk_assessment_rationale")),
            "config.json: governance.risk_assessment_rationale is required",
            errors,
        )
        require(
            isinstance(governance.get("unresolved_hazards"), list),
            "config.json: governance.unresolved_hazards must be an array",
            errors,
        )
    if strict_21 and risk_tier not in {"standard", "high"}:
        errors.append(
            "config.json: governance.risk_tier must be exactly standard or high"
        )
        return
    if not isinstance(risk_tier, str) or risk_tier.strip().lower() != "high":
        return
    required = [
        "domain_owner",
        "human_oversight",
        "deployment_decision",
        "approval_status",
        "prohibited_uses",
    ]
    if strict_21:
        required.extend(
            [
                "critical_harms",
                "label_provenance",
                "validation_plan",
                "appeal_path",
                "incident_owner",
                "monitoring_cadence",
                "rollback_plan",
            ]
        )
    for field in required:
        if not is_nonempty_collection(governance.get(field)):
            errors.append(f"config.json: high-stakes governance.{field} is required")
    if governance.get("deployment_decision") == "autonomous":
        if governance.get("approval_status") != "approved":
            errors.append(
                "config.json: autonomous high-stakes deployment requires recorded "
                "approval"
            )
        oversight = governance.get("human_oversight")
        if (
            oversight is None
            or oversight is False
            or (isinstance(oversight, str) and oversight.strip().lower() == "none")
        ):
            errors.append(
                "config.json: autonomous high-stakes deployment requires an "
                "explicit oversight/escalation design"
            )


def validate_run_manifest(
    config,
    metrics,
    manifest,
    artifacts: Path,
    artifacts_dir_name: str,
    errors: list[str],
):
    if not isinstance(manifest, dict):
        errors.append("run_manifest.json: object is required")
        return
    run_id = manifest.get("run_id")
    run_kind = manifest.get("run_kind")
    require(is_nonempty_string(run_id), "run_manifest.json: run_id is required", errors)
    require(
        run_kind in {"initial", "improvement"},
        "run_manifest.json: run_kind must be initial or improvement",
        errors,
    )
    expected_directory = Path(artifacts_dir_name).as_posix().rstrip("/")
    require(
        manifest.get("artifact_directory") == expected_directory,
        "run_manifest.json: artifact_directory must match --artifacts-dir",
        errors,
    )
    if is_nonempty_string(run_id):
        require(
            Path(expected_directory).name == run_id,
            "run_manifest.json: artifact_directory must be versioned by run_id",
            errors,
        )
    require(
        is_nonempty_string(manifest.get("code_revision")),
        "run_manifest.json: code_revision is required",
        errors,
    )
    for field, filename in [
        ("data_fingerprint", "data_fingerprint.json"),
        ("split_fingerprint", "split_manifest.json"),
    ]:
        digest = manifest.get(field)
        if require(
            is_sha256(digest),
            f"run_manifest.json: {field} must be a sha256 digest",
            errors,
        ):
            path = artifacts / filename
            if path.is_file():
                require(
                    sha256_file(path) == digest.lower().removeprefix("sha256:"),
                    f"run_manifest.json: {field} does not match {filename}",
                    errors,
                )
    parse_timestamp(manifest.get("created_at"), "run_manifest.json: created_at", errors)
    frozen_at = parse_timestamp(
        manifest.get("roster_frozen_at"),
        "run_manifest.json: roster_frozen_at",
        errors,
    )
    parent_hashes = manifest.get("parent_artifact_hashes")
    changes = manifest.get("changes")
    prior_evidence = manifest.get("prior_evidence")
    if not isinstance(prior_evidence, list):
        errors.append("run_manifest.json: prior_evidence must be an array")
        valid_prior_evidence = []
    else:
        valid_prior_evidence = []
        evidence_keys = set()
        for index, item in enumerate(prior_evidence):
            prefix = f"run_manifest.json: prior_evidence[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            source_run_id = item.get("source_run_id")
            final_set = item.get("final_set")
            population = item.get("population_fingerprint")
            require(
                is_nonempty_string(source_run_id),
                f"{prefix}.source_run_id is required",
                errors,
            )
            require(
                is_nonempty_string(final_set),
                f"{prefix}.final_set is required",
                errors,
            )
            require(
                is_sha256(population),
                f"{prefix}.population_fingerprint must be a sha256 digest",
                errors,
            )
            require(
                is_sha256(item.get("run_manifest_sha256")),
                f"{prefix}.run_manifest_sha256 must bind the source run manifest",
                errors,
            )
            require(
                is_sha256(item.get("metrics_sha256")),
                f"{prefix}.metrics_sha256 must bind the source final evidence",
                errors,
            )
            key = (source_run_id, final_set, population)
            require(
                key not in evidence_keys,
                f"{prefix} duplicates an earlier evidence reference",
                errors,
            )
            evidence_keys.add(key)
            status = item.get("status")
            require(
                status in {"sealed", "opened", "benchmark_selection", "pending_labels"},
                f"{prefix}.status is invalid",
                errors,
            )
            values = item.get("values_viewed")
            decisions = item.get("decisions_influenced")
            require(
                is_string_array(values),
                f"{prefix}.values_viewed must be a string array",
                errors,
            )
            require(
                is_string_array(decisions),
                f"{prefix}.decisions_influenced must be a string array",
                errors,
            )
            if status in {"opened", "benchmark_selection"}:
                parse_timestamp(item.get("opened_at"), f"{prefix}.opened_at", errors)
                require(
                    is_nonempty_string(item.get("opened_for")),
                    f"{prefix}.opened_for is required",
                    errors,
                )
                require(
                    is_string_array(values, nonempty=True),
                    f"{prefix}.values_viewed cannot be empty",
                    errors,
                )
                if status == "opened":
                    require(
                        decisions == [],
                        f"{prefix}: opened evidence cannot claim influenced decisions",
                        errors,
                    )
                else:
                    require(
                        is_string_array(decisions, nonempty=True),
                        f"{prefix}: benchmark evidence requires influenced decisions",
                        errors,
                    )
            else:
                require(
                    item.get("opened_at") is None
                    and item.get("opened_for") is None
                    and values == []
                    and decisions == [],
                    f"{prefix}: unopened evidence must have null opening fields "
                    "and empty audit arrays",
                    errors,
                )
            valid_prior_evidence.append(item)
    if run_kind == "initial":
        require(
            manifest.get("parent_run_id") is None,
            "run_manifest.json: initial parent_run_id must be null",
            errors,
        )
        require(
            parent_hashes == {},
            "run_manifest.json: initial parent_artifact_hashes must be empty",
            errors,
        )
        require(
            is_string_array(changes),
            "run_manifest.json: changes must be a string array",
            errors,
        )
        require(
            valid_prior_evidence == [],
            "run_manifest.json: initial run prior_evidence must be empty",
            errors,
        )
    elif run_kind == "improvement":
        parent_run_id = manifest.get("parent_run_id")
        require(
            is_nonempty_string(parent_run_id),
            "run_manifest.json: improvement requires parent_run_id",
            errors,
        )
        if is_nonempty_string(parent_run_id):
            require(
                parent_run_id != run_id,
                "run_manifest.json: parent_run_id must differ from run_id",
                errors,
            )
        valid_hashes = (
            isinstance(parent_hashes, dict)
            and bool(parent_hashes)
            and all(
                is_nonempty_string(path) and is_sha256(digest)
                for path, digest in parent_hashes.items()
            )
        )
        require(
            valid_hashes,
            "run_manifest.json: improvement requires parent artifact sha256 hashes",
            errors,
        )
        if isinstance(parent_hashes, dict):
            require(
                {"config.json", "metrics.json", "run_manifest.json"}.issubset(
                    parent_hashes
                ),
                "run_manifest.json: parent_artifact_hashes must bind the parent "
                "config, metrics, and run manifest",
                errors,
            )
        require(
            is_string_array(changes, nonempty=True),
            "run_manifest.json: improvement requires recorded changes",
            errors,
        )
        require(
            any(
                item.get("source_run_id") == parent_run_id
                for item in valid_prior_evidence
            ),
            "run_manifest.json: improvement prior_evidence must reference its "
            "direct parent run",
            errors,
        )

    exposure = manifest.get("evaluation_exposure")
    opened_at = None
    if not isinstance(exposure, dict):
        errors.append("run_manifest.json: evaluation_exposure object is required")
    else:
        exposure_fields = {
            "status",
            "final_set",
            "population_fingerprint",
            "opened_at",
            "opened_for",
            "values_viewed",
            "decisions_influenced",
        }
        require(
            exposure_fields.issubset(exposure),
            "run_manifest.json: evaluation_exposure is missing required audit fields",
            errors,
        )
        status = exposure.get("status")
        require(
            status in {"sealed", "opened", "benchmark_selection", "pending_labels"},
            "run_manifest.json: invalid evaluation_exposure.status",
            errors,
        )
        require(
            is_nonempty_string(exposure.get("final_set")),
            "run_manifest.json: evaluation_exposure.final_set is required",
            errors,
        )
        require(
            is_sha256(exposure.get("population_fingerprint")),
            "run_manifest.json: evaluation_exposure.population_fingerprint must "
            "be a sha256 digest",
            errors,
        )
        values_viewed = exposure.get("values_viewed")
        decisions = exposure.get("decisions_influenced")
        require(
            is_string_array(values_viewed),
            "run_manifest.json: evaluation_exposure.values_viewed must be a string array",
            errors,
        )
        require(
            is_string_array(decisions),
            "run_manifest.json: evaluation_exposure.decisions_influenced must be a string array",
            errors,
        )
        if status in {"opened", "benchmark_selection"}:
            opened_at = parse_timestamp(
                exposure.get("opened_at"),
                "run_manifest.json: evaluation_exposure.opened_at",
                errors,
            )
            require(
                is_nonempty_string(exposure.get("opened_for")),
                "run_manifest.json: opened evaluation requires opened_for",
                errors,
            )
            require(
                is_string_array(values_viewed, nonempty=True),
                "run_manifest.json: opened evaluation requires values_viewed",
                errors,
            )
            if status == "opened":
                require(
                    decisions == [],
                    "run_manifest.json: opened evaluation must not claim influenced decisions",
                    errors,
                )
            else:
                require(
                    is_string_array(decisions, nonempty=True),
                    "run_manifest.json: benchmark selection requires decisions_influenced",
                    errors,
                )
        else:
            require(
                exposure.get("opened_at") is None,
                "run_manifest.json: unopened evaluation must have opened_at null",
                errors,
            )
            require(
                exposure.get("opened_for") is None,
                "run_manifest.json: unopened evaluation must have opened_for null",
                errors,
            )
            require(
                values_viewed == [],
                "run_manifest.json: unopened evaluation must have no values_viewed",
                errors,
            )
            require(
                decisions == [],
                "run_manifest.json: unopened evaluation must have no decisions_influenced",
                errors,
            )
    if run_kind == "improvement" and isinstance(exposure, dict):
        current_population = exposure.get("population_fingerprint")
        reused = [
            item
            for item in valid_prior_evidence
            if item.get("status") in {"opened", "benchmark_selection"}
            and item.get("population_fingerprint") == current_population
        ]
        require(
            not reused,
            "run_manifest.json: an improvement cannot present previously opened "
            "or benchmark-selected evidence as its current final population",
            errors,
        )
    if frozen_at is not None and opened_at is not None and frozen_at >= opened_at:
        errors.append(
            "run_manifest.json: roster_frozen_at must precede final evaluation opening"
        )

    run_pointer = config.get("run")
    if not isinstance(run_pointer, dict):
        errors.append("config.json: run pointer object is required")
    else:
        expected_manifest = f"{expected_directory}/run_manifest.json"
        require(
            run_pointer.get("manifest") == expected_manifest,
            f"config.json: run.manifest must be '{expected_manifest}'",
            errors,
        )
        require(
            run_pointer.get("run_id") == run_id,
            "config.json: run.run_id must match run_manifest.json",
            errors,
        )
        require(
            run_pointer.get("run_kind") == run_kind,
            "config.json: run.run_kind must match run_manifest.json",
            errors,
        )

    mode = config.get("mode")
    if mode == "model-improvement" or run_kind == "improvement":
        require(
            mode == "model-improvement" and run_kind == "improvement",
            "config.json: model-improvement mode and run_kind improvement must agree",
            errors,
        )

    evaluation = config.get("evaluation")
    search = config.get("search")
    if isinstance(exposure, dict) and isinstance(evaluation, dict):
        require(
            exposure.get("final_set") == evaluation.get("final_eval_set"),
            "run_manifest.json: final_set must match config evaluation",
            errors,
        )
        if evaluation.get("cohort_fingerprint") is not None:
            require(
                exposure.get("population_fingerprint")
                == evaluation.get("cohort_fingerprint"),
                "run_manifest.json: population_fingerprint must match config evaluation",
                errors,
            )
        if evaluation.get("status") == "pending_labels":
            require(
                exposure.get("status") == "pending_labels",
                "run_manifest.json: pending labels must be recorded in evaluation_exposure",
                errors,
            )
        final_score = nested(metrics, "final", "score")
        if is_finite_number(final_score):
            require(
                exposure.get("status") in {"opened", "benchmark_selection"},
                "run_manifest.json: completed numeric final evidence requires "
                "evaluation_exposure.status opened or benchmark_selection",
                errors,
            )
        elif evaluation.get("status") == "pending_labels":
            require(
                exposure.get("status") == "pending_labels",
                "run_manifest.json: pending prospective evidence requires "
                "evaluation_exposure.status pending_labels",
                errors,
            )
    if isinstance(search, dict):
        require(
            search.get("roster_frozen_at") == manifest.get("roster_frozen_at"),
            "config.json: search.roster_frozen_at must match run_manifest.json",
            errors,
        )


def validate_split_manifest(
    config,
    manifest,
    fingerprint,
    schema,
    metrics,
    errors: list[str],
):
    if not isinstance(manifest, dict):
        errors.append("split_manifest.json: object is required")
        return
    strategy = manifest.get("strategy")
    require(
        strategy
        in (
            "random",
            "stratified_random",
            "grouped",
            "temporal",
            "grouped_temporal",
        ),
        "split_manifest.json: strategy must be random, stratified_random, "
        "grouped, temporal, or grouped_temporal",
        errors,
    )
    if is_nonempty_string(strategy):
        require(
            strategy == nested(config, "split", "strategy"),
            "split_manifest.json: strategy must match config split.strategy",
            errors,
        )
    if strategy in {"grouped", "grouped_temporal"}:
        require(
            is_nonempty_string(nested(config, "problem", "group_column")),
            "config.json: grouped split strategy requires problem.group_column",
            errors,
        )
    if strategy in {"temporal", "grouped_temporal"}:
        require(
            is_nonempty_string(nested(config, "problem", "time_column")),
            "config.json: temporal split strategy requires problem.time_column",
            errors,
        )
    group_overlap_policy = nested(config, "split", "group_overlap_policy")
    require(
        group_overlap_policy
        in {"disallow", "known_series_temporal", "known_entity_temporal"},
        "config.json: split.group_overlap_policy must be disallow, "
        "known_series_temporal, or known_entity_temporal",
        errors,
    )
    if group_overlap_policy == "known_series_temporal":
        require(
            nested(config, "problem", "task") == "time-series"
            and strategy == "grouped_temporal",
            "config.json: split.group_overlap_policy known_series_temporal is "
            "valid only for time-series grouped_temporal evaluation",
            errors,
        )
    if group_overlap_policy == "known_entity_temporal":
        require(
            nested(config, "problem", "task") in {"classification", "regression"}
            and strategy in {"temporal", "grouped_temporal"},
            "config.json: split.group_overlap_policy known_entity_temporal is "
            "valid only for temporally evaluated classification or regression",
            errors,
        )
    assignment = manifest.get("assignment")
    if not isinstance(assignment, dict):
        errors.append("split_manifest.json: assignment object is required")
    else:
        column = assignment.get("column")
        require(
            is_nonempty_string(column),
            "split_manifest.json: assignment.column is required",
            errors,
        )
        if is_nonempty_string(column):
            require(
                column == nested(config, "split", "assignment_column"),
                "split_manifest.json: assignment.column must match config",
                errors,
            )
        require(
            is_nonempty_string(assignment.get("source")),
            "split_manifest.json: assignment.source is required",
            errors,
        )
        assignment_fingerprint = assignment.get("fingerprint")
        if not isinstance(assignment_fingerprint, dict):
            errors.append(
                "split_manifest.json: assignment.fingerprint object is required"
            )
        else:
            require(
                assignment_fingerprint.get("algorithm") == "sha256",
                "split_manifest.json: assignment fingerprint algorithm must be sha256",
                errors,
            )
            require(
                is_sha256(assignment_fingerprint.get("value")),
                "split_manifest.json: assignment fingerprint value must be sha256",
                errors,
            )
            require(
                is_nonempty_string(assignment_fingerprint.get("basis")),
                "split_manifest.json: assignment fingerprint basis is required",
                errors,
            )

    partitions = manifest.get("partitions")
    valid_partitions: list[dict] = []
    allowed_roles = {
        "development",
        "final_evaluation",
        "outer_evaluation",
        "discovery_excluded",
    }
    if not isinstance(partitions, list) or len(partitions) < 2:
        errors.append(
            "split_manifest.json: partitions must contain at least two entries"
        )
    else:
        names = set()
        for index, partition in enumerate(partitions):
            prefix = f"split_manifest.json: partitions[{index}]"
            if not isinstance(partition, dict):
                errors.append(f"{prefix} must be an object")
                continue
            name = partition.get("name")
            require(is_nonempty_string(name), f"{prefix}.name is required", errors)
            if is_nonempty_string(name):
                require(name not in names, f"{prefix}.name must be unique", errors)
                require(
                    name != "<missing>",
                    f"{prefix}.name cannot represent missing split assignments",
                    errors,
                )
                names.add(name)
            require(
                partition.get("role") in allowed_roles,
                f"{prefix}.role must be development, final_evaluation, "
                "outer_evaluation, or discovery_excluded",
                errors,
            )
            require(
                is_positive_integer(partition.get("rows")),
                f"{prefix}.rows must be positive",
                errors,
            )
            valid_partitions.append(partition)

        total_rows = sum(
            partition["rows"]
            for partition in valid_partitions
            if is_positive_integer(partition.get("rows"))
        )
        fingerprint_rows = fingerprint.get("rows")
        if is_positive_integer(fingerprint_rows):
            require(
                total_rows == fingerprint_rows,
                "split_manifest.json: partition rows must sum to "
                "data_fingerprint.json rows",
                errors,
            )

    evaluation_design = nested(config, "evaluation", "design")
    folds = manifest.get("folds")
    if evaluation_design == "nested_cv":
        require(
            isinstance(folds, list) and len(folds) >= 2,
            "split_manifest.json: nested_cv requires at least two outer folds",
            errors,
        )
    valid_folds: list[dict] = []
    if folds is not None:
        if not isinstance(folds, list):
            errors.append("split_manifest.json: folds must be an array")
        else:
            fold_ids = set()
            for index, fold in enumerate(folds):
                prefix = f"split_manifest.json: folds[{index}]"
                if not isinstance(fold, dict):
                    errors.append(f"{prefix} must be an object")
                    continue
                fold_id = fold.get("id")
                valid_id = is_nonempty_string(fold_id)
                require(valid_id, f"{prefix}.id is required", errors)
                if valid_id:
                    require(
                        fold_id not in fold_ids, f"{prefix}.id must be unique", errors
                    )
                    fold_ids.add(fold_id)
                require(
                    fold.get("role") == "outer_evaluation",
                    f"{prefix}.role must be outer_evaluation",
                    errors,
                )
                require(
                    is_positive_integer(fold.get("rows")),
                    f"{prefix}.rows must be positive",
                    errors,
                )
                valid_folds.append(fold)

    if evaluation_design == "nested_cv":
        require(
            strategy not in {"temporal", "grouped_temporal"},
            "split_manifest.json: generic nested_cv cannot use temporal split "
            "mechanics because outer training is represented as fold complements",
            errors,
        )
        outer_partitions = [
            partition
            for partition in valid_partitions
            if partition.get("role") == "outer_evaluation"
        ]
        disallowed = [
            partition
            for partition in valid_partitions
            if partition.get("role") not in {"outer_evaluation", "discovery_excluded"}
        ]
        require(
            not disallowed,
            "split_manifest.json: nested_cv partitions may only use "
            "outer_evaluation or discovery_excluded roles",
            errors,
        )
        require(
            nested(config, "split", "development_label") is None,
            "config.json: nested_cv split.development_label must be null because "
            "outer training is the complement of each outer fold",
            errors,
        )
        require(
            len(outer_partitions) >= 2,
            "split_manifest.json: nested_cv requires at least two "
            "outer_evaluation partitions",
            errors,
        )
        require(
            len(outer_partitions) == len(valid_folds),
            "split_manifest.json: nested outer partition and fold counts must agree",
            errors,
        )
        partition_rows = {
            partition.get("name"): partition.get("rows")
            for partition in outer_partitions
            if is_nonempty_string(partition.get("name"))
        }
        fold_rows = {
            fold.get("id"): fold.get("rows")
            for fold in valid_folds
            if is_nonempty_string(fold.get("id"))
        }
        require(
            set(partition_rows) == set(fold_rows),
            "split_manifest.json: nested outer partition names must exactly "
            "match fold IDs",
            errors,
        )
        if set(partition_rows) == set(fold_rows):
            for fold_id, rows in fold_rows.items():
                require(
                    rows == partition_rows[fold_id],
                    f"split_manifest.json: fold {fold_id!r} rows must match its "
                    "outer partition",
                    errors,
                )
        fold_scores = nested(metrics, "final", "fold_scores")
        if isinstance(fold_scores, list) and isinstance(folds, list):
            require(
                len(fold_scores) == len(valid_folds),
                "metrics.json: final.fold_scores length must match "
                "split_manifest outer folds",
                errors,
            )
            require(
                bool(fold_scores)
                and all(is_finite_number(score) for score in fold_scores),
                "metrics.json: final.fold_scores must contain only finite numbers",
                errors,
            )
    else:
        development_label = nested(config, "split", "development_label")
        development = [
            partition
            for partition in valid_partitions
            if partition.get("role") == "development"
        ]
        final_partitions = [
            partition
            for partition in valid_partitions
            if partition.get("role") == "final_evaluation"
        ]
        require(
            len(development) == 1 and development[0].get("name") == development_label,
            "split_manifest.json: non-nested evaluation requires exactly one "
            "development partition named by config split.development_label",
            errors,
        )
        require(
            len(final_partitions) == 1,
            "split_manifest.json: non-nested evaluation requires exactly one "
            "final_evaluation partition",
            errors,
        )
        require(
            not any(
                partition.get("role") in {"outer_evaluation", "discovery_excluded"}
                for partition in valid_partitions
            ),
            "split_manifest.json: non-nested evaluation cannot contain outer "
            "evaluation or discovery-excluded partitions",
            errors,
        )

    audits = manifest.get("audits")
    if not isinstance(audits, dict):
        errors.append("split_manifest.json: audits object is required")
        return
    for name in ["group_overlap", "temporal_order", "duplicate_overlap"]:
        audit = audits.get(name)
        prefix = f"split_manifest.json: audits.{name}"
        if not isinstance(audit, dict):
            errors.append(f"{prefix} object is required")
            continue
        checked = audit.get("checked")
        if not isinstance(checked, bool):
            errors.append(f"{prefix}.checked must be boolean")
            continue
        if name == "group_overlap":
            require(
                isinstance(audit.get("allowed"), bool),
                f"{prefix}.allowed must be boolean",
                errors,
            )
            if group_overlap_policy not in {
                "known_series_temporal",
                "known_entity_temporal",
            }:
                require(
                    audit.get("allowed") is False,
                    f"{prefix}.allowed must be false unless "
                    "a known-entity temporal policy is declared",
                    errors,
                )
        if not checked:
            require(
                is_nonempty_string(audit.get("reason")),
                f"{prefix}.reason is required when unchecked",
                errors,
            )
            if name == "duplicate_overlap":
                errors.append(
                    f"{prefix} must be checked before a completed model handoff"
                )
            continue
        if name == "group_overlap":
            spanning = audit.get("groups_spanning_partitions")
            require(
                is_nonnegative_integer(spanning),
                f"{prefix}.groups_spanning_partitions must be non-negative",
                errors,
            )
            null_groups = audit.get("null_group_rows")
            require(
                is_nonnegative_integer(null_groups),
                f"{prefix}.null_group_rows must be non-negative",
                errors,
            )
            if strategy in {"grouped", "grouped_temporal"} or group_overlap_policy in {
                "known_series_temporal",
                "known_entity_temporal",
            }:
                require(
                    null_groups == 0,
                    f"{prefix}: grouped evaluation cannot contain missing group IDs",
                    errors,
                )
            if group_overlap_policy in {
                "known_series_temporal",
                "known_entity_temporal",
            }:
                require(
                    audit.get("allowed") is True,
                    f"{prefix}.allowed must be true for {group_overlap_policy}",
                    errors,
                )
                if is_nonnegative_integer(spanning) and spanning:
                    require(
                        is_nonempty_string(audit.get("reason")),
                        f"{prefix}.reason is required for allowed overlap",
                        errors,
                    )
            else:
                if is_nonnegative_integer(spanning):
                    require(
                        spanning == 0,
                        f"{prefix} found disallowed overlap",
                        errors,
                    )
        elif name == "temporal_order":
            require(audit.get("valid") is True, f"{prefix}.valid must be true", errors)
            if strategy in {"temporal", "grouped_temporal"}:
                require(
                    is_nonnegative_integer(audit.get("invalid_timestamp_rows"))
                    and audit.get("invalid_timestamp_rows") == 0,
                    f"{prefix}: temporal evaluation cannot contain missing or "
                    "unparseable timestamps",
                    errors,
                )
        else:
            crossing = audit.get("rows_crossing_partitions")
            require(
                is_nonnegative_integer(crossing),
                f"{prefix}.rows_crossing_partitions must be non-negative",
                errors,
            )
            if is_nonnegative_integer(crossing):
                require(
                    crossing == 0, f"{prefix} found cross-partition duplicates", errors
                )
    blockers = manifest.get("blockers")
    if blockers is not None:
        require(
            blockers == [],
            "split_manifest.json: unresolved profiler blockers must be empty",
            errors,
        )

    group_audit = audits.get("group_overlap")
    group_required = bool(nested(config, "problem", "group_column")) or (
        isinstance(strategy, str) and "group" in strategy.lower()
    )
    if group_required and isinstance(group_audit, dict):
        require(
            group_audit.get("checked") is True,
            "split_manifest.json: grouped evaluation requires a checked group audit",
            errors,
        )
    temporal_audit = audits.get("temporal_order")
    temporal_required = strategy in {"temporal", "grouped_temporal"}
    if temporal_required and isinstance(temporal_audit, dict):
        require(
            temporal_audit.get("checked") is True,
            "split_manifest.json: temporal evaluation requires a checked temporal audit",
            errors,
        )
        time_column = nested(config, "problem", "time_column")
        time_specification = (
            schema.get("columns", {}).get(time_column)
            if isinstance(schema.get("columns"), dict)
            else None
        )
        require(
            isinstance(time_specification, dict)
            and time_specification.get("semantic_type") == "datetime",
            "schema.json: temporal split column must have datetime semantic_type",
            errors,
        )
        ranges = temporal_audit.get("ranges")
        require(
            isinstance(ranges, list) and len(ranges) == len(valid_partitions),
            "split_manifest.json: temporal audit ranges must cover every partition",
            errors,
        )
        parsed_ranges: dict[str, tuple[datetime, datetime]] = {}
        if isinstance(ranges, list):
            partition_rows = {
                partition.get("name"): partition.get("rows")
                for partition in valid_partitions
                if is_nonempty_string(partition.get("name"))
            }
            for index, item in enumerate(ranges):
                prefix = f"split_manifest.json: audits.temporal_order.ranges[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{prefix} must be an object")
                    continue
                name = item.get("name")
                require(
                    is_nonempty_string(name)
                    and name in partition_rows
                    and name not in parsed_ranges,
                    f"{prefix}.name must uniquely match a partition",
                    errors,
                )
                require(
                    is_nonempty_string(name)
                    and item.get("rows") == partition_rows.get(name),
                    f"{prefix}.rows must match its partition",
                    errors,
                )
                start = parse_timestamp(item.get("start"), f"{prefix}.start", errors)
                end = parse_timestamp(item.get("end"), f"{prefix}.end", errors)
                if start is not None and end is not None:
                    require(
                        start <= end,
                        f"{prefix}.start cannot follow end",
                        errors,
                    )
                    if is_nonempty_string(name):
                        parsed_ranges[name] = (start, end)
            if set(parsed_ranges) == set(partition_rows):
                development_label = nested(config, "split", "development_label")
                development_range = parsed_ranges.get(development_label)
                evaluation_names = [
                    partition.get("name")
                    for partition in valid_partitions
                    if partition.get("role") == "final_evaluation"
                ]
                require(
                    development_range is not None and bool(evaluation_names),
                    "split_manifest.json: temporal audit needs development and "
                    "final-evaluation ranges",
                    errors,
                )
                if development_range is not None:
                    for evaluation_name in evaluation_names:
                        require(
                            development_range[1] < parsed_ranges[evaluation_name][0],
                            "split_manifest.json: temporal development rows must "
                            "end strictly before every final-evaluation range",
                            errors,
                        )


def validate_nested_discovery(config, split_manifest, errors: list[str]):
    if nested(config, "evaluation", "design") != "nested_cv":
        return
    target_partition = nested(config, "analysis", "target_aware_partition")
    partitions = split_manifest.get("partitions")
    discovery_partitions = (
        [
            partition
            for partition in partitions
            if isinstance(partition, dict)
            and partition.get("role") == "discovery_excluded"
        ]
        if isinstance(partitions, list)
        else []
    )
    if target_partition is None:
        require(
            not discovery_partitions,
            "split_manifest.json: discovery_excluded partition requires "
            "config analysis.target_aware_partition",
            errors,
        )
        return
    require(
        nested(config, "analysis", "discovery_excluded_from_outer") is True,
        "config.json: nested-CV discovery requires discovery_excluded_from_outer true",
        errors,
    )
    discovery = [
        partition
        for partition in discovery_partitions
        if partition.get("name") == target_partition
    ]
    require(
        len(discovery_partitions) == 1
        and len(discovery) == 1
        and is_sha256(discovery[0].get("fingerprint")),
        "split_manifest.json: nested-CV target-aware discovery needs one "
        "fingerprinted discovery_excluded partition",
        errors,
    )


def validate_pre_partition_target_exposure(config, run_manifest, errors: list[str]):
    exposure = nested(config, "analysis", "pre_partition_target_exposure")
    if not isinstance(exposure, dict):
        errors.append(
            "config.json: analysis.pre_partition_target_exposure object is required"
        )
        return
    status = exposure.get("status")
    require(
        status in {"none", "development_only", "full_population"},
        "config.json: pre_partition_target_exposure.status must be none, "
        "development_only, or full_population",
        errors,
    )
    overlap = exposure.get("final_population_overlap")
    require(
        isinstance(overlap, bool),
        "config.json: pre_partition_target_exposure.final_population_overlap "
        "must be boolean",
        errors,
    )
    values = exposure.get("values_viewed")
    decisions = exposure.get("decisions_influenced")
    require(
        is_string_array(values),
        "config.json: pre_partition_target_exposure.values_viewed must be a "
        "string array",
        errors,
    )
    require(
        is_string_array(decisions),
        "config.json: pre_partition_target_exposure.decisions_influenced must "
        "be a string array",
        errors,
    )
    if status == "none":
        require(
            exposure.get("source") is None
            and overlap is False
            and values == []
            and decisions == [],
            "config.json: no pre-partition target exposure requires null source, "
            "no final overlap, and empty audit arrays",
            errors,
        )
        return
    require(
        is_nonempty_string(exposure.get("source")),
        "config.json: prior target exposure requires a source",
        errors,
    )
    require(
        is_string_array(values, nonempty=True),
        "config.json: prior target exposure requires values_viewed",
        errors,
    )
    if status == "development_only":
        require(
            overlap is False,
            "config.json: development-only target exposure cannot overlap the "
            "final evaluation population",
            errors,
        )
        return
    if overlap is False:
        require(
            nested(config, "evaluation", "design")
            in {"external_test", "prospective_validation"},
            "config.json: full-population prior target exposure needs a disjoint "
            "external or prospective final evaluation",
            errors,
        )
    elif overlap is True:
        require(
            nested(config, "evaluation", "independent_test") is False,
            "config.json: a final population exposed before partitioning cannot "
            "be marked independent",
            errors,
        )
        current_exposure = run_manifest.get("evaluation_exposure")
        require(
            isinstance(current_exposure, dict)
            and current_exposure.get("status") == "benchmark_selection",
            "run_manifest.json: overlapping pre-partition target exposure must "
            "mark the current evaluation as benchmark_selection",
            errors,
        )
        require(
            is_string_array(decisions, nonempty=True),
            "config.json: overlapping full-population target exposure requires "
            "recorded decisions_influenced",
            errors,
        )


def normalize_family(value) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def validate_supervised_candidates(config, metrics, errors: list[str]):
    search = config.get("search")
    if not isinstance(search, dict):
        errors.append("config.json: supervised schema 2.1 runs require a search object")
        return
    parse_timestamp(
        search.get("roster_frozen_at"),
        "config.json: search.roster_frozen_at",
        errors,
    )
    candidates = search.get("candidates")
    if not isinstance(candidates, list):
        errors.append("config.json: search.candidates must be an array")
        return
    suitability_values = {"eligible", "excluded"}
    dependency_values = {
        "installed",
        "installed_for_run",
        "not_required",
        "installation_failed",
        "user_declined",
    }
    execution_values = {
        "attempted",
        "excluded",
        "installation_failed",
        "user_declined",
        "deferred_by_budget",
    }
    roster: dict[str, dict] = {}
    for index, candidate in enumerate(candidates):
        prefix = f"config.json: search.candidates[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix} must be an object")
            continue
        family = normalize_family(candidate.get("family"))
        if not family:
            errors.append(f"{prefix}.family is required")
            continue
        if family in roster:
            errors.append(f"{prefix}.family duplicates another candidate")
            continue
        roster[family] = candidate
        suitability = candidate.get("suitability_status")
        dependency = candidate.get("dependency_status")
        execution = candidate.get("execution_status")
        require(
            is_nonempty_string(candidate.get("consideration_basis")),
            f"{prefix}.consideration_basis must explain task/data/deployment fit "
            "independently of installed packages",
            errors,
        )
        require(
            suitability in suitability_values,
            f"{prefix}.suitability_status is invalid",
            errors,
        )
        require(
            dependency in dependency_values,
            f"{prefix}.dependency_status is invalid",
            errors,
        )
        require(
            execution in execution_values,
            f"{prefix}.execution_status is invalid",
            errors,
        )
        reason = candidate.get("reason")
        if execution != "attempted":
            require(
                is_nonempty_string(reason),
                f"{prefix}.reason is required when not attempted",
                errors,
            )
        if suitability == "excluded":
            require(
                execution == "excluded",
                f"{prefix}: unsuitable family must be excluded",
                errors,
            )
            require(
                dependency == "not_required",
                f"{prefix}: suitability must be independent of dependencies",
                errors,
            )
            if is_nonempty_string(reason) and re.search(
                r"\b(dependenc|install|package|import|module)\w*\b",
                reason,
                re.IGNORECASE,
            ):
                errors.append(
                    f"{prefix}: installed-package availability cannot define suitability"
                )
        elif suitability == "eligible":
            require(
                execution != "excluded",
                f"{prefix}: eligible family cannot be excluded",
                errors,
            )
            available = dependency in {"installed", "installed_for_run", "not_required"}
            if execution in {"attempted", "deferred_by_budget"}:
                require(
                    available,
                    f"{prefix}: execution status requires an available dependency",
                    errors,
                )
            if execution in {"installation_failed", "user_declined"}:
                require(
                    dependency == execution,
                    f"{prefix}: dependency and execution status must agree",
                    errors,
                )

    required_families = {"xgboost", "lightgbm", "catboost"}
    missing = sorted(required_families - roster.keys())
    if missing:
        errors.append(
            "config.json: search.candidates must consider XGBoost, LightGBM, "
            "and CatBoost; missing " + ", ".join(missing)
        )

    family_results = nested(metrics, "search", "family_results")
    if not isinstance(family_results, list):
        errors.append("metrics.json: search.family_results must be an array")
        return
    results: dict[str, dict] = {}
    result_statuses = {"attempted", "failed"}
    for index, result in enumerate(family_results):
        prefix = f"metrics.json: search.family_results[{index}]"
        if not isinstance(result, dict):
            errors.append(f"{prefix} must be an object")
            continue
        family = normalize_family(result.get("family"))
        if not family:
            errors.append(f"{prefix}.family is required")
            continue
        if family in results:
            errors.append(f"{prefix}.family duplicates another family result")
            continue
        results[family] = result
        status = result.get("status")
        require(
            status in result_statuses,
            f"{prefix}.status must be attempted or failed",
            errors,
        )
        completed_trials = result.get("completed_trials")
        best_validation = result.get("best_validation")
        if status == "attempted":
            require(
                is_positive_integer(completed_trials),
                f"{prefix}.completed_trials must be positive for a successful attempt",
                errors,
            )
            require(
                is_finite_number(best_validation),
                f"{prefix}.best_validation must be finite for a successful attempt",
                errors,
            )
        elif status == "failed":
            require(
                is_nonempty_string(result.get("reason")),
                f"{prefix}.reason is required for a failed attempt",
                errors,
            )
            if completed_trials is not None:
                require(
                    is_nonnegative_integer(completed_trials),
                    f"{prefix}.completed_trials must be non-negative",
                    errors,
                )
            if best_validation is not None:
                require(
                    is_finite_number(best_validation),
                    f"{prefix}.best_validation must be finite",
                    errors,
                )
    attempted = {
        family
        for family, candidate in roster.items()
        if candidate.get("execution_status") == "attempted"
    }
    missing_results = sorted(attempted - results.keys())
    if missing_results:
        errors.append(
            "metrics.json: missing attempted family results: "
            + ", ".join(missing_results)
        )
    unknown_results = sorted(results.keys() - roster.keys())
    if unknown_results:
        errors.append(
            "metrics.json: family results absent from candidate roster: "
            + ", ".join(unknown_results)
        )
    unattempted_results = sorted(
        family
        for family in results.keys() & roster.keys()
        if roster[family].get("execution_status") != "attempted"
    )
    if unattempted_results:
        errors.append(
            "metrics.json: results exist for non-attempted families: "
            + ", ".join(unattempted_results)
        )

    best_family_value = nested(metrics, "search", "best_family")
    if best_family_value is not None:
        best_family = normalize_family(best_family_value)
        require(
            bool(best_family) and best_family in roster,
            "metrics.json: search.best_family must name a candidate roster family",
            errors,
        )
        best_result = results.get(best_family)
        require(
            isinstance(best_result, dict)
            and best_result.get("status") == "attempted"
            and is_positive_integer(best_result.get("completed_trials"))
            and is_finite_number(best_result.get("best_validation")),
            "metrics.json: search.best_family must have a successful family result",
            errors,
        )


def validate_incumbent_baseline(config, metrics, run_manifest, errors: list[str]):
    incumbent = nested(config, "baselines", "incumbent")
    if not isinstance(incumbent, dict):
        errors.append(
            "config.json: baselines.incumbent must record whether an incumbent "
            "was available"
        )
        return
    available = incumbent.get("available")
    if not isinstance(available, bool):
        errors.append("config.json: baselines.incumbent.available must be boolean")
        return
    result = nested(metrics, "baselines", "incumbent")
    if available:
        if not is_nonempty_string(incumbent.get("name")):
            errors.append("config.json: available incumbent baseline requires a name")
        if not isinstance(result, dict) or not result:
            errors.append(
                "metrics.json: an available incumbent requires a baseline result"
            )
        else:
            primary_name = nested(metrics, "primary_metric", "name")
            require(
                is_finite_number(result.get("score")),
                "metrics.json: available incumbent score must be finite",
                errors,
            )
            require(
                is_nonempty_string(primary_name)
                and result.get("metric") == primary_name,
                "metrics.json: incumbent metric must match primary_metric.name",
                errors,
            )
            require(
                is_nonempty_string(result.get("eval_set"))
                and result.get("eval_set") == nested(metrics, "final", "eval_set"),
                "metrics.json: incumbent eval_set must match the candidate final "
                "evaluation set",
                errors,
            )
            expected_population = nested(
                run_manifest,
                "evaluation_exposure",
                "population_fingerprint",
            )
            require(
                is_nonempty_string(result.get("population_fingerprint"))
                and result.get("population_fingerprint") == expected_population,
                "metrics.json: incumbent population_fingerprint must match the "
                "candidate final evaluation population",
                errors,
            )
    else:
        if not is_nonempty_string(incumbent.get("reason")):
            errors.append(
                "config.json: unavailable incumbent baseline requires a reason"
            )
        metric_baselines = metrics.get("baselines")
        if isinstance(metric_baselines, dict) and "incumbent" in metric_baselines:
            errors.append(
                "metrics.json: incumbent result must be absent when no incumbent "
                "was available"
            )


def validate_capacity_selection(config, errors: list[str]):
    selection = config.get("selection")
    if not isinstance(selection, dict) or "capacity" not in selection:
        return
    capacity = selection.get("capacity")
    if not isinstance(capacity, dict):
        errors.append("config.json: selection.capacity must be an object")
        return
    enabled = capacity.get("enabled")
    if not isinstance(enabled, bool):
        errors.append("config.json: selection.capacity.enabled must be boolean")
        return
    if not enabled:
        return
    for field in [
        "unit",
        "timezone",
        "cutoff",
        "eligibility_rule",
        "tie_breaker",
        "sub_capacity_behavior",
    ]:
        if not is_nonempty_string(capacity.get(field)):
            errors.append(
                f"config.json: enabled selection.capacity.{field} is required"
            )
    if not is_positive_integer(capacity.get("limit")):
        errors.append(
            "config.json: enabled selection.capacity.limit must be a positive integer"
        )


def validate_probability_specification(
    value,
    prefix: str,
    schema_columns,
    errors: list[str],
    *,
    allow_null: bool,
):
    if value is None and allow_null:
        return
    if not isinstance(value, dict):
        errors.append(
            f"{prefix} must be an object" + (" or null" if allow_null else "")
        )
        return
    has_column = is_nonempty_string(value.get("column"))
    has_formula = is_nonempty_string(value.get("formula"))
    require(
        has_column != has_formula,
        f"{prefix} needs exactly one of column or formula",
        errors,
    )
    require(
        is_nonempty_string(value.get("scope")),
        f"{prefix}.scope is required",
        errors,
    )
    if has_column and isinstance(schema_columns, dict):
        require(
            value["column"] in schema_columns,
            f"{prefix} column is absent from schema.json",
            errors,
        )


def validate_selective_label_contract(config, cohort, schema, errors: list[str]):
    selective = cohort.get("selective_labels")
    if not isinstance(selective, bool):
        errors.append("config.json: problem.cohort.selective_labels must be boolean")
        return
    acquisition = cohort.get("label_acquisition")
    if not selective:
        require(
            acquisition is None,
            "config.json: label_acquisition must be null when labels are not selective",
            errors,
        )
        return
    if not isinstance(acquisition, dict):
        errors.append(
            "config.json: selective labels require problem.cohort.label_acquisition"
        )
        return
    schema_columns = schema.get("columns")
    development = acquisition.get("development")
    evaluation = acquisition.get("evaluation")
    if not isinstance(development, dict):
        errors.append("config.json: label_acquisition.development object is required")
    else:
        require(
            is_nonempty_string(development.get("mechanism")),
            "config.json: label_acquisition.development.mechanism is required",
            errors,
        )
        require(
            development.get("positivity")
            in {"satisfied", "violated", "unknown", "not_applicable"},
            "config.json: label_acquisition.development.positivity is invalid",
            errors,
        )
        validate_probability_specification(
            development.get("selection_probability"),
            "config.json: label_acquisition.development.selection_probability",
            schema_columns,
            errors,
            allow_null=True,
        )
    if not isinstance(evaluation, dict):
        errors.append("config.json: label_acquisition.evaluation object is required")
        return
    design = evaluation.get("design")
    require(
        design
        in {
            "complete_followup",
            "randomized_audit",
            "randomized_experiment",
            "interleaved_comparison",
            "independent_ascertainment",
            "off_policy_with_support",
            "observed_subset_only",
        },
        "config.json: label_acquisition.evaluation.design is invalid",
        errors,
    )
    positivity = evaluation.get("positivity")
    require(
        positivity in {"satisfied", "violated", "unknown", "not_applicable"},
        "config.json: label_acquisition.evaluation.positivity is invalid",
        errors,
    )
    require(
        is_nonempty_string(evaluation.get("selection_unit")),
        "config.json: label_acquisition.evaluation.selection_unit is required",
        errors,
    )
    require(
        is_nonempty_string(evaluation.get("claim_scope")),
        "config.json: label_acquisition.evaluation.claim_scope is required",
        errors,
    )
    population_support = evaluation.get("supports_population_performance")
    calibration_support = evaluation.get("supports_probability_calibration")
    require(
        isinstance(population_support, bool),
        "config.json: label_acquisition.evaluation."
        "supports_population_performance must be boolean",
        errors,
    )
    require(
        isinstance(calibration_support, bool),
        "config.json: label_acquisition.evaluation."
        "supports_probability_calibration must be boolean",
        errors,
    )
    require(
        population_support == cohort.get("evaluation_representative"),
        "config.json: selective-label population-support declaration must match "
        "cohort.evaluation_representative",
        errors,
    )
    require(
        calibration_support == cohort.get("calibration_representative"),
        "config.json: selective-label calibration-support declaration must match "
        "cohort.calibration_representative",
        errors,
    )
    evaluation_probability = evaluation.get("selection_probability")
    validate_probability_specification(
        evaluation_probability,
        "config.json: label_acquisition.evaluation.selection_probability",
        schema_columns,
        errors,
        allow_null=True,
    )
    if design == "observed_subset_only":
        require(
            population_support is False and calibration_support is False,
            "config.json: observed-subset-only labels cannot support population "
            "performance or probability calibration",
            errors,
        )
    if design == "off_policy_with_support":
        require(
            positivity == "satisfied" and isinstance(evaluation_probability, dict),
            "config.json: off-policy evaluation requires positivity and logged "
            "selection probabilities",
            errors,
        )
        if nested(config, "selection", "capacity", "enabled") is True and isinstance(
            evaluation_probability, dict
        ):
            require(
                evaluation_probability.get("scope") in {"queue", "slate"},
                "config.json: capacity policy off-policy evaluation requires "
                "queue- or slate-level support, not row-only propensities",
                errors,
            )
    if positivity in {"violated", "unknown"}:
        require(
            population_support is False and calibration_support is False,
            "config.json: violated or unknown evaluation positivity cannot "
            "support population performance or probability calibration",
            errors,
        )


def validate_supervised_cohort(config, schema, manifest, errors: list[str]):
    cohort = nested(config, "problem", "cohort")
    if not isinstance(cohort, dict):
        errors.append("config.json: supervised runs require problem.cohort")
        return
    for field in [
        "source_population",
        "inclusion_rule",
        "label_observation",
        "sampling_design",
    ]:
        require(
            is_nonempty_string(cohort.get(field)),
            f"config.json: problem.cohort.{field} is required",
            errors,
        )
    probability = cohort.get("inclusion_probability")
    if is_finite_number(probability):
        require(
            0 < probability <= 1,
            "config.json: cohort inclusion_probability must be in (0, 1]",
            errors,
        )
    elif isinstance(probability, dict):
        has_column = is_nonempty_string(probability.get("column"))
        has_formula = is_nonempty_string(probability.get("formula"))
        require(
            has_column != has_formula,
            "config.json: cohort inclusion_probability needs exactly one of column or formula",
            errors,
        )
        if "scope" in probability:
            require(
                is_nonempty_string(probability.get("scope")),
                "config.json: cohort inclusion_probability.scope must be non-empty",
                errors,
            )
        if has_column and isinstance(schema.get("columns"), dict):
            require(
                probability["column"] in schema["columns"],
                "config.json: inclusion-probability column is absent from schema.json",
                errors,
            )
    else:
        errors.append(
            "config.json: cohort inclusion_probability must be numeric or a computation object"
        )
    for field in ["evaluation_representative", "calibration_representative"]:
        require(
            isinstance(cohort.get(field), bool),
            f"config.json: problem.cohort.{field} must be boolean",
            errors,
        )
    validate_selective_label_contract(config, cohort, schema, errors)
    weight = cohort.get("sample_weight")
    if weight is not None:
        if not isinstance(weight, dict):
            errors.append(
                "config.json: problem.cohort.sample_weight must be null or an object"
            )
            return
        has_column = is_nonempty_string(weight.get("column"))
        has_formula = is_nonempty_string(weight.get("formula"))
        require(
            has_column != has_formula,
            "config.json: cohort sample_weight needs exactly one of column or formula",
            errors,
        )
        require(
            is_nonempty_string(weight.get("scope")),
            "config.json: cohort sample_weight.scope is required",
            errors,
        )
        column = weight.get("column")
        schema_columns = schema.get("columns")
        if has_column and isinstance(schema_columns, dict):
            require(
                column in schema_columns,
                "config.json: sample-weight column is absent from schema.json",
                errors,
            )
        if has_column and isinstance(manifest.get("raw_input_features"), list):
            require(
                column not in manifest["raw_input_features"],
                "feature_manifest.json: sample-weight column cannot be a model feature",
                errors,
            )


def validate_data_feature_schema_contract(
    config,
    fingerprint,
    schema,
    manifest,
    split_manifest,
    errors: list[str],
    warnings: list[str],
):
    run_directory = nested(config, "run", "manifest")
    run_directory = (
        str(Path(run_directory).parent) if is_nonempty_string(run_directory) else None
    )
    if run_directory:
        references = [
            (
                nested(config, "data", "fingerprint_file"),
                f"{run_directory}/data_fingerprint.json",
                "data.fingerprint_file",
            ),
            (
                nested(config, "data", "schema_file"),
                f"{run_directory}/schema.json",
                "data.schema_file",
            ),
            (
                nested(config, "data", "split_manifest"),
                f"{run_directory}/split_manifest.json",
                "data.split_manifest",
            ),
            (
                nested(config, "feature_contract", "manifest"),
                f"{run_directory}/feature_manifest.json",
                "feature_contract.manifest",
            ),
        ]
        for actual, expected, field in references:
            require(
                actual == expected,
                f"config.json: {field} must reference the selected run directory",
                errors,
            )
    rows = fingerprint.get("rows")
    columns_count = fingerprint.get("columns")
    if not is_positive_integer(rows):
        errors.append("data_fingerprint.json: rows must be a positive integer")
    if not is_positive_integer(columns_count):
        errors.append("data_fingerprint.json: columns must be a positive integer")
    input_contract = fingerprint.get("input")
    if not isinstance(input_contract, dict):
        errors.append("data_fingerprint.json: input object is required")
        limited_remote_source = False
    else:
        digest = input_contract.get("sha256")
        immutable_id = input_contract.get("immutable_source_id")
        remote = input_contract.get("remote_preflight")
        remote_override = (
            isinstance(remote, dict)
            and remote.get("override_used") is True
            and remote.get("status") == "overridden"
            and is_string_array(
                remote.get("unknown_fields"),
                nonempty=True,
                unique=True,
            )
        )
        declared_not_verified = input_contract.get(
            "version_verification"
        ) == "declared_not_verified" or (
            isinstance(remote, dict)
            and remote.get("version_verification") == "declared_not_verified"
        )
        limited_remote_source = remote_override or declared_not_verified
        if declared_not_verified:
            require(
                is_nonempty_string(immutable_id),
                "data_fingerprint.json: declared_not_verified source version "
                "requires input.immutable_source_id",
                errors,
            )
            if input_contract.get("reproducibility_status") != "limited_remote_source":
                errors.append(
                    "data_fingerprint.json: declared_not_verified source version "
                    "requires input.reproducibility_status limited_remote_source"
                )
            if (
                nested(config, "data", "reproducibility_status")
                != "limited_remote_source"
            ):
                errors.append(
                    "config.json: declared_not_verified source version requires "
                    "data.reproducibility_status limited_remote_source"
                )
            warnings.append(
                "remote source version was declared but not scan-verified; "
                "reproducibility is limited"
            )
        if digest is None and (
            not is_nonempty_string(immutable_id) or declared_not_verified
        ):
            if declared_not_verified:
                pass
            elif remote_override:
                if (
                    nested(config, "data", "reproducibility_status")
                    != "limited_remote_source"
                ):
                    errors.append(
                        "config.json: explicit remote fingerprint override requires "
                        "data.reproducibility_status limited_remote_source"
                    )
                warnings.append(
                    "remote source fingerprint/version is unknown under an explicit "
                    "override; reproducibility is limited"
                )
            else:
                errors.append(
                    "data_fingerprint.json: input.sha256 or "
                    "input.immutable_source_id is required unless the canonical "
                    "limited remote-source state is recorded"
                )
        if digest is not None and not is_sha256(digest):
            errors.append("data_fingerprint.json: input.sha256 must be a sha256 digest")

    schema_columns = schema.get("columns")
    if (
        not isinstance(schema_columns, dict)
        or not schema_columns
        or not all(
            is_nonempty_string(name) and isinstance(specification, dict)
            for name, specification in schema_columns.items()
        )
    ):
        errors.append("schema.json: columns must be a non-empty name-to-object mapping")
        schema_names: set[str] = set()
    else:
        schema_names = set(schema_columns)
        for column, specification in schema_columns.items():
            prefix = f"schema.json: columns[{column!r}].observational_completeness"
            completeness = specification.get("observational_completeness")
            if not isinstance(completeness, dict):
                errors.append(f"{prefix} object is required")
                continue
            require(
                is_nonempty_string(completeness.get("population")),
                f"{prefix}.population is required",
                errors,
            )
            population_rows = completeness.get("population_rows")
            require(
                is_positive_integer(population_rows),
                f"{prefix}.population_rows must be positive",
                errors,
            )
            status = completeness.get("status")
            require(
                status in ("observed", "not_assessed_target_blind"),
                f"{prefix}.status is invalid",
                errors,
            )
            if status == "observed":
                missing = completeness.get("missing_count")
                non_missing = completeness.get("non_missing_count")
                fraction = completeness.get("missing_fraction")
                valid_counts = (
                    is_nonnegative_integer(missing)
                    and is_nonnegative_integer(non_missing)
                    and is_positive_integer(population_rows)
                )
                require(
                    valid_counts and missing + non_missing == population_rows,
                    f"{prefix}: missing and non-missing counts must sum to "
                    "population_rows",
                    errors,
                )
                require(
                    is_finite_number(fraction)
                    and 0 <= fraction <= 1
                    and (
                        not valid_counts
                        or math.isclose(
                            fraction,
                            missing / population_rows,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                    ),
                    f"{prefix}.missing_fraction must match missing_count",
                    errors,
                )
            elif status == "not_assessed_target_blind":
                require(
                    completeness.get("missing_count") is None
                    and completeness.get("missing_fraction") is None
                    and completeness.get("non_missing_count") is None,
                    f"{prefix}: target-blind completeness counts must be null",
                    errors,
                )
        if is_positive_integer(columns_count) and len(schema_names) != columns_count:
            errors.append(
                "data_fingerprint.json: columns must match the number of "
                "schema.json columns"
            )

    inference = schema.get("inference")
    required_inputs = None
    optional_inputs = None
    if not isinstance(inference, dict):
        errors.append("schema.json: inference object is required")
    else:
        required_inputs = inference.get("required_inputs")
        optional_inputs = inference.get("optional_inputs")
        for name, values in [
            ("required_inputs", required_inputs),
            ("optional_inputs", optional_inputs),
        ]:
            if not is_string_array(values, unique=True):
                errors.append(
                    f"schema.json: inference.{name} must be a unique string array"
                )
        if isinstance(required_inputs, list) and isinstance(optional_inputs, list):
            overlap = set(required_inputs) & set(optional_inputs)
            if overlap:
                errors.append(
                    "schema.json: inference required_inputs and optional_inputs "
                    "must be disjoint"
                )

    raw_features = manifest.get("raw_input_features")
    if not isinstance(raw_features, list):
        errors.append("feature_manifest.json: raw_input_features must be a list")
        raw_set: set[str] = set()
    elif not is_string_array(raw_features, unique=True):
        errors.append(
            "feature_manifest.json: raw_input_features must be a unique string array"
        )
        raw_set = {value for value in raw_features if is_nonempty_string(value)}
    else:
        raw_set = set(raw_features)

    feature_contract = config.get("feature_contract")
    if isinstance(feature_contract, dict):
        for field in ["inference_unavailable", "target_sources_excluded"]:
            values = feature_contract.get(field)
            if not is_string_array(values, unique=True):
                errors.append(
                    f"config.json: feature_contract.{field} must be a unique "
                    "string array"
                )
                continue
            conflicted = sorted(raw_set & set(values))
            if conflicted:
                errors.append(
                    "feature_manifest.json: raw_input_features cannot also be "
                    f"feature_contract.{field}: " + ", ".join(conflicted)
                )

    if schema_names:
        unknown_raw = sorted(raw_set - schema_names)
        if unknown_raw:
            errors.append(
                "feature_manifest.json: raw_input_features absent from schema.json: "
                + ", ".join(unknown_raw)
            )
    if isinstance(required_inputs, list) and isinstance(optional_inputs, list):
        declared_inputs = set(required_inputs) | set(optional_inputs)
        unknown_inputs = sorted(declared_inputs - schema_names)
        if unknown_inputs:
            errors.append(
                "schema.json: inference inputs absent from columns: "
                + ", ".join(unknown_inputs)
            )
        undeclared_raw = sorted(raw_set - declared_inputs)
        if undeclared_raw:
            errors.append(
                "feature_manifest.json: raw_input_features absent from inference "
                "inputs: " + ", ".join(undeclared_raw)
            )

    target = nested(config, "problem", "target")
    partition = schema.get("partition_column")
    forbidden = {value for value in (target, partition) if is_nonempty_string(value)}
    leaked_features = sorted(raw_set & forbidden)
    if leaked_features:
        errors.append(
            "feature_manifest.json: target/partition columns cannot be raw model "
            "features: " + ", ".join(leaked_features)
        )
    if isinstance(required_inputs, list) and isinstance(optional_inputs, list):
        leaked_inputs = sorted(
            (set(required_inputs) | set(optional_inputs)) & forbidden
        )
        if leaked_inputs:
            errors.append(
                "schema.json: target/partition columns cannot be inference inputs: "
                + ", ".join(leaked_inputs)
            )

    schema_target = schema.get("target")
    if is_nonempty_string(target) and schema_target != target:
        errors.append("schema.json: target must match config.json problem.target")
    assignment_column = nested(split_manifest, "assignment", "column")
    if schema.get("partition_column") != assignment_column:
        errors.append(
            "schema.json: partition_column must match split_manifest assignment.column"
        )
    return limited_remote_source


def validate_inference_output_contract(output, prefix: str, errors: list[str]):
    if not isinstance(output, dict):
        errors.append(f"{prefix}.output must be an object")
        return
    required = output.get("required_columns")
    predictions = output.get("prediction_columns")
    rows = output.get("row_count")
    require(
        is_nonempty_string(output.get("path")),
        f"{prefix}.output.path is required",
        errors,
    )
    require(
        output.get("format") in {"csv", "json"},
        f"{prefix}.output.format must be csv or json",
        errors,
    )
    require(
        is_nonnegative_integer(rows),
        f"{prefix}.output.row_count must be a non-negative integer",
        errors,
    )
    require(
        is_string_array(required, nonempty=True, unique=True),
        f"{prefix}.output.required_columns must be an ordered unique string array",
        errors,
    )
    require(
        is_string_array(predictions, nonempty=True, unique=True),
        f"{prefix}.output.prediction_columns must be a unique string array",
        errors,
    )
    if isinstance(required, list) and isinstance(predictions, list):
        require(
            set(predictions).issubset(required),
            f"{prefix}.output.prediction_columns must be required columns",
            errors,
        )

    row_id = output.get("row_id_column")
    expected_ids = output.get("expected_row_ids")
    if row_id is not None or expected_ids is not None:
        require(
            is_nonempty_string(row_id)
            and isinstance(required, list)
            and row_id in required,
            f"{prefix}.output.row_id_column must name a required column",
            errors,
        )
        require(
            isinstance(expected_ids, list)
            and (not is_nonnegative_integer(rows) or len(expected_ids) == rows),
            f"{prefix}.output.expected_row_ids must match row_count",
            errors,
        )

    golden = output.get("golden_predictions")
    expected_hash = output.get("sha256")
    require(
        golden is not None or expected_hash is not None,
        f"{prefix}.output requires golden_predictions or sha256",
        errors,
    )
    if golden is not None:
        require(
            isinstance(golden, dict)
            and bool(golden)
            and isinstance(predictions, list)
            and set(golden) == set(predictions),
            f"{prefix}.output.golden_predictions must map every prediction column",
            errors,
        )
        if isinstance(golden, dict):
            for column, values in golden.items():
                require(
                    isinstance(values, list)
                    and (not is_nonnegative_integer(rows) or len(values) == rows)
                    and all(is_finite_number(value) for value in values),
                    f"{prefix}.output golden values for {column!r} must be finite and match row_count",
                    errors,
                )
        tolerance = output.get("absolute_tolerance")
        require(
            is_finite_number(tolerance) and tolerance >= 0,
            f"{prefix}.output.absolute_tolerance must be non-negative and finite",
            errors,
        )
    if expected_hash is not None:
        require(
            is_sha256(expected_hash),
            f"{prefix}.output.sha256 must be a sha256 digest",
            errors,
        )
    golden_values = output.get("golden_values")
    if golden_values is not None:
        require(
            isinstance(golden_values, dict)
            and bool(golden_values)
            and isinstance(required, list)
            and set(golden_values).issubset(required),
            f"{prefix}.output.golden_values must map required output columns",
            errors,
        )
        if isinstance(golden_values, dict):
            for column, values in golden_values.items():
                require(
                    isinstance(values, list)
                    and (not is_nonnegative_integer(rows) or len(values) == rows)
                    and all(
                        value is None or isinstance(value, (str, int, float, bool))
                        for value in values
                    ),
                    f"{prefix}.output golden values for {column!r} must be JSON "
                    "scalars and match row_count",
                    errors,
                )


def validate_inference_case_contract(
    project: Path,
    artifacts: Path,
    inference_test,
    config,
    errors: list[str],
):
    prediction_constraints = inference_test.get("prediction_constraints")
    if not isinstance(prediction_constraints, dict) or not prediction_constraints:
        errors.append(
            "inference_test.json: prediction_constraints must map prediction "
            "columns to semantic contracts"
        )
        prediction_constraints = {}
    else:
        for column, specification in prediction_constraints.items():
            prefix = f"inference_test.json: prediction_constraints[{column!r}]"
            if not is_nonempty_string(column) or not isinstance(specification, dict):
                errors.append(f"{prefix} must be a named object")
                continue
            semantic = specification.get("semantic")
            require(
                semantic in {"probability", "numeric", "forecast", "anomaly_score"},
                f"{prefix}.semantic is invalid",
                errors,
            )
            minimum = specification.get("minimum")
            maximum = specification.get("maximum")
            if minimum is not None:
                require(
                    is_finite_number(minimum),
                    f"{prefix}.minimum must be finite",
                    errors,
                )
            if maximum is not None:
                require(
                    is_finite_number(maximum),
                    f"{prefix}.maximum must be finite",
                    errors,
                )
            if is_finite_number(minimum) and is_finite_number(maximum):
                require(
                    minimum <= maximum,
                    f"{prefix}.minimum cannot exceed maximum",
                    errors,
                )
            if semantic == "probability":
                require(
                    (minimum is None or (is_finite_number(minimum) and minimum == 0))
                    and (
                        maximum is None or (is_finite_number(maximum) and maximum == 1)
                    ),
                    f"{prefix}: probability bounds are fixed at [0, 1]",
                    errors,
                )
    cases = inference_test.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append(
            "inference_test.json: schema 2.1 requires a non-empty cases array"
        )
        return []
    expected_infer = artifacts / "infer.py"
    require(
        expected_infer.is_file() and not expected_infer.is_symlink(),
        "inference_test.json: selected run infer.py must be a regular non-symlink file",
        errors,
    )
    names = set()
    declared_output_paths: set[Path] = set()
    inference_output_root = (artifacts / "inference_outputs").resolve()
    valid_cases = []
    for index, case in enumerate(cases):
        prefix = f"inference_test.json: cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        name = case.get("name")
        require(is_nonempty_string(name), f"{prefix}.name is required", errors)
        if is_nonempty_string(name):
            require(name not in names, f"{prefix}.name must be unique", errors)
            names.add(name)
        argv = case.get("argv")
        argv_valid = require(
            is_string_array(argv, nonempty=True) and len(argv) >= 2,
            f"{prefix}.argv must contain {{python}}, the selected infer.py, and "
            "optional arguments",
            errors,
        )
        if argv_valid:
            require(
                argv[0] == "{python}",
                f"{prefix}.argv must start with exactly {{python}} followed by "
                "the selected run's infer.py",
                errors,
            )
            declared_infer = resolve_artifact_path(
                project,
                artifacts,
                argv[1],
                f"{prefix}.argv[1]",
                errors,
            )
            if declared_infer is not None:
                require(
                    declared_infer == expected_infer,
                    f"{prefix}.argv[1] must be the selected run's own infer.py",
                    errors,
                )
        expected_exit = case.get("expected_exit_code")
        require(
            isinstance(expected_exit, int) and not isinstance(expected_exit, bool),
            f"{prefix}.expected_exit_code must be an integer",
            errors,
        )
        stderr_contains = case.get("stderr_contains")
        if stderr_contains is not None:
            require(
                is_nonempty_string(stderr_contains),
                f"{prefix}.stderr_contains must be a non-empty string",
                errors,
            )
        if "output" in case:
            output = case.get("output")
            validate_inference_output_contract(output, prefix, errors)
            if isinstance(output, dict):
                output_path = resolve_artifact_path(
                    project,
                    artifacts,
                    output.get("path"),
                    f"{prefix}.output.path",
                    errors,
                )
                if output_path is not None:
                    require(
                        output_path != inference_output_root
                        and inference_output_root in output_path.parents,
                        f"{prefix}.output.path must be below the selected run's "
                        "inference_outputs/ directory",
                        errors,
                    )
                    require(
                        output_path not in declared_output_paths,
                        f"{prefix}.output.path duplicates another inference case",
                        errors,
                    )
                    declared_output_paths.add(output_path)
                    if output_path.exists() and not output_path.is_file():
                        errors.append(
                            f"{prefix}.output.path must identify a file, not a "
                            "directory"
                        )
        valid_cases.append(case)

    missing = sorted(CORE_INFERENCE_CASES - names)
    if missing:
        errors.append(
            "inference_test.json: missing core edge cases: " + ", ".join(missing)
        )
    declared_prediction_columns = {
        column
        for case in valid_cases
        if isinstance(case.get("output"), dict)
        for column in (
            case["output"].get("prediction_columns", [])
            if isinstance(case["output"].get("prediction_columns"), list)
            else []
        )
        if is_nonempty_string(column)
    }
    require(
        declared_prediction_columns == set(prediction_constraints),
        "inference_test.json: prediction_constraints must exactly cover all "
        "declared prediction columns",
        errors,
    )
    indexed_cases = {
        case.get("name"): case
        for case in valid_cases
        if is_nonempty_string(case.get("name"))
    }
    for case_name in ("representative_batch", "one_row"):
        case = indexed_cases.get(case_name)
        if case is None:
            continue
        require(
            case.get("expected_exit_code") == 0,
            f"inference_test.json: {case_name} must expect exit code 0",
            errors,
        )
        output = case.get("output")
        require(
            isinstance(output, dict) and is_positive_integer(output.get("row_count")),
            f"inference_test.json: {case_name} requires positive parsed output",
            errors,
        )
        if case_name == "one_row" and isinstance(output, dict):
            require(
                output.get("row_count") == 1,
                "inference_test.json: one_row output.row_count must be 1",
                errors,
            )

    for case_name in ("missing_required", "wrong_dtypes"):
        case = indexed_cases.get(case_name)
        if case is None:
            continue
        require(
            isinstance(case.get("expected_exit_code"), int)
            and not isinstance(case.get("expected_exit_code"), bool)
            and case.get("expected_exit_code") > 0,
            f"inference_test.json: {case_name} must expect a controlled non-zero "
            "exit code",
            errors,
        )
        require(
            is_nonempty_string(case.get("stderr_contains")),
            f"inference_test.json: {case_name} requires an actionable "
            "stderr_contains substring",
            errors,
        )

    flexible_core = CORE_INFERENCE_CASES - {
        "representative_batch",
        "one_row",
        "missing_required",
        "wrong_dtypes",
    }
    for case_name in sorted(flexible_core):
        case = indexed_cases.get(case_name)
        if case is None:
            continue
        expected_exit = case.get("expected_exit_code")
        if expected_exit == 0:
            require(
                isinstance(case.get("output"), dict),
                f"inference_test.json: successful {case_name} requires parsed output",
                errors,
            )
        elif (
            isinstance(expected_exit, int)
            and not isinstance(expected_exit, bool)
            and expected_exit > 0
        ):
            require(
                is_nonempty_string(case.get("stderr_contains")),
                f"inference_test.json: failing {case_name} requires an actionable "
                "stderr_contains substring",
                errors,
            )
        elif isinstance(expected_exit, int) and not isinstance(expected_exit, bool):
            errors.append(
                f"inference_test.json: {case_name} must expect success or a "
                "controlled positive non-zero exit code"
            )

    if nested(config, "selection", "capacity", "enabled") is True:
        capacity = nested(config, "selection", "capacity")
        for case_name in sorted(CAPACITY_INFERENCE_CASES):
            case = indexed_cases.get(case_name)
            valid_case = require(
                isinstance(case, dict)
                and case.get("expected_exit_code") == 0
                and isinstance(case.get("output"), dict),
                "inference_test.json: enabled selection capacity requires a "
                f"successful {case_name} case with parsed output",
                errors,
            )
            if not valid_case or case_name == "score_rows":
                continue
            output = case["output"]
            required_columns = output.get("required_columns")
            require(
                isinstance(required_columns, list)
                and {"selection_rank", "selected"}.issubset(required_columns),
                f"inference_test.json: {case_name} output must include "
                "selection_rank and selected",
                errors,
            )
            golden_values = output.get("golden_values")
            require(
                isinstance(golden_values, dict)
                and {"selection_rank", "selected"}.issubset(golden_values),
                f"inference_test.json: {case_name} must verify actual "
                "selection_rank and selected values",
                errors,
            )
            eligible = output.get("eligible_count")
            selected = output.get("selected_count")
            limit = capacity.get("limit") if isinstance(capacity, dict) else None
            require(
                is_nonnegative_integer(eligible)
                and is_nonnegative_integer(selected)
                and is_positive_integer(limit),
                f"inference_test.json: {case_name} requires non-negative "
                "eligible_count/selected_count and the configured capacity limit",
                errors,
            )
            if (
                is_nonnegative_integer(eligible)
                and is_nonnegative_integer(selected)
                and is_positive_integer(limit)
            ):
                require(
                    selected == min(eligible, limit),
                    f"inference_test.json: {case_name} selected_count must equal "
                    "min(eligible_count, capacity limit)",
                    errors,
                )
            for field in ["timezone", "cutoff", "tie_breaker"]:
                require(
                    isinstance(capacity, dict)
                    and output.get(field) == capacity.get(field),
                    f"inference_test.json: {case_name} output.{field} must match "
                    "the capacity contract",
                    errors,
                )
            require(
                output.get("capacity_limit") == limit,
                f"inference_test.json: {case_name} output.capacity_limit must "
                "match the capacity contract",
                errors,
            )
            if case_name == "capacity_empty":
                require(
                    eligible == 0 and selected == 0 and output.get("row_count") == 0,
                    "inference_test.json: capacity_empty must verify an empty queue",
                    errors,
                )
            elif case_name == "capacity_sub_capacity" and is_positive_integer(limit):
                require(
                    is_positive_integer(eligible)
                    and eligible < limit
                    and selected == eligible,
                    "inference_test.json: capacity_sub_capacity must select every "
                    "eligible row below capacity",
                    errors,
                )
            if case_name == "capacity_duplicates":
                expected_ids = output.get("expected_row_ids")
                require(
                    isinstance(expected_ids, list)
                    and len(expected_ids) == len(set(map(str, expected_ids))),
                    "inference_test.json: capacity_duplicates must produce unique "
                    "action row IDs",
                    errors,
                )
    return valid_cases


def read_inference_output(path: Path, output, prefix: str, errors: list[str]):
    try:
        if output.get("format") == "csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                rows = list(reader)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                if not payload:
                    errors.append(
                        f"{prefix}: zero-row JSON output must use the "
                        'schema-bearing {"columns": [...], "rows": []} envelope'
                    )
                    return None, None
                if not all(isinstance(row, dict) for row in payload):
                    errors.append(
                        f"{prefix}: JSON output must be an array of row objects"
                    )
                    return None, None
                rows = payload
                columns = list(rows[0])
            elif isinstance(payload, dict):
                columns = payload.get("columns")
                rows = payload.get("rows")
                if not (
                    is_string_array(columns, nonempty=True, unique=True)
                    and isinstance(rows, list)
                    and all(isinstance(row, dict) for row in rows)
                ):
                    errors.append(
                        f"{prefix}: JSON envelope requires unique string columns "
                        "and an array of row objects"
                    )
                    return None, None
            else:
                errors.append(
                    f"{prefix}: JSON output must be row objects or a "
                    "schema-bearing columns/rows envelope"
                )
                return None, None
            if any(list(row) != columns for row in rows):
                errors.append(
                    f"{prefix}: JSON rows must use one consistent column order"
                )
                return None, None
    except (OSError, csv.Error, json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"{prefix}: could not parse output ({exc})")
        return None, None
    return columns, rows


def validate_inference_output(
    path: Path,
    output,
    prefix: str,
    prediction_constraints,
    errors: list[str],
):
    if not path.is_file():
        errors.append(f"{prefix}: expected output file was not created: {path}")
        return
    columns, rows = read_inference_output(path, output, prefix, errors)
    if columns is None or rows is None:
        return
    required_columns = output.get("required_columns")
    if columns != required_columns:
        errors.append(
            f"{prefix}: output columns/order {columns!r} do not match "
            f"{required_columns!r}"
        )
    if len(rows) != output.get("row_count"):
        errors.append(
            f"{prefix}: output row count {len(rows)} does not match "
            f"{output.get('row_count')}"
        )

    prediction_columns = output.get("prediction_columns")
    if isinstance(prediction_columns, list):
        for row_index, row in enumerate(rows):
            for column in prediction_columns:
                value = row.get(column)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = math.nan
                if not math.isfinite(numeric):
                    errors.append(
                        f"{prefix}: prediction {column!r} at row {row_index} "
                        "must be finite"
                    )
                    break
                specification = (
                    prediction_constraints.get(column)
                    if isinstance(prediction_constraints, dict)
                    else None
                )
                if isinstance(specification, dict):
                    minimum = specification.get("minimum")
                    maximum = specification.get("maximum")
                    if specification.get("semantic") == "probability":
                        minimum = 0.0
                        maximum = 1.0
                    if is_finite_number(minimum) and numeric < float(minimum):
                        errors.append(
                            f"{prefix}: prediction {column!r} at row {row_index} "
                            f"is below {minimum}"
                        )
                    if is_finite_number(maximum) and numeric > float(maximum):
                        errors.append(
                            f"{prefix}: prediction {column!r} at row {row_index} "
                            f"is above {maximum}"
                        )

    row_id_column = output.get("row_id_column")
    expected_row_ids = output.get("expected_row_ids")
    if is_nonempty_string(row_id_column) and isinstance(expected_row_ids, list):
        actual_ids = [str(row.get(row_id_column)) for row in rows]
        if actual_ids != [str(value) for value in expected_row_ids]:
            errors.append(
                f"{prefix}: output row identifiers/order do not match expected_row_ids"
            )

    golden = output.get("golden_predictions")
    if isinstance(golden, dict):
        tolerance_value = output.get("absolute_tolerance")
        if not is_finite_number(tolerance_value) or tolerance_value < 0:
            return
        tolerance = float(tolerance_value)
        for column, expected_values in golden.items():
            for row_index, expected in enumerate(expected_values):
                if row_index >= len(rows):
                    break
                try:
                    actual = float(rows[row_index].get(column))
                    expected_numeric = float(expected)
                except (TypeError, ValueError):
                    continue
                if (
                    math.isfinite(actual)
                    and math.isfinite(expected_numeric)
                    and abs(actual - expected_numeric) > tolerance
                ):
                    errors.append(
                        f"{prefix}: prediction {column!r} at row {row_index} "
                        f"differs from golden value by more than {tolerance}"
                    )
                    break
    golden_values = output.get("golden_values")
    if isinstance(golden_values, dict):
        for column, expected_values in golden_values.items():
            for row_index, expected in enumerate(expected_values):
                if row_index >= len(rows):
                    break
                actual = rows[row_index].get(column)
                if str(actual) != str(expected):
                    errors.append(
                        f"{prefix}: output {column!r} at row {row_index} differs "
                        "from its golden value"
                    )
                    break
    expected_hash = output.get("sha256")
    if is_sha256(expected_hash):
        expected_digest = expected_hash.lower().removeprefix("sha256:")
        if sha256_file(path) != expected_digest:
            errors.append(f"{prefix}: output sha256 does not match")


def execute_inference_cases(
    project: Path,
    artifacts: Path,
    cases,
    prediction_constraints,
    timeout_seconds: int,
    errors: list[str],
):
    for index, case in enumerate(cases):
        prefix = f"inference round trip case {case.get('name', index)!r}"
        argv = case.get("argv")
        expected_exit_code = case.get("expected_exit_code")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(is_nonempty_string(value) for value in argv)
            or not isinstance(expected_exit_code, int)
            or isinstance(expected_exit_code, bool)
        ):
            continue
        output = case.get("output")
        output_path = None
        if isinstance(output, dict):
            output_path = resolve_artifact_path(
                project,
                artifacts,
                output.get("path"),
                f"inference_test.json: cases[{index}].output.path",
                errors,
            )
            if (
                expected_exit_code == 0
                and output_path is not None
                and output_path.exists()
            ):
                if not output_path.is_file() or output_path.is_symlink():
                    errors.append(
                        f"{prefix}: existing output must be a regular non-symlink file"
                    )
                    continue
                try:
                    output_path.unlink()
                except OSError as exc:
                    errors.append(
                        f"{prefix}: could not replace declared output ({exc})"
                    )
                    continue
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
            errors.append(f"{prefix} could not run: {exc}")
            continue
        if completed.returncode != expected_exit_code:
            detail = (completed.stderr or completed.stdout).strip()[-500:]
            errors.append(
                f"{prefix} exited {completed.returncode}, expected "
                f"{expected_exit_code}: {detail}"
            )
            continue
        stderr_contains = case.get("stderr_contains")
        if (
            is_nonempty_string(stderr_contains)
            and stderr_contains not in completed.stderr
        ):
            errors.append(
                f"{prefix}: stderr did not contain expected text {stderr_contains!r}"
            )
        if (
            expected_exit_code == 0
            and isinstance(output, dict)
            and output_path is not None
        ):
            if not output_path.exists():
                errors.append(
                    f"{prefix}: expected output file was not created: {output_path}"
                )
                continue
            if output_path.is_symlink() or not output_path.is_file():
                errors.append(
                    f"{prefix}: created output must be a regular non-symlink file"
                )
                continue
            validate_inference_output(
                output_path,
                output,
                prefix,
                prediction_constraints,
                errors,
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


def immutable_direct_requirement(item: str):
    if " @ " not in item:
        return None
    name, location = item.split(" @ ", 1)
    if not DIRECT_REQUIREMENT_NAME_RE.fullmatch(name):
        return None
    if location.startswith("git+"):
        revision = location.rsplit("@", 1)[-1].split("#", 1)[0]
        return name if re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) else None
    if location.startswith(("https://", "http://")):
        fragment = location.partition("#")[2]
        return (
            name
            if re.search(r"(?:^|&)sha256=[0-9a-fA-F]{64}(?:&|$)", fragment)
            else None
        )
    return None


def is_exact_version(value) -> bool:
    return is_nonempty_string(value) and EXACT_VERSION_RE.fullmatch(value) is not None


def validate_pinned_requirements(
    path: Path,
    errors: list[str],
    warnings: list[str],
    strict_21: bool = False,
):
    if not path.exists():
        errors.append("missing pinned inference environment: requirements.lock")
        return
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"requirements.lock could not be read: {exc}")
        return
    unpinned = []
    packages: dict[str, str] = {}
    active_entries = 0
    for line in contents.splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        active_entries += 1
        if not strict_21:
            if item.startswith(("-", "git+", "http://", "https://")):
                continue
            if "==" not in item or item.startswith(("-e ", ".")):
                unpinned.append(item)
            continue
        match = PINNED_REQUIREMENT_RE.fullmatch(item)
        direct_name = immutable_direct_requirement(item)
        if direct_name is not None:
            name = re.sub(r"[-_.]+", "-", direct_name.split("[", 1)[0].lower())
            version = item.split(" @ ", 1)[1]
        elif (
            match is not None
            and "\\" not in item
            and not item.startswith(("-", ".", "/", "git+", "http://", "https://"))
        ):
            version = match.group("version")
            if not is_exact_version(version):
                unpinned.append(item)
                continue
            name = re.sub(r"[-_.]+", "-", match.group("name").lower())
        else:
            unpinned.append(item)
            continue
        previous = packages.get(name)
        if previous is not None:
            errors.append(f"requirements.lock contains duplicate package entry: {name}")
        packages[name] = version
    if unpinned:
        errors.append(
            "requirements.lock contains unpinned entries: " + ", ".join(unpinned[:5])
        )
    if strict_21 and active_entries == 0:
        errors.append("requirements.lock contains no pinned packages")
    elif not contents.strip():
        warnings.append("requirements.lock is empty")


def validate_environment_lock(path: Path, errors: list[str], strict_21: bool):
    if not path.exists():
        errors.append("missing pinned inference environment: environment.lock")
        return
    if not strict_21:
        try:
            if not path.read_text(encoding="utf-8").strip():
                errors.append("environment.lock is empty")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"environment.lock could not be read: {exc}")
        return
    document = read_json(path, errors)
    if not isinstance(document, dict):
        return
    if document.get("schema_version") != CURRENT_SCHEMA_VERSION:
        errors.append(
            f"environment.lock: schema_version must be {CURRENT_SCHEMA_VERSION}"
        )
    if not (
        is_nonempty_string(document.get("python"))
        and re.fullmatch(r"\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]*)?", document["python"])
    ):
        errors.append("environment.lock: python must be an exact interpreter version")
    if not is_nonempty_string(document.get("platform")):
        errors.append("environment.lock: platform is required")
    packages = document.get("packages")
    if not isinstance(packages, list) or not packages:
        errors.append("environment.lock: packages must be a non-empty array")
        return
    seen = set()
    for index, package in enumerate(packages):
        prefix = f"environment.lock: packages[{index}]"
        if not isinstance(package, dict):
            errors.append(f"{prefix} must be an object")
            continue
        name = package.get("name")
        version = package.get("version")
        if not is_nonempty_string(name):
            errors.append(f"{prefix}.name is required")
        else:
            normalized = re.sub(r"[-_.]+", "-", name.lower())
            if normalized in seen:
                errors.append(f"{prefix}.name duplicates another package")
            seen.add(normalized)
        if not is_exact_version(version):
            errors.append(f"{prefix}.version must be one exact version")


def validate_model_directory_manifest(
    model_dir: Path,
    manifest_path: Path,
    errors: list[str],
):
    manifest = read_json(manifest_path, errors)
    if not isinstance(manifest, dict):
        return
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("model/manifest.json: files must be a non-empty array")
        return
    recorded: set[str] = set()
    for index, item in enumerate(files):
        prefix = f"model/manifest.json: files[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        relative_value = item.get("path")
        if not is_nonempty_string(relative_value):
            errors.append(f"{prefix}.path is required")
            continue
        relative = Path(relative_value)
        candidate = (model_dir / relative).resolve()
        if (
            relative.is_absolute()
            or candidate == model_dir
            or model_dir not in candidate.parents
            or relative.as_posix() == "manifest.json"
        ):
            errors.append(
                f"{prefix}.path must identify a model file below the model directory"
            )
            continue
        normalized = relative.as_posix()
        if normalized in recorded:
            errors.append(f"{prefix}.path duplicates another manifest entry")
            continue
        recorded.add(normalized)
        declared_path = model_dir / relative
        if declared_path.is_symlink():
            errors.append(f"{prefix}.path cannot be a symlink: {normalized}")
            continue
        if not candidate.is_file():
            errors.append(f"{prefix}.path does not exist: {normalized}")
            continue
        digest = item.get("sha256")
        if not is_sha256(digest):
            errors.append(f"{prefix}.sha256 must be a sha256 digest")
        elif sha256_file(candidate) != digest.lower().removeprefix("sha256:"):
            errors.append(f"{prefix}.sha256 does not match {normalized}")
    actual = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    unrecorded = sorted(actual - recorded)
    missing = sorted(recorded - actual)
    if unrecorded:
        errors.append(
            "model/manifest.json: unrecorded model files: " + ", ".join(unrecorded[:10])
        )
    if missing:
        errors.append(
            "model/manifest.json: recorded files are missing: "
            + ", ".join(missing[:10])
        )


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
    data_profile = documents.get("data_profile.json") or {}
    fingerprint = documents.get("data_fingerprint.json") or {}
    split_manifest = documents.get("split_manifest.json") or {}
    run_manifest = documents.get("run_manifest.json") or {}
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
    strict_21 = config.get("schema_version") == CURRENT_SCHEMA_VERSION
    task = nested(config, "problem", "task")
    unlabeled_anomaly = (
        task == "anomaly" and nested(config, "problem", "labels_available") is False
    )
    cases = []

    if strict_21:
        require(
            config.get("mode") in {"model-building", "model-improvement"},
            "config.json: model schema 2.1 mode must be model-building or "
            "model-improvement",
            errors,
        )
        require(
            task in {"classification", "regression", "time-series", "anomaly"},
            "config.json: problem.task must be classification, regression, "
            "time-series, or anomaly",
            errors,
        )
        require(
            metrics.get("task") == task,
            "metrics.json: task must match config.json problem.task",
            errors,
        )
        require(
            data_profile.get("task") == task,
            "data_profile.json: task must match config.json problem.task",
            errors,
        )
    if not nested(config, "problem", "prediction_moment"):
        errors.append("config.json: problem.prediction_moment is required")
    if not task:
        errors.append("config.json: problem.task is required")
    if not nested(config, "split", "assignment_column"):
        errors.append("config.json: split.assignment_column is required")
    if unlabeled_anomaly:
        allowed_anomaly_sets = {
            "future_scoring_window",
            "prospective_review_window",
        }
        if nested(config, "analysis", "population_partition") not in {
            "train",
            "reference",
        }:
            errors.append(
                "config.json: unlabeled anomaly analysis requires a historical "
                "population_partition"
            )
        metric_final_set = nested(metrics, "final", "eval_set")
        if metric_final_set not in allowed_anomaly_sets:
            errors.append(
                "metrics.json: unlabeled anomaly final.eval_set must be a future "
                "scoring/review window"
            )
        if strict_21:
            evaluation_final_set = nested(
                config,
                "evaluation",
                "final_eval_set",
            )
            if evaluation_final_set not in allowed_anomaly_sets:
                errors.append(
                    "config.json: unlabeled anomaly evaluation.final_eval_set "
                    "must be a future scoring/review window"
                )
            elif metric_final_set != evaluation_final_set:
                errors.append(
                    "metrics.json: unlabeled anomaly final.eval_set must match "
                    "config evaluation.final_eval_set"
                )
            if (
                "primary_metric" not in metrics
                or metrics.get("primary_metric") is not None
            ):
                errors.append(
                    "metrics.json: unlabeled anomaly primary_metric must be null"
                )
            final = metrics.get("final")
            if (
                not isinstance(final, dict)
                or "score" not in final
                or final.get("score") is not None
            ):
                errors.append(
                    "metrics.json: unlabeled anomaly final.score must be null"
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
            review_capacity = anomaly_evaluation.get("review_capacity")
            if not is_positive_integer(review_capacity):
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
            if strict_21:
                require(
                    nested(config, "selection", "capacity", "enabled") is True
                    and nested(config, "selection", "capacity", "limit")
                    == review_capacity,
                    "config.json: unlabeled anomaly review_capacity requires a "
                    "matching enabled selection.capacity queue contract",
                    errors,
                )
                constraints = inference_test.get("prediction_constraints")
                require(
                    isinstance(constraints, dict)
                    and bool(constraints)
                    and all(
                        isinstance(specification, dict)
                        and specification.get("semantic") == "anomaly_score"
                        for specification in constraints.values()
                    ),
                    "inference_test.json: unlabeled anomaly predictions must use "
                    "anomaly_score semantics",
                    errors,
                )
    else:
        development_label = nested(config, "split", "development_label") or "train"
        nested_21 = strict_21 and nested(config, "evaluation", "design") == "nested_cv"
        if (
            not nested_21
            and nested(config, "analysis", "target_aware_partition")
            != development_label
        ):
            errors.append(
                "config.json: analysis.target_aware_partition must match "
                "split.development_label"
            )
        pending_labels = validate_evaluation_contract(
            config,
            metrics,
            errors,
            strict_21=strict_21,
        )
        validate_metric_contract(
            metrics,
            errors,
            pending_labels=bool(pending_labels),
        )
        if strict_21:
            validate_strict_metric_evidence(
                metrics,
                errors,
                pending_labels=bool(pending_labels),
                config=config,
            )
            require(
                nested(config, "selection", "primary_metric")
                == nested(metrics, "primary_metric", "name"),
                "config.json: selection.primary_metric must match "
                "metrics.json primary_metric.name",
                errors,
            )
    validate_high_stakes(config, errors, strict_21=strict_21)
    if not strict_21 and not isinstance(manifest.get("raw_input_features"), list):
        errors.append("feature_manifest.json: raw_input_features must be a list")
    if schema.get("partition_column") is None:
        errors.append("schema.json: partition_column is required in model mode")

    if strict_21:
        if inference_test.get("schema_version") != CURRENT_SCHEMA_VERSION:
            errors.append(
                f"inference_test.json: schema_version must be {CURRENT_SCHEMA_VERSION}"
            )
        cases = validate_inference_case_contract(
            project,
            artifacts,
            inference_test,
            config,
            errors,
        )
        if not args.run_inference_test:
            warnings.append(
                "inference was not executed; rerun with --run-inference-test "
                "before handoff"
            )
    else:
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
                "inference was not executed; rerun with --run-inference-test "
                "before handoff"
            )

    model_path = artifacts / "model.joblib"
    model_directory = artifacts / "model"
    model_manifest = model_directory / "manifest.json"
    if strict_21 and model_path.is_symlink():
        errors.append("schema 2.1 model.joblib must be a regular non-symlink file")
    if strict_21 and model_directory.is_symlink():
        errors.append("schema 2.1 model/ must be a regular non-symlink directory")
    if strict_21 and model_path.exists() and model_directory.exists():
        errors.append(
            "schema 2.1 run must not contain both deployable model forms: "
            "model.joblib and model/"
        )
    trusted_path = model_path if model_path.exists() else model_manifest
    if strict_21 and model_directory.exists():
        if model_manifest.exists():
            validate_model_directory_manifest(
                model_directory,
                model_manifest,
                errors,
            )
        else:
            errors.append("model directory requires model/manifest.json")
    if not trusted_path.exists():
        errors.append("missing artefacts/model.joblib or artefacts/model/manifest.json")
    else:
        digest = sha256_file(trusted_path)
        recorded = inference_test.get("trusted_model_sha256")
        if not recorded:
            if strict_21:
                errors.append("inference_test.json: trusted_model_sha256 is required")
            else:
                warnings.append(
                    "inference_test.json: trusted_model_sha256 is not recorded"
                )
        elif not is_sha256(recorded):
            errors.append(
                "inference_test.json: trusted_model_sha256 must be a sha256 digest"
            )
        elif recorded.lower().removeprefix("sha256:") != digest:
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
        validate_pinned_requirements(
            requirements,
            errors,
            warnings,
            strict_21=strict_21,
        )
        if environment_lock.exists():
            validate_environment_lock(environment_lock, errors, strict_21)
    elif environment_lock.exists():
        validate_environment_lock(environment_lock, errors, strict_21)

    limited_remote_source = False
    if strict_21:
        validate_run_manifest(
            config,
            metrics,
            run_manifest,
            artifacts,
            args.artifacts_dir,
            errors,
        )
        validate_split_manifest(
            config,
            split_manifest,
            fingerprint,
            schema,
            metrics,
            errors,
        )
        validate_nested_discovery(config, split_manifest, errors)
        validate_pre_partition_target_exposure(config, run_manifest, errors)
        limited_remote_source = validate_data_feature_schema_contract(
            config,
            fingerprint,
            schema,
            manifest,
            split_manifest,
            errors,
            warnings,
        )
        if task in {"classification", "regression"}:
            if not is_nonempty_string(nested(config, "problem", "target")):
                errors.append(
                    "config.json: supervised model runs require problem.target"
                )
            validate_supervised_candidates(config, metrics, errors)
            validate_incumbent_baseline(config, metrics, run_manifest, errors)
            validate_supervised_cohort(config, schema, manifest, errors)
        validate_capacity_selection(config, errors)
    results_path = (artifacts if strict_21 else project) / "results.md"
    if not results_path.exists():
        location = "run artifact" if strict_21 else "project-level"
        errors.append(f"missing {location} results.md")
    elif not re.search(
        r"prediction moment", results_path.read_text(encoding="utf-8"), re.IGNORECASE
    ):
        warnings.append("results.md does not mention the prediction moment")
    if (
        limited_remote_source
        and results_path.exists()
        and "limited_remote_source" not in results_path.read_text(encoding="utf-8")
    ):
        errors.append(
            "results.md: explicit remote fingerprint override must disclose "
            "reproducibility_status limited_remote_source"
        )
    if strict_21 and args.run_inference_test:
        if errors:
            warnings.append(
                "inference execution was skipped because static artifact "
                "validation failed"
            )
        else:
            execute_inference_cases(
                project,
                artifacts,
                cases,
                inference_test.get("prediction_constraints"),
                args.inference_timeout_seconds,
                errors,
            )


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    artifacts_argument = Path(args.artifacts_dir)
    if artifacts_argument.is_absolute() or ".." in artifacts_argument.parts:
        print("ERROR: --artifacts-dir must be a project-relative directory")
        return 1
    artifacts = project / artifacts_argument
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
    config_object = config if isinstance(config, dict) else {}
    mode = config_object.get("mode", "model-building")
    analysis_only = mode == "analysis-only"
    legacy_run = isinstance(config, dict) and config.get("schema_version") is None
    strict_21 = (
        isinstance(config, dict)
        and config.get("schema_version") == CURRENT_SCHEMA_VERSION
    )
    if strict_21:
        try:
            resolved_artifacts = artifacts.resolve()
        except (OSError, RuntimeError) as exc:
            print(f"ERROR: could not resolve selected artifact directory: {exc}")
            return 1
        if (
            resolved_artifacts != artifacts.absolute()
            or not path_is_within(resolved_artifacts, project)
            or path_has_symlink_component(artifacts, project)
        ):
            print(
                "ERROR: schema 2.1 --artifacts-dir must resolve without symlinks "
                "inside the project directory"
            )
            return 1
        artifacts = resolved_artifacts
    if legacy_run and not analysis_only:
        required = LEGACY_MODEL_FILES
    else:
        required = ANALYSIS_FILES if analysis_only else MODEL_FILES
        if strict_21 and not analysis_only:
            required = required | STRICT_MODEL_FILES
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

    documents = {"config.json": config_object}
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
        if version not in COMPATIBLE_SCHEMA_VERSIONS
    ]
    if legacy_files:
        warnings.append(
            "legacy v1 artifacts lack schema_version: " + ", ".join(legacy_files)
        )
    if unsupported:
        errors.append("unsupported schema versions: " + ", ".join(unsupported))
    if strict_21:
        mismatched = [
            f"{filename}={version}"
            for filename, version in versions.items()
            if version != CURRENT_SCHEMA_VERSION
        ]
        if mismatched:
            errors.append(
                "schema 2.1 runs require consistent versioned JSON artifacts: "
                + ", ".join(mismatched)
            )
    elif isinstance(config, dict) and config.get("schema_version") == "2.0":
        warnings.append(
            "schema v2.0 accepted for compatibility; strict v2.1 split, lineage, "
            "candidate, inference-output, and environment safeguards were not "
            "applied"
        )

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
    try:
        exit_code = main()
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        print(
            "ERROR: malformed artifact contract could not be validated safely: "
            f"{type(exc).__name__}: {exc}"
        )
        exit_code = 1
    sys.exit(exit_code)
