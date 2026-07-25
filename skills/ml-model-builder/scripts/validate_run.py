#!/usr/bin/env python3
"""Validate ml-model-builder artifacts without importing the trained model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    project: Path, artifacts: Path, documents, errors, warnings, legacy=False
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
        if nested(config, "split", "holdout_target_sealed") is not True:
            errors.append("config.json: split.holdout_target_sealed must be true")
        if nested(config, "analysis", "target_aware_partition") != "train":
            errors.append(
                "config.json: analysis.target_aware_partition must be 'train'"
            )
        if nested(metrics, "final", "eval_set") != "holdout_test":
            errors.append("metrics.json: final.eval_set must be 'holdout_test'")
        if nested(metrics, "final", "score") is None:
            errors.append("metrics.json: final.score is required")
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
    legacy = isinstance(config, dict) and config.get("schema_version") is None
    if legacy and not analysis_only:
        required = LEGACY_MODEL_FILES
    else:
        required = ANALYSIS_FILES if analysis_only else MODEL_FILES
    for filename in sorted(required):
        path = artifacts / filename
        if not path.exists():
            errors.append(f"missing artefacts/{filename}")
    if (
        not legacy
        and not analysis_only
        and not (
            (artifacts / "requirements.lock").exists()
            or (artifacts / "environment.lock").exists()
        )
    ):
        errors.append("missing pinned inference environment")

    if not legacy:
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
    legacy = [filename for filename, version in versions.items() if version is None]
    unsupported = [
        f"{filename}={version}"
        for filename, version in versions.items()
        if version not in {None, "2.0"}
    ]
    if legacy:
        warnings.append("legacy v1 artifacts lack schema_version: " + ", ".join(legacy))
    if unsupported:
        errors.append("unsupported schema versions: " + ", ".join(unsupported))

    profile = documents.get("data_profile.json") or {}
    if not legacy and profile.get("mode") != (
        "analysis-only" if analysis_only else "model"
    ):
        errors.append("data_profile.json: mode does not match config.json")
    if not analysis_only:
        validate_model(project, artifacts, documents, errors, warnings, legacy=legacy)

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        print(f"Validation failed with {len(errors)} error(s).")
        return 1
    print(
        f"Validated {'analysis-only' if analysis_only else 'model'} artifacts "
        f"at {artifacts} ({len(warnings)} warning(s))."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
