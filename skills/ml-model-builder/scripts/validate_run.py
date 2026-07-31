#!/usr/bin/env python3
"""Validate one project-local ML run and optionally exercise its inference CLI.

The contract shipped with the project uses one run manifest, one report,
backend-appropriate artifacts, and inline inference cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

BACKEND_NAMES = ("classical", "autogluon", "sap_rpt")
BACKEND_SET = set(BACKEND_NAMES)
NATIVE_THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
REQUIRED_INFERENCE_CASE_KINDS = {
    "representative",
    "single_row",
    "empty_input",
    "missing_required_column",
}
REQUIRED_ROOT_FILES = {
    "run.json",
    "report.html",
    "results.md",
    "infer.py",
    "requirements.lock",
    "validation.json",
}
ALLOWED_ROOT_ENTRIES = REQUIRED_ROOT_FILES | {"train.py", "backends"}
FORBIDDEN_NAMES = {
    "README.md",
    "data_profile.json",
    "data_report.html",
    "data_summary.md",
    "diagnostics",
    "eda_report.html",
    "figures",
    "inference_outputs",
    "__pycache__",
}
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
PINNED_REQUIREMENT_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?"
    r"==[A-Za-z0-9][A-Za-z0-9._+!-]*(?:\s*;\s*.+)?$"
)
RPT_FORBIDDEN_KEY_RE = re.compile(
    r"(?:^|_)(?:train(?:ed|ing)?|fit(?:ted|ting)?|hyperparameters?|"
    r"search|optuna|trials?)(?:_|$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", default=".")
    parser.add_argument("--artifacts-dir", default="artefacts")
    parser.add_argument(
        "--run-inference-test",
        action="store_true",
        help="Run validation.json inference cases without a shell",
    )
    parser.add_argument("--inference-timeout-seconds", type=int, default=120)
    return parser.parse_args()


def read_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: invalid JSON ({exc})")
        return None


def set_validation_status(path: Path, status: str) -> str | None:
    """Atomically update validation status without touching case definitions."""
    temporary_path: Path | None = None
    try:
        existing_mode = path.stat().st_mode & 0o7777
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            return "validation.json: root value must be an object"
        document["status"] = status
        document["validated_at"] = (
            datetime.now(timezone.utc).isoformat() if status == "passed" else None
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".validation-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.chmod(temporary_path, existing_mode)
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return f"validation.json: cannot atomically set status {status!r} ({exc})"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return None


def require(condition: bool, message: str, errors: list[str]) -> bool:
    if not condition:
        errors.append(message)
    return condition


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_string_list(value: Any, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (not nonempty or bool(value))
        and all(is_nonempty_string(item) for item in value)
        and len(value) == len(set(value))
    )


def is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value.strip()))


def parse_timestamp(value: Any, field: str, errors: list[str]) -> datetime | None:
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


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def iter_keys(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{path}.{key}" if path else str(key)
            yield current, str(key)
            yield from iter_keys(item, current)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_keys(item, f"{path}[{index}]")


def reject_key(value: Any, key_name: str, label: str, errors: list[str]) -> None:
    for path, key in iter_keys(value):
        if key == key_name:
            errors.append(f"{label}: forbidden field {path!r}")


def validate_known_keys(
    value: Any,
    allowed: set[str],
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        return
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append(f"{label}: unsupported fields: {', '.join(unknown)}")


def resolve_artifact_path(
    run_dir: Path,
    raw_path: Any,
    field: str,
    errors: list[str],
    *,
    kind: str,
) -> Path | None:
    if not is_nonempty_string(raw_path):
        errors.append(f"{field} must be a non-empty relative path")
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        errors.append(f"{field} must be relative to the run directory")
        return None
    resolved = (run_dir / candidate).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError:
        errors.append(f"{field} escapes the run directory")
        return None
    if kind == "file" and not resolved.is_file():
        errors.append(f"{field} does not identify a file: {raw_path}")
    elif kind == "dir" and not resolved.is_dir():
        errors.append(f"{field} does not identify a directory: {raw_path}")
    return resolved


def directory_file_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class _ReportAssetParser(HTMLParser):
    ASSET_ATTRIBUTES: ClassVar[dict[str, str]] = {
        "script": "src",
        "img": "src",
        "link": "href",
        "iframe": "src",
        "source": "src",
        "video": "src",
        "audio": "src",
        "object": "data",
        "embed": "src",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attribute = self.ASSET_ATTRIBUTES.get(tag.lower())
        if not attribute:
            return
        for name, value in attrs:
            if name.lower() == attribute and value is not None:
                self.references.append((tag.lower(), attribute, value.strip()))


def _is_embedded_asset(reference: str) -> bool:
    lowered = reference.lower()
    return not reference or lowered.startswith(("data:", "#", "about:blank"))


def validate_report(path: Path, errors: list[str]) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"report.html: cannot read ({exc})")
        return
    require(
        "<html" in source.lower() and "</html>" in source.lower(),
        "report.html: must be a complete HTML document",
        errors,
    )
    parser = _ReportAssetParser()
    parser.feed(source)
    for tag, attribute, reference in parser.references:
        if not _is_embedded_asset(reference):
            errors.append(
                "report.html: must be self-contained; "
                f"<{tag}> {attribute} references {reference!r}"
            )
    for match in re.finditer(
        r"url\(\s*(['\"]?)(.*?)\1\s*\)",
        source,
        re.IGNORECASE | re.DOTALL,
    ):
        reference = match.group(2).strip()
        if not _is_embedded_asset(reference):
            errors.append(
                "report.html: must be self-contained; CSS url() references "
                f"{reference!r}"
            )
    if re.search(
        r"@import\s+(?:url\()?\s*['\"]?(?!data:|#)",
        source,
        re.IGNORECASE,
    ):
        errors.append("report.html: must not use external CSS @import")
    if re.search(r"\b(?:fetch|XMLHttpRequest)\s*\(", source):
        errors.append("report.html: must not fetch external or local assets")


def normalized_handoff_text(path: Path, *, html_document: bool) -> str | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if html_document:
        source = re.sub(r"<[^>]+>", " ", source)
        source = unescape(source)
    return re.sub(r"[\s_-]+", " ", source.lower()).strip()


def _contains_numeric_score(
    text: str,
    backend_label: str,
    expected_score: float,
) -> bool:
    """Match a displayed score numerically near its backend label."""
    number_pattern = re.compile(
        r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?\s*%?",
        re.IGNORECASE,
    )
    for backend_match in re.finditer(re.escape(backend_label), text):
        window = text[backend_match.start() : backend_match.end() + 240]
        for number_match in number_pattern.finditer(window):
            token = number_match.group(0).strip()
            percent = token.endswith("%")
            if percent:
                token = token[:-1].strip()
            try:
                value = float(token)
            except ValueError:
                continue
            if percent:
                value /= 100.0
            if math.isclose(
                value,
                expected_score,
                rel_tol=1e-4,
                abs_tol=5e-4,
            ):
                return True
    return False


def validate_handoff_content(
    run_document: Any,
    report_path: Path,
    results_path: Path,
    errors: list[str],
) -> None:
    if not isinstance(run_document, dict):
        return
    report = normalized_handoff_text(report_path, html_document=True)
    results = normalized_handoff_text(results_path, html_document=False)
    if report is None or results is None:
        return
    backends = run_document.get("backends")
    if not isinstance(backends, dict):
        return
    documents = (("report.html", report), ("results.md", results))

    for backend, payload in backends.items():
        backend_label = backend.replace("_", " ")
        for name, text in documents:
            if backend_label not in text:
                errors.append(
                    f"{name}: must include the approved {backend_label} result or status"
                )
            if isinstance(payload, dict) and payload.get("status") not in text:
                errors.append(
                    f"{name}: must include the {backend_label} backend status "
                    f"{payload.get('status')!r}"
                )
        if isinstance(payload, dict) and payload.get("status") == "completed":
            score = nested(payload, "evaluation", "score")
            if is_finite_number(score):
                for name, text in documents:
                    if not _contains_numeric_score(
                        text,
                        backend_label,
                        float(score),
                    ):
                        errors.append(
                            f"{name}: must include the {backend_label} "
                            "primary-metric score as a numeric value within "
                            "display precision"
                        )

    metric = nested(run_document, "evaluation", "primary_metric", "name")
    if is_nonempty_string(metric):
        metric_label = metric.replace("_", " ").lower()
        for name, text in documents:
            if metric_label not in text and metric.lower() not in text:
                errors.append(
                    f"{name}: must identify the shared primary metric {metric!r}"
                )

    selection = run_document.get("selection")
    for name, text in documents:
        for field, phrase in (
            ("predictive_winner", "predictive winner"),
            ("operational_recommendation", "operational recommendation"),
        ):
            selected_backend = (
                selection.get(field) if isinstance(selection, dict) else None
            )
            if phrase not in text:
                errors.append(f"{name}: must state the {phrase}")
            if is_nonempty_string(selected_backend):
                label = selected_backend.replace("_", " ").lower()
                if label not in text:
                    errors.append(f"{name}: must identify the {phrase} backend")
        if "infer.py" not in text:
            errors.append(f"{name}: must include the unified infer.py command")
        required_concepts = {
            "limitations": ("limitation", "known constraint"),
            "uncertainty": (
                "uncertainty",
                "confidence interval",
                "fold variation",
                "bootstrap",
            ),
            "monitoring": ("monitoring", "monitor ", "drift"),
            "intended use": ("intended use",),
            "prohibited use": ("prohibited use", "not for"),
        }
        for concept, alternatives in required_concepts.items():
            if not any(alternative in text for alternative in alternatives):
                errors.append(f"{name}: must discuss {concept}")

        if "classical" in backends:
            for concept in ("baseline", "leaderboard"):
                if concept not in text:
                    errors.append(f"{name}: must include the classical {concept}")
        if "autogluon" in backends:
            preset = nested(backends, "autogluon", "build", "preset")
            if "preset" not in text or (
                is_nonempty_string(preset)
                and preset.replace("_", " ").lower() not in text
            ):
                errors.append(f"{name}: must include the AutoGluon preset")
            concepts = {
                "deployment packaging": (
                    "deployment clone",
                    "clone for deployment",
                    "clone_for_deployment",
                ),
                "internal component failures": (
                    "internal failure",
                    "component failure",
                    "failed component",
                    "skipped component",
                ),
            }
            for concept, alternatives in concepts.items():
                if not any(alternative in text for alternative in alternatives):
                    errors.append(f"{name}: must include AutoGluon {concept}")
            failures = nested(backends, "autogluon", "internal_failures")
            if isinstance(failures, list):
                for failure in failures:
                    component = (
                        failure.get("component") if isinstance(failure, dict) else None
                    )
                    if is_nonempty_string(component) and component.lower() not in text:
                        errors.append(
                            f"{name}: must include AutoGluon internal failure "
                            f"component {component!r}"
                        )
        if "sap_rpt" in backends:
            concepts = {
                "context": ("context",),
                "access": ("access",),
                "latency": ("latency",),
                "model ID": ("model id", "model_id"),
                "retrieval": ("retrieval", "vectorsearch", "random::"),
                "configuration coverage": (
                    "approved configurations",
                    "configuration ledger",
                ),
            }
            for concept, alternatives in concepts.items():
                if not any(alternative in text for alternative in alternatives):
                    errors.append(f"{name}: must include the SAP RPT {concept} details")
            configurations = nested(backends, "sap_rpt", "configurations")
            if isinstance(configurations, list):
                model_ids = {
                    configuration.get("model_id")
                    for configuration in configurations
                    if isinstance(configuration, dict)
                    and is_nonempty_string(configuration.get("model_id"))
                }
                for model_id in model_ids:
                    normalized_model_id = re.sub(
                        r"[\s_-]+",
                        " ",
                        model_id.lower(),
                    ).strip()
                    if normalized_model_id not in text:
                        errors.append(
                            f"{name}: must include SAP RPT model ID {model_id!r}"
                        )


def validate_requirements(path: Path, errors: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"requirements.lock: cannot read ({exc})")
        return
    requirements = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    comments = [
        line.lstrip()[1:].strip().lower()
        for line in lines
        if line.lstrip().startswith("#")
    ]
    if not requirements and not any(
        "no third-party" in comment or "standard library only" in comment
        for comment in comments
    ):
        errors.append(
            "requirements.lock: add exact package pins or explicitly state "
            "'No third-party dependencies'"
        )
    for line in requirements:
        if not PINNED_REQUIREMENT_RE.fullmatch(line):
            errors.append(
                "requirements.lock: every dependency must use an exact "
                f"'package==version' pin; found {line!r}"
            )


def validate_problem_and_data(document: dict[str, Any], errors: list[str]) -> None:
    problem = document.get("problem")
    require(isinstance(problem, dict), "run.json: problem must be an object", errors)
    if isinstance(problem, dict):
        for field in ("task", "target", "prediction_moment", "row_grain"):
            require(
                is_nonempty_string(problem.get(field)),
                f"run.json: problem.{field} must be a non-empty string",
                errors,
            )
        require(
            is_nonempty_string(problem.get("intended_use")),
            "run.json: problem.intended_use must be a non-empty string",
            errors,
        )
        require(
            is_string_list(problem.get("prohibited_uses"), nonempty=True),
            "run.json: problem.prohibited_uses must be a non-empty unique string list",
            errors,
        )
        contract = problem.get("feature_contract")
        require(
            isinstance(contract, dict),
            "run.json: problem.feature_contract must be an object",
            errors,
        )
        if isinstance(contract, dict):
            included = contract.get("included")
            require(
                is_string_list(included, nonempty=True),
                "run.json: problem.feature_contract.included must be a "
                "non-empty unique string list",
                errors,
            )
            require(
                is_string_list(contract.get("excluded")),
                "run.json: problem.feature_contract.excluded must be a "
                "unique string list",
                errors,
            )
            if (
                is_nonempty_string(problem.get("target"))
                and isinstance(included, list)
                and problem["target"] in included
            ):
                errors.append(
                    "run.json: target must not appear in "
                    "problem.feature_contract.included"
                )

    data = document.get("data")
    require(isinstance(data, dict), "run.json: data must be an object", errors)
    if isinstance(data, dict):
        require(
            is_nonempty_string(data.get("source")),
            "run.json: data.source must be a non-empty string",
            errors,
        )
        require(
            is_sha256(data.get("fingerprint")),
            "run.json: data.fingerprint must be a SHA-256 digest",
            errors,
        )
        require(
            is_positive_integer(data.get("row_count")),
            "run.json: data.row_count must be a positive integer",
            errors,
        )


def validate_preflight(document: dict[str, Any], errors: list[str]) -> None:
    preflight = document.get("modeling_preflight")
    require(
        isinstance(preflight, dict),
        "run.json: modeling_preflight must be an object",
        errors,
    )
    if not isinstance(preflight, dict):
        return
    require(
        preflight.get("status") == "passed",
        "run.json: modeling_preflight.status must be 'passed'",
        errors,
    )
    checks = (
        "target_validated",
        "row_grain_validated",
        "prediction_moment_validated",
        "leakage_reviewed",
        "feature_availability_reviewed",
        "split_suitable",
    )
    for check in checks:
        require(
            preflight.get(check) is True,
            f"run.json: modeling_preflight.{check} must be true",
            errors,
        )
    require(
        isinstance(preflight.get("findings"), list),
        "run.json: modeling_preflight.findings must be a list",
        errors,
    )


def validate_evaluation(
    document: dict[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    evaluation = document.get("evaluation")
    require(
        isinstance(evaluation, dict),
        "run.json: evaluation must be an object",
        errors,
    )
    if not isinstance(evaluation, dict):
        return None
    require(
        is_nonempty_string(evaluation.get("design")),
        "run.json: evaluation.design must be a non-empty string",
        errors,
    )
    for field in ("split_fingerprint", "evaluation_rows_fingerprint"):
        require(
            is_sha256(evaluation.get(field)),
            f"run.json: evaluation.{field} must be a SHA-256 digest",
            errors,
        )
    metric = evaluation.get("primary_metric")
    require(
        isinstance(metric, dict),
        "run.json: evaluation.primary_metric must be an object",
        errors,
    )
    if isinstance(metric, dict):
        require(
            is_nonempty_string(metric.get("name")),
            "run.json: evaluation.primary_metric.name must be non-empty",
            errors,
        )
        require(
            metric.get("direction") in {"maximize", "minimize"},
            "run.json: evaluation.primary_metric.direction must be "
            "'maximize' or 'minimize'",
            errors,
        )
    return evaluation


def _validate_budget(
    backend: str,
    budget: Any,
    errors: list[str],
) -> None:
    field = f"run.json: approval.tracks.{backend}.budget"
    if not require(
        isinstance(budget, dict) and bool(budget),
        f"{field} must be a non-empty object for an approved track",
        errors,
    ):
        return
    common_fields = {
        "cpu_count",
        "parallel_jobs",
        "memory_gb",
        "gpu_enabled",
    }
    track_fields = {
        "classical": {
            "candidate_families",
            "time_limit_seconds",
            "optuna_trials",
            "minimum_family_coverage",
        },
        "autogluon": {
            "preset",
            "run_mode",
            "time_limit_seconds",
            "runtime_estimate",
            "disk_gb",
        },
        "sap_rpt": {
            "max_requests",
            "max_context_rows",
            "max_request_rows",
            "max_query_batch_rows",
            "max_columns",
            "max_retries",
            "timeout_seconds",
        },
    }
    validate_known_keys(
        budget,
        common_fields | track_fields[backend],
        field,
        errors,
    )
    for name in ("cpu_count", "parallel_jobs"):
        require(
            is_positive_integer(budget.get(name)),
            f"{field}.{name} must be a positive integer",
            errors,
        )
    require(
        is_finite_number(budget.get("memory_gb")) and float(budget["memory_gb"]) > 0,
        f"{field}.memory_gb must be a positive finite number",
        errors,
    )
    require(
        isinstance(budget.get("gpu_enabled"), bool),
        f"{field}.gpu_enabled must be boolean",
        errors,
    )
    if backend == "classical":
        require(
            is_string_list(budget.get("candidate_families"), nonempty=True),
            f"{field}.candidate_families must be a non-empty unique string list",
            errors,
        )
        require(
            is_positive_integer(budget.get("time_limit_seconds")),
            f"{field}.time_limit_seconds must be a positive integer",
            errors,
        )
        require(
            is_positive_integer(budget.get("optuna_trials")),
            f"{field}.optuna_trials must be a positive integer",
            errors,
        )
        require(
            is_positive_integer(budget.get("minimum_family_coverage")),
            f"{field}.minimum_family_coverage must be a positive integer",
            errors,
        )
    elif backend == "autogluon":
        require(
            is_nonempty_string(budget.get("preset")),
            f"{field}.preset must be a non-empty string",
            errors,
        )
        run_mode = budget.get("run_mode")
        require(
            run_mode in {"run_to_completion", "time_limited"},
            f"{field}.run_mode must be 'run_to_completion' or 'time_limited'",
            errors,
        )
        require(
            "time_limit_seconds" in budget,
            f"{field}.time_limit_seconds must be present",
            errors,
        )
        if run_mode == "run_to_completion":
            require(
                budget.get("time_limit_seconds") is None,
                f"{field}.time_limit_seconds must be null when run_mode is "
                "'run_to_completion'",
                errors,
            )
        elif run_mode == "time_limited":
            require(
                is_positive_integer(budget.get("time_limit_seconds")),
                f"{field}.time_limit_seconds must be a positive integer when "
                "run_mode is 'time_limited'",
                errors,
            )
        estimate = budget.get("runtime_estimate")
        if require(
            isinstance(estimate, dict),
            f"{field}.runtime_estimate must be an object",
            errors,
        ):
            validate_known_keys(
                estimate,
                {"lower_seconds", "upper_seconds", "basis"},
                f"{field}.runtime_estimate",
                errors,
            )
            lower = estimate.get("lower_seconds")
            upper = estimate.get("upper_seconds")
            require(
                is_positive_integer(lower),
                f"{field}.runtime_estimate.lower_seconds must be a positive "
                "integer",
                errors,
            )
            require(
                is_positive_integer(upper),
                f"{field}.runtime_estimate.upper_seconds must be a positive "
                "integer",
                errors,
            )
            if is_positive_integer(lower) and is_positive_integer(upper):
                require(
                    lower <= upper,
                    f"{field}.runtime_estimate.lower_seconds cannot exceed "
                    "upper_seconds",
                    errors,
                )
            require(
                is_nonempty_string(estimate.get("basis")),
                f"{field}.runtime_estimate.basis must be non-empty",
                errors,
            )
        require(
            is_finite_number(budget.get("disk_gb")) and float(budget["disk_gb"]) > 0,
            f"{field}.disk_gb must be a positive finite number",
            errors,
        )
    else:
        for name in (
            "max_requests",
            "max_context_rows",
            "max_request_rows",
            "max_query_batch_rows",
            "max_columns",
            "timeout_seconds",
        ):
            require(
                is_positive_integer(budget.get(name)),
                f"{field}.{name} must be a positive integer",
                errors,
            )
        require(
            is_nonnegative_integer(budget.get("max_retries")),
            f"{field}.max_retries must be a non-negative integer",
            errors,
        )
        if all(
            is_positive_integer(budget.get(name))
            for name in (
                "max_context_rows",
                "max_request_rows",
                "max_query_batch_rows",
            )
        ):
            require(
                budget["max_query_batch_rows"] <= budget["max_request_rows"],
                f"{field}.max_query_batch_rows cannot exceed max_request_rows",
                errors,
            )
            require(
                budget["max_context_rows"] + budget["max_query_batch_rows"]
                <= budget["max_request_rows"],
                f"{field}.max_context_rows plus max_query_batch_rows cannot "
                "exceed max_request_rows",
                errors,
            )


def _validate_rpt_plan(
    plan: Any,
    budget: Any,
    errors: list[str],
) -> None:
    field = "run.json: approval.tracks.sap_rpt.plan"
    if not require(
        isinstance(plan, dict) and bool(plan),
        f"{field} must be a non-empty object for an approved SAP RPT track",
        errors,
    ):
        return
    validate_known_keys(
        plan,
        {
            "model_ids",
            "full_context_fits",
            "use_full_context_when_supported",
            "context_size_candidates",
            "retrieval_strategies",
            "context_seed",
            "input_format",
            "retrieval_extra_status",
            "estimated_configurations",
        },
        field,
        errors,
    )
    require(
        is_string_list(plan.get("model_ids"), nonempty=True),
        f"{field}.model_ids must be a non-empty unique string list",
        errors,
    )
    require(
        isinstance(plan.get("full_context_fits"), bool),
        f"{field}.full_context_fits must be boolean",
        errors,
    )
    require(
        plan.get("use_full_context_when_supported") is True,
        f"{field}.use_full_context_when_supported must be true",
        errors,
    )
    sizes = plan.get("context_size_candidates")
    valid_sizes = (
        isinstance(sizes, list)
        and bool(sizes)
        and all(is_positive_integer(value) for value in sizes)
        and len(sizes) == len(set(sizes))
    )
    require(
        valid_sizes,
        f"{field}.context_size_candidates must be a non-empty unique list of "
        "positive integers",
        errors,
    )
    if valid_sizes and isinstance(budget, dict):
        maximum = budget.get("max_context_rows")
        if is_positive_integer(maximum):
            require(
                max(sizes) <= maximum,
                f"{field}.context_size_candidates cannot exceed the approved "
                "max_context_rows",
                errors,
            )
    strategies = plan.get("retrieval_strategies")
    valid_strategies = is_string_list(strategies, nonempty=True) and set(
        strategies
    ).issubset({"full", "random", "vectorsearch"})
    require(
        valid_strategies,
        f"{field}.retrieval_strategies must be a non-empty unique subset of "
        "full, random, and vectorsearch",
        errors,
    )
    if valid_strategies and plan.get("full_context_fits") is True:
        require(
            "full" in strategies,
            f"{field}.retrieval_strategies must include full when full context "
            "fits",
            errors,
        )
    if valid_strategies and plan.get("full_context_fits") is False:
        require(
            "random" in strategies,
            f"{field}.retrieval_strategies must include random when full "
            "context does not fit",
            errors,
        )
    require(
        is_nonnegative_integer(plan.get("context_seed")),
        f"{field}.context_seed must be a non-negative integer",
        errors,
    )
    require(
        plan.get("input_format") in {"column_json", "csv", "parquet"},
        f"{field}.input_format must be column_json, csv, or parquet",
        errors,
    )
    extra_status = plan.get("retrieval_extra_status")
    require(
        extra_status
        in {"installed", "approved_install", "unavailable", "not_required"},
        f"{field}.retrieval_extra_status is invalid",
        errors,
    )
    if valid_strategies and "vectorsearch" in strategies:
        require(
            extra_status in {"installed", "approved_install"},
            f"{field}.retrieval_extra_status must be installed or "
            "approved_install when vectorsearch is planned",
            errors,
        )
    require(
        is_positive_integer(plan.get("estimated_configurations")),
        f"{field}.estimated_configurations must be a positive integer",
        errors,
    )


def _validate_approval_amendments(
    amendments: Any,
    errors: list[str],
) -> None:
    field = "run.json: approval.amendments"
    if not require(isinstance(amendments, list), f"{field} must be a list", errors):
        return
    identifiers: list[str] = []
    for index, amendment in enumerate(amendments):
        prefix = f"{field}[{index}]"
        if not require(
            isinstance(amendment, dict),
            f"{prefix} must be an object",
            errors,
        ):
            continue
        validate_known_keys(
            amendment,
            {"id", "approved_at", "reason", "changes"},
            prefix,
            errors,
        )
        identifier = amendment.get("id")
        require(
            is_nonempty_string(identifier),
            f"{prefix}.id must be non-empty",
            errors,
        )
        if is_nonempty_string(identifier):
            identifiers.append(identifier)
        parse_timestamp(amendment.get("approved_at"), f"{prefix}.approved_at", errors)
        require(
            is_nonempty_string(amendment.get("reason")),
            f"{prefix}.reason must be non-empty",
            errors,
        )
        changes = amendment.get("changes")
        if not require(
            isinstance(changes, list) and bool(changes),
            f"{prefix}.changes must be a non-empty list",
            errors,
        ):
            continue
        paths: list[str] = []
        for change_index, change in enumerate(changes):
            change_prefix = f"{prefix}.changes[{change_index}]"
            if not require(
                isinstance(change, dict),
                f"{change_prefix} must be an object",
                errors,
            ):
                continue
            validate_known_keys(
                change,
                {"path", "before", "after"},
                change_prefix,
                errors,
            )
            path = change.get("path")
            require(
                is_nonempty_string(path)
                and path.startswith(
                    (
                        "scope.",
                        "tracks.classical.",
                        "tracks.autogluon.",
                        "tracks.sap_rpt.",
                    )
                ),
                f"{change_prefix}.path must identify an approval scope or track field",
                errors,
            )
            if is_nonempty_string(path):
                paths.append(path)
            require(
                "before" in change and "after" in change,
                f"{change_prefix} must record before and after values",
                errors,
            )
            if "before" in change and "after" in change:
                require(
                    change["before"] != change["after"],
                    f"{change_prefix}.before and after must differ",
                    errors,
                )
        require(
            len(paths) == len(set(paths)),
            f"{prefix}.changes paths must be unique",
            errors,
        )
    require(
        len(identifiers) == len(set(identifiers)),
        f"{field} ids must be unique",
        errors,
    )


def _validate_remote_transfers(
    transfers: Any,
    selected: set[str],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    field = "run.json: approval.remote_transfers"
    if not require(isinstance(transfers, list), f"{field} must be a list", errors):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for index, transfer in enumerate(transfers):
        prefix = f"{field}[{index}]"
        if not require(
            isinstance(transfer, dict),
            f"{prefix} must be an object",
            errors,
        ):
            continue
        validate_known_keys(
            transfer,
            {
                "id",
                "approved_at",
                "backend",
                "destination",
                "purpose",
                "data_scope",
            },
            prefix,
            errors,
        )
        identifier = transfer.get("id")
        require(
            is_nonempty_string(identifier),
            f"{prefix}.id must be non-empty",
            errors,
        )
        if is_nonempty_string(identifier):
            require(
                identifier not in indexed,
                f"{field} ids must be unique",
                errors,
            )
            indexed[identifier] = transfer
        parse_timestamp(transfer.get("approved_at"), f"{prefix}.approved_at", errors)
        backend = transfer.get("backend")
        require(
            backend == "sap_rpt",
            f"{prefix}.backend must be 'sap_rpt'",
            errors,
        )
        require(
            backend in selected,
            f"{prefix}.backend must name an approved track",
            errors,
        )
        for name in ("destination", "purpose"):
            require(
                is_nonempty_string(transfer.get(name)),
                f"{prefix}.{name} must be non-empty",
                errors,
            )
        data_scope = transfer.get("data_scope")
        if not require(
            isinstance(data_scope, dict),
            f"{prefix}.data_scope must be an object",
            errors,
        ):
            continue
        validate_known_keys(
            data_scope,
            {"features", "labels", "query_rows", "identifiers"},
            f"{prefix}.data_scope",
            errors,
        )
        require(
            is_string_list(data_scope.get("features"), nonempty=True),
            f"{prefix}.data_scope.features must be a non-empty unique string list",
            errors,
        )
        require(
            data_scope.get("labels") is True,
            f"{prefix}.data_scope.labels must be true for labelled RPT context",
            errors,
        )
        require(
            data_scope.get("query_rows") is True,
            f"{prefix}.data_scope.query_rows must be true",
            errors,
        )
        require(
            is_string_list(data_scope.get("identifiers")),
            f"{prefix}.data_scope.identifiers must be a unique string list",
            errors,
        )
    return indexed


def validate_approval(
    document: dict[str, Any],
    errors: list[str],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    approval = document.get("approval")
    require(
        isinstance(approval, dict),
        "run.json: approval must be an object",
        errors,
    )
    if not isinstance(approval, dict):
        return set(), {}
    validate_known_keys(
        approval,
        {"approved_at", "scope", "tracks", "amendments", "remote_transfers"},
        "run.json: approval",
        errors,
    )
    parse_timestamp(
        approval.get("approved_at"), "run.json: approval.approved_at", errors
    )
    scope = approval.get("scope")
    require(
        isinstance(scope, dict),
        "run.json: approval.scope must be an object",
        errors,
    )
    if isinstance(scope, dict):
        for field in (
            "target",
            "feature_contract",
            "split_design",
            "primary_metric",
        ):
            require(
                scope.get(field) is True,
                f"run.json: approval.scope.{field} must be true",
                errors,
            )
    tracks = approval.get("tracks")
    require(
        isinstance(tracks, dict),
        "run.json: approval.tracks must be an object",
        errors,
    )
    if not isinstance(tracks, dict):
        return set(), {}
    require(
        set(tracks) == BACKEND_SET,
        "run.json: approval.tracks must contain exactly classical, "
        "autogluon, and sap_rpt",
        errors,
    )
    selected: set[str] = set()
    for backend in BACKEND_NAMES:
        track = tracks.get(backend)
        require(
            isinstance(track, dict),
            f"run.json: approval.tracks.{backend} must be an object",
            errors,
        )
        if not isinstance(track, dict):
            continue
        allowed_track_fields = {"selected", "status", "budget"}
        if backend == "sap_rpt":
            allowed_track_fields.add("plan")
        validate_known_keys(
            track,
            allowed_track_fields,
            f"run.json: approval.tracks.{backend}",
            errors,
        )
        chosen = track.get("selected")
        require(
            isinstance(chosen, bool),
            f"run.json: approval.tracks.{backend}.selected must be boolean",
            errors,
        )
        expected_status = "approved" if chosen is True else "declined"
        require(
            track.get("status") == expected_status,
            f"run.json: approval.tracks.{backend}.status must be {expected_status!r}",
            errors,
        )
        if chosen is True:
            selected.add(backend)
            _validate_budget(backend, track.get("budget"), errors)
            if backend == "sap_rpt":
                _validate_rpt_plan(track.get("plan"), track.get("budget"), errors)
        else:
            require(
                track.get("budget") in (None, {}),
                f"run.json: approval.tracks.{backend}.budget must be null or "
                "empty when declined",
                errors,
            )
            if backend == "sap_rpt":
                require(
                    track.get("plan") in (None, {}),
                    "run.json: approval.tracks.sap_rpt.plan must be null or "
                    "empty when declined",
                    errors,
                )
    require(
        bool(selected),
        "run.json: approval must select at least one modeling track",
        errors,
    )
    _validate_approval_amendments(approval.get("amendments"), errors)
    remote_transfers = _validate_remote_transfers(
        approval.get("remote_transfers"),
        selected,
        errors,
    )
    return selected, remote_transfers


def validate_backend_evaluation(
    backend: str,
    value: dict[str, Any],
    evaluation: dict[str, Any] | None,
    errors: list[str],
) -> float | None:
    backend_eval = value.get("evaluation")
    require(
        isinstance(backend_eval, dict),
        f"run.json: backends.{backend}.evaluation must be an object for a "
        "completed backend",
        errors,
    )
    if not isinstance(backend_eval, dict):
        return None
    if isinstance(evaluation, dict):
        for field in ("split_fingerprint", "evaluation_rows_fingerprint"):
            require(
                backend_eval.get(field) == evaluation.get(field),
                f"run.json: backends.{backend}.evaluation.{field} must match "
                "the shared evaluation contract",
                errors,
            )
        expected_metric = nested(evaluation, "primary_metric", "name")
        require(
            backend_eval.get("primary_metric") == expected_metric,
            f"run.json: backends.{backend}.evaluation.primary_metric must "
            "match evaluation.primary_metric.name",
            errors,
        )
    score = backend_eval.get("score")
    require(
        is_finite_number(score),
        f"run.json: backends.{backend}.evaluation.score must be a finite number",
        errors,
    )
    return float(score) if is_finite_number(score) else None


def validate_classical(
    value: dict[str, Any],
    approved_budget: Any,
    run_dir: Path,
    errors: list[str],
) -> None:
    preprocessing = value.get("preprocessing")
    require(
        isinstance(preprocessing, dict) and preprocessing.get("scope") == "fold_local",
        "run.json: backends.classical.preprocessing.scope must be 'fold_local'",
        errors,
    )
    search = value.get("search")
    require(
        isinstance(search, dict),
        "run.json: backends.classical.search must be an object",
        errors,
    )
    if isinstance(search, dict):
        method = search.get("method")
        require(
            method in {"optuna", "none"},
            "run.json: backends.classical.search.method must be 'optuna' or 'none'",
            errors,
        )
        if method == "optuna":
            budget = search.get("trials_budget")
            completed = search.get("trials_completed")
            require(
                is_positive_integer(budget),
                "run.json: backends.classical.search.trials_budget must be "
                "a positive integer",
                errors,
            )
            require(
                is_nonnegative_integer(completed),
                "run.json: backends.classical.search.trials_completed must be "
                "a non-negative integer",
                errors,
            )
            if is_positive_integer(budget) and is_nonnegative_integer(completed):
                require(
                    completed <= budget,
                    "run.json: backends.classical.search.trials_completed "
                    "cannot exceed trials_budget",
                    errors,
                )
            if isinstance(approved_budget, dict):
                require(
                    budget == approved_budget.get("optuna_trials"),
                    "run.json: backends.classical.search.trials_budget must "
                    "match the approved Optuna trial budget",
                    errors,
                )
        elif method == "none":
            require(
                is_nonempty_string(search.get("reason")),
                "run.json: backends.classical.search.reason is required when "
                "search.method is 'none'",
                errors,
            )
    candidates = value.get("candidates")
    require(
        isinstance(candidates, list) and bool(candidates),
        "run.json: backends.classical.candidates must be a non-empty list",
        errors,
    )
    if isinstance(candidates, list):
        candidate_names: list[str] = []
        candidate_families: set[str] = set()
        for index, candidate in enumerate(candidates):
            prefix = f"run.json: backends.classical.candidates[{index}]"
            if not require(
                isinstance(candidate, dict), f"{prefix} must be an object", errors
            ):
                continue
            require(
                is_nonempty_string(candidate.get("name")),
                f"{prefix}.name must be non-empty",
                errors,
            )
            require(
                is_nonempty_string(candidate.get("family")),
                f"{prefix}.family must be non-empty",
                errors,
            )
            require(
                is_nonempty_string(candidate.get("consideration_basis")),
                f"{prefix}.consideration_basis must be non-empty",
                errors,
            )
            if is_nonempty_string(candidate.get("name")):
                candidate_names.append(candidate["name"])
            if is_nonempty_string(candidate.get("family")):
                candidate_families.add(candidate["family"])
            require(
                candidate.get("status") in {"completed", "failed", "excluded"},
                f"{prefix}.status must be completed, failed, or excluded",
                errors,
            )
            if candidate.get("status") == "completed":
                require(
                    is_finite_number(candidate.get("score")),
                    f"{prefix}.score must be finite for a completed candidate",
                    errors,
                )
            else:
                require(
                    is_nonempty_string(candidate.get("reason")),
                    f"{prefix}.reason is required when not completed",
                    errors,
                )
        require(
            len(candidate_names) == len(set(candidate_names)),
            "run.json: classical candidate names must be unique",
            errors,
        )
        if isinstance(approved_budget, dict) and is_string_list(
            approved_budget.get("candidate_families"),
            nonempty=True,
        ):
            require(
                set(approved_budget["candidate_families"]).issubset(candidate_families),
                "run.json: classical candidate ledger must cover every family "
                "in approval.tracks.classical.budget.candidate_families",
                errors,
            )
    artifacts = value.get("artifacts")
    require(
        isinstance(artifacts, dict),
        "run.json: backends.classical.artifacts must be an object",
        errors,
    )
    if isinstance(artifacts, dict):
        path = artifacts.get("model")
        resolved = resolve_artifact_path(
            run_dir,
            path,
            "run.json: backends.classical.artifacts.model",
            errors,
            kind="file",
        )
        if resolved is not None:
            expected_root = (run_dir / "backends" / "classical").resolve()
            try:
                resolved.relative_to(expected_root)
            except ValueError:
                errors.append(
                    "run.json: backends.classical.artifacts.model must be "
                    "inside backends/classical"
                )


def validate_autogluon(
    value: dict[str, Any],
    approved_budget: Any,
    run_dir: Path,
    errors: list[str],
) -> None:
    require(
        "search" not in value,
        "run.json: backends.autogluon must not declare an external search",
        errors,
    )
    require(
        "preprocessing" not in value,
        "run.json: backends.autogluon must not declare classical preprocessing",
        errors,
    )
    build = value.get("build")
    require(
        isinstance(build, dict),
        "run.json: backends.autogluon.build must be an object",
        errors,
    )
    if isinstance(build, dict):
        require(
            is_nonempty_string(build.get("preset")),
            "run.json: backends.autogluon.build.preset must be non-empty",
            errors,
        )
        run_mode = build.get("run_mode")
        require(
            run_mode in {"run_to_completion", "time_limited"},
            "run.json: backends.autogluon.build.run_mode must be "
            "'run_to_completion' or 'time_limited'",
            errors,
        )
        require(
            "time_limit_seconds" in build,
            "run.json: backends.autogluon.build.time_limit_seconds must be present",
            errors,
        )
        if run_mode == "run_to_completion":
            require(
                build.get("time_limit_seconds") is None,
                "run.json: backends.autogluon.build.time_limit_seconds must be "
                "null when run_mode is 'run_to_completion'",
                errors,
            )
        elif run_mode == "time_limited":
            require(
                is_positive_integer(build.get("time_limit_seconds")),
                "run.json: backends.autogluon.build.time_limit_seconds must be "
                "a positive integer when run_mode is 'time_limited'",
                errors,
            )
        if isinstance(approved_budget, dict):
            require(
                build.get("preset") == approved_budget.get("preset"),
                "run.json: backends.autogluon.build.preset must match the "
                "approved preset",
                errors,
            )
            require(
                build.get("run_mode") == approved_budget.get("run_mode"),
                "run.json: backends.autogluon.build.run_mode must match the "
                "approved run mode",
                errors,
            )
            require(
                build.get("time_limit_seconds")
                == approved_budget.get("time_limit_seconds"),
                "run.json: backends.autogluon.build.time_limit_seconds must "
                "match the approved time limit",
                errors,
            )
            if approved_budget.get("parallel_jobs") == 1:
                require(
                    build.get("fold_fitting_strategy") == "sequential_local",
                    "run.json: backends.autogluon.build.fold_fitting_strategy "
                    "must be 'sequential_local' when approved parallel_jobs is 1",
                    errors,
                )
        require(
            is_nonempty_string(build.get("fold_fitting_strategy")),
            "run.json: backends.autogluon.build.fold_fitting_strategy must be "
            "non-empty",
            errors,
        )
        require(
            is_nonempty_string(build.get("fold_fitting_strategy_reason")),
            "run.json: backends.autogluon.build.fold_fitting_strategy_reason "
            "must be non-empty",
            errors,
        )
        training_diagnostics = build.get("training_diagnostics")
        if require(
            isinstance(training_diagnostics, dict),
            "run.json: backends.autogluon.build.training_diagnostics must be "
            "an object",
            errors,
        ):
            validate_known_keys(
                training_diagnostics,
                {
                    "fit_summary_captured",
                    "elapsed_seconds",
                    "completion_status",
                    "stop_reason",
                },
                "run.json: backends.autogluon.build.training_diagnostics",
                errors,
            )
            require(
                training_diagnostics.get("fit_summary_captured") is True,
                "run.json: backends.autogluon.build.training_diagnostics."
                "fit_summary_captured must be true",
                errors,
            )
            require(
                is_finite_number(training_diagnostics.get("elapsed_seconds"))
                and float(training_diagnostics["elapsed_seconds"]) >= 0,
                "run.json: backends.autogluon.build.training_diagnostics."
                "elapsed_seconds must be a non-negative finite number",
                errors,
            )
            completion_status = training_diagnostics.get("completion_status")
            require(
                completion_status
                in {"completed_configuration", "time_limit_reached"},
                "run.json: backends.autogluon.build.training_diagnostics."
                "completion_status must be 'completed_configuration' or "
                "'time_limit_reached'",
                errors,
            )
            if run_mode == "run_to_completion":
                require(
                    completion_status == "completed_configuration",
                    "run.json: backends.autogluon.build.training_diagnostics."
                    "completion_status must be 'completed_configuration' for "
                    "run_to_completion",
                    errors,
                )
            require(
                is_nonempty_string(training_diagnostics.get("stop_reason")),
                "run.json: backends.autogluon.build.training_diagnostics."
                "stop_reason must be non-empty",
                errors,
            )
        predictor = resolve_artifact_path(
            run_dir,
            build.get("predictor_path"),
            "run.json: backends.autogluon.build.predictor_path",
            errors,
            kind="dir",
        )
        if predictor is not None:
            expected_root = (run_dir / "backends" / "autogluon").resolve()
            try:
                predictor.relative_to(expected_root)
            except ValueError:
                errors.append(
                    "run.json: backends.autogluon.build.predictor_path must be "
                    "inside backends/autogluon"
                )
            require(
                build.get("predictor_path") == "backends/autogluon/predictor",
                "run.json: backends.autogluon.build.predictor_path must point "
                "to the deployment clone at backends/autogluon/predictor",
                errors,
            )
        packaging = build.get("packaging")
        if require(
            isinstance(packaging, dict),
            "run.json: backends.autogluon.build.packaging must be an object",
            errors,
        ):
            validate_known_keys(
                packaging,
                {
                    "method",
                    "model",
                    "diagnostics_captured_before_clone",
                    "prediction_equivalence",
                    "training_predictor_retained",
                    "training_predictor_path",
                    "retention_reason",
                    "deployment_predictor_bytes",
                    "peak_packaging_disk_bytes",
                },
                "run.json: backends.autogluon.build.packaging",
                errors,
            )
            require(
                packaging.get("method") == "clone_for_deployment",
                "run.json: backends.autogluon.build.packaging.method must be "
                "'clone_for_deployment'",
                errors,
            )
            require(
                packaging.get("model") == "best",
                "run.json: backends.autogluon.build.packaging.model must be 'best'",
                errors,
            )
            require(
                packaging.get("diagnostics_captured_before_clone") is True,
                "run.json: backends.autogluon.build.packaging."
                "diagnostics_captured_before_clone must be true",
                errors,
            )
            equivalence = packaging.get("prediction_equivalence")
            if require(
                isinstance(equivalence, dict),
                "run.json: backends.autogluon.build.packaging."
                "prediction_equivalence must be an object",
                errors,
            ):
                validate_known_keys(
                    equivalence,
                    {"validated", "rows", "absolute_tolerance"},
                    "run.json: backends.autogluon.build.packaging."
                    "prediction_equivalence",
                    errors,
                )
                require(
                    equivalence.get("validated") is True,
                    "run.json: backends.autogluon.build.packaging."
                    "prediction_equivalence.validated must be true",
                    errors,
                )
                require(
                    is_positive_integer(equivalence.get("rows")),
                    "run.json: backends.autogluon.build.packaging."
                    "prediction_equivalence.rows must be a positive integer",
                    errors,
                )
                require(
                    is_finite_number(equivalence.get("absolute_tolerance"))
                    and float(equivalence["absolute_tolerance"]) >= 0,
                    "run.json: backends.autogluon.build.packaging."
                    "prediction_equivalence.absolute_tolerance must be a "
                    "non-negative finite number",
                    errors,
                )
            deployment_bytes = packaging.get("deployment_predictor_bytes")
            peak_bytes = packaging.get("peak_packaging_disk_bytes")
            require(
                is_positive_integer(deployment_bytes),
                "run.json: backends.autogluon.build.packaging."
                "deployment_predictor_bytes must be a positive integer",
                errors,
            )
            if predictor is not None and is_positive_integer(deployment_bytes):
                require(
                    directory_file_bytes(predictor) == deployment_bytes,
                    "run.json: backends.autogluon.build.packaging."
                    "deployment_predictor_bytes must match the retained "
                    "deployment clone size",
                    errors,
                )
            require(
                is_positive_integer(peak_bytes),
                "run.json: backends.autogluon.build.packaging."
                "peak_packaging_disk_bytes must be a positive integer",
                errors,
            )
            if is_positive_integer(deployment_bytes) and is_positive_integer(
                peak_bytes
            ):
                require(
                    peak_bytes >= deployment_bytes,
                    "run.json: backends.autogluon.build.packaging."
                    "peak_packaging_disk_bytes cannot be smaller than "
                    "deployment_predictor_bytes",
                    errors,
                )
            retained_training = packaging.get("training_predictor_retained")
            require(
                isinstance(retained_training, bool),
                "run.json: backends.autogluon.build.packaging."
                "training_predictor_retained must be boolean",
                errors,
            )
            if retained_training is True:
                require(
                    is_nonempty_string(packaging.get("retention_reason")),
                    "run.json: backends.autogluon.build.packaging."
                    "retention_reason must explain why the full training "
                    "predictor is retained",
                    errors,
                )
                training_predictor = resolve_artifact_path(
                    run_dir,
                    packaging.get("training_predictor_path"),
                    "run.json: backends.autogluon.build.packaging."
                    "training_predictor_path",
                    errors,
                    kind="dir",
                )
                require(
                    packaging.get("training_predictor_path")
                    == "backends/autogluon/training_predictor",
                    "run.json: backends.autogluon.build.packaging."
                    "training_predictor_path must be "
                    "backends/autogluon/training_predictor when retained",
                    errors,
                )
                if training_predictor is not None and predictor is not None:
                    require(
                        training_predictor != predictor,
                        "run.json: AutoGluon training and deployment predictors "
                        "must use different paths",
                        errors,
                    )
            elif retained_training is False:
                require(
                    packaging.get("training_predictor_path") is None,
                    "run.json: backends.autogluon.build.packaging."
                    "training_predictor_path must be null when the training "
                    "predictor is not retained",
                    errors,
                )
                require(
                    packaging.get("retention_reason") is None,
                    "run.json: backends.autogluon.build.packaging."
                    "retention_reason must be null when the training predictor "
                    "is not retained",
                    errors,
                )
        backend_dir = run_dir / "backends" / "autogluon"
        if backend_dir.is_dir() and isinstance(packaging, dict):
            allowed_entries = {"predictor"}
            if packaging.get("training_predictor_retained") is True:
                training_path = packaging.get("training_predictor_path")
                if is_nonempty_string(training_path):
                    allowed_entries.add(Path(training_path).name)
            unknown_entries = sorted(
                child.name
                for child in backend_dir.iterdir()
                if child.name not in allowed_entries
            )
            if unknown_entries:
                errors.append(
                    "backends/autogluon contains unsupported retained training "
                    "clutter: " + ", ".join(unknown_entries)
                )
    handling = value.get("data_handling")
    require(
        isinstance(handling, dict),
        "run.json: backends.autogluon.data_handling must be an object",
        errors,
    )
    if isinstance(handling, dict):
        require(
            handling.get("raw_tabular") is True,
            "run.json: backends.autogluon.data_handling.raw_tabular must be true",
            errors,
        )
        require(
            handling.get("external_preprocessing") is False,
            "run.json: backends.autogluon.data_handling.external_preprocessing "
            "must be false",
            errors,
        )
        require(
            handling.get("external_optuna") is False,
            "run.json: backends.autogluon.data_handling.external_optuna must be false",
            errors,
        )
    runtime = value.get("runtime")
    require(
        isinstance(runtime, dict),
        "run.json: backends.autogluon.runtime must be an object",
        errors,
    )
    if isinstance(runtime, dict):
        validate_known_keys(
            runtime,
            {
                "cold_start_subprocess",
                "native_thread_limits",
                "limits_set_before_imports",
            },
            "run.json: backends.autogluon.runtime",
            errors,
        )
        require(
            runtime.get("cold_start_subprocess") is True,
            "run.json: backends.autogluon.runtime.cold_start_subprocess must be true",
            errors,
        )
        require(
            runtime.get("limits_set_before_imports") is True,
            "run.json: backends.autogluon.runtime.limits_set_before_imports must be true",
            errors,
        )
        limits = runtime.get("native_thread_limits")
        require(
            isinstance(limits, dict)
            and set(limits) == set(NATIVE_THREAD_LIMITS)
            and all(value == 1 for value in limits.values()),
            "run.json: backends.autogluon.runtime.native_thread_limits must "
            "set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, and "
            "VECLIB_MAXIMUM_THREADS to 1",
            errors,
        )
    leaderboard = value.get("native_leaderboard")
    require(
        isinstance(leaderboard, list) and bool(leaderboard),
        "run.json: backends.autogluon.native_leaderboard must be a non-empty list",
        errors,
    )
    if isinstance(leaderboard, list):
        for index, item in enumerate(leaderboard):
            prefix = f"run.json: backends.autogluon.native_leaderboard[{index}]"
            if not require(
                isinstance(item, dict), f"{prefix} must be an object", errors
            ):
                continue
            require(
                is_nonempty_string(item.get("model")),
                f"{prefix}.model must be non-empty",
                errors,
            )
            require(
                is_finite_number(item.get("score_val")),
                f"{prefix}.score_val must be finite",
                errors,
            )
    failures = value.get("internal_failures")
    require(
        isinstance(failures, list),
        "run.json: backends.autogluon.internal_failures must be a list",
        errors,
    )
    if isinstance(failures, list):
        for index, failure in enumerate(failures):
            prefix = f"run.json: backends.autogluon.internal_failures[{index}]"
            if not require(
                isinstance(failure, dict),
                f"{prefix} must be an object",
                errors,
            ):
                continue
            validate_known_keys(
                failure,
                {"component", "stage", "status", "reason", "track_impact"},
                prefix,
                errors,
            )
            for name in ("component", "stage", "reason", "track_impact"):
                require(
                    is_nonempty_string(failure.get(name)),
                    f"{prefix}.{name} must be non-empty",
                    errors,
                )
            require(
                failure.get("status") in {"failed", "skipped", "unavailable"},
                f"{prefix}.status must be failed, skipped, or unavailable",
                errors,
            )


def _validate_rpt_configurations(
    value: dict[str, Any],
    model: Any,
    context: Any,
    approved_plan: Any,
    approved_budget: Any,
    errors: list[str],
) -> None:
    field = "run.json: backends.sap_rpt.configurations"
    configurations = value.get("configurations")
    if not require(
        isinstance(configurations, list) and bool(configurations),
        f"{field} must be a non-empty list",
        errors,
    ):
        return
    if isinstance(approved_plan, dict):
        expected_count = approved_plan.get("estimated_configurations")
        if is_positive_integer(expected_count):
            require(
                len(configurations) == expected_count,
                f"{field} must contain one row for every approved estimated "
                "configuration",
                errors,
            )

    identifiers: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    allowed_keys = {
        "id",
        "status",
        "model_id",
        "context_candidate_rows",
        "context_rows_planned",
        "context_rows_sent",
        "context_strategy",
        "cli_strategy",
        "context_seed",
        "input_format",
        "fold_eligibility_policy",
        "score",
        "latency_ms",
        "throughput_queries_per_second",
        "request_count",
        "retrieval_extra_used",
        "selected",
        "failure_reason",
    }
    planned_models = (
        set(approved_plan.get("model_ids", []))
        if isinstance(approved_plan, dict)
        else set()
    )
    planned_strategies = (
        set(approved_plan.get("retrieval_strategies", []))
        if isinstance(approved_plan, dict)
        else set()
    )
    planned_format = (
        approved_plan.get("input_format")
        if isinstance(approved_plan, dict)
        else None
    )
    planned_seed = (
        approved_plan.get("context_seed")
        if isinstance(approved_plan, dict)
        else None
    )
    maximum_context = (
        approved_budget.get("max_context_rows")
        if isinstance(approved_budget, dict)
        else None
    )

    for index, configuration in enumerate(configurations):
        prefix = f"{field}[{index}]"
        if not require(
            isinstance(configuration, dict),
            f"{prefix} must be an object",
            errors,
        ):
            continue
        validate_known_keys(configuration, allowed_keys, prefix, errors)
        identifier = configuration.get("id")
        require(
            is_nonempty_string(identifier),
            f"{prefix}.id must be non-empty",
            errors,
        )
        if is_nonempty_string(identifier):
            identifiers.append(identifier)
        status = configuration.get("status")
        require(
            status in {"completed", "failed", "skipped", "unavailable"},
            f"{prefix}.status must be completed, failed, skipped, or unavailable",
            errors,
        )
        model_id = configuration.get("model_id")
        require(
            is_nonempty_string(model_id),
            f"{prefix}.model_id must be non-empty",
            errors,
        )
        if planned_models and is_nonempty_string(model_id):
            require(
                model_id in planned_models,
                f"{prefix}.model_id was not included in the approved RPT plan",
                errors,
            )
        candidate_rows = configuration.get("context_candidate_rows")
        planned_rows = configuration.get("context_rows_planned")
        sent_rows = configuration.get("context_rows_sent")
        require(
            is_positive_integer(candidate_rows),
            f"{prefix}.context_candidate_rows must be a positive integer",
            errors,
        )
        require(
            is_positive_integer(planned_rows),
            f"{prefix}.context_rows_planned must be a positive integer",
            errors,
        )
        require(
            is_nonnegative_integer(sent_rows),
            f"{prefix}.context_rows_sent must be a non-negative integer",
            errors,
        )
        if is_positive_integer(candidate_rows) and is_positive_integer(planned_rows):
            require(
                planned_rows <= candidate_rows,
                f"{prefix}.context_rows_planned cannot exceed "
                "context_candidate_rows",
                errors,
            )
        if is_positive_integer(planned_rows) and is_positive_integer(maximum_context):
            require(
                planned_rows <= maximum_context,
                f"{prefix}.context_rows_planned exceeds the approved "
                "max_context_rows",
                errors,
            )
        strategy = configuration.get("context_strategy")
        require(
            strategy in {"full", "random", "vectorsearch"},
            f"{prefix}.context_strategy must be full, random, or vectorsearch",
            errors,
        )
        if planned_strategies and strategy in {"full", "random", "vectorsearch"}:
            require(
                strategy in planned_strategies,
                f"{prefix}.context_strategy was not included in the approved "
                "RPT plan",
                errors,
            )
        if (
            strategy == "full"
            and is_positive_integer(candidate_rows)
            and is_positive_integer(planned_rows)
        ):
            require(
                planned_rows == candidate_rows,
                f"{prefix}: full context must plan every candidate row",
                errors,
            )
            require(
                configuration.get("cli_strategy") is None,
                f"{prefix}.cli_strategy must be null for full context",
                errors,
            )
            require(
                configuration.get("context_seed") is None,
                f"{prefix}.context_seed must be null for full context",
                errors,
            )
        elif strategy in {"random", "vectorsearch"}:
            if is_positive_integer(candidate_rows) and is_positive_integer(
                planned_rows
            ):
                require(
                    planned_rows < candidate_rows,
                    f"{prefix}: a retrieval strategy must reduce context rows",
                    errors,
                )
                require(
                    configuration.get("cli_strategy")
                    == f"{strategy}::{planned_rows}",
                    f"{prefix}.cli_strategy must match "
                    f"{strategy}::context_rows_planned",
                    errors,
                )
            require(
                configuration.get("context_seed") == planned_seed,
                f"{prefix}.context_seed must match the approved RPT plan",
                errors,
            )
        input_format = configuration.get("input_format")
        require(
            input_format in {"column_json", "csv", "parquet"},
            f"{prefix}.input_format must be column_json, csv, or parquet",
            errors,
        )
        if planned_format is not None:
            require(
                input_format == planned_format,
                f"{prefix}.input_format must match the approved RPT plan",
                errors,
            )
        require(
            is_nonempty_string(configuration.get("fold_eligibility_policy")),
            f"{prefix}.fold_eligibility_policy must be non-empty",
            errors,
        )
        require(
            isinstance(configuration.get("retrieval_extra_used"), bool),
            f"{prefix}.retrieval_extra_used must be boolean",
            errors,
        )
        if strategy == "vectorsearch":
            require(
                configuration.get("retrieval_extra_used") is True,
                f"{prefix}.retrieval_extra_used must be true for vectorsearch",
                errors,
            )
        elif strategy in {"full", "random"}:
            require(
                configuration.get("retrieval_extra_used") is False,
                f"{prefix}.retrieval_extra_used must be false for {strategy}",
                errors,
            )
        require(
            isinstance(configuration.get("selected"), bool),
            f"{prefix}.selected must be boolean",
            errors,
        )
        request_count = configuration.get("request_count")
        require(
            is_nonnegative_integer(request_count),
            f"{prefix}.request_count must be a non-negative integer",
            errors,
        )
        if status == "completed":
            completed_rows.append(configuration)
            require(
                sent_rows == planned_rows,
                f"{prefix}.context_rows_sent must match context_rows_planned "
                "when completed",
                errors,
            )
            require(
                is_finite_number(configuration.get("score")),
                f"{prefix}.score must be finite when completed",
                errors,
            )
            require(
                is_positive_integer(request_count),
                f"{prefix}.request_count must be positive when completed",
                errors,
            )
            require(
                configuration.get("failure_reason") in (None, ""),
                f"{prefix}.failure_reason must be null when completed",
                errors,
            )
            latency = configuration.get("latency_ms")
            if require(
                isinstance(latency, dict),
                f"{prefix}.latency_ms must be an object when completed",
                errors,
            ):
                validate_known_keys(
                    latency,
                    {"median", "p95"},
                    f"{prefix}.latency_ms",
                    errors,
                )
                median = latency.get("median")
                p95 = latency.get("p95")
                require(
                    is_finite_number(median) and float(median) >= 0,
                    f"{prefix}.latency_ms.median must be non-negative and finite",
                    errors,
                )
                require(
                    is_finite_number(p95) and float(p95) >= 0,
                    f"{prefix}.latency_ms.p95 must be non-negative and finite",
                    errors,
                )
                if is_finite_number(median) and is_finite_number(p95):
                    require(
                        float(p95) >= float(median),
                        f"{prefix}.latency_ms.p95 cannot be below median",
                        errors,
                    )
            throughput = configuration.get("throughput_queries_per_second")
            require(
                is_finite_number(throughput) and float(throughput) > 0,
                f"{prefix}.throughput_queries_per_second must be positive and "
                "finite when completed",
                errors,
            )
        else:
            require(
                is_nonempty_string(configuration.get("failure_reason")),
                f"{prefix}.failure_reason must be non-empty when not completed",
                errors,
            )
        if configuration.get("selected") is True:
            require(
                status == "completed",
                f"{prefix}: only a completed configuration may be selected",
                errors,
            )
            selected_rows.append(configuration)

    observed_models = {
        row.get("model_id")
        for row in configurations
        if isinstance(row, dict)
    }
    observed_strategies = {
        row.get("context_strategy")
        for row in configurations
        if isinstance(row, dict)
    }
    observed_sizes = {
        row.get("context_rows_planned")
        for row in configurations
        if isinstance(row, dict)
    }
    if isinstance(approved_plan, dict):
        require(
            set(approved_plan.get("model_ids", [])) <= observed_models,
            f"{field} must record every model ID in the approved RPT plan",
            errors,
        )
        require(
            set(approved_plan.get("retrieval_strategies", []))
            <= observed_strategies,
            f"{field} must record every retrieval strategy in the approved "
            "RPT plan",
            errors,
        )
        require(
            set(approved_plan.get("context_size_candidates", []))
            <= observed_sizes,
            f"{field} must record every context size in the approved RPT plan",
            errors,
        )
    if len(identifiers) != len(set(identifiers)):
        errors.append(f"{field} IDs must be unique")
    require(
        len(selected_rows) == 1,
        f"{field} must select exactly one completed configuration",
        errors,
    )
    if len(selected_rows) == 1:
        selected = selected_rows[0]
        if isinstance(model, dict):
            require(
                model.get("id") == selected.get("model_id"),
                "run.json: backends.sap_rpt.model.id must match the selected "
                "configuration model_id",
                errors,
            )
        if isinstance(context, dict):
            require(
                context.get("selected_configuration_id") == selected.get("id"),
                "run.json: backends.sap_rpt.context.selected_configuration_id "
                "must reference the selected configuration",
                errors,
            )

    coverage = value.get("evaluation_coverage")
    coverage_field = "run.json: backends.sap_rpt.evaluation_coverage"
    if not require(
        isinstance(coverage, dict),
        f"{coverage_field} must be an object",
        errors,
    ):
        return
    validate_known_keys(
        coverage,
        {
            "summary",
            "context_scale_tested",
            "retrieval_comparison_tested",
            "model_variants_tested",
            "full_context_tested",
            "coverage_gaps",
        },
        coverage_field,
        errors,
    )
    require(
        isinstance(coverage.get("summary"), str)
        and "evaluated under the approved configurations"
        in coverage["summary"].lower(),
        f"{coverage_field}.summary must state evaluated under the approved "
        "configurations",
        errors,
    )
    completed_context_sizes = {
        row.get("context_rows_planned") for row in completed_rows
    }
    completed_strategies = {
        row.get("context_strategy") for row in completed_rows
    }
    completed_models = {row.get("model_id") for row in completed_rows}
    expected_flags = {
        "context_scale_tested": len(completed_context_sizes) > 1,
        "retrieval_comparison_tested": {
            "random",
            "vectorsearch",
        }.issubset(completed_strategies),
        "model_variants_tested": len(completed_models) > 1,
        "full_context_tested": "full" in completed_strategies,
    }
    for name, expected in expected_flags.items():
        require(
            coverage.get(name) is expected,
            f"{coverage_field}.{name} must match the completed configuration "
            "ledger",
            errors,
        )
    require(
        is_string_list(coverage.get("coverage_gaps")),
        f"{coverage_field}.coverage_gaps must be a unique string list",
        errors,
    )


def validate_sap_rpt(
    value: dict[str, Any],
    run_dir: Path,
    remote_transfers: dict[str, dict[str, Any]],
    approved_plan: Any,
    approved_budget: Any,
    errors: list[str],
) -> None:
    reject_sap_rpt_operation_fields(value, errors)
    model = value.get("model")
    require(
        isinstance(model, dict),
        "run.json: backends.sap_rpt.model must be an object",
        errors,
    )
    if isinstance(model, dict):
        require(
            is_nonempty_string(model.get("name")),
            "run.json: backends.sap_rpt.model.name must be non-empty",
            errors,
        )
        require(
            is_nonempty_string(model.get("id")),
            "run.json: backends.sap_rpt.model.id must be non-empty",
            errors,
        )
        require(
            is_nonempty_string(model.get("version")),
            "run.json: backends.sap_rpt.model.version must be non-empty",
            errors,
        )
        require(
            model.get("production_capable") is True,
            "run.json: backends.sap_rpt.model.production_capable must be true",
            errors,
        )
    access = value.get("access")
    require(
        isinstance(access, dict),
        "run.json: backends.sap_rpt.access must be an object",
        errors,
    )
    if isinstance(access, dict):
        require(
            is_nonempty_string(access.get("route")),
            "run.json: backends.sap_rpt.access.route must be non-empty",
            errors,
        )
        require(
            is_nonempty_string(access.get("client")),
            "run.json: backends.sap_rpt.access.client must be non-empty",
            errors,
        )
        require(
            is_nonempty_string(access.get("customer_production_route")),
            "run.json: backends.sap_rpt.access.customer_production_route must "
            "be non-empty",
            errors,
        )
        if isinstance(model, dict):
            require(
                access.get("route") != model.get("name"),
                "run.json: SAP RPT model identity and access route must be "
                "recorded separately",
                errors,
            )
    context = value.get("context")
    require(
        isinstance(context, dict),
        "run.json: backends.sap_rpt.context must be an object",
        errors,
    )
    if isinstance(context, dict):
        require(
            is_sha256(context.get("fingerprint")),
            "run.json: backends.sap_rpt.context.fingerprint must be a SHA-256 digest",
            errors,
        )
        require(
            is_nonempty_string(context.get("policy")),
            "run.json: backends.sap_rpt.context.policy must be non-empty",
            errors,
        )
        require(
            is_nonempty_string(context.get("selected_configuration_id")),
            "run.json: backends.sap_rpt.context.selected_configuration_id "
            "must be non-empty",
            errors,
        )
        manifest = resolve_artifact_path(
            run_dir,
            context.get("manifest"),
            "run.json: backends.sap_rpt.context.manifest",
            errors,
            kind="file",
        )
        if manifest is not None:
            expected_root = (run_dir / "backends" / "sap_rpt").resolve()
            try:
                manifest.relative_to(expected_root)
            except ValueError:
                errors.append(
                    "run.json: backends.sap_rpt.context.manifest must be "
                    "inside backends/sap_rpt"
                )
    confirmation = value.get("transfer_confirmation")
    require(
        isinstance(confirmation, dict),
        "run.json: backends.sap_rpt.transfer_confirmation must be an object",
        errors,
    )
    if isinstance(confirmation, dict):
        approval_id = confirmation.get("approval_id")
        require(
            is_nonempty_string(approval_id),
            "run.json: backends.sap_rpt.transfer_confirmation.approval_id "
            "must be non-empty",
            errors,
        )
        require(
            approval_id in remote_transfers,
            "run.json: backends.sap_rpt.transfer_confirmation.approval_id "
            "must reference approval.remote_transfers",
            errors,
        )
        for field in (
            "schema_validated",
            "labels_validated",
            "query_rows_excluded_from_context",
        ):
            require(
                confirmation.get(field) is True,
                "run.json: backends.sap_rpt.transfer_confirmation."
                f"{field} must be true",
                errors,
            )
    _validate_rpt_configurations(
        value,
        model,
        context,
        approved_plan,
        approved_budget,
        errors,
    )


def reject_sap_rpt_operation_fields(
    value: dict[str, Any],
    errors: list[str],
) -> None:
    for path, key in iter_keys(value):
        if RPT_FORBIDDEN_KEY_RE.search(key):
            errors.append(
                "run.json: SAP RPT is pretrained and must not declare "
                f"training, fit, hyperparameter, or search fields; found {path!r}"
            )


def validate_backends(
    document: dict[str, Any],
    selected: set[str],
    remote_transfers: dict[str, dict[str, Any]],
    evaluation: dict[str, Any] | None,
    run_dir: Path,
    errors: list[str],
) -> tuple[dict[str, float], set[str]]:
    backends = document.get("backends")
    require(
        isinstance(backends, dict),
        "run.json: backends must be an object",
        errors,
    )
    if not isinstance(backends, dict):
        return {}, set()
    unknown = sorted(set(backends) - BACKEND_SET)
    if unknown:
        errors.append(f"run.json: unsupported backend entries: {', '.join(unknown)}")
    unapproved = sorted(set(backends) - selected)
    if unapproved:
        errors.append(
            "run.json: execution evidence exists for unapproved tracks: "
            + ", ".join(unapproved)
        )
    missing = sorted(selected - set(backends))
    if missing:
        errors.append(
            "run.json: approved tracks must record backend status/evidence: "
            + ", ".join(missing)
        )

    scores: dict[str, float] = {}
    retained: set[str] = set()
    backend_root = run_dir / "backends"
    for backend in BACKEND_NAMES:
        value = backends.get(backend)
        if not isinstance(value, dict):
            if backend in backends:
                errors.append(f"run.json: backends.{backend} must be an object")
            continue
        status = value.get("status")
        require(
            status in {"completed", "failed", "unavailable"},
            f"run.json: backends.{backend}.status must be completed, failed, "
            "or unavailable",
            errors,
        )
        require(
            isinstance(value.get("retained"), bool),
            f"run.json: backends.{backend}.retained must be boolean",
            errors,
        )
        evidence = value.get("evidence")
        require(
            isinstance(evidence, dict) and bool(evidence),
            f"run.json: backends.{backend}.evidence must be a non-empty object",
            errors,
        )
        backend_dir = backend_root / backend
        if status != "unavailable":
            require(
                backend_dir.is_dir(),
                f"missing backend directory: backends/{backend}",
                errors,
            )
        if status == "completed":
            score = validate_backend_evaluation(backend, value, evaluation, errors)
            if score is not None:
                scores[backend] = score
            if value.get("retained") is True:
                retained.add(backend)
        else:
            require(
                value.get("retained") is False,
                f"run.json: backends.{backend}.retained must be false when "
                f"status is {status!r}",
                errors,
            )
            require(
                "evaluation" not in value,
                f"run.json: backends.{backend} must not claim evaluation "
                f"metrics when status is {status!r}",
                errors,
            )
        if status == "completed":
            approved_budget = nested(
                document,
                "approval",
                "tracks",
                backend,
                "budget",
            )
            if backend == "classical":
                validate_classical(value, approved_budget, run_dir, errors)
            elif backend == "autogluon":
                validate_autogluon(value, approved_budget, run_dir, errors)
            else:
                approved_plan = nested(
                    document,
                    "approval",
                    "tracks",
                    "sap_rpt",
                    "plan",
                )
                validate_sap_rpt(
                    value,
                    run_dir,
                    remote_transfers,
                    approved_plan,
                    approved_budget,
                    errors,
                )
        elif backend == "sap_rpt":
            reject_sap_rpt_operation_fields(value, errors)

    if backend_root.exists():
        for child in backend_root.iterdir():
            if not child.is_dir() or child.name not in BACKEND_SET:
                errors.append(
                    "backends/: only classical, autogluon, and sap_rpt "
                    f"directories are allowed; found {child.name!r}"
                )
            elif child.name not in selected:
                errors.append(
                    f"backends/{child.name}: directory exists for an unapproved track"
                )
    require(
        bool(retained),
        "run.json: at least one completed backend must be retained",
        errors,
    )
    return scores, retained


def validate_selection(
    document: dict[str, Any],
    scores: dict[str, float],
    retained: set[str],
    evaluation: dict[str, Any] | None,
    errors: list[str],
) -> None:
    selection = document.get("selection")
    require(
        isinstance(selection, dict),
        "run.json: selection must be an object",
        errors,
    )
    if not isinstance(selection, dict):
        return
    winner = selection.get("predictive_winner")
    recommendation = selection.get("operational_recommendation")
    require(
        winner in retained,
        "run.json: selection.predictive_winner must name a completed retained backend",
        errors,
    )
    require(
        recommendation in retained,
        "run.json: selection.operational_recommendation must name a completed "
        "retained backend",
        errors,
    )
    require(
        is_nonempty_string(selection.get("rationale")),
        "run.json: selection.rationale must be non-empty",
        errors,
    )
    metric_name = nested(evaluation, "primary_metric", "name")
    require(
        selection.get("primary_metric") == metric_name,
        "run.json: selection.primary_metric must match evaluation.primary_metric.name",
        errors,
    )
    if winner in scores and scores:
        direction = nested(evaluation, "primary_metric", "direction")
        best_score = (
            max(scores.values()) if direction == "maximize" else min(scores.values())
        )
        require(
            math.isclose(scores[winner], best_score, rel_tol=1e-12, abs_tol=1e-12),
            "run.json: selection.predictive_winner does not have the best "
            "shared primary-metric score",
            errors,
        )


def validate_inference_contract(
    document: dict[str, Any],
    retained: set[str],
    errors: list[str],
) -> None:
    inference = document.get("inference")
    require(
        isinstance(inference, dict),
        "run.json: inference must be an object",
        errors,
    )
    if not isinstance(inference, dict):
        return
    require(
        inference.get("entrypoint") == "infer.py",
        "run.json: inference.entrypoint must be 'infer.py'",
        errors,
    )
    require(
        inference.get("default_backend") in retained,
        "run.json: inference.default_backend must name a retained backend",
        errors,
    )
    require(
        inference.get("default_backend")
        == nested(document, "selection", "operational_recommendation"),
        "run.json: inference.default_backend must match "
        "selection.operational_recommendation",
        errors,
    )
    input_contract = inference.get("input")
    require(
        isinstance(input_contract, dict),
        "run.json: inference.input must be an object",
        errors,
    )
    if isinstance(input_contract, dict):
        require(
            input_contract.get("format") in {"csv", "json"},
            "run.json: inference.input.format must be csv or json",
            errors,
        )
        required_columns = input_contract.get("required_columns")
        optional_columns = input_contract.get("optional_columns")
        require(
            is_string_list(required_columns, nonempty=True),
            "run.json: inference.input.required_columns must be a non-empty "
            "unique string list",
            errors,
        )
        require(
            is_string_list(optional_columns),
            "run.json: inference.input.optional_columns must be a unique string list",
            errors,
        )
        all_inputs: set[str] = set()
        if isinstance(required_columns, list):
            all_inputs.update(required_columns)
        if isinstance(optional_columns, list):
            all_inputs.update(optional_columns)
        if isinstance(required_columns, list) and isinstance(optional_columns, list):
            require(
                not set(required_columns) & set(optional_columns),
                "run.json: inference input required and optional columns must "
                "be disjoint",
                errors,
            )
        dtypes = input_contract.get("dtypes")
        require(
            isinstance(dtypes, dict)
            and set(dtypes) == all_inputs
            and all(is_nonempty_string(dtype) for dtype in dtypes.values()),
            "run.json: inference.input.dtypes must define a non-empty dtype "
            "for every required and optional input column",
            errors,
        )
        require(
            is_nonempty_string(input_contract.get("missing_value_policy"))
            or (
                isinstance(input_contract.get("missing_value_policy"), dict)
                and bool(input_contract["missing_value_policy"])
            ),
            "run.json: inference.input.missing_value_policy must be a "
            "non-empty string or object",
            errors,
        )
        require(
            input_contract.get("extra_column_policy") in {"reject", "ignore"},
            "run.json: inference.input.extra_column_policy must be 'reject' or 'ignore'",
            errors,
        )
        target_column = input_contract.get("target_column")
        require(
            target_column == nested(document, "problem", "target"),
            "run.json: inference.input.target_column must match problem.target",
            errors,
        )
        if is_nonempty_string(target_column):
            require(
                target_column not in all_inputs,
                "run.json: inference input must exclude the target column",
                errors,
            )
        identifiers = input_contract.get("identifier_columns")
        require(
            is_string_list(identifiers),
            "run.json: inference.input.identifier_columns must be a unique string list",
            errors,
        )
        if isinstance(identifiers, list):
            require(
                set(identifiers).issubset(all_inputs),
                "run.json: inference.input.identifier_columns must be declared "
                "input columns",
                errors,
            )
        feature_order = input_contract.get("feature_order")
        require(
            is_string_list(feature_order, nonempty=True),
            "run.json: inference.input.feature_order must be a non-empty "
            "unique string list",
            errors,
        )
        if isinstance(feature_order, list) and isinstance(identifiers, list):
            require(
                set(feature_order) == all_inputs - set(identifiers),
                "run.json: inference.input.feature_order must contain every "
                "non-identifier input exactly once",
                errors,
            )

    output_contract = inference.get("output")
    require(
        isinstance(output_contract, dict),
        "run.json: inference.output must be an object",
        errors,
    )
    if isinstance(output_contract, dict):
        require(
            output_contract.get("format") in {"csv", "json"},
            "run.json: inference.output.format must be csv or json",
            errors,
        )
        prediction_column = output_contract.get("prediction_column")
        require(
            is_nonempty_string(prediction_column),
            "run.json: inference.output.prediction_column must be non-empty",
            errors,
        )
        probability_columns = output_contract.get("probability_columns")
        require(
            is_string_list(probability_columns),
            "run.json: inference.output.probability_columns must be a unique "
            "string list",
            errors,
        )
        row_id_column = output_contract.get("row_id_column")
        require(
            row_id_column is None or is_nonempty_string(row_id_column),
            "run.json: inference.output.row_id_column must be null or non-empty",
            errors,
        )
        require(
            output_contract.get("finite_values") is True,
            "run.json: inference.output.finite_values must be true",
            errors,
        )
        output_names = [
            name
            for name in [prediction_column, row_id_column]
            if is_nonempty_string(name)
        ]
        if isinstance(probability_columns, list):
            output_names.extend(probability_columns)
        require(
            len(output_names) == len(set(output_names)),
            "run.json: inference output column roles must be distinct",
            errors,
        )
        bounds = output_contract.get("probability_bounds")
        if isinstance(probability_columns, list) and probability_columns:
            require(
                isinstance(bounds, list)
                and len(bounds) == 2
                and all(is_finite_number(item) for item in bounds)
                and float(bounds[0]) == 0.0
                and float(bounds[1]) == 1.0,
                "run.json: inference.output.probability_bounds must be [0, 1] "
                "when probability columns are declared",
                errors,
            )
        else:
            require(
                bounds in (None, []),
                "run.json: inference.output.probability_bounds must be omitted, "
                "null, or empty when no probability columns are declared",
                errors,
            )
    commands = inference.get("backends")
    require(
        isinstance(commands, dict),
        "run.json: inference.backends must be an object",
        errors,
    )
    if isinstance(commands, dict):
        require(
            set(commands) == retained,
            "run.json: inference.backends must contain exactly the retained backends",
            errors,
        )
        for backend, command in commands.items():
            require(
                is_nonempty_string(command),
                f"run.json: inference.backends.{backend} must be a non-empty command",
                errors,
            )


def validate_lineage(document: dict[str, Any], errors: list[str]) -> None:
    lineage = document.get("lineage")
    require(
        isinstance(lineage, dict),
        "run.json: lineage must be an object",
        errors,
    )
    if not isinstance(lineage, dict):
        return
    require(
        lineage.get("source_data_fingerprint")
        == nested(document, "data", "fingerprint"),
        "run.json: lineage.source_data_fingerprint must match data.fingerprint",
        errors,
    )
    require(
        lineage.get("parent_run_id") is None
        or is_nonempty_string(lineage.get("parent_run_id")),
        "run.json: lineage.parent_run_id must be null or a non-empty string",
        errors,
    )
    require(
        isinstance(lineage.get("notes"), list),
        "run.json: lineage.notes must be a list",
        errors,
    )


def validate_run_document(
    document: Any,
    run_dir: Path,
    errors: list[str],
) -> set[str]:
    if not require(
        isinstance(document, dict),
        "run.json: root value must be an object",
        errors,
    ):
        return set()
    for path, key in iter_keys(document):
        if key == "duplicated_parent_files":
            errors.append(
                f"run.json: {path} is forbidden; parent artifacts must not be copied"
            )
    required = {
        "run_id",
        "created_at",
        "problem",
        "data",
        "modeling_preflight",
        "evaluation",
        "approval",
        "backends",
        "selection",
        "inference",
        "lineage",
    }
    missing = sorted(required - set(document))
    if missing:
        errors.append(f"run.json: missing required fields: {', '.join(missing)}")
    validate_known_keys(document, required, "run.json", errors)
    require(
        is_nonempty_string(document.get("run_id")),
        "run.json: run_id must be a non-empty string",
        errors,
    )
    parse_timestamp(document.get("created_at"), "run.json: created_at", errors)
    validate_problem_and_data(document, errors)
    validate_preflight(document, errors)
    evaluation = validate_evaluation(document, errors)
    selected, remote_transfers = validate_approval(document, errors)
    scores, retained = validate_backends(
        document,
        selected,
        remote_transfers,
        evaluation,
        run_dir,
        errors,
    )
    validate_selection(document, scores, retained, evaluation, errors)
    validate_inference_contract(document, retained, errors)
    validate_lineage(document, errors)
    return retained


def read_inference_output(
    path: Path,
    output_contract: dict[str, Any],
    case_name: str,
    errors: list[str],
) -> tuple[list[str] | None, list[dict[str, Any]] | None]:
    output_format = output_contract.get("format")
    if output_format == "csv":
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    errors.append(
                        f"validation.json: inference case {case_name!r} output "
                        "CSV has no header"
                    )
                    return None, None
                return list(reader.fieldnames), list(reader)
        except (OSError, csv.Error) as exc:
            errors.append(
                f"validation.json: inference case {case_name!r} output "
                f"cannot be read as CSV ({exc})"
            )
            return None, None
    if output_format == "json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                f"validation.json: inference case {case_name!r} output "
                f"cannot be read as JSON ({exc})"
            )
            return None, None
        columns: list[str] | None = None
        rows: Any = value
        if isinstance(value, dict):
            columns = value.get("columns")
            rows = value.get("rows")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            errors.append(
                f"validation.json: inference case {case_name!r} JSON output "
                "must be a row-object list or {columns, rows} envelope"
            )
            return None, None
        if columns is None:
            if not rows:
                errors.append(
                    f"validation.json: inference case {case_name!r} zero-row "
                    "JSON output must use a schema-bearing {columns, rows} envelope"
                )
                return None, None
            columns = list(dict.fromkeys(key for row in rows for key in row))
        if not is_string_list(columns):
            errors.append(
                f"validation.json: inference case {case_name!r} JSON columns "
                "must be a unique string list"
            )
            return None, None
        return columns, rows
    errors.append(
        f"validation.json: inference case {case_name!r} output format must be "
        "csv or json"
    )
    return None, None


def validate_inference_case(
    case: Any,
    index: int,
    retained: set[str],
    inference_contract: Any,
    errors: list[str],
) -> tuple[str | None, str | None]:
    prefix = f"validation.json: inference_cases[{index}]"
    if not require(isinstance(case, dict), f"{prefix} must be an object", errors):
        return None, None
    name = case.get("name")
    require(is_nonempty_string(name), f"{prefix}.name must be non-empty", errors)
    backend = case.get("backend")
    require(
        backend in retained,
        f"{prefix}.backend must name a retained backend",
        errors,
    )
    kind = case.get("kind")
    require(
        kind in REQUIRED_INFERENCE_CASE_KINDS,
        f"{prefix}.kind must be one of "
        + ", ".join(sorted(REQUIRED_INFERENCE_CASE_KINDS)),
        errors,
    )
    argv = case.get("argv")
    require(
        isinstance(argv, list)
        and bool(argv)
        and all(is_nonempty_string(arg) for arg in argv),
        f"{prefix}.argv must be a non-empty string list",
        errors,
    )
    if isinstance(argv, list) and all(isinstance(arg, str) for arg in argv):
        joined = "\0".join(argv)
        require(
            "{input}" in joined,
            f"{prefix}.argv must use the temporary {{input}} placeholder",
            errors,
        )
        require(
            "{output}" in joined,
            f"{prefix}.argv must use the temporary {{output}} placeholder",
            errors,
        )
        dispatches_backend = any(
            argument == "--backend"
            and position + 1 < len(argv)
            and argv[position + 1] == backend
            for position, argument in enumerate(argv)
        )
        require(
            dispatches_backend,
            f"{prefix}.argv must select its declared backend with "
            f"'--backend {backend}'",
            errors,
        )
    inline_input = case.get("input")
    require(
        isinstance(inline_input, dict),
        f"{prefix}.input must define inline data",
        errors,
    )
    if isinstance(inline_input, dict):
        global_input = (
            inference_contract.get("input")
            if isinstance(inference_contract, dict)
            else None
        )
        require(
            isinstance(global_input, dict)
            and inline_input.get("format") == global_input.get("format"),
            f"{prefix}.input.format must match run.json inference.input.format",
            errors,
        )
        columns = inline_input.get("columns")
        rows = inline_input.get("rows")
        require(
            is_string_list(columns, nonempty=True),
            f"{prefix}.input.columns must be a non-empty unique string list",
            errors,
        )
        require(
            isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
            f"{prefix}.input.rows must be a list of objects",
            errors,
        )
        if (
            isinstance(global_input, dict)
            and isinstance(columns, list)
            and all(isinstance(column, str) for column in columns)
        ):
            required_inputs = set(global_input.get("required_columns", []))
            optional_inputs = set(global_input.get("optional_columns", []))
            if kind == "missing_required_column":
                require(
                    bool(required_inputs - set(columns)),
                    f"{prefix}: missing_required_column must omit at least one "
                    "required input column",
                    errors,
                )
            else:
                require(
                    required_inputs.issubset(columns),
                    f"{prefix}: successful cases must include every required "
                    "input column",
                    errors,
                )
            require(
                set(columns).issubset(required_inputs | optional_inputs),
                f"{prefix}: inline input contains columns outside the inference "
                "input contract",
                errors,
            )
        if isinstance(rows, list):
            expected_rows = {
                "single_row": 1,
                "empty_input": 0,
            }.get(kind)
            if expected_rows is not None:
                require(
                    len(rows) == expected_rows,
                    f"{prefix}: {kind} must define exactly {expected_rows} input rows",
                    errors,
                )
            if kind == "representative":
                require(
                    len(rows) >= 2,
                    f"{prefix}: representative input must contain at least two rows",
                    errors,
                )
    expect = case.get("expect")
    require(
        isinstance(expect, dict),
        f"{prefix}.expect must be an object",
        errors,
    )
    if isinstance(expect, dict):
        require(
            is_nonnegative_integer(expect.get("exit_code")),
            f"{prefix}.expect.exit_code must be a non-negative integer",
            errors,
        )
        stderr_contains = expect.get("stderr_contains")
        require(
            stderr_contains is None
            or is_nonempty_string(stderr_contains)
            or is_string_list(stderr_contains, nonempty=True),
            f"{prefix}.expect.stderr_contains must be a string or string list",
            errors,
        )
        output = expect.get("output")
        expected_success = kind != "missing_required_column"
        if expected_success:
            require(
                expect.get("exit_code") == 0,
                f"{prefix}.expect.exit_code must be 0 for {kind}",
                errors,
            )
            require(
                isinstance(output, dict),
                f"{prefix}.expect.output is required for a successful case",
                errors,
            )
        else:
            require(
                is_nonnegative_integer(expect.get("exit_code"))
                and expect.get("exit_code") > 0,
                f"{prefix}.expect.exit_code must be non-zero for a missing "
                "required column",
                errors,
            )
            require(
                stderr_contains is not None,
                f"{prefix}.expect.stderr_contains is required for a missing "
                "required column",
                errors,
            )
        if isinstance(output, dict):
            global_output = (
                inference_contract.get("output")
                if isinstance(inference_contract, dict)
                else None
            )
            require(
                isinstance(global_output, dict)
                and output.get("format") == global_output.get("format"),
                f"{prefix}.expect.output.format must match run.json "
                "inference.output.format",
                errors,
            )
            required_output_columns = output.get("required_columns")
            require(
                is_string_list(required_output_columns, nonempty=True),
                f"{prefix}.expect.output.required_columns must be a non-empty "
                "unique string list",
                errors,
            )
            if isinstance(global_output, dict) and isinstance(
                required_output_columns,
                list,
            ):
                manifest_columns = set(global_output.get("probability_columns") or [])
                manifest_columns.add(global_output.get("prediction_column"))
                manifest_columns.add(global_output.get("row_id_column"))
                manifest_columns.discard(None)
                require(
                    manifest_columns.issubset(required_output_columns),
                    f"{prefix}.expect.output.required_columns must cover the "
                    "declared inference output columns",
                    errors,
                )
            minimum = output.get("min_rows")
            maximum = output.get("max_rows")
            require(
                is_nonnegative_integer(minimum),
                f"{prefix}.expect.output.min_rows must be a non-negative integer",
                errors,
            )
            require(
                is_nonnegative_integer(maximum),
                f"{prefix}.expect.output.max_rows must be a non-negative integer",
                errors,
            )
            if is_nonnegative_integer(minimum) and is_nonnegative_integer(maximum):
                require(
                    minimum <= maximum,
                    f"{prefix}.expect.output.min_rows cannot exceed max_rows",
                    errors,
                )
            if isinstance(inline_input, dict) and isinstance(
                inline_input.get("rows"),
                list,
            ):
                row_count = len(inline_input["rows"])
                require(
                    minimum == row_count and maximum == row_count,
                    f"{prefix}.expect.output row bounds must preserve input row count",
                    errors,
                )
    repeat_runs = case.get("repeat_runs", 1)
    require(
        is_positive_integer(repeat_runs),
        f"{prefix}.repeat_runs must be a positive integer",
        errors,
    )
    if kind in {"representative", "single_row"}:
        require(
            is_positive_integer(repeat_runs) and repeat_runs >= 2,
            f"{prefix}.repeat_runs must be at least 2 for deterministic "
            f"{kind} inference",
            errors,
        )
    return (
        backend if backend in retained else None,
        kind if kind in REQUIRED_INFERENCE_CASE_KINDS else None,
    )


def validate_validation_document(
    document: Any,
    retained: set[str],
    inference_contract: Any,
    *,
    executing_cases: bool,
    errors: list[str],
) -> list[dict[str, Any]]:
    if not require(
        isinstance(document, dict),
        "validation.json: root value must be an object",
        errors,
    ):
        return []
    validate_known_keys(
        document,
        {"status", "validated_at", "inference_cases"},
        "validation.json",
        errors,
    )
    status = document.get("status")
    require(
        status in {"pending", "passed"},
        "validation.json: status must be 'pending' or 'passed'",
        errors,
    )
    if executing_cases:
        require(
            status == "pending",
            "validation.json: status must be 'pending' before executable tests",
            errors,
        )
    if status == "passed":
        parse_timestamp(
            document.get("validated_at"),
            "validation.json: validated_at",
            errors,
        )
    elif status == "pending":
        require(
            document.get("validated_at") is None,
            "validation.json: validated_at must be null while status is 'pending'",
            errors,
        )
    cases = document.get("inference_cases")
    require(
        isinstance(cases, list) and bool(cases),
        "validation.json: inference_cases must be a non-empty list",
        errors,
    )
    if not isinstance(cases, list):
        return []
    names: list[str] = []
    covered_kinds = {backend: set() for backend in retained}
    for index, case in enumerate(cases):
        backend, kind = validate_inference_case(
            case,
            index,
            retained,
            inference_contract,
            errors,
        )
        if isinstance(case, dict) and is_nonempty_string(case.get("name")):
            names.append(case["name"])
        if backend and kind:
            covered_kinds[backend].add(kind)
    if len(names) != len(set(names)):
        errors.append("validation.json: inference case names must be unique")
    for backend in sorted(retained):
        missing = sorted(REQUIRED_INFERENCE_CASE_KINDS - covered_kinds[backend])
        if missing:
            errors.append(
                f"validation.json: retained backend {backend!r} is missing "
                "required inference case kinds: " + ", ".join(missing)
            )
    return [case for case in cases if isinstance(case, dict)]


def _write_inline_input(path: Path, value: dict[str, Any]) -> None:
    columns = value["columns"]
    rows = value["rows"]
    if value["format"] == "csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text(
            json.dumps({"columns": columns, "rows": rows}),
            encoding="utf-8",
        )


def _validate_actual_inference_output(
    *,
    name: str,
    output_path: Path,
    case_output_contract: dict[str, Any],
    inference_contract: dict[str, Any],
    inline_input: dict[str, Any],
    errors: list[str],
) -> None:
    columns, rows = read_inference_output(
        output_path,
        case_output_contract,
        name,
        errors,
    )
    if columns is None or rows is None:
        return
    required = set(case_output_contract.get("required_columns", []))
    missing = sorted(required - set(columns))
    if missing:
        errors.append(
            f"validation.json: inference case {name!r} output is "
            "missing columns: " + ", ".join(missing)
        )
    minimum = case_output_contract.get("min_rows")
    maximum = case_output_contract.get("max_rows")
    if is_nonnegative_integer(minimum) and len(rows) < minimum:
        errors.append(
            f"validation.json: inference case {name!r} returned "
            f"{len(rows)} rows; expected at least {minimum}"
        )
    if is_nonnegative_integer(maximum) and len(rows) > maximum:
        errors.append(
            f"validation.json: inference case {name!r} returned "
            f"{len(rows)} rows; expected at most {maximum}"
        )

    global_output = inference_contract.get("output")
    if not isinstance(global_output, dict):
        return
    row_id_column = global_output.get("row_id_column")
    global_input = inference_contract.get("input")
    identifier_columns = (
        global_input.get("identifier_columns", [])
        if isinstance(global_input, dict)
        else []
    )
    if is_nonempty_string(row_id_column):
        input_id_column = (
            row_id_column
            if row_id_column in identifier_columns
            else identifier_columns[0]
            if len(identifier_columns) == 1
            else None
        )
        if not is_nonempty_string(input_id_column):
            errors.append(
                "run.json: inference output row_id_column must map to an input "
                "identifier column"
            )
        elif row_id_column in columns:
            expected_ids = [
                str(row.get(input_id_column, "")) for row in inline_input["rows"]
            ]
            actual_ids = [str(row.get(row_id_column, "")) for row in rows]
            if actual_ids != expected_ids:
                errors.append(
                    f"validation.json: inference case {name!r} did not preserve "
                    "input row identifier alignment"
                )

    probability_columns = global_output.get("probability_columns")
    bounds = global_output.get("probability_bounds")
    prediction_column = global_output.get("prediction_column")
    if is_nonempty_string(prediction_column):
        for row_index, row in enumerate(rows):
            raw_prediction = row.get(prediction_column)
            try:
                numeric_prediction = float(raw_prediction)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_prediction):
                errors.append(
                    f"validation.json: inference case {name!r} row {row_index} "
                    f"prediction {prediction_column!r} is not finite"
                )
    if not (
        isinstance(probability_columns, list)
        and isinstance(bounds, list)
        and len(bounds) == 2
        and all(is_finite_number(bound) for bound in bounds)
    ):
        return
    lower, upper = map(float, bounds)
    for row_index, row in enumerate(rows):
        for column in probability_columns:
            raw_value = row.get(column)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                errors.append(
                    f"validation.json: inference case {name!r} row {row_index} "
                    f"probability {column!r} is not numeric"
                )
                continue
            if not math.isfinite(value):
                errors.append(
                    f"validation.json: inference case {name!r} row {row_index} "
                    f"probability {column!r} is not finite"
                )
            elif not lower <= value <= upper:
                errors.append(
                    f"validation.json: inference case {name!r} row {row_index} "
                    f"probability {column!r} is outside [{lower:g}, {upper:g}]"
                )


def run_inference_cases(
    cases: list[dict[str, Any]],
    project: Path,
    run_dir: Path,
    inference_contract: dict[str, Any],
    timeout_seconds: int,
    errors: list[str],
) -> None:
    for index, case in enumerate(cases):
        name = case.get("name", f"case-{index}")
        inline_input = case.get("input")
        expect = case.get("expect")
        argv = case.get("argv")
        if not (
            isinstance(inline_input, dict)
            and isinstance(expect, dict)
            and isinstance(argv, list)
        ):
            continue
        input_format = inline_input.get("format")
        output_contract = expect.get("output")
        output_format = (
            output_contract.get("format")
            if isinstance(output_contract, dict)
            else "csv"
        )
        with tempfile.TemporaryDirectory(prefix="ml-run-inference-") as directory:
            case_dir = Path(directory)
            input_path = case_dir / f"input.{input_format}"
            try:
                _write_inline_input(input_path, inline_input)
            except (OSError, csv.Error, TypeError, ValueError) as exc:
                errors.append(
                    f"validation.json: inference case {name!r} inline input "
                    f"cannot be materialized ({exc})"
                )
                continue
            repeat_runs = case.get("repeat_runs", 1)
            output_bytes: list[bytes] = []
            for repeat_index in range(repeat_runs):
                output_path = case_dir / f"output-{repeat_index}.{output_format}"
                replacements = {
                    "{python}": sys.executable,
                    "{project}": str(project),
                    "{run_dir}": str(run_dir),
                    "{case_dir}": str(case_dir),
                    "{input}": str(input_path),
                    "{output}": str(output_path),
                }
                command: list[str] = []
                for raw_argument in argv:
                    argument = raw_argument
                    for placeholder, replacement in replacements.items():
                        argument = argument.replace(placeholder, replacement)
                    command.append(argument)
                try:
                    process_environment = os.environ.copy()
                    process_environment.update(NATIVE_THREAD_LIMITS)
                    completed = subprocess.run(
                        command,
                        cwd=run_dir,
                        env=process_environment,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=timeout_seconds,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(
                        f"validation.json: inference case {name!r} repeat "
                        f"{repeat_index + 1} could not run ({exc})"
                    )
                    break
                expected_exit = expect.get("exit_code")
                if completed.returncode != expected_exit:
                    errors.append(
                        f"validation.json: inference case {name!r} repeat "
                        f"{repeat_index + 1} returned {completed.returncode}; "
                        f"expected {expected_exit}. "
                        f"stderr: {completed.stderr.strip()[:500]}"
                    )
                expected_stderr = expect.get("stderr_contains")
                if isinstance(expected_stderr, str):
                    expected_stderr = [expected_stderr]
                if isinstance(expected_stderr, list):
                    for fragment in expected_stderr:
                        if fragment not in completed.stderr:
                            errors.append(
                                f"validation.json: inference case {name!r} "
                                f"repeat {repeat_index + 1} stderr does not "
                                f"contain {fragment!r}"
                            )
                if isinstance(output_contract, dict):
                    if not output_path.is_file():
                        errors.append(
                            f"validation.json: inference case {name!r} repeat "
                            f"{repeat_index + 1} did not create its temporary output"
                        )
                        continue
                    _validate_actual_inference_output(
                        name=name,
                        output_path=output_path,
                        case_output_contract=output_contract,
                        inference_contract=inference_contract,
                        inline_input=inline_input,
                        errors=errors,
                    )
                    output_bytes.append(output_path.read_bytes())
            if len(output_bytes) >= 2 and any(
                output != output_bytes[0] for output in output_bytes[1:]
            ):
                errors.append(
                    f"validation.json: inference case {name!r} is not "
                    f"deterministic across {len(output_bytes)} repeat runs"
                )


def validate_root_layout(run_dir: Path, errors: list[str]) -> None:
    if not run_dir.is_dir():
        errors.append(f"run directory does not exist: {run_dir}")
        return
    entries = {entry.name for entry in run_dir.iterdir()}
    for filename in sorted(REQUIRED_ROOT_FILES):
        if not (run_dir / filename).is_file():
            errors.append(f"missing required run artifact: {filename}")
    unknown = sorted(entries - ALLOWED_ROOT_ENTRIES)
    if unknown:
        errors.append("run root contains unsupported clutter: " + ", ".join(unknown))
    for path in run_dir.rglob("*"):
        if path.name in FORBIDDEN_NAMES:
            errors.append(
                f"forbidden run artifact or directory: {path.relative_to(run_dir)}"
            )
        if path.is_symlink():
            errors.append(
                f"run artifacts must not use symlinks: {path.relative_to(run_dir)}"
            )


def validate(
    project: Path,
    run_dir: Path,
    *,
    run_inference_test: bool,
    inference_timeout_seconds: int,
) -> list[str]:
    errors: list[str] = []
    validate_root_layout(run_dir, errors)
    if not run_dir.is_dir():
        return errors
    report_path = run_dir / "report.html"
    results_path = run_dir / "results.md"
    requirements_path = run_dir / "requirements.lock"
    if report_path.is_file():
        validate_report(report_path, errors)
    if requirements_path.is_file():
        validate_requirements(requirements_path, errors)
    run_document = (
        read_json(run_dir / "run.json", errors)
        if (run_dir / "run.json").is_file()
        else None
    )
    retained = (
        validate_run_document(run_document, run_dir, errors)
        if run_document is not None
        else set()
    )
    if report_path.is_file() and results_path.is_file():
        validate_handoff_content(
            run_document,
            report_path,
            results_path,
            errors,
        )
    needs_train = bool(retained & {"classical", "autogluon"})
    train_path = run_dir / "train.py"
    if needs_train and not train_path.is_file():
        errors.append(
            "missing train.py: a retained classical or AutoGluon backend "
            "needs a rebuild entry point"
        )
    if not needs_train and train_path.exists():
        errors.append(
            "train.py is not allowed for an SAP RPT-only run because SAP RPT "
            "does not train or fit"
        )
    validation_document = (
        read_json(run_dir / "validation.json", errors)
        if (run_dir / "validation.json").is_file()
        else None
    )
    inference_contract = (
        run_document.get("inference")
        if isinstance(run_document, dict)
        and isinstance(run_document.get("inference"), dict)
        else {}
    )
    cases = (
        validate_validation_document(
            validation_document,
            retained,
            inference_contract,
            executing_cases=run_inference_test,
            errors=errors,
        )
        if validation_document is not None
        else []
    )
    if run_inference_test and cases and not errors:
        run_inference_cases(
            cases,
            project,
            run_dir,
            inference_contract,
            inference_timeout_seconds,
            errors,
        )
    return errors


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    raw_run_dir = Path(args.artifacts_dir).expanduser()
    run_dir = (
        raw_run_dir.resolve()
        if raw_run_dir.is_absolute()
        else (project / raw_run_dir).resolve()
    )
    if args.inference_timeout_seconds <= 0:
        print("ERROR: --inference-timeout-seconds must be positive")
        return 2
    validation_path = run_dir / "validation.json"
    if args.run_inference_test and validation_path.is_file():
        status_error = set_validation_status(validation_path, "pending")
        if status_error:
            print("Validation failed with 1 error(s):")
            print(f"- {status_error}")
            return 1
    errors = validate(
        project,
        run_dir,
        run_inference_test=args.run_inference_test,
        inference_timeout_seconds=args.inference_timeout_seconds,
    )
    if errors:
        print(f"Validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1
    if args.run_inference_test:
        status_error = set_validation_status(validation_path, "passed")
        if status_error:
            print("Validation failed with 1 error(s):")
            print(f"- {status_error}")
            return 1
    print(f"Validated ML run: {run_dir}")
    if args.run_inference_test:
        print("All declared inference cases passed using temporary inputs/outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
