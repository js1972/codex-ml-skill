#!/usr/bin/env python3
"""Generate a leakage-aware dataset analysis report."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlparse

SCHEMA_VERSION = "2.1"
ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|uuid|guid|key$)", re.IGNORECASE)
REMOTE_PREFLIGHT_OVERRIDE = "--allow-unknown-remote-preflight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Local or supported remote CSV/Parquet source"
    )
    parser.add_argument("--output-dir", default="artefacts")
    parser.add_argument(
        "--mode", choices=["analysis-only", "model"], default="analysis-only"
    )
    parser.add_argument(
        "--run-kind",
        choices=["initial", "improvement"],
        default="initial",
        help="Use improvement when profiling a new immutable child model run",
    )
    parser.add_argument(
        "--task",
        choices=[
            "analysis",
            "classification",
            "regression",
            "time-series",
            "anomaly",
        ],
        default="analysis",
    )
    parser.add_argument("--target")
    parser.add_argument("--time-column")
    parser.add_argument("--group-column")
    parser.add_argument("--partition-column", default="_ml_partition")
    parser.add_argument("--train-label", default="train")
    parser.add_argument(
        "--evaluation-design",
        choices=[
            "holdout",
            "nested_cv",
            "external_test",
            "prospective_validation",
        ],
        default="holdout",
        help="Predeclared final evaluation design for model mode",
    )
    parser.add_argument(
        "--split-strategy",
        choices=[
            "random",
            "stratified_random",
            "grouped",
            "temporal",
            "grouped_temporal",
        ],
        help="Required persisted split mechanics for model mode",
    )
    parser.add_argument(
        "--group-overlap-policy",
        choices=[
            "disallow",
            "known_series_temporal",
            "known_entity_temporal",
        ],
        default="disallow",
        help="Explicit policy for groups spanning persisted partitions",
    )
    parser.add_argument("--max-plot-rows", type=int, default=10_000)
    parser.add_argument("--max-numeric-plots", type=int, default=12)
    parser.add_argument("--max-categorical-plots", type=int, default=12)
    parser.add_argument(
        "--max-panel-series",
        type=int,
        default=12,
        help="Maximum representative panel series to render",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--risk-tier",
        choices=["not_assessed", "standard", "high"],
        default=None,
        help="Governance risk tier; defaults to not_assessed for a new config",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "pandas", "duckdb"],
        default="auto",
        help="Use pandas in memory or DuckDB with disk spilling",
    )
    parser.add_argument(
        "--max-in-memory-bytes",
        type=int,
        default=0,
        help="Override the auto-routing memory budget; 0 derives it from the host",
    )
    parser.add_argument(
        "--expected-source-bytes",
        type=int,
        default=0,
        help="Expected remote/object-store input size for disk preflight",
    )
    parser.add_argument(
        "--expected-source-rows",
        type=int,
        default=0,
        help="Optional expected remote/object-store row count metadata",
    )
    parser.add_argument(
        "--remote-source-version",
        help=(
            "Caller-declared version, ETag, snapshot, or table-version "
            "identifier; recorded as unverified by the generic profiler"
        ),
    )
    parser.add_argument(
        REMOTE_PREFLIGHT_OVERRIDE,
        action="store_true",
        help=(
            "Explicitly accept an unknown remote byte-scan or immutable-version "
            "preflight; the override is recorded in generated artifacts"
        ),
    )
    parser.add_argument("--duckdb-memory-limit", default="4GB")
    parser.add_argument("--duckdb-temp-directory")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def is_remote_location(location: str) -> bool:
    parsed = urlparse(location)
    return bool(parsed.scheme and parsed.scheme != "file")


def remote_preflight(args: argparse.Namespace) -> dict:
    """Fail before a remote scan unless cost and source identity are bounded."""
    if not is_remote_location(args.input):
        return {
            "applies": False,
            "status": "not_applicable",
            "override_used": False,
            "unknown_fields": [],
        }
    unknown_fields = []
    if args.expected_source_bytes <= 0:
        unknown_fields.append("expected_source_bytes")
    if not args.remote_source_version:
        unknown_fields.append("remote_source_version")
    override_used = bool(args.allow_unknown_remote_preflight)
    if unknown_fields and not override_used:
        missing = ", ".join(unknown_fields)
        raise SystemExit(
            "Remote/object-store scan preflight failed closed before scanning: "
            f"missing {missing}. Supply --expected-source-bytes and "
            "--remote-source-version, or explicitly accept the unknown scan "
            f"cost/source identity with {REMOTE_PREFLIGHT_OVERRIDE}."
        )
    return {
        "applies": True,
        "status": "overridden" if unknown_fields else "passed",
        "override_used": override_used,
        "unknown_fields": unknown_fields,
        "expected_source_bytes": args.expected_source_bytes or None,
        "expected_source_rows": args.expected_source_rows or None,
        "immutable_source_version": args.remote_source_version,
        "version_verification": (
            "declared_not_verified" if args.remote_source_version else "not_available"
        ),
        "reproducibility_status": "limited_remote_source",
    }


def validate_profiler_args(args) -> dict:
    if args.max_panel_series <= 0:
        raise SystemExit("--max-panel-series must be greater than zero.")
    if args.expected_source_bytes < 0 or args.expected_source_rows < 0:
        raise SystemExit("Expected remote source bytes/rows cannot be negative.")
    if args.mode == "model" and not args.split_strategy:
        raise SystemExit("--split-strategy is required in model mode.")
    if args.mode != "model" and args.split_strategy:
        raise SystemExit("--split-strategy applies only in model mode.")
    if args.mode != "model" and args.run_kind != "initial":
        raise SystemExit("--run-kind improvement applies only in model mode.")
    grouped = args.split_strategy in {"grouped", "grouped_temporal"}
    temporal = args.split_strategy in {"temporal", "grouped_temporal"}
    if args.evaluation_design == "nested_cv" and temporal:
        raise SystemExit(
            "The generic profiler does not represent prior-only training windows "
            "for temporal nested evaluation. Use a holdout, external, or "
            "prospective final design with rolling-origin development folds."
        )
    if args.split_strategy == "stratified_random" and not args.target:
        raise SystemExit(
            "--split-strategy stratified_random requires --target because the "
            "generic profiler has no separate stratification-column option."
        )
    if grouped and not args.group_column:
        raise SystemExit(
            f"--split-strategy {args.split_strategy} requires --group-column."
        )
    if temporal and not args.time_column:
        raise SystemExit(
            f"--split-strategy {args.split_strategy} requires --time-column."
        )
    if args.group_overlap_policy == "known_series_temporal" and not (
        args.mode == "model"
        and args.task == "time-series"
        and args.split_strategy == "grouped_temporal"
        and args.group_column
        and args.time_column
    ):
        raise SystemExit(
            "--group-overlap-policy known_series_temporal requires model-mode "
            "time-series profiling with --split-strategy grouped_temporal, "
            "--group-column, and --time-column."
        )
    if args.group_overlap_policy == "known_entity_temporal" and not (
        args.mode == "model"
        and args.task in {"classification", "regression"}
        and args.split_strategy in {"temporal", "grouped_temporal"}
        and args.group_column
        and args.time_column
    ):
        raise SystemExit(
            "--group-overlap-policy known_entity_temporal requires model-mode "
            "classification or regression with a temporal split, "
            "--group-column, and --time-column."
        )
    return remote_preflight(args)


def remote_source_contract(args, preflight: dict) -> dict:
    clean_source = re.sub(r"[?#].*$", "", args.input)
    version = args.remote_source_version
    return {
        "kind": (
            "remote_declared_version"
            if version
            else "remote_reference_preflight_overridden"
        ),
        "sha256": None,
        "bytes": args.expected_source_bytes or None,
        "expected_rows": args.expected_source_rows or None,
        "source": clean_source,
        "remote_source_version": version,
        "immutable_source_id": f"{clean_source}@{version}" if version else None,
        "source_version_provenance": (
            "user_declared_not_verified" if version else "not_supplied"
        ),
        "version_verification": (
            "declared_not_verified" if version else "not_available"
        ),
        "reproducibility_status": "limited_remote_source",
        "remote_preflight": preflight,
    }


def expected_config_mode(args) -> str:
    if args.mode == "analysis-only":
        return "analysis-only"
    return "model-improvement" if args.run_kind == "improvement" else "model-building"


def _nested_config_value(config: dict, *path):
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_compatible_value(config: dict, expected, *path) -> None:
    observed = _nested_config_value(config, *path)
    if observed is not None and observed != expected:
        dotted = ".".join(path)
        raise SystemExit(
            f"Existing config.json conflicts at '{dotted}': "
            f"expected {expected!r}, found {observed!r}. "
            "The profiler will not relabel or overwrite an established contract."
        )


def local_source_sha256(location: str) -> str | None:
    parsed = urlparse(location)
    if parsed.scheme not in {"", "file"}:
        return None
    local = (
        Path(parsed.path if parsed.scheme == "file" else location)
        .expanduser()
        .resolve()
    )
    digest = hashlib.sha256()
    with local.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_existing_config(output_dir: Path, args) -> dict | None:
    """Validate an established 2.1 contract before reading or scanning data."""
    path = output_dir / "config.json"
    if not path.exists():
        return None
    if (output_dir / "run_manifest.json").exists():
        raise SystemExit(
            "The selected output directory already contains run_manifest.json "
            "and is immutable. Profile into a new versioned run directory."
        )
    fingerprint_path = output_dir / "data_fingerprint.json"
    if fingerprint_path.exists():
        try:
            fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Existing data_fingerprint.json is invalid: {exc}"
            ) from exc
        recorded_digest = _nested_config_value(fingerprint, "input", "sha256")
        current_digest = local_source_sha256(args.input)
        if (
            isinstance(recorded_digest, str)
            and bool(recorded_digest.strip())
            and current_digest is not None
            and recorded_digest.lower().removeprefix("sha256:") != current_digest
        ):
            raise SystemExit(
                "The input contents changed since data_fingerprint.json was "
                "written. Profile into a new output/run directory."
            )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Existing config.json is invalid: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit("Existing config.json must contain a JSON object.")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(
            "Existing config.json must use schema_version "
            f"{SCHEMA_VERSION!r}; found {config.get('schema_version')!r}. "
            "Migrate it explicitly before profiling."
        )
    for section in ("problem", "data", "split", "evaluation", "governance", "analysis"):
        value = config.get(section)
        if value is not None and not isinstance(value, dict):
            raise SystemExit(
                f"Existing config.json field '{section}' must be an object or null."
            )

    model_mode = args.mode == "model"
    checks = [
        (expected_config_mode(args), ("mode",)),
        (args.task, ("problem", "task")),
        (args.target, ("problem", "target")),
        (args.group_column, ("problem", "group_column")),
        (args.time_column, ("problem", "time_column")),
        (
            args.evaluation_design if model_mode else None,
            ("evaluation", "design"),
        ),
        (
            args.partition_column if model_mode else None,
            ("split", "assignment_column"),
        ),
        (
            args.train_label
            if model_mode and args.evaluation_design != "nested_cv"
            else None,
            ("split", "development_label"),
        ),
        (
            declared_split_strategy(args) if model_mode else None,
            ("split", "strategy"),
        ),
        (
            args.group_overlap_policy if model_mode else None,
            ("split", "group_overlap_policy"),
        ),
        ([args.input], ("data", "locations")),
    ]
    if args.risk_tier is not None:
        checks.append((args.risk_tier, ("governance", "risk_tier")))
    if is_remote_location(args.input):
        checks.extend(
            [
                ("limited_remote_source", ("data", "reproducibility_status")),
                (
                    args.remote_source_version,
                    ("data", "remote_source_version"),
                ),
                (
                    "declared_not_verified"
                    if args.remote_source_version
                    else "not_available",
                    ("data", "version_verification"),
                ),
            ]
        )
    for expected, path_parts in checks:
        _validate_compatible_value(config, expected, *path_parts)
    return config


def fill_missing_values(target: dict, defaults: dict) -> None:
    """Recursively fill absent/null fields while preserving established values."""
    for key, default in defaults.items():
        current = target.get(key)
        if key not in target or current is None:
            target[key] = default
        elif isinstance(default, dict):
            if not isinstance(current, dict):
                raise SystemExit(
                    f"Existing config.json field '{key}' cannot be merged with "
                    "the generated object contract."
                )
            fill_missing_values(current, default)


def available_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or available_pages <= 0:
        return None
    return int(page_size * available_pages)


def estimated_in_memory_bytes(location: str) -> int | None:
    parsed = urlparse(location)
    if parsed.scheme and parsed.scheme != "file":
        return None
    path = Path(location).expanduser()
    if not path.is_file():
        return None
    lower = path.name.lower()
    multiplier = 6 if lower.endswith((".csv", ".csv.gz", ".csv.zst")) else 3
    return path.stat().st_size * multiplier


def should_use_duckdb(args: argparse.Namespace) -> tuple[bool, str]:
    if args.engine == "duckdb":
        return True, "explicit --engine duckdb"
    if args.engine == "pandas":
        return False, "explicit --engine pandas"
    parsed = urlparse(args.input)
    if parsed.scheme and parsed.scheme != "file":
        return True, "remote/object-store input has unknown in-memory size"
    estimate = estimated_in_memory_bytes(args.input)
    if estimate is None:
        return False, "input size unavailable"
    available = available_memory_bytes()
    derived_budget = (
        min(2 * 1024**3, int(available * 0.25)) if available else 512 * 1024**2
    )
    budget = args.max_in_memory_bytes or derived_budget
    if estimate > budget:
        return (
            True,
            (
                f"estimated in-memory footprint {estimate:,} bytes exceeds "
                f"budget {budget:,} bytes"
            ),
        )
    return False, f"estimated footprint {estimate:,} bytes fits budget {budget:,} bytes"


def run_duckdb_profiler(reason: str) -> int:
    if importlib.util.find_spec("duckdb") is None:
        raise SystemExit(
            "Dataset requires the disk-backed EDA route "
            f"({reason}), but DuckDB is not installed. Install it in the project "
            "environment with `python -m pip install duckdb`, or run the analysis "
            "inside a remote warehouse/cluster. Do not force pandas unless the "
            "dataset is known to fit memory."
        )
    script = Path(__file__).with_name("profile_large_dataset.py")
    forwarded = []
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument == "--engine":
            skip_next = True
            continue
        if argument.startswith("--engine="):
            continue
        forwarded.append(argument)
    print(f"Routing EDA to DuckDB: {reason}.")
    completed = subprocess.run(
        [sys.executable, str(script), *forwarded],
        check=False,
    )
    return completed.returncode


def import_analysis_packages():
    try:
        import matplotlib
        import pandas as pd

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "profile_dataset.py requires pandas and matplotlib in the "
            f"project environment: {exc}"
        ) from exc
    return pd, plt


def load_frame(pd, location: str):
    parsed = urlparse(location)
    path_part = parsed.path if parsed.scheme in {"http", "https"} else location
    lower = path_part.lower()
    if lower.endswith((".csv", ".csv.gz", ".csv.zst")):
        return pd.read_csv(location)
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(location)
    raise SystemExit("Input must be a CSV or Parquet file/URL.")


def source_fingerprint(pd, location: str, frame) -> dict:
    parsed = urlparse(location)
    if not parsed.scheme or parsed.scheme == "file":
        local = (
            Path(parsed.path if parsed.scheme == "file" else location)
            .expanduser()
            .resolve()
        )
        return {
            "kind": "file_sha256",
            "sha256": local_source_sha256(location),
            "bytes": local.stat().st_size,
            "source": str(local),
        }

    row_hashes = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return {
        "kind": "loaded_frame_sha256",
        "sha256": hashlib.sha256(row_hashes).hexdigest(),
        "bytes": None,
        "source": re.sub(r"[?#].*$", "", location),
    }


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def semantic_type(pd, series, column: str, time_column: str | None) -> str:
    if column == time_column or pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    non_null = series.dropna()
    if not non_null.empty:
        average_length = non_null.astype(str).str.len().mean()
        if average_length > 80:
            return "free_text"
    return "categorical"


def add_finding(findings, severity, code, message, recommendation, column=None):
    findings.append(
        {
            "severity": severity,
            "code": code,
            "column": column,
            "message": message,
            "recommendation": recommendation,
        }
    )


def add_contract_findings(findings, args, preflight, observed_rows, manifest=None):
    if manifest:
        for severity in ("blocker", "warning"):
            for item in manifest[f"{severity}s"]:
                add_finding(
                    findings,
                    severity,
                    item["code"],
                    item["message"],
                    (
                        "Resolve the persisted split contract before modeling."
                        if severity == "blocker"
                        else "Review the target-blind audit in split_manifest.json."
                    ),
                    args.partition_column,
                )
    if preflight["override_used"]:
        add_finding(
            findings,
            "warning",
            "remote_preflight_overridden",
            "Unknown remote preflight fields were explicitly accepted: "
            f"{', '.join(preflight['unknown_fields']) or 'none'}.",
            "Record the eventual bytes scanned and immutable source identity.",
        )
    if preflight["applies"] and args.remote_source_version:
        add_finding(
            findings,
            "warning",
            "remote_version_binding_not_verified",
            "The remote source version was declared by the caller but could not "
            "be verified against the bytes scanned by this profiler.",
            "Verify the object version, ETag, snapshot, or table version in the "
            "source platform before treating this run as exactly reproducible.",
        )
    if preflight["applies"] and not args.remote_source_version:
        add_finding(
            findings,
            "warning",
            "remote_source_version_unknown",
            "The remote preflight override accepted an unversioned source.",
            "Capture and independently verify an object version, ETag, snapshot, "
            "or table version.",
        )
    if args.expected_source_rows and args.expected_source_rows != observed_rows:
        add_finding(
            findings,
            "warning",
            "remote_row_preflight_mismatch",
            f"Expected {args.expected_source_rows:,} rows but observed "
            f"{observed_rows:,}.",
            "Reconcile the source metadata and immutable version before reuse.",
        )


def target_blind_semantic_type(pd, series, column: str, time_column: str | None):
    """Infer only from declared dtype/name without examining sealed values."""
    if column == time_column or pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    return "not_assessed"


def profile_columns(
    pd,
    frame,
    time_column: str | None,
    findings: list[dict],
    blind_columns: set[str] | None = None,
):
    profiles = {}
    rows = len(frame)
    blind_columns = blind_columns or set()
    for column in frame.columns:
        series = frame[column]
        if column in blind_columns:
            profiles[column] = {
                "dtype": str(series.dtype),
                "semantic_type": target_blind_semantic_type(
                    pd, series, column, time_column
                ),
                "missing_count": None,
                "missing_fraction": None,
                "unique_count": None,
                "unique_fraction": None,
                "identifier_like": bool(ID_NAME_RE.search(column)),
                "values_inspected": False,
                "profile_status": "target_blind_in_model_mode",
            }
            continue
        non_null = series.dropna()
        unique = int(non_null.nunique(dropna=True))
        missing = int(series.isna().sum())
        kind = semantic_type(pd, series, column, time_column)
        id_like = bool(ID_NAME_RE.search(column)) or (
            rows >= 20 and unique / max(rows, 1) >= 0.98
        )
        item = {
            "dtype": str(series.dtype),
            "semantic_type": kind,
            "missing_count": missing,
            "missing_fraction": missing / max(rows, 1),
            "unique_count": unique,
            "unique_fraction": unique / max(len(non_null), 1),
            "identifier_like": id_like,
            "values_inspected": True,
        }
        if kind == "numeric" and not non_null.empty:
            quantiles = non_null.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
            item["numeric"] = {
                "mean": finite_number(non_null.mean()),
                "std": finite_number(non_null.std()),
                "skew": finite_number(non_null.skew()),
                "zero_fraction": finite_number((non_null == 0).mean()),
                "quantiles": {
                    str(index): finite_number(value)
                    for index, value in quantiles.items()
                },
            }
        profiles[column] = item

        if missing / max(rows, 1) > 0.5:
            add_finding(
                findings,
                "warning",
                "high_missingness",
                f"Column '{column}' is {missing / max(rows, 1):.1%} missing.",
                "Determine why values are missing before dropping or imputing.",
                column,
            )
        if unique <= 1:
            add_finding(
                findings,
                "warning",
                "constant_column",
                f"Column '{column}' has no variation.",
                "Confirm whether it is required by the inference contract.",
                column,
            )
        elif id_like:
            add_finding(
                findings,
                "information",
                "identifier_like",
                f"Column '{column}' appears identifier-like.",
                "Use for grouping/joining if appropriate; do not model it blindly.",
                column,
            )
        if kind == "categorical" and unique > 100:
            add_finding(
                findings,
                "warning",
                "high_cardinality",
                f"Column '{column}' has {unique} observed categories.",
                "Use a high-cardinality-safe encoder/model and handle unseen values.",
                column,
            )
    return profiles


def build_schema_contract(
    profiles, args, population_label, population_rows, generated_at
):
    columns = {}
    for column, info in profiles.items():
        missing = info["missing_count"]
        columns[column] = {
            "dtype": info["dtype"],
            "semantic_type": info["semantic_type"],
            "identifier_like": info["identifier_like"],
            "inference_requiredness": "not_assessed",
            "observational_completeness": {
                "population": population_label,
                "population_rows": population_rows,
                "missing_count": missing,
                "missing_fraction": info["missing_fraction"],
                "non_missing_count": (
                    population_rows - missing if missing is not None else None
                ),
                "status": (
                    "observed" if missing is not None else "not_assessed_target_blind"
                ),
            },
        }
    unassessed = [
        name for name in profiles if name not in {args.target, args.partition_column}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "columns": columns,
        "inference": {
            "required_inputs": [],
            "optional_inputs": [],
            "unassessed_inputs": unassessed,
            "requiredness_status": "not_assessed",
        },
        "target": args.target,
        "time_column": args.time_column,
        "group_column": args.group_column,
        "partition_column": (args.partition_column if args.mode == "model" else None),
    }


def save_figure(plt, figure, path: Path, figures: list[dict], caption: str):
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    figures.append({"file": f"figures/{path.name}", "caption": caption})


def plot_missingness(plt, frame, output, figures):
    missing = frame.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(30)
    if missing.empty:
        return
    fig, axis = plt.subplots(figsize=(10, max(3, len(missing) * 0.28)))
    axis.barh(missing.index[::-1], missing.values[::-1], color="#4472C4")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Missing fraction")
    axis.set_title("Columns with missing values")
    save_figure(
        plt,
        fig,
        output / "missingness.png",
        figures,
        "Missing-value fractions for up to 30 columns.",
    )


def plot_numeric(plt, sample, profiles, output, figures, limit):
    columns = [
        column
        for column, info in profiles.items()
        if info["semantic_type"] == "numeric"
        and info.get("values_inspected", True)
        and not info["identifier_like"]
    ][:limit]
    if not columns:
        return
    cols = 3
    rows = math.ceil(len(columns) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        values = sample[column].dropna()
        axis.hist(values, bins=30, color="#4472C4", alpha=0.85)
        axis.set_title(column)
        axis.set_ylabel("Rows")
    for axis in axes.flat[len(columns) :]:
        axis.axis("off")
    fig.suptitle("Numeric distributions (plot sample)", y=1.01)
    save_figure(
        plt,
        fig,
        output / "numeric_distributions.png",
        figures,
        "Numeric histograms from the deterministic plot sample.",
    )


def plot_categorical(
    plt,
    sample,
    profiles,
    output,
    figures,
    limit,
):
    columns = [
        column
        for column, info in profiles.items()
        if info["semantic_type"] in {"categorical", "boolean"}
        and info.get("values_inspected", True)
        and not info["identifier_like"]
    ][:limit]
    if not columns:
        return
    cols = 2
    rows = math.ceil(len(columns) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        counts = (
            sample[column].astype("string").fillna("<missing>").value_counts().head(10)
        )
        labels = [str(value)[:30] for value in counts.index]
        axis.barh(labels[::-1], counts.values[::-1], color="#70AD47")
        axis.set_title(column)
        axis.set_xlabel("Rows")
    for axis in axes.flat[len(columns) :]:
        axis.axis("off")
    fig.suptitle("Top category frequencies (plot sample)", y=1.01)
    save_figure(
        plt,
        fig,
        output / "categorical_frequencies.png",
        figures,
        "Top category frequencies with their observed labels.",
    )


def plot_correlation(plt, analysis_frame, profiles, output, figures, sampled=False):
    numeric = [
        column
        for column, info in profiles.items()
        if info["semantic_type"] == "numeric"
        and info.get("values_inspected", True)
        and not info["identifier_like"]
    ][:20]
    if len(numeric) < 2:
        return
    correlation = analysis_frame[numeric].corr()
    fig, axis = plt.subplots(figsize=(max(7, len(numeric) * 0.45), 6))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(numeric)), numeric, rotation=90)
    axis.set_yticks(range(len(numeric)), numeric)
    population = "deterministic plot sample" if sampled else "analysis population"
    axis.set_title(f"Numeric correlation ({population})")
    fig.colorbar(image, ax=axis, fraction=0.03)
    save_figure(
        plt,
        fig,
        output / "numeric_correlation.png",
        figures,
        "Pearson correlations "
        f"from the {population} are screening clues, not proof of redundancy "
        "or leakage.",
    )


def plot_target(
    pd,
    plt,
    analysis_frame,
    plot_sample,
    mode,
    task,
    target,
    profiles,
    output,
    figures,
    findings,
):
    if not target:
        return None
    if target not in analysis_frame.columns:
        raise SystemExit(f"Target column '{target}' does not exist.")
    full_values = analysis_frame[target].dropna()
    plot_values = plot_sample[target].dropna()
    if full_values.empty:
        add_finding(
            findings,
            "blocker" if mode == "model" else "warning",
            "empty_target",
            "The permitted target analysis population has no non-missing labels.",
            "Fix label generation or partitioning before modeling.",
            target,
        )
        return {"non_missing_count": 0}

    if task in {"classification", "anomaly"}:
        counts = full_values.astype("string").value_counts()
        plotted_counts = counts.head(30)
        labels = [str(value)[:30] for value in plotted_counts.index]
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.bar(labels, plotted_counts.values, color="#ED7D31")
        axis.set_title("Target support (analysis population)")
        axis.set_ylabel("Rows")
        axis.tick_params(axis="x", rotation=45)
        save_figure(
            plt,
            fig,
            output / "target_distribution.png",
            figures,
            "Target support from the permitted target-aware population.",
        )
        if len(counts) > len(plotted_counts):
            add_finding(
                findings,
                "information",
                "target_classes_truncated_in_chart",
                f"The target has {len(counts)} classes; the chart shows the 30 "
                "most frequent.",
                "Use data_profile.json for complete target support.",
                target,
            )
        minority = counts.min() / max(counts.sum(), 1)
        if minority < 0.1:
            add_finding(
                findings,
                "warning",
                "class_imbalance",
                f"The smallest class represents {minority:.1%} of labeled rows.",
                "Use rare-event metrics and verify event counts in every fold.",
                target,
            )
        if len(counts) < 2:
            add_finding(
                findings,
                "blocker" if mode == "model" else "warning",
                "single_class_target",
                "The target contains fewer than two classes in the analysis population.",
                "Fix the split or target before classification modeling.",
                target,
            )
        return {
            "non_missing_count": int(counts.sum()),
            "class_count": len(counts),
            "class_support": [
                {"label": str(label), "rows": int(rows)}
                for label, rows in counts.items()
            ],
        }

    if profiles[target]["semantic_type"] == "numeric" and not plot_values.empty:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.hist(plot_values, bins=30, color="#ED7D31")
        axis.set_title("Target distribution (analysis population)")
        axis.set_xlabel(target)
        axis.set_ylabel("Rows")
        save_figure(
            plt,
            fig,
            output / "target_distribution.png",
            figures,
            "Target distribution from the permitted target-aware population.",
        )
    return {
        "non_missing_count": len(full_values),
        "numeric": profiles[target].get("numeric"),
    }


def plot_feature_relationships(
    pd,
    plt,
    analysis_sample,
    task,
    target,
    profiles,
    output,
    figures,
):
    """Show a small, screened set of target relationships without claiming causality."""
    if (
        not target
        or target not in analysis_sample.columns
        or task not in {"classification", "regression"}
    ):
        return
    numeric = [
        column
        for column, info in profiles.items()
        if column != target
        and info["semantic_type"] == "numeric"
        and info.get("values_inspected", True)
        and not info["identifier_like"]
    ]
    if not numeric:
        return

    if task == "classification":
        target_values = analysis_sample[target].dropna()
        classes = list(target_values.unique())
        if not 2 <= len(classes) <= 10:
            return
        scores = []
        for column in numeric:
            grouped = analysis_sample[[column, target]].dropna().groupby(target)[column]
            means = grouped.mean()
            pooled_std = analysis_sample[column].std()
            if len(means) >= 2 and pooled_std and math.isfinite(float(pooled_std)):
                scores.append(
                    (
                        float((means.max() - means.min()) / pooled_std),
                        column,
                    )
                )
        selected = [column for _, column in sorted(scores, reverse=True)[:6]]
        if not selected:
            return
        cols = 2
        rows = math.ceil(len(selected) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.7 * rows), squeeze=False)
        class_order = sorted(classes, key=lambda value: str(value))
        for axis, column in zip(axes.flat, selected):
            groups = [
                analysis_sample.loc[analysis_sample[target] == value, column].dropna()
                for value in class_order
            ]
            axis.boxplot(
                groups,
                tick_labels=[str(value)[:30] for value in class_order],
            )
            axis.set_title(column)
            axis.set_ylabel(column)
        for axis in axes.flat[len(selected) :]:
            axis.axis("off")
        fig.suptitle("Screened numeric features by target class", y=1.01)
        save_figure(
            plt,
            fig,
            output / "feature_target_relationships.png",
            figures,
            "Highest standardized mean-separation screens; evidence for investigation, not feature selection by itself.",
        )
        return

    numeric_target = pd.to_numeric(analysis_sample[target], errors="coerce")
    scores = []
    for column in numeric:
        pair = pd.DataFrame(
            {
                "feature": pd.to_numeric(analysis_sample[column], errors="coerce"),
                "target": numeric_target,
            }
        ).dropna()
        if len(pair) >= 3 and pair["feature"].nunique() > 1:
            correlation = pair["feature"].corr(pair["target"])
            if correlation is not None and math.isfinite(float(correlation)):
                scores.append((abs(float(correlation)), column))
    selected = [column for _, column in sorted(scores, reverse=True)[:6]]
    if not selected:
        return
    cols = 2
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.7 * rows), squeeze=False)
    for axis, column in zip(axes.flat, selected):
        pair = pd.DataFrame(
            {
                "feature": pd.to_numeric(analysis_sample[column], errors="coerce"),
                "target": numeric_target,
            }
        ).dropna()
        axis.scatter(pair["feature"], pair["target"], s=10, alpha=0.35)
        axis.set_title(column)
        axis.set_xlabel(column)
        axis.set_ylabel(target)
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    fig.suptitle("Screened numeric feature–target relationships", y=1.01)
    save_figure(
        plt,
        fig,
        output / "feature_target_relationships.png",
        figures,
        "Highest absolute Pearson-correlation screens; inspect nonlinearity, outliers and leakage before use.",
    )


def _representative_positions(total: int, limit: int) -> list[int]:
    if total <= 0:
        return []
    selected = min(total, max(limit, 1))
    if selected == 1:
        return [total // 2]
    return sorted(
        {round(index * (total - 1) / (selected - 1)) for index in range(selected)}
    )


def panel_coverage_contract(
    group_column,
    series_count,
    null_group_rows,
    row_summary,
    start,
    end,
    representatives,
    limit,
    method,
):
    return {
        "group_column": group_column,
        "series_count": int(series_count),
        "null_group_rows": int(null_group_rows),
        "rows_per_series": row_summary,
        "time_range": {"start": start, "end": end},
        "representative_selection": {
            "method": method,
            "limit": max(int(limit), 1),
            "selected": len(representatives),
        },
        "representative_series": representatives,
    }


def plot_representative_panel(
    plt,
    points,
    target,
    group_column,
    output,
    figures,
    max_plot_rows,
    caption,
):
    if points.empty:
        return
    labels = list(points["series"].dropna().unique())
    per_series_limit = max(2, max(int(max_plot_rows), 1) // max(len(labels), 1))
    fig, axis = plt.subplots(figsize=(12, 5.5))
    for series_label, series_frame in points.groupby("series", sort=True):
        series_frame = series_frame.sort_values("time")
        if len(series_frame) > per_series_limit:
            series_frame = series_frame.iloc[
                _representative_positions(len(series_frame), per_series_limit)
            ]
        axis.plot(
            series_frame["time"],
            series_frame["target"],
            label=str(series_label)[:40],
            linewidth=1.2,
            alpha=0.85,
        )
    axis.set_title("Representative panel series (analysis population)")
    axis.set_xlabel("Time")
    axis.set_ylabel(target)
    axis.legend(title=group_column, fontsize="small", ncol=2)
    save_figure(
        plt,
        fig,
        output / "representative_panel_series.png",
        figures,
        caption,
    )


def plot_time_series(
    pd,
    plt,
    structural_frame,
    target_frame,
    time_column,
    target,
    group_column,
    output,
    figures,
    findings,
    max_panel_series=12,
    max_plot_rows=10_000,
):
    if not time_column:
        return None
    if time_column not in structural_frame.columns:
        raise SystemExit(f"Time column '{time_column}' does not exist.")
    parsed = pd.to_datetime(structural_frame[time_column], errors="coerce", utc=True)
    invalid = int(parsed.isna().sum() - structural_frame[time_column].isna().sum())
    if invalid > 0:
        add_finding(
            findings,
            "warning",
            "invalid_timestamps",
            f"{invalid} non-missing values in '{time_column}' could not be parsed.",
            "Resolve formats/timezones before temporal splitting.",
            time_column,
        )
    coverage = pd.DataFrame({"time": parsed}).dropna()
    if not coverage.empty:
        coverage["rows"] = 1
        coverage = coverage.set_index("time").resample("D")["rows"].sum()
        fig, axis = plt.subplots(figsize=(11, 3.5))
        axis.plot(coverage.index, coverage.values, color="#70AD47")
        axis.set_title("Row coverage over time")
        axis.set_xlabel("Time")
        axis.set_ylabel("Rows per day")
        save_figure(
            plt,
            fig,
            output / "time_coverage.png",
            figures,
            "Daily row coverage reveals gaps, bursts and changing collection cadence.",
        )
    panel_coverage = None
    if group_column:
        if group_column not in structural_frame.columns:
            raise SystemExit(f"Group column '{group_column}' does not exist.")
        labels = structural_frame[group_column].astype("string")
        valid_group = labels.notna()
        panel = pd.DataFrame(
            {
                "series": labels.loc[valid_group],
                "time": parsed.loc[valid_group],
            }
        )
        stats = (
            panel.groupby("series", sort=False)
            .agg(
                rows=("series", "size"),
                parseable_time_rows=("time", "count"),
                start=("time", "min"),
                end=("time", "max"),
            )
            .reset_index()
        )
        stats = stats.sort_values(["rows", "series"], kind="stable").reset_index(
            drop=True
        )
        representative = stats.iloc[
            _representative_positions(len(stats), max_panel_series)
        ].copy()
        row_counts = stats["rows"] if not stats.empty else None
        representatives = [
            {
                "label": str(row.series),
                "rows": int(row.rows),
                "parseable_time_rows": int(row.parseable_time_rows),
                "start": row.start.isoformat() if pd.notna(row.start) else None,
                "end": row.end.isoformat() if pd.notna(row.end) else None,
            }
            for row in representative.itertuples(index=False)
        ]
        panel_coverage = panel_coverage_contract(
            group_column,
            len(stats),
            (~valid_group).sum(),
            {
                "minimum": int(row_counts.min()) if row_counts is not None else None,
                "median": finite_number(row_counts.median())
                if row_counts is not None
                else None,
                "maximum": int(row_counts.max()) if row_counts is not None else None,
            },
            parsed.min().isoformat() if parsed.notna().any() else None,
            parsed.max().isoformat() if parsed.notna().any() else None,
            representatives,
            max_panel_series,
            (
                "deterministic evenly spaced ranks after ordering series by "
                "row coverage and observed label"
            ),
        )
        if len(stats) > len(representative):
            add_finding(
                findings,
                "information",
                "panel_series_plot_limited",
                f"The panel contains {len(stats):,} series; the chart renders "
                f"{len(representative):,} representative series.",
                "Use panel_coverage in data_profile.json for the recorded coverage "
                "summary and adjust --max-panel-series when a larger readable "
                "view is needed.",
                group_column,
            )
    if (
        not target
        or target_frame is None
        or target not in target_frame.columns
        or time_column not in target_frame.columns
    ):
        return panel_coverage
    target_parsed = pd.to_datetime(target_frame[time_column], errors="coerce", utc=True)
    numeric_target = pd.to_numeric(target_frame[target], errors="coerce")
    if group_column:
        plot_frame = pd.DataFrame(
            {
                "time": target_parsed,
                "series": target_frame[group_column].astype("string"),
                "target": numeric_target,
            }
        ).dropna()
        if plot_frame.empty or not panel_coverage:
            return panel_coverage
        selected_labels = {
            item["label"] for item in panel_coverage["representative_series"]
        }
        plot_frame = plot_frame.loc[plot_frame["series"].isin(selected_labels)]
        plot_frame = plot_frame.groupby(["series", "time"], as_index=False)[
            "target"
        ].mean()
        plot_representative_panel(
            plt,
            plot_frame,
            target,
            group_column,
            output,
            figures,
            max_plot_rows,
            "Deterministically selected low-, middle- and high-coverage series; "
            "values are aggregated only within each series and timestamp.",
        )
        return panel_coverage
    plot_frame = pd.DataFrame(
        {"time": target_parsed, "target": numeric_target}
    ).dropna()
    if plot_frame.empty:
        return panel_coverage
    plot_frame = plot_frame.groupby("time", as_index=False)["target"].mean()
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.plot(plot_frame["time"], plot_frame["target"], color="#4472C4")
    axis.set_title("Target over time (analysis population)")
    axis.set_xlabel("Time")
    axis.set_ylabel(target)
    save_figure(
        plt,
        fig,
        output / "target_over_time.png",
        figures,
        "Target aggregated by timestamp for the permitted analysis population.",
    )
    return panel_coverage


def _partition_role(args, name: str) -> str:
    if args.evaluation_design == "nested_cv":
        return "outer_evaluation"
    if name == args.train_label:
        return "development"
    return "final_evaluation"


def declared_split_strategy(args) -> str | None:
    return args.split_strategy if args.mode == "model" else None


def panel_group_overlap_allowed(args) -> bool:
    return args.group_overlap_policy in {
        "known_series_temporal",
        "known_entity_temporal",
    }


def group_overlap_reason(args) -> str:
    if args.group_overlap_policy == "known_series_temporal":
        return "Panel forecasting evaluates future periods for known series."
    if args.group_overlap_policy == "known_entity_temporal":
        return (
            "Deployment scores future events for known entities; temporal "
            "ordering and feature as-of availability remain mandatory."
        )
    return "No deployment-specific overlap exception was established."


def split_is_temporal(args) -> bool:
    return args.split_strategy in {"temporal", "grouped_temporal"}


def _assignment_labels(series):
    return series.astype("string").fillna("<missing>")


def update_assignment_digest(digest, ordinal: int, value) -> None:
    digest.update(ordinal.to_bytes(8, "big", signed=False))
    null_value = value is None or value.__class__.__name__ in {"NAType", "NaTType"}
    if not null_value and isinstance(value, (float,)):
        null_value = math.isnan(value)
    if null_value:
        digest.update(b"N")
        return
    payload = str(value).encode("utf-8")
    digest.update(b"V" + len(payload).to_bytes(8, "big") + payload)


def _assignment_fingerprint(series) -> str:
    digest = hashlib.sha256()
    for ordinal, value in enumerate(series):
        update_assignment_digest(digest, ordinal, value)
    return digest.hexdigest()


def assemble_split_manifest(
    args,
    generated_at: str,
    partitions: list[dict],
    fingerprint_value: str,
    fingerprint_basis: str,
    audits: dict,
    warnings: list[dict],
    blockers: list[dict],
    generated_by: str,
):
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "strategy": declared_split_strategy(args),
        "evaluation_design": args.evaluation_design,
        "group_overlap_policy": args.group_overlap_policy,
        "assignment": {
            "column": args.partition_column,
            "source": "persisted_input_column",
            "fingerprint": {
                "algorithm": "sha256",
                "value": fingerprint_value,
                "basis": fingerprint_basis,
            },
        },
        "partitions": partitions,
        "audits": audits,
        "warnings": warnings,
        "blockers": blockers,
        "provenance": {
            "generated_by": generated_by,
            "seed": args.seed,
            "sealed_target_values_inspected": False,
            "target_excluded_from_duplicate_audit": bool(args.target),
            "target_excluded_from_duplicate_screening": bool(args.target),
            "partition_excluded_from_duplicate_screening": True,
            "duplicate_screening_excluded_columns": list(
                dict.fromkeys(
                    column
                    for column in (args.target, args.partition_column)
                    if column is not None
                )
            ),
        },
    }
    if args.evaluation_design == "nested_cv":
        outer_folds = [
            {"id": item["name"], "role": "outer_evaluation", "rows": item["rows"]}
            for item in partitions
            if item["role"] == "outer_evaluation"
        ]
        if outer_folds:
            manifest["folds"] = outer_folds
    return manifest


def initial_split_findings(
    args, partitions: list[dict], missing_rows: int
) -> tuple[list[dict], list[dict]]:
    warnings = []
    blockers = []
    if missing_rows:
        blockers.append(
            {
                "code": "missing_partition_assignments",
                "message": f"{missing_rows:,} rows have no split assignment.",
            }
        )
    if len(partitions) < 2:
        if args.evaluation_design == "nested_cv":
            warnings.append(
                {
                    "code": "nested_cv_outer_assignments_not_materialized",
                    "message": (
                        "Only one persisted population label is present; outer-fold "
                        "assignments must be materialized before target-aware fold "
                        "work."
                    ),
                }
            )
        else:
            blockers.append(
                {
                    "code": "insufficient_partitions",
                    "message": (
                        "Model profiling found fewer than two split partitions."
                    ),
                }
            )
    partition_names = {item["name"] for item in partitions}
    if (
        args.evaluation_design != "nested_cv"
        and args.train_label not in partition_names
    ):
        blockers.append(
            {
                "code": "development_partition_missing",
                "message": (
                    f"No rows use the configured development label "
                    f"'{args.train_label}'."
                ),
            }
        )
    return warnings, blockers


def temporal_audit_from_ranges(args, ranges: list[dict], invalid: int):
    comparable = [item for item in ranges if item["start"] and item["end"]]
    valid_order = bool(comparable) and invalid == 0
    if valid_order and args.evaluation_design == "nested_cv":
        ordered = sorted(comparable, key=lambda item: (item["start"], item["name"]))
        valid_order = all(
            left["end"] < right["start"] for left, right in pairwise(ordered)
        )
    elif valid_order:
        development = [item for item in comparable if item["name"] == args.train_label]
        evaluation = [item for item in comparable if item["name"] != args.train_label]
        valid_order = bool(development and evaluation) and all(
            development[0]["end"] < item["start"] for item in evaluation
        )
    reason = (
        "Persisted partition time ranges are chronologically ordered."
        if valid_order
        else (
            "Partition ranges overlap, timestamps are invalid, or a "
            "development/evaluation range is missing."
        )
    )
    audit = {
        "checked": True,
        "time_column": args.time_column,
        "valid": valid_order,
        "invalid_timestamp_rows": invalid,
        "ranges": ranges,
        "purge_gap": None,
        "reason": reason,
    }
    warning = (
        None
        if valid_order
        else {"code": "temporal_partition_order_unverified", "message": reason}
    )
    return audit, warning


def build_split_manifest_pandas(pd, frame, args, generated_at: str):
    """Audit persisted split assignments without reading target values."""
    partition = args.partition_column
    labels = _assignment_labels(frame[partition])
    counts = labels.value_counts(dropna=False, sort=False)
    partitions = [
        {
            "name": str(name),
            "role": _partition_role(args, str(name)),
            "rows": int(rows),
        }
        for name, rows in sorted(counts.items(), key=lambda item: str(item[0]))
    ]
    missing_rows = int(labels.eq("<missing>").sum())
    warnings, blockers = initial_split_findings(args, partitions, missing_rows)

    if args.group_column:
        group_values = frame[args.group_column]
        valid = group_values.notna()
        group_assignments = pd.DataFrame(
            {"group": group_values.loc[valid], "partition": labels.loc[valid]}
        ).drop_duplicates()
        partitions_per_group = group_assignments.groupby("group", dropna=False)[
            "partition"
        ].nunique()
        spanning = int((partitions_per_group > 1).sum())
        allowed = panel_group_overlap_allowed(args)
        group_audit = {
            "checked": True,
            "group_column": args.group_column,
            "unique_groups": int(group_values.loc[valid].nunique(dropna=True)),
            "null_group_rows": int((~valid).sum()),
            "groups_spanning_partitions": spanning,
            "allowed": allowed,
            "reason": (
                group_overlap_reason(args)
                if spanning and allowed
                else "No deployment-specific overlap exception was established."
                if spanning
                else "No non-null groups span persisted partitions."
            ),
        }
        if spanning:
            destination = warnings if allowed else blockers
            destination.append(
                {
                    "code": "groups_span_partitions",
                    "message": (
                        f"{spanning:,} groups occur in more than one partition; "
                        "confirm whether this matches deployment."
                    ),
                }
            )
        group_required = args.split_strategy in {
            "grouped",
            "grouped_temporal",
        } or args.group_overlap_policy in {
            "known_series_temporal",
            "known_entity_temporal",
        }
        if group_required and group_audit["null_group_rows"]:
            blockers.append(
                {
                    "code": "missing_split_group",
                    "message": (
                        f"{group_audit['null_group_rows']:,} rows have no "
                        "group/entity ID, so overlap cannot be audited."
                    ),
                }
            )
    else:
        group_audit = {
            "checked": False,
            "groups_spanning_partitions": None,
            "allowed": False,
            "reason": "No group column was supplied.",
        }

    if split_is_temporal(args):
        parsed = pd.to_datetime(frame[args.time_column], errors="coerce", utc=True)
        invalid = int(parsed.isna().sum())
        temporal_frame = pd.DataFrame({"partition": labels, "time": parsed})
        ranges = []
        for name, subset in temporal_frame.groupby("partition", sort=True):
            valid_time = subset["time"].dropna()
            ranges.append(
                {
                    "name": str(name),
                    "role": _partition_role(args, str(name)),
                    "rows": len(subset),
                    "parseable_time_rows": len(valid_time),
                    "start": valid_time.min().isoformat()
                    if not valid_time.empty
                    else None,
                    "end": valid_time.max().isoformat()
                    if not valid_time.empty
                    else None,
                }
            )
        temporal_audit, temporal_warning = temporal_audit_from_ranges(
            args, ranges, invalid
        )
        if temporal_warning:
            blockers.append(temporal_warning)
    else:
        temporal_audit = {
            "checked": False,
            "valid": None,
            "purge_gap": None,
            "reason": (
                "The declared split strategy is not temporal; the time column "
                "is profiled structurally only."
            ),
        }

    duplicate_audit = {
        "checked": False,
        "rows_crossing_partitions": None,
        "reason": (
            "The generic profiler does not make a target-blind near/exact "
            "duplicate identity decision; audit approved feature/key columns "
            "before training."
        ),
    }

    return assemble_split_manifest(
        args,
        generated_at,
        partitions,
        _assignment_fingerprint(frame[partition]),
        (
            "source row ordinal plus a null marker or length-prefixed UTF-8 "
            "assignment value"
        ),
        {
            "group_overlap": group_audit,
            "temporal_order": temporal_audit,
            "duplicate_overlap": duplicate_audit,
        },
        warnings,
        blockers,
        Path(__file__).name,
    )


def markdown_summary(profile: dict) -> str:
    severity_order = ["blocker", "warning", "information"]
    target_population = profile.get("target_aware_population")
    lines = [
        "# Dataset Summary",
        "",
        f"- Mode: {profile['mode']}",
        f"- Task: {profile['task']}",
        f"- Rows: {profile['shape']['rows']:,}",
        f"- Columns: {profile['shape']['columns']:,}",
        f"- Structural population: {profile['analysis_population']['label']}",
        "- Target-aware population: "
        + (
            target_population["label"]
            if target_population is not None
            else "not generated"
        ),
        (
            f"- Plot sample: {profile['plot_sampling']['rows']:,} rows "
            f"(seed {profile['plot_sampling']['seed']})"
        ),
        "",
        "## Findings",
        "",
    ]
    for severity in severity_order:
        matching = [
            item for item in profile["findings"] if item["severity"] == severity
        ]
        lines.append(f"### {severity.title()}s")
        lines.append("")
        if not matching:
            lines.append("- None detected by the automated checks.")
        else:
            for item in matching:
                suffix = f" ({item['column']})" if item.get("column") else ""
                lines.append(
                    f"- **{item['code']}{suffix}:** {item['message']} "
                    f"{item['recommendation']}"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Automated findings are screening evidence, not domain conclusions.",
            (
                "- In model mode, target-aware analysis, when generated, uses only "
                "the declared development population."
            ),
            "- Review the HTML charts and data contract before choosing transformations.",
            "",
        ]
    )
    return "\n".join(lines)


def html_report(profile: dict) -> str:
    target_population = profile.get("target_aware_population")
    target_population_label = (
        target_population["label"] if target_population is not None else "not generated"
    )
    counts = {
        severity: sum(item["severity"] == severity for item in profile["findings"])
        for severity in ["blocker", "warning", "information"]
    }
    findings_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['severity'])}</td>"
        f"<td>{html.escape(item['code'])}</td>"
        f"<td>{html.escape(item.get('column') or '')}</td>"
        f"<td>{html.escape(item['message'])}</td>"
        f"<td>{html.escape(item['recommendation'])}</td>"
        "</tr>"
        for item in profile["findings"]
    )
    column_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(column)}</td>"
        f"<td>{html.escape(info['semantic_type'])}</td>"
        "<td>"
        + (
            f"{info['missing_fraction']:.1%}"
            if info["missing_fraction"] is not None
            else "not assessed (target blind)"
        )
        + "</td><td>"
        + (
            str(info["unique_count"])
            if info["unique_count"] is not None
            else "not assessed"
        )
        + "</td>"
        + "</tr>"
        for column, info in profile["columns"].items()
    )
    figures = "\n".join(
        "<figure>"
        f"<img src='{html.escape(item['file'])}' alt='{html.escape(item['caption'])}'>"
        f"<figcaption>{html.escape(item['caption'])}</figcaption>"
        "</figure>"
        for item in profile["figures"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset analysis report</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 1100px; margin: auto; padding: 2rem; color: #222; }}
h1, h2 {{ color: #17365d; }}
.cards {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
.card {{ border: 1px solid #ccc; border-radius: 8px; padding: 1rem; min-width: 150px; }}
table {{ border-collapse: collapse; width: 100%; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid #ddd; padding: .5rem; text-align: left; vertical-align: top; }}
th {{ background: #eef3f8; }}
img {{ max-width: 100%; height: auto; }}
figcaption {{ color: #555; margin-bottom: 2rem; }}
</style>
</head>
<body>
<h1>Dataset analysis report</h1>
<p>Mode: <strong>{html.escape(profile["mode"])}</strong>; task:
<strong>{html.escape(profile["task"])}</strong>. Structural population:
<strong>{html.escape(profile["analysis_population"]["label"])}</strong>.
Target-aware population: <strong>{html.escape(target_population_label)}</strong>.</p>
<div class="cards">
<div class="card"><strong>{profile["shape"]["rows"]:,}</strong><br>rows</div>
<div class="card"><strong>{profile["shape"]["columns"]:,}</strong><br>columns</div>
<div class="card"><strong>{counts["blocker"]}</strong><br>blockers</div>
<div class="card"><strong>{counts["warning"]}</strong><br>warnings</div>
</div>
<h2>Findings</h2>
<table><thead><tr><th>Severity</th><th>Code</th><th>Column</th><th>Evidence</th><th>Recommendation</th></tr></thead>
<tbody>{findings_rows}</tbody></table>
<h2>Columns</h2>
<table><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead>
<tbody>{column_rows}</tbody></table>
<h2>Charts</h2>
{figures or "<p>No applicable charts were generated.</p>"}
<h2>Limits</h2>
<ul>
<li>Automated findings are screening evidence, not domain conclusions.</li>
<li>Plots may use a deterministic sample; statistics use all permitted rows.</li>
<li>In model mode, target-aware analysis, when generated, uses only the declared
development population.</li>
</ul>
</body>
</html>
"""


def artifact_path_context(output_dir: Path):
    try:
        relative = output_dir.relative_to(Path.cwd().resolve())
        return relative, {
            "directory": relative.as_posix(),
            "reference_base": "project_root",
        }
    except ValueError:
        return None, {
            "directory": str(output_dir),
            "reference_base": "artifact_directory",
        }


def update_config(
    output_dir: Path,
    args,
    structural_population_label: str,
    structural_rows: int,
    structural_plot_rows: int,
    target_population_label: str | None,
    target_rows: int,
    target_plot_rows: int,
    existing_config: dict | None = None,
    analysis_engine: str | None = None,
    approximate_aggregates: list[str] | None = None,
):
    """Create the EDA contract or fill only absent fields in a compatible one."""
    path = output_dir / "config.json"
    prefix, artifact_paths = artifact_path_context(output_dir)

    def reference(name):
        return (prefix / name).as_posix() if prefix is not None else name

    fingerprint_file = reference("data_fingerprint.json")
    schema_file = reference("schema.json")
    split_manifest_file = reference("split_manifest.json")
    report_file = reference("data_report.html")
    model_mode = args.mode == "model"
    preflight = remote_preflight(args)
    data_defaults = {
        "locations": [args.input],
        "fingerprint_file": fingerprint_file,
        "schema_file": schema_file,
        "split_manifest": split_manifest_file if model_mode else None,
        "reproducibility_status": (
            "limited_remote_source"
            if is_remote_location(args.input)
            else "reproducible_source"
        ),
    }
    if is_remote_location(args.input):
        data_defaults.update(
            {
                "remote_source_version": args.remote_source_version,
                "version_verification": (
                    "declared_not_verified"
                    if args.remote_source_version
                    else "not_available"
                ),
                "remote_preflight": preflight,
            }
        )
    analysis_defaults = {
        "report": report_file,
        "population_partition": None,
        "target_aware_partition": (
            args.train_label
            if model_mode and args.target and args.evaluation_design != "nested_cv"
            else None
        ),
        "population": structural_population_label,
        "structural_population": {
            "label": structural_population_label,
            "rows": structural_rows,
            "target_values_inspected": not (model_mode and args.target),
        },
        "target_aware_population": (
            {
                "label": target_population_label,
                "rows": target_rows,
                "partition": args.train_label if model_mode else None,
            }
            if target_population_label is not None
            else None
        ),
        "plot_sample_size": structural_plot_rows,
        "target_plot_sample_size": target_plot_rows,
        "plot_sample_seed": args.seed,
        "category_labels_rendered": True,
        "nested_cv_global_profile_target_blind": (
            model_mode and args.evaluation_design == "nested_cv"
        ),
    }
    if analysis_engine is not None:
        analysis_defaults["engine"] = analysis_engine
    if approximate_aggregates is not None:
        analysis_defaults["approximate_aggregates"] = approximate_aggregates
    defaults = {
        "schema_version": SCHEMA_VERSION,
        "mode": expected_config_mode(args),
        "artifact_paths": artifact_paths,
        "problem": {
            "task": args.task,
            "target": args.target,
            "prediction_moment": None,
            "row_grain": None,
            "group_column": args.group_column,
            "time_column": args.time_column,
        },
        "data": data_defaults,
        "split": {
            "strategy": declared_split_strategy(args),
            "group_overlap_policy": (args.group_overlap_policy if model_mode else None),
            "assignment_column": args.partition_column if model_mode else None,
            "development_label": (
                args.train_label
                if model_mode and args.evaluation_design != "nested_cv"
                else None
            ),
            "manifest_file": split_manifest_file if model_mode else None,
            "holdout_target_sealed": (
                model_mode and args.evaluation_design == "holdout"
            ),
            "seed": args.seed,
        },
        "evaluation": {
            "design": args.evaluation_design if model_mode else None,
            "final_eval_set": (
                {
                    "holdout": "holdout_test",
                    "nested_cv": "outer_cv",
                    "external_test": "external_test",
                    "prospective_validation": "prospective_validation",
                }[args.evaluation_design]
                if model_mode
                else None
            ),
            "independent_test": (
                args.evaluation_design != "nested_cv" if model_mode else None
            ),
            "selection_nested": (
                args.evaluation_design == "nested_cv" if model_mode else None
            ),
        },
        "governance": {
            "risk_tier": args.risk_tier or "not_assessed",
            "deployment_decision": "not_assessed",
            "approval_status": "not_assessed",
        },
        "analysis": analysis_defaults,
    }
    config = existing_config if existing_config is not None else {}
    fill_missing_values(config, defaults)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    preflight = validate_profiler_args(args)
    output_dir = Path(args.output_dir).resolve()
    existing_config = load_and_validate_existing_config(output_dir, args)
    use_duckdb, routing_reason = should_use_duckdb(args)
    if use_duckdb:
        return run_duckdb_profiler(routing_reason)
    pd, plt = import_analysis_packages()
    frame = load_frame(pd, args.input)
    if frame.empty:
        raise SystemExit("Input dataset contains no rows.")
    if len(set(frame.columns)) != len(frame.columns):
        raise SystemExit("Input dataset contains duplicate column names.")
    if args.target and args.target not in frame.columns:
        raise SystemExit(f"Target column '{args.target}' does not exist.")
    if args.group_column and args.group_column not in frame.columns:
        raise SystemExit(f"Group column '{args.group_column}' does not exist.")
    if args.time_column and args.time_column not in frame.columns:
        raise SystemExit(f"Time column '{args.time_column}' does not exist.")

    nested_cv_target_blind = (
        args.mode == "model" and args.evaluation_design == "nested_cv"
    )
    structural_frame = frame
    target_frame = frame if args.target else None
    structural_population_label = (
        "full permitted dataset (target-blind structural analysis)"
        if args.mode == "model"
        else "full permitted dataset (descriptive analysis)"
    )
    target_population_label = (
        "full permitted dataset (descriptive target-aware analysis)"
        if args.mode == "analysis-only" and args.target
        else None
    )
    if args.mode == "model":
        if args.partition_column not in frame.columns:
            raise SystemExit(
                "Model mode requires a persisted partition column before EDA. "
                f"Missing: '{args.partition_column}'."
            )
        if nested_cv_target_blind:
            target_frame = None
            target_population_label = None
        else:
            development_frame = frame.loc[
                frame[args.partition_column].astype(str) == args.train_label
            ].copy()
            if development_frame.empty:
                raise SystemExit(
                    f"Partition column '{args.partition_column}' has no "
                    f"'{args.train_label}' rows."
                )
            target_frame = development_frame if args.target else None
            target_population_label = (
                f"{args.partition_column}={args.train_label}" if args.target else None
            )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    split_manifest = (
        build_split_manifest_pandas(pd, frame, args, generated_at)
        if args.mode == "model"
        else None
    )

    findings: list[dict] = []
    add_contract_findings(findings, args, preflight, len(frame), split_manifest)
    duplicate_columns = list(frame.columns)
    duplicate_exclusions = []
    if args.mode == "model":
        duplicate_exclusions = list(
            dict.fromkeys(
                column
                for column in (args.target, args.partition_column)
                if column is not None
            )
        )
        duplicate_columns = [
            column
            for column in duplicate_columns
            if column not in set(duplicate_exclusions)
        ]
    duplicates = (
        int(frame[duplicate_columns].duplicated().sum()) if duplicate_columns else 0
    )
    if duplicates:
        add_finding(
            findings,
            "warning",
            "exact_duplicates",
            f"The raw dataset contains {duplicates:,} exact "
            f"{'feature-only ' if args.mode == 'model' else ''}"
            "duplicate rows.",
            "Determine whether these are accidental copies or legitimate repeated events.",
        )

    blind_columns = {args.target} if args.mode == "model" and args.target else set()
    profiles = profile_columns(
        pd,
        structural_frame,
        args.time_column,
        findings,
        blind_columns=blind_columns,
    )
    structural_columns = [
        column for column in structural_frame.columns if column not in blind_columns
    ]
    structural_plot_frame = structural_frame[structural_columns]
    structural_plot_rows = min(len(structural_plot_frame), max(args.max_plot_rows, 1))
    structural_plot_sample = (
        structural_plot_frame.sample(n=structural_plot_rows, random_state=args.seed)
        if structural_plot_rows < len(structural_plot_frame)
        else structural_plot_frame.copy()
    )
    target_profiles = None
    target_plot_rows = 0
    target_plot_sample = None
    if target_frame is not None and args.target:
        target_profiles = profile_columns(
            pd,
            target_frame[[args.target]],
            args.time_column,
            findings,
        )
        target_plot_rows = min(len(target_frame), max(args.max_plot_rows, 1))
        target_plot_sample = (
            target_frame.sample(n=target_plot_rows, random_state=args.seed)
            if target_plot_rows < len(target_frame)
            else target_frame.copy()
        )

    figures: list[dict] = []
    plot_missingness(plt, structural_plot_frame, figures_dir, figures)
    plot_numeric(
        plt,
        structural_plot_sample,
        profiles,
        figures_dir,
        figures,
        args.max_numeric_plots,
    )
    plot_categorical(
        plt,
        structural_plot_sample,
        profiles,
        figures_dir,
        figures,
        args.max_categorical_plots,
    )
    plot_correlation(plt, structural_plot_frame, profiles, figures_dir, figures)
    if nested_cv_target_blind:
        target_summary = {
            "status": "not_generated",
            "reason": (
                "Nested-CV global profiling is target-blind because every row "
                "may become an outer evaluation row."
            ),
            "target_values_inspected": False,
        }
    else:
        target_summary = plot_target(
            pd,
            plt,
            target_frame,
            target_plot_sample,
            args.mode,
            args.task,
            args.target,
            target_profiles or profiles,
            figures_dir,
            figures,
            findings,
        )
        plot_feature_relationships(
            pd,
            plt,
            target_plot_sample
            if target_plot_sample is not None
            else structural_plot_sample,
            args.task,
            args.target,
            profiles,
            figures_dir,
            figures,
        )
    panel_coverage = None
    if args.task == "time-series" or args.time_column:
        panel_coverage = plot_time_series(
            pd,
            plt,
            structural_frame,
            target_frame,
            args.time_column,
            args.target,
            args.group_column,
            figures_dir,
            figures,
            findings,
            max_panel_series=args.max_panel_series,
            max_plot_rows=args.max_plot_rows,
        )

    if is_remote_location(args.input):
        source = remote_source_contract(args, preflight)
    else:
        source = source_fingerprint(pd, args.input, frame)
        source["reproducibility_status"] = "reproducible_source"
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "input": source,
        "rows": len(frame),
        "columns": len(frame.columns),
    }
    schema = build_schema_contract(
        profiles,
        args,
        structural_population_label,
        len(structural_frame),
        generated_at,
    )
    target_population = (
        {
            "label": target_population_label,
            "rows": len(target_frame),
            "partition": args.train_label if args.mode == "model" else None,
        }
        if target_frame is not None and args.target
        else None
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "mode": args.mode,
        "task": args.task,
        "shape": {"rows": len(frame), "columns": len(frame.columns)},
        "analysis_population": {
            "label": structural_population_label,
            "rows": len(structural_frame),
        },
        "structural_population": {
            "label": structural_population_label,
            "rows": len(structural_frame),
            "target_values_inspected": not bool(blind_columns),
        },
        "target_aware_population": target_population,
        "plot_sampling": {
            "population_rows": len(structural_frame),
            "rows": structural_plot_rows,
            "method": "deterministic random sample without replacement"
            if structural_plot_rows < len(structural_frame)
            else "all permitted rows",
            "seed": args.seed,
        },
        "target_plot_sampling": (
            {
                "population_rows": len(target_frame),
                "rows": target_plot_rows,
                "method": "deterministic random sample without replacement"
                if target_plot_rows < len(target_frame)
                else "all permitted target-aware rows",
                "seed": args.seed,
            }
            if target_frame is not None and args.target
            else None
        ),
        "duplicates_full_dataset": duplicates,
        "duplicate_screening": {
            "population": "full permitted dataset",
            "method": "exact duplicated-row comparison",
            "columns": duplicate_columns,
            "excluded_columns": duplicate_exclusions,
            "target_excluded": bool(args.mode == "model" and args.target),
            "partition_excluded": args.mode == "model",
        },
        "target_summary": target_summary,
        "nested_cv_global_profile_target_blind": nested_cv_target_blind,
        "panel_coverage": panel_coverage,
        "split_manifest": "split_manifest.json" if split_manifest else None,
        "remote_preflight": preflight,
        "columns": profiles,
        "findings": findings,
        "figures": figures,
    }

    update_config(
        output_dir,
        args,
        structural_population_label,
        len(structural_frame),
        structural_plot_rows,
        target_population_label,
        len(target_frame) if target_frame is not None else 0,
        target_plot_rows,
        existing_config=existing_config,
    )
    (output_dir / "data_profile.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "data_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
    )
    if split_manifest:
        (output_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output_dir / "data_summary.md").write_text(
        markdown_summary(report), encoding="utf-8"
    )
    (output_dir / "data_report.html").write_text(html_report(report), encoding="utf-8")

    blockers = sum(item["severity"] == "blocker" for item in findings)
    warnings = sum(item["severity"] == "warning" for item in findings)
    print(
        f"Wrote dataset report to {output_dir} "
        f"({blockers} blocker(s), {warnings} warning(s))."
    )
    return 2 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
