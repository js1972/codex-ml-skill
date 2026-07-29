#!/usr/bin/env python3
"""Run a modeling-only preflight over a local tabular dataset.

This utility deliberately does not perform exploratory data analysis, create
charts, or retain artifacts. It prints a compact JSON assessment to stdout so
an agent can discuss blockers with the user before asking for approval to run
model backends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    missing = getattr(exc, "name", None) or "pandas/numpy"
    print(
        "Modeling preflight cannot start because "
        f"{missing!r} is unavailable. Re-run this script with a Python interpreter "
        "where pandas and numpy already exist; do not install dependencies as part "
        "of this skill.",
        file=sys.stderr,
    )
    raise SystemExit(2) from None


ID_NAME_RE = re.compile(
    r"(^id$|^id[_\-\s]|[_\-\s]id$|uuid|guid|identifier|"
    r"(?:^|[_\-\s])(?:row|record|sample|observation)[_\-\s]?id(?:$|[_\-\s]))",
    re.IGNORECASE,
)
PIPELINE_METADATA_RE = re.compile(
    r"(^|[_\-\s])(fold|split|partition)([_\-\s]|$)"
    r"|(^|[_\-\s])duplicate[_\-\s]?group([_\-\s]|$)"
    r"|^_?ml[_\-\s]",
    re.IGNORECASE,
)
FOLD_METADATA_RE = re.compile(
    r"(^|[_\-\s])(fold|split|partition)([_\-\s]|$)", re.IGNORECASE
)
GROUP_METADATA_RE = re.compile(
    r"(^|[_\-\s])duplicate[_\-\s]?group([_\-\s]|$)"
    r"|(^|[_\-\s])(?:group|entity|subject|patient|customer)[_\-\s]?id"
    r"([_\-\s]|$)",
    re.IGNORECASE,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Local CSV, TSV, or Parquet file")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument(
        "--task",
        choices=("auto", "classification", "regression"),
        default="auto",
        help="Modeling task; auto infers a conservative default",
    )
    parser.add_argument("--row-grain", required=True, help="Meaning of one dataset row")
    parser.add_argument(
        "--prediction-moment",
        required=True,
        help="When predictions will be made in the real workflow",
    )
    parser.add_argument("--group-column", help="Entity/group column for split review")
    parser.add_argument(
        "--time-column", help="Timestamp column for temporal split review"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Feature excluded from modeling; may be repeated",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional inspection cap; omission inspects the full local dataset",
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    except (AttributeError, OSError, ValueError):
        return None


def read_table(path: Path, max_rows: int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python", nrows=max_rows)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=max_rows)
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return frame if max_rows is None else frame.head(max_rows)
    raise ValueError("Supported inputs are CSV, TSV, and Parquet files")


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    return str(value)


def infer_task(target: pd.Series) -> str:
    non_null = target.dropna()
    unique = int(non_null.nunique())
    if (
        pd.api.types.is_bool_dtype(non_null)
        or isinstance(non_null.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(non_null)
        or unique <= max(20, int(math.sqrt(max(len(non_null), 1))))
    ):
        return "classification"
    return "regression"


def likely_identifier(name: str, series: pd.Series, row_count: int) -> bool:
    """Return True for strong identifier signals, not merely unique measurements."""
    non_null = int(series.notna().sum())
    if non_null == 0 or row_count == 0:
        return False
    if ID_NAME_RE.search(str(name)):
        return True
    uniqueness = float(series.nunique(dropna=True)) / non_null
    return bool(
        non_null >= 20
        and uniqueness >= 0.98
        and not pd.api.types.is_numeric_dtype(series.dtype)
    )


def named_pipeline_metadata(
    columns: list[str],
    *,
    target: str,
    group_column: str | None,
    time_column: str | None,
) -> list[str]:
    explicit_roles = {target, group_column, time_column}
    return [
        column
        for column in columns
        if column not in explicit_roles and PIPELINE_METADATA_RE.search(str(column))
    ]


def group_signature_audit(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    group_column: str,
) -> dict[str, Any]:
    """Describe whether equivalent feature vectors are split across groups."""
    groups = frame[group_column]
    present = groups.notna()
    group_sizes = groups[present].value_counts()
    summary: dict[str, Any] = {
        "column": group_column,
        "unique_groups": int(groups.nunique(dropna=True)),
        "missing_rows": int(groups.isna().sum()),
        "minimum_group_rows": int(group_sizes.min()) if not group_sizes.empty else 0,
        "maximum_group_rows": int(group_sizes.max()) if not group_sizes.empty else 0,
        "singleton_groups": int((group_sizes == 1).sum()),
        "identical_feature_signatures_across_groups": 0,
        "conflicting_label_signatures_across_groups": 0,
        "target_is_function_of_group": False,
    }

    target_by_group = (
        frame.loc[present, [group_column, target]]
        .groupby(group_column, dropna=False)[target]
        .nunique(dropna=True)
    )
    summary["target_is_function_of_group"] = bool(
        len(target_by_group) >= 2
        and int(frame[target].nunique(dropna=True)) >= 2
        and bool(target_by_group.le(1).all())
    )

    if features:
        signatures = (
            frame[features + [group_column, target]]
            .groupby(features, dropna=False, sort=False)
            .agg(
                group_count=(group_column, lambda values: values.nunique(dropna=True)),
                target_count=(target, lambda values: values.nunique(dropna=True)),
            )
        )
        cross_groups = signatures["group_count"] > 1
        summary["identical_feature_signatures_across_groups"] = int(cross_groups.sum())
        summary["conflicting_label_signatures_across_groups"] = int(
            (cross_groups & (signatures["target_count"] > 1)).sum()
        )
    return summary


def classification_fold_audit(
    frame: pd.DataFrame,
    *,
    fold_column: str,
    target: str,
    group_candidates: list[str],
) -> dict[str, Any]:
    """Audit supplied split metadata without admitting it into model features."""
    expected_classes = {
        str(label) for label in frame[target].dropna().drop_duplicates().tolist()
    }
    partitions: list[dict[str, Any]] = []
    missing_class_partitions: list[str] = []
    low_support_partitions: list[str] = []
    for value, partition in frame.groupby(fold_column, dropna=False, sort=False):
        counts = partition[target].dropna().value_counts()
        observed = {str(label) for label in counts.index.tolist()}
        missing_classes = sorted(expected_classes.difference(observed))
        label = "(missing)" if pd.isna(value) else str(value)
        if missing_classes:
            missing_class_partitions.append(label)
        if not counts.empty and int(counts.min()) < 2:
            low_support_partitions.append(label)
        partitions.append(
            {
                "value": label,
                "rows": len(partition),
                "classes": len(observed),
                "class_counts": {
                    str(class_label): int(count)
                    for class_label, count in counts.items()
                },
                "missing_classes": missing_classes,
            }
        )

    crossing: dict[str, dict[str, Any]] = {}
    for group_column in group_candidates:
        if group_column == fold_column:
            continue
        usable = frame[[group_column, fold_column]].dropna()
        if usable.empty:
            continue
        folds_per_group = usable.groupby(group_column, dropna=False)[
            fold_column
        ].nunique(dropna=True)
        crossing_groups = folds_per_group[folds_per_group > 1]
        crossing[group_column] = {
            "groups_observed": int(folds_per_group.size),
            "groups_crossing_partitions": int(crossing_groups.size),
            "examples": [str(value) for value in crossing_groups.index[:5]],
        }

    return {
        "column": fold_column,
        "partitions": partitions,
        "missing_class_partitions": missing_class_partitions,
        "low_support_partitions": low_support_partitions,
        "group_overlap": crossing,
    }


def add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    columns: list[str] | None = None,
) -> None:
    finding: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if columns:
        finding["columns"] = columns
    findings.append(finding)


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    path = args.dataset.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")

    frame = read_table(path, args.max_rows)
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names are not supported: {duplicates}")
    if args.target not in frame.columns:
        raise ValueError(f"Target column not found: {args.target}")

    referenced = [*args.exclude]
    if args.group_column:
        referenced.append(args.group_column)
    if args.time_column:
        referenced.append(args.time_column)
    missing_references = sorted(set(referenced).difference(frame.columns))
    if missing_references:
        raise ValueError(f"Referenced columns not found: {missing_references}")

    target = frame[args.target]
    task = infer_task(target) if args.task == "auto" else args.task
    findings: list[dict[str, Any]] = []

    explicit_exclusions = set(args.exclude)
    role_exclusions = {args.target}
    if args.group_column:
        role_exclusions.add(args.group_column)
    if args.time_column:
        role_exclusions.add(args.time_column)

    pipeline_columns = named_pipeline_metadata(
        list(frame.columns),
        target=args.target,
        group_column=args.group_column,
        time_column=args.time_column,
    )
    identifier_columns = [
        column
        for column in frame.columns
        if column not in role_exclusions
        and column not in explicit_exclusions
        and column not in pipeline_columns
        and likely_identifier(column, frame[column], len(frame))
    ]
    excluded = (
        explicit_exclusions
        | role_exclusions
        | set(pipeline_columns)
        | set(identifier_columns)
    )
    features = [column for column in frame.columns if column not in excluded]

    if pipeline_columns:
        add_finding(
            findings,
            "warning",
            "pipeline_metadata_excluded",
            "Likely pipeline metadata was excluded from modeling and duplicate comparison",
            pipeline_columns,
        )
    if identifier_columns:
        add_finding(
            findings,
            "warning",
            "identifier_features_excluded",
            "Likely identifiers were excluded from modeling and duplicate comparison",
            identifier_columns,
        )
    if not features:
        add_finding(
            findings, "blocker", "no_features", "No eligible feature columns remain"
        )

    target_non_null = target.dropna()
    target_unique = int(target_non_null.nunique())
    missing_target_rows = int(target.isna().sum())
    if missing_target_rows:
        add_finding(
            findings,
            "blocker",
            "missing_target",
            f"{missing_target_rows} rows have a missing target",
            [args.target],
        )
    if task == "classification" and target_unique < 2:
        add_finding(
            findings,
            "blocker",
            "invalid_class_target",
            "Classification requires at least two observed target classes",
            [args.target],
        )
    if task == "regression" and not pd.api.types.is_numeric_dtype(target):
        add_finding(
            findings,
            "blocker",
            "nonnumeric_regression_target",
            "Regression requires a numeric target",
            [args.target],
        )

    if task == "classification":
        class_counts = target_non_null.value_counts(dropna=False)
        if not class_counts.empty and int(class_counts.min()) < 2:
            add_finding(
                findings,
                "blocker",
                "singleton_class",
                "At least one class has fewer than two rows, preventing reliable splitting",
                [args.target],
            )
        elif not class_counts.empty and int(class_counts.min()) < 10:
            add_finding(
                findings,
                "warning",
                "small_class",
                "At least one class has fewer than ten rows; use repeated or carefully constrained validation",
                [args.target],
            )

    exact_duplicate_rows = int(frame.duplicated().sum())
    comparison_columns = features + [args.target]
    substantive_duplicate_rows = (
        int(frame.duplicated(subset=comparison_columns).sum())
        if comparison_columns
        else exact_duplicate_rows
    )
    excluded_comparison_columns = [
        column for column in frame.columns if column not in comparison_columns
    ]
    if substantive_duplicate_rows:
        add_finding(
            findings,
            "warning",
            "substantive_duplicate_observations",
            f"Found {substantive_duplicate_rows} repeated observations across eligible substantive features and the target",
            comparison_columns,
        )

    conflicting_label_signatures = 0
    if task == "classification" and features:
        target_counts = (
            frame[features + [args.target]]
            .groupby(features, dropna=False, sort=False)[args.target]
            .nunique(dropna=True)
        )
        conflicting_label_signatures = int((target_counts > 1).sum())
        if conflicting_label_signatures:
            add_finding(
                findings,
                "blocker",
                "conflicting_labels_for_identical_features",
                f"{conflicting_label_signatures} eligible feature signatures map to multiple labels",
                features,
            )

    constant_columns: list[str] = []
    high_missing_columns: list[str] = []
    suspicious_name_columns: list[str] = []
    direct_target_copies: list[str] = []
    near_perfect_numeric: list[str] = []
    target_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", args.target.lower())
        if len(token) >= 3
    }
    leakage_terms = {"target", "label", "outcome", "result", "final", "actual"}

    for column in features:
        series = frame[column]
        non_null = series.dropna()
        unique = int(non_null.nunique())
        if unique <= 1:
            constant_columns.append(column)
        if float(series.isna().mean()) >= 0.5:
            high_missing_columns.append(column)

        name_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", column.lower())
            if len(token) >= 3
        }
        if name_tokens.intersection(target_tokens | leakage_terms):
            suspicious_name_columns.append(column)

        comparable = frame[[column, args.target]].dropna()
        if not comparable.empty and comparable[column].astype(str).equals(
            comparable[args.target].astype(str)
        ):
            direct_target_copies.append(column)
        if (
            len(comparable) >= 3
            and pd.api.types.is_numeric_dtype(comparable[column])
            and pd.api.types.is_numeric_dtype(comparable[args.target])
        ):
            correlation = comparable[column].corr(comparable[args.target])
            if pd.notna(correlation) and abs(float(correlation)) >= 0.995:
                near_perfect_numeric.append(column)

    if constant_columns:
        add_finding(
            findings,
            "warning",
            "constant_features",
            "Constant features carry no predictive signal",
            constant_columns,
        )
    if high_missing_columns:
        add_finding(
            findings,
            "warning",
            "high_missingness",
            "Features with at least 50% missing values need an explicit retention decision",
            high_missing_columns,
        )
    if suspicious_name_columns:
        add_finding(
            findings,
            "warning",
            "leakage_name_signal",
            "Feature names suggest possible target or post-outcome information",
            suspicious_name_columns,
        )
    if direct_target_copies:
        add_finding(
            findings,
            "blocker",
            "direct_target_copy",
            "One or more eligible features directly reproduce the target",
            direct_target_copies,
        )
    if near_perfect_numeric:
        add_finding(
            findings,
            "warning",
            "near_perfect_target_association",
            "Near-perfect numeric association with the target requires a prediction-time availability review",
            near_perfect_numeric,
        )

    group_summary: dict[str, Any] | None = None
    if args.group_column:
        group_summary = group_signature_audit(
            frame,
            features=features,
            target=args.target,
            group_column=args.group_column,
        )
        unique_groups = int(group_summary["unique_groups"])
        if unique_groups < 2:
            add_finding(
                findings,
                "blocker",
                "insufficient_groups",
                "Group-aware validation requires at least two groups",
                [args.group_column],
            )
        if int(group_summary["missing_rows"]):
            add_finding(
                findings,
                "blocker",
                "missing_group_values",
                "Group-aware validation cannot safely assign rows with missing groups",
                [args.group_column],
            )
        if unique_groups and int(group_summary["singleton_groups"]) == unique_groups:
            add_finding(
                findings,
                "warning",
                "singleton_only_groups",
                "Every group contains one row, so group-aware validation adds no protection",
                [args.group_column],
            )
        if int(group_summary["identical_feature_signatures_across_groups"]):
            add_finding(
                findings,
                "warning",
                "feature_signatures_cross_groups",
                "Identical eligible feature vectors occur in multiple groups; review group construction before splitting",
                [args.group_column],
            )
        if int(group_summary["conflicting_label_signatures_across_groups"]):
            add_finding(
                findings,
                "blocker",
                "conflicting_labels_cross_groups",
                "Identical eligible feature vectors cross groups with conflicting labels",
                [args.group_column],
            )
        if bool(group_summary["target_is_function_of_group"]):
            add_finding(
                findings,
                "warning",
                "target_dependent_group_assignment",
                "Each observed group maps to only one target class; confirm the group was not derived from the label",
                [args.group_column],
            )

    time_summary: dict[str, Any] | None = None
    if args.time_column:
        parsed = pd.to_datetime(frame[args.time_column], errors="coerce", utc=True)
        invalid_rows = int(parsed.isna().sum())
        time_summary = {
            "column": args.time_column,
            "invalid_or_missing_rows": invalid_rows,
            "minimum": scalar(parsed.min()),
            "maximum": scalar(parsed.max()),
            "monotonic_in_source_order": bool(parsed.is_monotonic_increasing),
        }
        if invalid_rows:
            add_finding(
                findings,
                "blocker",
                "invalid_time_values",
                f"{invalid_rows} time values are missing or cannot be parsed",
                [args.time_column],
            )

    fold_columns = [
        column
        for column in frame.columns
        if column != args.target and FOLD_METADATA_RE.search(str(column))
    ]
    discovered_groups = [
        column
        for column in frame.columns
        if column != args.target and GROUP_METADATA_RE.search(str(column))
    ]
    group_candidates = list(
        dict.fromkeys(
            ([args.group_column] if args.group_column else []) + discovered_groups
        )
    )
    fold_audits: list[dict[str, Any]] = []
    if task == "classification":
        for fold_column in fold_columns:
            audit = classification_fold_audit(
                frame,
                fold_column=fold_column,
                target=args.target,
                group_candidates=group_candidates,
            )
            fold_audits.append(audit)
            if audit["missing_class_partitions"]:
                add_finding(
                    findings,
                    "blocker",
                    "fold_missing_classes",
                    f"{fold_column} has partitions missing target classes",
                    [fold_column],
                )
            elif audit["low_support_partitions"]:
                add_finding(
                    findings,
                    "warning",
                    "fold_low_class_support",
                    f"{fold_column} has partitions with fewer than two rows in a class",
                    [fold_column],
                )
            crossing_columns = [
                column
                for column, overlap in audit["group_overlap"].items()
                if overlap["groups_crossing_partitions"] > 0
            ]
            if crossing_columns:
                add_finding(
                    findings,
                    "blocker",
                    "group_ids_cross_folds",
                    f"Group identifiers cross partitions in {fold_column}",
                    [fold_column, *crossing_columns],
                )

    file_bytes = path.stat().st_size
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    available_bytes = available_memory_bytes()
    projected_working_bytes = max(frame_bytes * 4, file_bytes * 4)
    if available_bytes is not None and projected_working_bytes > available_bytes * 0.7:
        add_finding(
            findings,
            "warning",
            "memory_pressure",
            "Projected in-memory modeling work may exceed 70% of currently available memory",
        )

    blockers = sum(finding["severity"] == "blocker" for finding in findings)
    warnings = sum(finding["severity"] == "warning" for finding in findings)
    status = "blocked" if blockers else ("review_required" if warnings else "passed")

    return {
        "source": {
            "path": str(path),
            "fingerprint": f"sha256:{file_sha256(path)}",
            "file_bytes": file_bytes,
            "inspection_row_cap": args.max_rows,
            "full_dataset_inspected": args.max_rows is None,
        },
        "problem": {
            "task": task,
            "target": args.target,
            "row_grain": args.row_grain,
            "prediction_moment": args.prediction_moment,
            "feature_contract": {
                "included": features,
                "excluded": sorted(excluded),
                "explicitly_excluded": sorted(explicit_exclusions),
                "auto_excluded_identifiers": sorted(identifier_columns),
                "auto_excluded_pipeline_metadata": sorted(pipeline_columns),
            },
        },
        "dataset": {
            "rows_inspected": len(frame),
            "columns": len(frame.columns),
            "duplicate_rows": exact_duplicate_rows,
            "duplicates": {
                "exact_rows": exact_duplicate_rows,
                "substantive_rows": substantive_duplicate_rows,
                "comparison_columns": comparison_columns,
                "excluded_comparison_columns": excluded_comparison_columns,
                "conflicting_label_signatures": conflicting_label_signatures,
            },
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "missing_rows_by_column": {
                column: int(count)
                for column, count in frame.isna().sum().items()
                if int(count) > 0
            },
        },
        "target": {
            "observed_rows": len(target_non_null),
            "missing_rows": missing_target_rows,
            "unique_values": target_unique,
            "class_counts": (
                {
                    str(label): int(count)
                    for label, count in target_non_null.value_counts().items()
                }
                if task == "classification"
                else None
            ),
        },
        "split_context": {
            "group": group_summary,
            "time": time_summary,
            "fold_metadata": fold_audits,
            "discovered_group_metadata": discovered_groups,
            "requires_group_aware_split": args.group_column is not None,
            "requires_temporal_split": args.time_column is not None,
        },
        "memory": {
            "frame_bytes": frame_bytes,
            "projected_working_bytes": projected_working_bytes,
            "available_bytes": available_bytes,
        },
        "modeling_preflight": {
            "status": status,
            "blocker_count": blockers,
            "warning_count": warnings,
            "findings": findings,
            "human_review_required": [
                "Confirm the target matches the intended prediction",
                "Confirm every included feature exists at the stated prediction moment",
                "Confirm the row grain and split design prevent entity or temporal leakage",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        payload = inspect(parse_args(argv))
    except (OSError, ValueError, ImportError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
