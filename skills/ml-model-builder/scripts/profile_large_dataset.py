#!/usr/bin/env python3
"""Profile CSV/Parquet datasets with DuckDB and bounded local memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import profile_dataset as common

NUMERIC_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
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
    )
    parser.add_argument(
        "--group-overlap-policy",
        choices=[
            "disallow",
            "known_series_temporal",
            "known_entity_temporal",
        ],
        default="disallow",
    )
    parser.add_argument("--max-plot-rows", type=int, default=10_000)
    parser.add_argument("--max-numeric-plots", type=int, default=12)
    parser.add_argument("--max-categorical-plots", type=int, default=12)
    parser.add_argument("--max-panel-series", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--risk-tier",
        choices=["not_assessed", "standard", "high"],
        default=None,
    )
    parser.add_argument("--max-in-memory-bytes", type=int, default=0)
    parser.add_argument("--expected-source-bytes", type=int, default=0)
    parser.add_argument("--expected-source-rows", type=int, default=0)
    parser.add_argument(
        "--remote-source-version",
        help=(
            "Caller-declared version, ETag, snapshot, or table-version "
            "identifier; recorded as unverified by the generic profiler"
        ),
    )
    parser.add_argument(
        "--allow-unknown-remote-preflight",
        action="store_true",
    )
    parser.add_argument("--duckdb-memory-limit", default="4GB")
    parser.add_argument("--duckdb-temp-directory")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def scan_expression(location: str) -> str:
    parsed = urlparse(location)
    path_part = parsed.path if parsed.scheme else location
    lower = path_part.lower()
    literal = quote_literal(location)
    if lower.endswith((".parquet", ".pq")):
        return f"read_parquet({literal}, union_by_name=true)"
    if lower.endswith((".csv", ".csv.gz", ".csv.zst")):
        return (
            f"read_csv_auto({literal}, header=true, sample_size=100000, "
            "union_by_name=true)"
        )
    raise SystemExit("Input must be a CSV or Parquet file/URL.")


def duckdb_semantic_type(data_type: str, average_length: float | None) -> str:
    upper = data_type.upper()
    if any(upper.startswith(prefix) for prefix in NUMERIC_TYPES):
        return "numeric"
    if upper == "BOOLEAN":
        return "boolean"
    if any(token in upper for token in ["DATE", "TIME", "TIMESTAMP"]):
        return "datetime"
    if average_length is not None and average_length > 80:
        return "free_text"
    return "categorical"


def configure_connection(duckdb, args, output_dir: Path):
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit = {quote_literal(args.duckdb_memory_limit)}")
    connection.execute("SET preserve_insertion_order = true")
    if args.threads > 0:
        connection.execute(f"SET threads = {int(args.threads)}")
    temp_directory = Path(
        args.duckdb_temp_directory or output_dir / ".duckdb_tmp"
    ).resolve()
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory = {quote_literal(str(temp_directory))}")
    parsed = urlparse(args.input)
    local_path = (
        Path(parsed.path if parsed.scheme == "file" else args.input).expanduser()
        if not parsed.scheme or parsed.scheme == "file"
        else None
    )
    source_bytes = args.expected_source_bytes
    if not source_bytes and local_path is not None and local_path.is_file():
        source_bytes = local_path.stat().st_size
    lower = (parsed.path if parsed.scheme else args.input).lower()
    multiplier = 3.0 if lower.endswith((".csv", ".csv.gz", ".csv.zst")) else 1.5
    required_free = int(source_bytes * multiplier) if source_bytes else None
    free_disk = shutil.disk_usage(temp_directory).free
    if required_free and free_disk < required_free:
        raise SystemExit(
            f"DuckDB working disk has {free_disk:,} free bytes, below the "
            f"conservative {required_free:,}-byte preflight requirement. "
            "Choose a larger temporary disk or execute near the data in a "
            "warehouse, lakehouse, cluster, or cloud VM."
        )
    return connection, temp_directory, source_bytes, required_free, free_disk


def build_profiles(
    connection,
    columns,
    rows: int,
    sample,
    findings,
    blind_columns: set[str] | None = None,
    relation: str = "structural",
):
    expressions = ["count(*)::BIGINT AS row_count"]
    numeric_indexes = set()
    blind_columns = blind_columns or set()
    for index, (column, data_type) in enumerate(columns):
        if column in blind_columns:
            continue
        quoted = quote_identifier(column)
        upper = data_type.upper()
        numeric = any(upper.startswith(prefix) for prefix in NUMERIC_TYPES)
        if numeric:
            numeric_indexes.add(index)
            missing = f"({quoted} IS NULL OR isnan(try_cast({quoted} AS DOUBLE)))"
        else:
            missing = f"({quoted} IS NULL)"
        expressions.extend(
            [
                f"count(*) FILTER (WHERE {missing})::BIGINT AS c{index}_missing",
                (
                    "approx_count_distinct(cast("
                    f"{quoted} AS VARCHAR))::BIGINT AS c{index}_unique"
                ),
            ]
        )
        if numeric:
            numeric_value = f"try_cast({quoted} AS DOUBLE)"
            expressions.extend(
                [
                    f"avg({numeric_value}) AS c{index}_mean",
                    f"stddev_samp({numeric_value}) AS c{index}_std",
                    (
                        f"avg(CASE WHEN {numeric_value} = 0 THEN 1.0 ELSE 0.0 END) "
                        f"AS c{index}_zero"
                    ),
                    (
                        "approx_quantile("
                        f"{numeric_value}, [0.0,0.01,0.25,0.5,0.75,0.99,1.0]"
                        f") AS c{index}_quantiles"
                    ),
                ]
            )
    aggregate = connection.execute(
        "SELECT " + ", ".join(expressions) + f" FROM {relation}"
    ).fetchone()
    names = [item[0] for item in connection.description]
    values = dict(zip(names, aggregate))

    profiles = {}
    for index, (column, data_type) in enumerate(columns):
        if column in blind_columns:
            inferred = duckdb_semantic_type(data_type, None)
            profiles[column] = {
                "dtype": data_type,
                "semantic_type": inferred
                if inferred in {"numeric", "boolean", "datetime"}
                else "not_assessed",
                "missing_count": None,
                "missing_fraction": None,
                "unique_count": None,
                "unique_fraction": None,
                "identifier_like": bool(common.ID_NAME_RE.search(column)),
                "values_inspected": False,
                "profile_status": "target_blind_in_model_mode",
            }
            continue
        missing = int(values[f"c{index}_missing"])
        unique = int(values[f"c{index}_unique"])
        sample_values = (
            sample[column].dropna().astype(str) if column in sample.columns else []
        )
        average_length = (
            float(sample_values.str.len().mean())
            if hasattr(sample_values, "empty") and not sample_values.empty
            else None
        )
        kind = duckdb_semantic_type(data_type, average_length)
        id_like = bool(common.ID_NAME_RE.search(column)) or (
            rows >= 20 and unique / max(rows, 1) >= 0.98
        )
        item = {
            "dtype": data_type,
            "semantic_type": kind,
            "missing_count": missing,
            "missing_fraction": missing / max(rows, 1),
            "unique_count": unique,
            "unique_fraction": unique / max(rows - missing, 1),
            "identifier_like": id_like,
            "cardinality_estimate": "approx_count_distinct",
            "values_inspected": True,
        }
        if index in numeric_indexes:
            quantiles = values.get(f"c{index}_quantiles") or []
            keys = ["0.0", "0.01", "0.25", "0.5", "0.75", "0.99", "1.0"]
            item["numeric"] = {
                "mean": common.finite_number(values.get(f"c{index}_mean")),
                "std": common.finite_number(values.get(f"c{index}_std")),
                "zero_fraction": common.finite_number(values.get(f"c{index}_zero")),
                "quantiles": {
                    key: common.finite_number(value)
                    for key, value in zip(keys, quantiles)
                },
                "quantile_method": "DuckDB approx_quantile",
            }
        profiles[column] = item
        if missing / max(rows, 1) > 0.5:
            common.add_finding(
                findings,
                "warning",
                "high_missingness",
                f"Column '{column}' is {missing / max(rows, 1):.1%} missing.",
                "Determine the missingness mechanism before dropping or imputing.",
                column,
            )
        if unique <= 1:
            common.add_finding(
                findings,
                "warning",
                "constant_column",
                f"Column '{column}' has no observed variation.",
                "Confirm whether it is required by the inference contract.",
                column,
            )
        elif id_like:
            common.add_finding(
                findings,
                "information",
                "identifier_like",
                f"Column '{column}' appears identifier-like.",
                "Retain for joins/output if needed; do not model it blindly.",
                column,
            )
        if kind == "categorical" and unique > 100:
            common.add_finding(
                findings,
                "warning",
                "high_cardinality",
                f"Column '{column}' has approximately {unique} categories.",
                "Use a high-cardinality-safe representation and unseen-value policy.",
                column,
            )
    return profiles


def plot_full_missingness(plt, profiles, output, figures):
    missing = sorted(
        (
            (info["missing_fraction"], column)
            for column, info in profiles.items()
            if info["missing_fraction"] is not None and info["missing_fraction"] > 0
        ),
        reverse=True,
    )[:30]
    if not missing:
        return
    fractions = [item[0] for item in missing][::-1]
    labels = [item[1] for item in missing][::-1]
    figure, axis = plt.subplots(figsize=(10, max(3, len(labels) * 0.28)))
    axis.barh(labels, fractions, color="#4472C4")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Missing fraction")
    axis.set_title("Columns with missing values (full analysis population)")
    common.save_figure(
        plt,
        figure,
        output / "missingness.png",
        figures,
        "Full-population missing-value fractions for up to 30 columns.",
    )


def target_profile(
    connection,
    pd,
    plt,
    args,
    profiles,
    output,
    figures,
    findings,
    plot_sample,
    relation="target_analysis",
):
    if not args.target:
        return None
    quoted = quote_identifier(args.target)
    non_missing = connection.execute(
        f"SELECT count(*) FROM {relation} WHERE {quoted} IS NOT NULL"
    ).fetchone()[0]
    if not non_missing:
        common.add_finding(
            findings,
            "blocker" if args.mode == "model" else "warning",
            "empty_target",
            "The target has no non-missing values in the analysis population.",
            "Fix target generation or partitioning before modeling.",
            args.target,
        )
        return {"non_missing_count": 0}
    if args.task in {"classification", "anomaly"}:
        grouped = (
            f"SELECT cast({quoted} AS VARCHAR) AS label, count(*)::BIGINT AS rows "
            f"FROM {relation} WHERE {quoted} IS NOT NULL GROUP BY 1"
        )
        class_count, total, minority_rows = connection.execute(
            "SELECT count(*)::BIGINT, sum(rows)::BIGINT, min(rows)::BIGINT "
            f"FROM ({grouped}) AS class_counts"
        ).fetchone()
        counts = connection.execute(
            f"{grouped} ORDER BY rows DESC, label LIMIT 30"
        ).fetchall()
        plotted = counts
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar([str(row[0])[:30] for row in plotted], [row[1] for row in plotted])
        axis.set_title("Target support (target-aware population)")
        axis.set_ylabel("Rows")
        axis.tick_params(axis="x", rotation=45)
        common.save_figure(
            plt,
            figure,
            output / "target_distribution.png",
            figures,
            "Exact counts for up to the 30 largest classes in the permitted "
            "target-aware population.",
        )
        total = int(total)
        class_count = int(class_count)
        minority = int(minority_rows) / max(total, 1)
        if minority < 0.1:
            common.add_finding(
                findings,
                "warning",
                "class_imbalance",
                f"The smallest class represents {minority:.1%} of labeled rows.",
                "Verify event counts in every deployment-matched fold.",
                args.target,
            )
        if class_count < 2:
            common.add_finding(
                findings,
                "blocker" if args.mode == "model" else "warning",
                "single_class_target",
                "The target has fewer than two classes.",
                "Fix the target or evaluation population before classification.",
                args.target,
            )
        return {
            "non_missing_count": int(total),
            "class_count": class_count,
            "class_support": [
                {"label": str(label), "rows": int(rows)} for label, rows in counts
            ],
            "class_support_limit": 30,
            "class_support_truncated": class_count > len(counts),
        }
    elif profiles[args.target]["semantic_type"] == "numeric":
        values = pd.to_numeric(plot_sample[args.target], errors="coerce").dropna()
        if not values.empty:
            figure, axis = plt.subplots(figsize=(8, 4))
            axis.hist(values, bins=30, color="#ED7D31")
            axis.set_title("Target distribution (deterministic plot sample)")
            axis.set_xlabel(args.target)
            common.save_figure(
                plt,
                figure,
                output / "target_distribution.png",
                figures,
                "Target histogram from the recorded deterministic plot sample.",
            )
    return {
        "non_missing_count": int(non_missing),
        "numeric": profiles[args.target].get("numeric"),
    }


def _isoformat_or_none(value):
    return value.isoformat() if value is not None else None


def plot_time_coverage(
    connection,
    plt,
    args,
    output,
    figures,
    findings,
    target_relation: str | None,
):
    if not args.time_column:
        return None
    quoted_time = quote_identifier(args.time_column)
    timestamp = f"try_cast({quoted_time} AS TIMESTAMP)"
    invalid = connection.execute(
        f"SELECT count(*) FROM structural WHERE {quoted_time} IS NOT NULL "
        f"AND {timestamp} IS NULL"
    ).fetchone()[0]
    if invalid:
        common.add_finding(
            findings,
            "warning",
            "invalid_timestamps",
            f"{invalid:,} non-missing values in '{args.time_column}' could not "
            "be parsed.",
            "Resolve formats and timezones before temporal splitting.",
            args.time_column,
        )
    daily = connection.execute(
        f"SELECT date_trunc('day', {timestamp}) AS day, count(*)::BIGINT AS rows "
        f"FROM structural WHERE {timestamp} IS NOT NULL GROUP BY 1 ORDER BY 1"
    ).fetchdf()
    if not daily.empty:
        figure, axis = plt.subplots(figsize=(11, 3.5))
        axis.plot(daily["day"], daily["rows"], color="#70AD47")
        axis.set_title("Row coverage over time (full analysis population)")
        axis.set_xlabel("Time")
        axis.set_ylabel("Rows per day")
        common.save_figure(
            plt,
            figure,
            output / "time_coverage.png",
            figures,
            "Exact daily row coverage from the full permitted analysis population.",
        )
    panel_coverage = None
    if args.group_column:
        quoted_group = quote_identifier(args.group_column)
        series_sql = (
            "SELECT cast("
            f"{quoted_group} AS VARCHAR) AS series, count(*)::BIGINT AS rows, "
            f"count({timestamp})::BIGINT AS parseable_time_rows, "
            f"min({timestamp}) AS start_time, max({timestamp}) AS end_time "
            f"FROM structural WHERE {quoted_group} IS NOT NULL GROUP BY 1"
        )
        summary = connection.execute(
            "SELECT count(*)::BIGINT AS series_count, "
            "min(rows)::BIGINT AS minimum_rows, "
            "approx_quantile(rows, 0.5) AS median_rows, "
            "max(rows)::BIGINT AS maximum_rows, min(start_time) AS start_time, "
            f"max(end_time) AS end_time FROM ({series_sql})"
        ).fetchone()
        null_groups = int(
            connection.execute(
                f"SELECT count(*) FROM structural WHERE {quoted_group} IS NULL"
            ).fetchone()[0]
        )
        series_count = int(summary[0] or 0)
        representative = connection.execute(
            f"WITH series AS ({series_sql}), bucketed AS ("
            "SELECT *, ntile("
            f"{max(int(args.max_panel_series), 1)}) OVER "
            "(ORDER BY rows, series) AS coverage_bucket FROM series"
            "), ranked AS ("
            "SELECT *, row_number() OVER (PARTITION BY coverage_bucket "
            "ORDER BY hash(series)) AS bucket_rank FROM bucketed"
            ") SELECT series, rows, parseable_time_rows, start_time, end_time "
            "FROM ranked WHERE bucket_rank = 1 ORDER BY rows, series"
        ).fetchall()
        representatives = [
            {
                "label": str(label),
                "rows": int(rows),
                "parseable_time_rows": int(parseable_rows),
                "start": _isoformat_or_none(start),
                "end": _isoformat_or_none(end),
            }
            for label, rows, parseable_rows, start, end in representative
        ]
        panel_coverage = common.panel_coverage_contract(
            args.group_column,
            series_count,
            null_groups,
            {
                "minimum": int(summary[1]) if summary[1] is not None else None,
                "median": common.finite_number(summary[2]),
                "maximum": int(summary[3]) if summary[3] is not None else None,
            },
            _isoformat_or_none(summary[4]),
            _isoformat_or_none(summary[5]),
            representatives,
            args.max_panel_series,
            "deterministic one-series sample from ranked row-coverage buckets",
        )
        if series_count > len(representative):
            common.add_finding(
                findings,
                "information",
                "panel_series_plot_limited",
                f"The panel contains {series_count:,} series; the chart renders "
                f"{len(representative):,} representative series.",
                "Use panel_coverage in data_profile.json for the bounded summary "
                "or adjust --max-panel-series.",
                args.group_column,
            )
    if args.target and target_relation and args.group_column and panel_coverage:
        selected = [item["label"] for item in panel_coverage["representative_series"]]
        if selected:
            quoted_target = quote_identifier(args.target)
            values = ", ".join(quote_literal(value) for value in selected)
            per_series_limit = max(
                2, max(int(args.max_plot_rows), 1) // max(len(selected), 1)
            )
            panel_points = connection.execute(
                "WITH points AS (SELECT "
                f"{timestamp} AS time, cast({quote_identifier(args.group_column)} "
                "AS VARCHAR) AS series, "
                f"avg(try_cast({quoted_target} AS DOUBLE)) AS target "
                f"FROM {target_relation} WHERE "
                f"{timestamp} IS NOT NULL AND {quoted_target} IS NOT NULL "
                f"AND cast({quote_identifier(args.group_column)} AS VARCHAR) "
                f"IN ({values}) GROUP BY 1, 2), numbered AS ("
                "SELECT *, row_number() OVER (PARTITION BY series ORDER BY time) "
                "AS point_number, count(*) OVER (PARTITION BY series) AS "
                "series_points FROM points WHERE target IS NOT NULL"
                ") SELECT time, series, target FROM numbered WHERE "
                "point_number = 1 OR point_number = series_points OR "
                "((point_number - 1) % greatest(1, "
                f"ceil(series_points / {float(per_series_limit)})::BIGINT) = 0) "
                "ORDER BY series, time"
            ).fetchdf()
            common.plot_representative_panel(
                plt,
                panel_points,
                args.target,
                args.group_column,
                output,
                figures,
                args.max_plot_rows,
                "Bounded, deterministic panel sample; values are aggregated "
                "only within each series and timestamp.",
            )
        return panel_coverage
    if args.target and target_relation:
        quoted_target = quote_identifier(args.target)
        target_daily = connection.execute(
            f"SELECT date_trunc('day', {timestamp}) AS day, "
            f"avg(try_cast({quoted_target} AS DOUBLE)) AS target "
            f"FROM {target_relation} WHERE {timestamp} IS NOT NULL "
            "GROUP BY 1 ORDER BY 1"
        ).fetchdf()
        if target_daily["target"].notna().any():
            figure, axis = plt.subplots(figsize=(11, 4))
            axis.plot(target_daily["day"], target_daily["target"], color="#4472C4")
            axis.set_title("Target over time (full analysis population)")
            axis.set_xlabel("Time")
            axis.set_ylabel(args.target)
            common.save_figure(
                plt,
                figure,
                output / "target_over_time.png",
                figures,
                "Full-population target mean by day; model mode includes only "
                "the selected development partition.",
            )
    return panel_coverage


def approximate_duplicates(duckdb, connection, columns, rows: int) -> int | None:
    if not columns or rows == 0:
        return 0
    arguments = ", ".join(quote_identifier(column) for column, _ in columns)
    try:
        distinct = connection.execute(
            f"SELECT approx_count_distinct(hash({arguments})) FROM raw_data"
        ).fetchone()[0]
    except duckdb.Error:
        return None
    return max(0, rows - int(distinct))


def assignment_fingerprint(connection, partition_column: str) -> str:
    digest = hashlib.sha256()
    cursor = connection.execute(
        f"SELECT {quote_identifier(partition_column)} FROM raw_data"
    )
    ordinal = 0
    while True:
        rows = cursor.fetchmany(10_000)
        if not rows:
            break
        for (value,) in rows:
            common.update_assignment_digest(digest, ordinal, value)
            ordinal += 1
    return digest.hexdigest()


def build_split_manifest_duckdb(connection, columns, args, generated_at: str):
    """Run exact, target-blind split audits with bounded result materialization."""
    partition = quote_identifier(args.partition_column)
    partition_rows = connection.execute(
        "SELECT coalesce(cast("
        f"{partition} AS VARCHAR), '<missing>') AS name, "
        "count(*)::BIGINT AS rows FROM raw_data GROUP BY 1 ORDER BY 1"
    ).fetchall()
    partitions = [
        {
            "name": str(name),
            "role": common._partition_role(args, str(name)),
            "rows": int(rows),
        }
        for name, rows in partition_rows
    ]
    missing_rows = int(
        connection.execute(
            f"SELECT count(*) FROM raw_data WHERE {partition} IS NULL"
        ).fetchone()[0]
    )
    warnings, blockers = common.initial_split_findings(args, partitions, missing_rows)

    if args.group_column:
        group = quote_identifier(args.group_column)
        unique_groups, null_groups = connection.execute(
            f"SELECT count(DISTINCT {group}), count(*) FILTER "
            f"(WHERE {group} IS NULL) FROM raw_data"
        ).fetchone()
        spanning = int(
            connection.execute(
                "SELECT count(*) FROM (SELECT "
                f"{group} FROM raw_data WHERE {group} IS NOT NULL GROUP BY "
                f"{group} HAVING count(DISTINCT coalesce(cast({partition} AS "
                "VARCHAR), '<missing>')) > 1)"
            ).fetchone()[0]
        )
        allowed = common.panel_group_overlap_allowed(args)
        group_audit = {
            "checked": True,
            "group_column": args.group_column,
            "unique_groups": int(unique_groups),
            "null_group_rows": int(null_groups),
            "groups_spanning_partitions": spanning,
            "allowed": allowed,
            "reason": (
                common.group_overlap_reason(args)
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

    if common.split_is_temporal(args):
        time_column = quote_identifier(args.time_column)
        timestamp = f"try_cast({time_column} AS TIMESTAMP)"
        invalid = int(
            connection.execute(
                f"SELECT count(*) FROM raw_data WHERE {timestamp} IS NULL"
            ).fetchone()[0]
        )
        range_rows = connection.execute(
            "SELECT coalesce(cast("
            f"{partition} AS VARCHAR), '<missing>') AS name, "
            "count(*)::BIGINT AS rows, "
            f"count({timestamp})::BIGINT AS parseable_time_rows, "
            f"min({timestamp}) AS start_time, max({timestamp}) AS end_time "
            "FROM raw_data GROUP BY 1 ORDER BY 1"
        ).fetchall()
        ranges = [
            {
                "name": str(name),
                "role": common._partition_role(args, str(name)),
                "rows": int(rows),
                "parseable_time_rows": int(parseable_rows),
                "start": _isoformat_or_none(start),
                "end": _isoformat_or_none(end),
            }
            for name, rows, parseable_rows, start, end in range_rows
        ]
        temporal_audit, temporal_warning = common.temporal_audit_from_ranges(
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

    return common.assemble_split_manifest(
        args,
        generated_at,
        partitions,
        assignment_fingerprint(connection, args.partition_column),
        (
            "source scan row ordinal plus a null marker or length-prefixed "
            "UTF-8 assignment value; hashed incrementally in 10,000-row batches"
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


def main() -> int:
    args = parse_args()
    preflight = common.validate_profiler_args(args)
    output_dir = Path(args.output_dir).resolve()
    existing_config = common.load_and_validate_existing_config(output_dir, args)
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit(
            "profile_large_dataset.py requires DuckDB: `python -m pip install duckdb`."
        ) from exc
    pd, plt = common.import_analysis_packages()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (
        connection,
        temp_directory,
        source_bytes,
        required_free,
        free_disk,
    ) = configure_connection(duckdb, args, output_dir)
    scan = scan_expression(args.input)
    connection.execute(f"CREATE VIEW raw_data AS SELECT * FROM {scan}")
    connection.execute("CREATE VIEW structural AS SELECT * FROM raw_data")
    raw_columns = [
        (row[0], row[1])
        for row in connection.execute("DESCRIBE SELECT * FROM raw_data").fetchall()
    ]
    column_names = [column for column, _ in raw_columns]
    if len(set(column_names)) != len(column_names):
        raise SystemExit("Input dataset contains duplicate column names.")
    for name, value in [
        ("target", args.target),
        ("time column", args.time_column),
        ("group column", args.group_column),
    ]:
        if value and value not in column_names:
            raise SystemExit(f"{name.title()} '{value}' does not exist.")

    nested_cv_target_blind = (
        args.mode == "model" and args.evaluation_design == "nested_cv"
    )
    structural_population_label = (
        "full permitted dataset (target-blind structural analysis)"
        if args.mode == "model"
        else "full permitted dataset (descriptive analysis)"
    )
    target_relation = "target_analysis" if args.target else None
    target_population_label = (
        "full permitted dataset (descriptive target-aware analysis)"
        if args.mode == "analysis-only" and args.target
        else None
    )
    if args.mode == "model":
        if args.partition_column not in column_names:
            raise SystemExit(
                "Model mode requires a persisted partition column before EDA. "
                f"Missing: '{args.partition_column}'."
            )
        if nested_cv_target_blind:
            target_relation = None
            target_population_label = None
        else:
            partition = quote_identifier(args.partition_column)
            label = quote_literal(args.train_label)
            connection.execute(
                "CREATE VIEW development AS SELECT * FROM raw_data "
                f"WHERE cast({partition} AS VARCHAR) = {label}"
            )
            development_rows = int(
                connection.execute("SELECT count(*) FROM development").fetchone()[0]
            )
            if development_rows == 0:
                raise SystemExit(
                    f"Partition column '{args.partition_column}' has no "
                    f"'{args.train_label}' rows."
                )
            if args.target:
                connection.execute(
                    "CREATE VIEW target_analysis AS SELECT * FROM development"
                )
                target_population_label = f"{args.partition_column}={args.train_label}"
    elif args.target:
        connection.execute("CREATE VIEW target_analysis AS SELECT * FROM raw_data")

    raw_rows = int(connection.execute("SELECT count(*) FROM raw_data").fetchone()[0])
    structural_rows = int(
        connection.execute("SELECT count(*) FROM structural").fetchone()[0]
    )
    if raw_rows == 0:
        raise SystemExit("The input dataset contains no rows.")
    target_rows = (
        int(connection.execute(f"SELECT count(*) FROM {target_relation}").fetchone()[0])
        if target_relation
        else 0
    )
    structural_plot_rows = min(structural_rows, max(args.max_plot_rows, 1))
    structural_sample_columns = [
        column
        for column in column_names
        if not (args.mode == "model" and column == args.target)
    ]
    structural_sample_projection = ", ".join(
        quote_identifier(column) for column in structural_sample_columns
    )
    structural_plot_sample = connection.execute(
        f"SELECT {structural_sample_projection} FROM structural USING SAMPLE "
        f"reservoir({structural_plot_rows} ROWS) REPEATABLE ({int(args.seed)})"
    ).fetchdf()
    target_plot_rows = min(target_rows, max(args.max_plot_rows, 1))
    target_plot_sample = (
        connection.execute(
            f"SELECT * FROM {target_relation} USING SAMPLE "
            f"reservoir({target_plot_rows} ROWS) REPEATABLE ({int(args.seed)})"
        ).fetchdf()
        if target_relation
        else None
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    split_manifest = (
        build_split_manifest_duckdb(connection, raw_columns, args, generated_at)
        if args.mode == "model"
        else None
    )
    findings: list[dict] = []
    common.add_contract_findings(findings, args, preflight, raw_rows, split_manifest)
    blind_columns = {args.target} if args.mode == "model" and args.target else set()
    profiles = build_profiles(
        connection,
        raw_columns,
        structural_rows,
        structural_plot_sample,
        findings,
        blind_columns=blind_columns,
        relation="structural",
    )
    target_profiles = None
    if target_relation and args.target:
        target_columns = [item for item in raw_columns if item[0] == args.target]
        target_profiles = build_profiles(
            connection,
            target_columns,
            target_rows,
            target_plot_sample,
            findings,
            relation=target_relation,
        )
    duplicate_columns = raw_columns
    duplicate_exclusions = []
    if args.mode == "model":
        duplicate_exclusions = list(
            dict.fromkeys(
                column
                for column in (args.target, args.partition_column)
                if column is not None
            )
        )
        excluded = set(duplicate_exclusions)
        duplicate_columns = [item for item in raw_columns if item[0] not in excluded]
    duplicate_estimate = approximate_duplicates(
        duckdb, connection, duplicate_columns, raw_rows
    )
    if duplicate_estimate:
        common.add_finding(
            findings,
            "warning",
            "possible_exact_duplicates",
            f"Hash/cardinality screening estimates about {duplicate_estimate:,} "
            "duplicate rows.",
            "Run exact key/full-row verification before deleting records.",
        )
    if duplicate_estimate is None:
        common.add_finding(
            findings,
            "information",
            "duplicate_screen_skipped",
            "The automatic whole-row duplicate screen could not be completed.",
            "Run an explicit DuckDB key/full-row duplicate audit.",
        )

    figures: list[dict] = []
    plot_full_missingness(plt, profiles, figures_dir, figures)
    common.plot_numeric(
        plt,
        structural_plot_sample,
        profiles,
        figures_dir,
        figures,
        args.max_numeric_plots,
    )
    common.plot_categorical(
        plt,
        structural_plot_sample,
        profiles,
        figures_dir,
        figures,
        args.max_categorical_plots,
    )
    common.plot_correlation(
        plt,
        structural_plot_sample,
        profiles,
        figures_dir,
        figures,
        sampled=True,
    )
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
        target_summary = target_profile(
            connection,
            pd,
            plt,
            args,
            target_profiles or profiles,
            figures_dir,
            figures,
            findings,
            target_plot_sample,
        )
    panel_coverage = plot_time_coverage(
        connection,
        plt,
        args,
        figures_dir,
        figures,
        findings,
        target_relation,
    )
    if target_plot_sample is not None:
        common.plot_feature_relationships(
            pd,
            plt,
            target_plot_sample,
            args.task,
            args.target,
            profiles,
            figures_dir,
            figures,
        )

    parsed = urlparse(args.input)
    if parsed.scheme and parsed.scheme != "file":
        source = common.remote_source_contract(args, preflight)
        source["bytes"] = source_bytes or None
    else:
        source = common.source_fingerprint(pd, args.input, structural_plot_sample)
        source["reproducibility_status"] = "reproducible_source"
    fingerprint = {
        "schema_version": common.SCHEMA_VERSION,
        "generated_at": generated_at,
        "input": source,
        "rows": raw_rows,
        "columns": len(raw_columns),
    }
    schema = common.build_schema_contract(
        profiles,
        args,
        structural_population_label,
        structural_rows,
        generated_at,
    )
    target_population = (
        {
            "label": target_population_label,
            "rows": target_rows,
            "partition": args.train_label if args.mode == "model" else None,
        }
        if target_relation and args.target
        else None
    )
    report = {
        "schema_version": common.SCHEMA_VERSION,
        "generated_at": generated_at,
        "mode": args.mode,
        "task": args.task,
        "engine": "duckdb",
        "shape": {"rows": raw_rows, "columns": len(raw_columns)},
        "analysis_population": {
            "label": structural_population_label,
            "rows": structural_rows,
        },
        "structural_population": {
            "label": structural_population_label,
            "rows": structural_rows,
            "target_values_inspected": not bool(blind_columns),
        },
        "target_aware_population": target_population,
        "plot_sampling": {
            "population_rows": structural_rows,
            "rows": structural_plot_rows,
            "method": "DuckDB deterministic reservoir sample",
            "seed": args.seed,
        },
        "target_plot_sampling": (
            {
                "population_rows": target_rows,
                "rows": target_plot_rows,
                "method": "DuckDB deterministic reservoir sample",
                "seed": args.seed,
            }
            if target_relation and args.target
            else None
        ),
        "aggregate_methods": {
            "row_and_missing_counts": "exact",
            "cardinality": "approx_count_distinct",
            "quantiles": "approx_quantile",
            "duplicate_rows": "hash plus approximate distinct screening",
            "correlation": "plot sample",
        },
        "duplicates_full_dataset_estimate": duplicate_estimate,
        "duplicate_screening": {
            "population": "full permitted dataset",
            "method": "hash plus approximate distinct screening",
            "columns": [column for column, _ in duplicate_columns],
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
        "resource_controls": {
            "memory_limit": args.duckdb_memory_limit,
            "threads": args.threads or "DuckDB default",
            "temp_directory": str(temp_directory),
            "source_bytes_for_preflight": source_bytes,
            "required_free_disk_bytes": required_free,
            "observed_free_disk_bytes": free_disk,
            "expected_source_rows": args.expected_source_rows or None,
            "remote_preflight": preflight,
        },
    }

    common.update_config(
        output_dir,
        args,
        structural_population_label,
        structural_rows,
        structural_plot_rows,
        target_population_label,
        target_rows,
        target_plot_rows,
        existing_config=existing_config,
        analysis_engine="duckdb",
        approximate_aggregates=[
            "cardinality",
            "quantiles",
            "duplicate screening",
        ],
    )
    for filename, document in [
        ("data_profile.json", report),
        ("data_fingerprint.json", fingerprint),
        ("schema.json", schema),
    ]:
        (output_dir / filename).write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )
    if split_manifest:
        (output_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output_dir / "data_summary.md").write_text(
        common.markdown_summary(report), encoding="utf-8"
    )
    (output_dir / "data_report.html").write_text(
        common.html_report(report), encoding="utf-8"
    )
    connection.close()
    try:
        temp_directory.rmdir()
    except OSError:
        pass

    blockers = sum(item["severity"] == "blocker" for item in findings)
    warnings = sum(item["severity"] == "warning" for item in findings)
    print(
        f"Wrote DuckDB dataset report to {output_dir} "
        f"({blockers} blocker(s), {warnings} warning(s))."
    )
    return 2 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
