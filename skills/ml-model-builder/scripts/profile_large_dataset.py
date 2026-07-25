#!/usr/bin/env python3
"""Profile CSV/Parquet datasets with DuckDB and bounded local memory."""

from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("--max-plot-rows", type=int, default=10_000)
    parser.add_argument("--max-numeric-plots", type=int, default=12)
    parser.add_argument("--max-categorical-plots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-in-memory-bytes", type=int, default=0)
    parser.add_argument("--expected-source-bytes", type=int, default=0)
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


def build_profiles(connection, columns, rows: int, sample, findings):
    expressions = ["count(*)::BIGINT AS row_count"]
    numeric_indexes = set()
    for index, (column, data_type) in enumerate(columns):
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
        "SELECT " + ", ".join(expressions) + " FROM analysis"
    ).fetchone()
    names = [item[0] for item in connection.description]
    values = dict(zip(names, aggregate))

    profiles = {}
    for index, (column, data_type) in enumerate(columns):
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
            if info["missing_fraction"] > 0
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
):
    if not args.target:
        return None
    quoted = quote_identifier(args.target)
    non_missing = connection.execute(
        f"SELECT count(*) FROM analysis WHERE {quoted} IS NOT NULL"
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
        counts = connection.execute(
            f"SELECT cast({quoted} AS VARCHAR) AS label, count(*)::BIGINT AS rows "
            f"FROM analysis WHERE {quoted} IS NOT NULL GROUP BY 1 "
            "ORDER BY rows DESC"
        ).fetchall()
        plotted = counts[:30]
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar([str(row[0])[:30] for row in plotted], [row[1] for row in plotted])
        axis.set_title("Target support (full analysis population)")
        axis.set_ylabel("Rows")
        axis.tick_params(axis="x", rotation=45)
        common.save_figure(
            plt,
            figure,
            output / "target_distribution.png",
            figures,
            "Exact class counts from the full permitted analysis population.",
        )
        total = sum(row[1] for row in counts)
        minority = min(row[1] for row in counts) / max(total, 1)
        if minority < 0.1:
            common.add_finding(
                findings,
                "warning",
                "class_imbalance",
                f"The smallest class represents {minority:.1%} of labeled rows.",
                "Verify event counts in every deployment-matched fold.",
                args.target,
            )
        if len(counts) < 2:
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
            "class_count": len(counts),
            "class_support": [
                {"label": str(label), "rows": int(rows)} for label, rows in counts
            ],
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


def plot_time_coverage(connection, plt, args, output, figures, findings):
    if not args.time_column:
        return
    quoted_time = quote_identifier(args.time_column)
    timestamp = f"try_cast({quoted_time} AS TIMESTAMP)"
    invalid = connection.execute(
        f"SELECT count(*) FROM analysis WHERE {quoted_time} IS NOT NULL "
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
        f"FROM analysis WHERE {timestamp} IS NOT NULL GROUP BY 1 ORDER BY 1"
    ).fetchdf()
    if daily.empty:
        return
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
    if args.target:
        quoted_target = quote_identifier(args.target)
        target_daily = connection.execute(
            f"SELECT date_trunc('day', {timestamp}) AS day, "
            f"avg(try_cast({quoted_target} AS DOUBLE)) AS target "
            f"FROM analysis WHERE {timestamp} IS NOT NULL GROUP BY 1 ORDER BY 1"
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


def main() -> int:
    args = parse_args()
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit(
            "profile_large_dataset.py requires DuckDB: `python -m pip install duckdb`."
        ) from exc
    pd, plt = common.import_analysis_packages()

    output_dir = Path(args.output_dir).resolve()
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

    if args.mode == "model":
        if args.partition_column not in column_names:
            raise SystemExit(
                "Model mode requires a persisted partition column before EDA. "
                f"Missing: '{args.partition_column}'."
            )
        partition = quote_identifier(args.partition_column)
        label = quote_literal(args.train_label)
        connection.execute(
            "CREATE VIEW analysis AS SELECT * FROM raw_data "
            f"WHERE cast({partition} AS VARCHAR) = {label}"
        )
        population_label = f"{args.partition_column}={args.train_label}"
    else:
        connection.execute("CREATE VIEW analysis AS SELECT * FROM raw_data")
        population_label = "full permitted dataset (descriptive analysis)"

    raw_rows = int(connection.execute("SELECT count(*) FROM raw_data").fetchone()[0])
    analysis_rows = int(
        connection.execute("SELECT count(*) FROM analysis").fetchone()[0]
    )
    if raw_rows == 0 or analysis_rows == 0:
        raise SystemExit("The input or selected analysis population contains no rows.")
    plot_rows = min(analysis_rows, max(args.max_plot_rows, 1))
    plot_sample = connection.execute(
        "SELECT * FROM analysis USING SAMPLE "
        f"reservoir({plot_rows} ROWS) REPEATABLE ({int(args.seed)})"
    ).fetchdf()

    findings: list[dict] = []
    profiles = build_profiles(
        connection, raw_columns, analysis_rows, plot_sample, findings
    )
    duplicate_estimate = approximate_duplicates(
        duckdb, connection, raw_columns, raw_rows
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
        plot_sample,
        profiles,
        figures_dir,
        figures,
        args.max_numeric_plots,
    )
    common.plot_categorical(
        plt,
        plot_sample,
        profiles,
        figures_dir,
        figures,
        args.max_categorical_plots,
    )
    common.plot_correlation(
        plt,
        plot_sample,
        profiles,
        figures_dir,
        figures,
        sampled=True,
    )
    target_summary = target_profile(
        connection,
        pd,
        plt,
        args,
        profiles,
        figures_dir,
        figures,
        findings,
        plot_sample,
    )
    plot_time_coverage(
        connection,
        plt,
        args,
        figures_dir,
        figures,
        findings,
    )
    common.plot_feature_relationships(
        pd,
        plt,
        plot_sample,
        args.task,
        args.target,
        profiles,
        figures_dir,
        figures,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    parsed = urlparse(args.input)
    if parsed.scheme and parsed.scheme != "file":
        source = {
            "kind": "remote_reference_unhashed",
            "sha256": None,
            "bytes": None,
            "source": re.sub(r"[?#].*$", "", args.input),
        }
        common.add_finding(
            findings,
            "warning",
            "remote_source_not_content_hashed",
            "The remote input could not be content-hashed without downloading it.",
            "Pin an immutable object version/ETag or materialize and hash the source.",
        )
        if not source_bytes:
            common.add_finding(
                findings,
                "warning",
                "remote_source_size_unknown",
                "Remote input size was not supplied, so temporary-disk safety "
                "could not be preflighted.",
                "Pass --expected-source-bytes or execute in the source platform.",
            )
    else:
        source = common.source_fingerprint(pd, args.input, plot_sample)
    fingerprint = {
        "schema_version": common.SCHEMA_VERSION,
        "generated_at": generated_at,
        "input": source,
        "rows": raw_rows,
        "columns": len(raw_columns),
    }
    schema = {
        "schema_version": common.SCHEMA_VERSION,
        "generated_at": generated_at,
        "columns": {
            column: {
                "dtype": info["dtype"],
                "semantic_type": info["semantic_type"],
                "required": bool(info["missing_count"] == 0),
                "identifier_like": info["identifier_like"],
            }
            for column, info in profiles.items()
        },
        "target": args.target,
        "time_column": args.time_column,
        "group_column": args.group_column,
        "partition_column": args.partition_column if args.mode == "model" else None,
    }
    report = {
        "schema_version": common.SCHEMA_VERSION,
        "generated_at": generated_at,
        "mode": args.mode,
        "task": args.task,
        "engine": "duckdb",
        "shape": {"rows": raw_rows, "columns": len(raw_columns)},
        "analysis_population": {
            "label": population_label,
            "rows": analysis_rows,
        },
        "plot_sampling": {
            "population_rows": analysis_rows,
            "rows": plot_rows,
            "method": "DuckDB deterministic reservoir sample",
            "seed": args.seed,
        },
        "aggregate_methods": {
            "row_and_missing_counts": "exact",
            "cardinality": "approx_count_distinct",
            "quantiles": "approx_quantile",
            "duplicate_rows": "hash plus approximate distinct screening",
            "correlation": "plot sample",
        },
        "duplicates_full_dataset_estimate": duplicate_estimate,
        "target_summary": target_summary,
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
        },
    }

    common.update_config(output_dir, args, population_label, plot_rows)
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["analysis"]["engine"] = "duckdb"
    config["analysis"]["approximate_aggregates"] = [
        "cardinality",
        "quantiles",
        "duplicate screening",
    ]
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    for filename, document in [
        ("data_profile.json", report),
        ("data_fingerprint.json", fingerprint),
        ("schema.json", schema),
    ]:
        (output_dir / filename).write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
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
