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
from pathlib import Path
from urllib.parse import urlparse

SCHEMA_VERSION = "2.0"
ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|uuid|guid|key$)", re.IGNORECASE)


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
    parser.add_argument("--max-plot-rows", type=int, default=10_000)
    parser.add_argument("--max-numeric-plots", type=int, default=12)
    parser.add_argument("--max-categorical-plots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--duckdb-memory-limit", default="4GB")
    parser.add_argument("--duckdb-temp-directory")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


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
        digest = hashlib.sha256()
        with local.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "kind": "file_sha256",
            "sha256": digest.hexdigest(),
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


def profile_columns(pd, frame, time_column: str | None, findings: list[dict]):
    profiles = {}
    rows = len(frame)
    for column in frame.columns:
        series = frame[column]
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
        if info["semantic_type"] == "numeric" and not info["identifier_like"]
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
        if info["semantic_type"] == "numeric" and not info["identifier_like"]
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
                tick_labels=[f"class {index + 1}" for index in range(len(groups))],
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


def plot_time_series(
    pd, plt, analysis_frame, time_column, target, output, figures, findings
):
    if not time_column:
        return
    if time_column not in analysis_frame.columns:
        raise SystemExit(f"Time column '{time_column}' does not exist.")
    parsed = pd.to_datetime(analysis_frame[time_column], errors="coerce", utc=True)
    invalid = int(parsed.isna().sum() - analysis_frame[time_column].isna().sum())
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
    if not target or target not in analysis_frame.columns:
        return
    numeric_target = pd.to_numeric(analysis_frame[target], errors="coerce")
    plot_frame = pd.DataFrame({"time": parsed, "target": numeric_target}).dropna()
    if plot_frame.empty:
        return
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


def markdown_summary(profile: dict) -> str:
    severity_order = ["blocker", "warning", "information"]
    lines = [
        "# Dataset Summary",
        "",
        f"- Mode: {profile['mode']}",
        f"- Task: {profile['task']}",
        f"- Rows: {profile['shape']['rows']:,}",
        f"- Columns: {profile['shape']['columns']:,}",
        f"- Target-aware population: {profile['analysis_population']['label']}",
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
            "- In model mode, target-aware analysis uses training rows only.",
            "- Review the HTML charts and data contract before choosing transformations.",
            "",
        ]
    )
    return "\n".join(lines)


def html_report(profile: dict) -> str:
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
        f"<td>{info['missing_fraction']:.1%}</td>"
        f"<td>{info['unique_count']}</td>"
        "</tr>"
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
<strong>{html.escape(profile["task"])}</strong>. Target-aware population:
<strong>{html.escape(profile["analysis_population"]["label"])}</strong>.</p>
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
<li>In model mode, target-aware analysis uses training rows only.</li>
</ul>
</body>
</html>
"""


def update_config(output_dir: Path, args, population_label: str, plot_rows: int):
    """Create the EDA contract or add EDA details without erasing user decisions."""
    path = output_dir / "config.json"
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Existing config.json is invalid: {exc}") from exc
        if not isinstance(config, dict):
            raise SystemExit("Existing config.json must contain a JSON object.")
    else:
        config = {
            "schema_version": SCHEMA_VERSION,
            "mode": "analysis-only"
            if args.mode == "analysis-only"
            else "model-building",
            "problem": {
                "task": args.task,
                "target": args.target,
                "prediction_moment": None,
                "row_grain": None,
                "group_column": args.group_column,
                "time_column": args.time_column,
            },
            "data": {
                "locations": [args.input],
                "fingerprint_file": "artefacts/data_fingerprint.json",
                "schema_file": "artefacts/schema.json",
            },
            "split": {
                "assignment_column": args.partition_column
                if args.mode == "model"
                else None,
                "development_label": args.train_label if args.mode == "model" else None,
                "holdout_target_sealed": (
                    args.mode == "model" and args.evaluation_design == "holdout"
                ),
                "seed": args.seed,
            },
            "evaluation": {
                "design": args.evaluation_design if args.mode == "model" else None,
                "final_eval_set": (
                    {
                        "holdout": "holdout_test",
                        "nested_cv": "outer_cv",
                        "external_test": "external_test",
                        "prospective_validation": "prospective_validation",
                    }[args.evaluation_design]
                    if args.mode == "model"
                    else None
                ),
                "independent_test": (
                    args.evaluation_design != "nested_cv"
                    if args.mode == "model"
                    else None
                ),
                "selection_nested": (
                    args.evaluation_design == "nested_cv"
                    if args.mode == "model"
                    else None
                ),
            },
            "governance": {
                "risk_tier": "standard",
                "deployment_decision": "not_assessed",
                "approval_status": "not_assessed",
            },
        }
    config.setdefault("schema_version", SCHEMA_VERSION)
    config["analysis"] = {
        "report": "artefacts/data_report.html",
        "population_partition": args.train_label if args.mode == "model" else None,
        "target_aware_partition": args.train_label
        if args.mode == "model" and args.target
        else None,
        "population": population_label,
        "plot_sample_size": plot_rows,
        "plot_sample_seed": args.seed,
        "category_labels_rendered": True,
    }
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
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

    if args.mode == "model":
        if args.partition_column not in frame.columns:
            raise SystemExit(
                "Model mode requires a persisted partition column before EDA. "
                f"Missing: '{args.partition_column}'."
            )
        analysis_frame = frame.loc[
            frame[args.partition_column].astype(str) == args.train_label
        ].copy()
        if analysis_frame.empty:
            raise SystemExit(
                f"Partition column '{args.partition_column}' has no "
                f"'{args.train_label}' rows."
            )
        population_label = f"{args.partition_column}={args.train_label}"
    else:
        analysis_frame = frame
        population_label = "full permitted dataset (descriptive analysis)"

    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    findings: list[dict] = []
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        add_finding(
            findings,
            "warning",
            "exact_duplicates",
            f"The raw dataset contains {duplicates:,} exact duplicate rows.",
            "Determine whether these are accidental copies or legitimate repeated events.",
        )

    profiles = profile_columns(pd, analysis_frame, args.time_column, findings)
    plot_rows = min(len(analysis_frame), max(args.max_plot_rows, 1))
    plot_sample = (
        analysis_frame.sample(n=plot_rows, random_state=args.seed)
        if plot_rows < len(analysis_frame)
        else analysis_frame.copy()
    )

    figures: list[dict] = []
    plot_missingness(plt, analysis_frame, figures_dir, figures)
    plot_numeric(
        plt,
        plot_sample,
        profiles,
        figures_dir,
        figures,
        args.max_numeric_plots,
    )
    plot_categorical(
        plt,
        plot_sample,
        profiles,
        figures_dir,
        figures,
        args.max_categorical_plots,
    )
    plot_correlation(plt, analysis_frame, profiles, figures_dir, figures)
    target_summary = plot_target(
        pd,
        plt,
        analysis_frame,
        plot_sample,
        args.mode,
        args.task,
        args.target,
        profiles,
        figures_dir,
        figures,
        findings,
    )
    plot_feature_relationships(
        pd,
        plt,
        plot_sample,
        args.task,
        args.target,
        profiles,
        figures_dir,
        figures,
    )
    if args.task == "time-series" or args.time_column:
        plot_time_series(
            pd,
            plt,
            analysis_frame,
            args.time_column,
            args.target,
            figures_dir,
            figures,
            findings,
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "input": source_fingerprint(pd, args.input, frame),
        "rows": len(frame),
        "columns": len(frame.columns),
    }
    schema = {
        "schema_version": SCHEMA_VERSION,
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
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "mode": args.mode,
        "task": args.task,
        "shape": {"rows": len(frame), "columns": len(frame.columns)},
        "analysis_population": {
            "label": population_label,
            "rows": len(analysis_frame),
        },
        "plot_sampling": {
            "population_rows": len(analysis_frame),
            "rows": plot_rows,
            "method": "deterministic random sample without replacement"
            if plot_rows < len(analysis_frame)
            else "all permitted rows",
            "seed": args.seed,
        },
        "duplicates_full_dataset": duplicates,
        "target_summary": target_summary,
        "columns": profiles,
        "findings": findings,
        "figures": figures,
    }

    update_config(output_dir, args, population_label, plot_rows)
    (output_dir / "data_profile.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "data_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
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
