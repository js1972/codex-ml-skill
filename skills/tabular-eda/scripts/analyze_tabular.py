#!/usr/bin/env python3
"""Explore a local CSV or Parquet dataset and write one self-contained HTML report."""

from __future__ import annotations

import argparse
import hashlib
import html
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised only in minimal runtimes
    raise SystemExit(
        "tabular-eda requires pandas and numpy. Install them in the active Python environment."
    ) from exc


DEFAULT_MAX_PLOT_ROWS = 5_000
DEFAULT_MAX_NUMERIC_CHARTS = 8
DEFAULT_MAX_CATEGORICAL_CHARTS = 6
ID_NAME_RE = re.compile(
    r"(^id$|^id[_-]|[_-]id$|uuid|guid|identifier|(?:^|[_-])key$)",
    re.IGNORECASE,
)
DATE_NAME_RE = re.compile(
    r"(date|time|timestamp|datetime|created|updated|occurred)", re.IGNORECASE
)
PIPELINE_METADATA_RE = re.compile(
    r"(^|[_\-\s])(fold|split|partition)([_\-\s]|$)"
    r"|(^|[_\-\s])duplicate[_\-\s]?group([_\-\s]|$)"
    r"|^_?ml[_\-\s]",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Finding:
    severity: str
    title: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local CSV or Parquet file")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination HTML file, normally eda_report.html",
    )
    parser.add_argument("--target", help="Optional target column to describe")
    parser.add_argument("--time-column", help="Optional time column to describe")
    parser.add_argument(
        "--group-column", help="Optional entity/group column to describe"
    )
    parser.add_argument(
        "--delimiter",
        help="CSV delimiter; omit to detect it from the file",
    )
    parser.add_argument(
        "--max-plot-rows",
        type=int,
        default=DEFAULT_MAX_PLOT_ROWS,
        help="Maximum deterministic sample size used for charts",
    )
    parser.add_argument(
        "--max-numeric-charts",
        type=int,
        default=DEFAULT_MAX_NUMERIC_CHARTS,
    )
    parser.add_argument(
        "--max-categorical-charts",
        type=int,
        default=DEFAULT_MAX_CATEGORICAL_CHARTS,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def clean_scalar(value: Any) -> Any:
    """Return a finite built-in Python scalar suitable for report summaries."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def display_number(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) >= 1_000_000 or abs(number) < 0.001:
        return f"{number:.{digits}g}"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")


def pct(count: float, total: float) -> str:
    return "0%" if not total else f"{100 * float(count) / float(total):.1f}%"


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataframe(path: Path, delimiter: str | None) -> pd.DataFrame:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise SystemExit(
                "Parquet support is unavailable. Install pyarrow or fastparquet in "
                "the active Python environment."
            ) from exc
        except Exception as exc:
            raise SystemExit(f"Could not read Parquet input: {exc}") from exc

    if not suffixes or suffixes[-1] not in {".csv", ".tsv", ".txt", ".gz", ".bz2"}:
        raise SystemExit(
            "Input must be a local CSV, TSV, compressed CSV, or Parquet file."
        )
    selected_delimiter = delimiter
    if selected_delimiter == r"\t":
        selected_delimiter = "\t"
    try:
        if selected_delimiter is not None:
            return pd.read_csv(path, sep=selected_delimiter, low_memory=False)
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise SystemExit(f"Could not read delimited input: {exc}") from exc


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    parsed = urlparse(args.input)
    if parsed.scheme and parsed.scheme != "file":
        raise SystemExit("Only local CSV and Parquet files are supported.")
    source = Path(parsed.path if parsed.scheme == "file" else args.input).expanduser()
    if not source.is_file():
        raise SystemExit(f"Input file does not exist: {source}")
    output = Path(args.output).expanduser()
    if output.suffix.lower() not in {".html", ".htm"}:
        raise SystemExit("--output must name an HTML file.")
    if source.resolve() == output.resolve():
        raise SystemExit("--output must not overwrite the source dataset.")
    for option in ("max_plot_rows", "max_numeric_charts", "max_categorical_charts"):
        if getattr(args, option) <= 0:
            raise SystemExit(f"--{option.replace('_', '-')} must be greater than zero.")
    return source.resolve(), output.resolve()


def normalized_label(value: Any, max_length: int = 48) -> str:
    if pd.isna(value):
        return "(missing)"
    label = SPACE_RE.sub(" ", str(value)).strip() or "(blank)"
    if len(label) > max_length:
        return label[: max_length - 1] + "…"
    return label


def logical_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    return "categorical/text"


def potential_identifier(name: str, series: pd.Series, row_count: int) -> bool:
    non_null = int(series.notna().sum())
    if non_null == 0 or row_count == 0:
        return False
    if ID_NAME_RE.search(str(name)):
        return True
    ratio = float(series.nunique(dropna=True)) / non_null
    return bool(
        non_null >= 20
        and ratio >= 0.98
        and not pd.api.types.is_numeric_dtype(series.dtype)
    )


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    if valid.empty:
        return {}
    quantiles = valid.quantile([0.25, 0.5, 0.75])
    q1, median, q3 = (float(quantiles.loc[value]) for value in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    outliers = 0
    if iqr > 0:
        outliers = int(((valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)).sum())
    return {
        "min": clean_scalar(valid.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": clean_scalar(valid.max()),
        "mean": clean_scalar(valid.mean()),
        "std": clean_scalar(valid.std(ddof=1)),
        "outliers_iqr": outliers,
        "infinite": int(np.isinf(pd.to_numeric(series, errors="coerce")).sum()),
    }


def series_role(
    name: str,
    series: pd.Series,
    row_count: int,
    *,
    target: str | None,
    time_column: str | None,
    group_column: str | None,
) -> str:
    if name == target:
        return "target"
    if name == time_column:
        return "time"
    if name == group_column:
        return "group"
    if PIPELINE_METADATA_RE.search(str(name)):
        return "possible pipeline metadata"
    if potential_identifier(name, series, row_count):
        return "possible identifier"
    if DATE_NAME_RE.search(str(name)) and not pd.api.types.is_numeric_dtype(
        series.dtype
    ):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if int(parsed.notna().sum()) >= max(1, int(series.notna().sum() * 0.8)):
            return "date/time-like"
    unique = int(series.nunique(dropna=True))
    if unique <= 1:
        return "constant"
    if logical_type(series) == "numeric":
        return "numeric measure"
    return "category/text"


def build_column_profiles(
    frame: pd.DataFrame,
    *,
    target: str | None,
    time_column: str | None,
    group_column: str | None,
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    rows = len(frame)
    for name in frame.columns:
        series = frame[name]
        missing = int(series.isna().sum())
        profile: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "type": logical_type(series),
            "role": series_role(
                str(name),
                series,
                rows,
                target=target,
                time_column=time_column,
                group_column=group_column,
            ),
            "missing": missing,
            "missing_rate": float(missing / rows) if rows else 0.0,
            "unique": int(series.nunique(dropna=True)),
        }
        if profile["type"] == "numeric":
            profile["numeric"] = numeric_summary(series)
        profiles.append(profile)
    return profiles


def duplicate_summary(
    frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    excluded_roles = {"possible identifier", "possible pipeline metadata"}
    excluded = [
        profile["name"] for profile in profiles if profile["role"] in excluded_roles
    ]
    included = [str(name) for name in frame.columns if str(name) not in excluded]
    exact = int(frame.duplicated().sum())
    repeated = int(frame.duplicated(subset=included).sum()) if included else exact
    return {
        "exact": exact,
        "repeated": repeated,
        "included": included,
        "excluded": excluded,
    }


def make_findings(
    frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
    duplicates: dict[str, Any],
    target: str | None,
    time_column: str | None,
    group_column: str | None,
) -> list[Finding]:
    rows, columns = frame.shape
    findings: list[Finding] = [
        Finding(
            "info",
            "Dataset shape",
            f"The dataset contains {rows:,} rows and {columns:,} columns.",
        )
    ]
    repeated_rows = int(duplicates["repeated"])
    exact_rows = int(duplicates["exact"])
    excluded = list(duplicates["excluded"])
    exclusion_note = (
        " The comparison excluded likely identifiers or pipeline metadata: "
        + ", ".join(excluded[:8])
        + ("." if len(excluded) <= 8 else ", and others.")
        if excluded
        else ""
    )
    if repeated_rows:
        exact_note = (
            f" {exact_rows:,} of these are also identical across every loaded column."
            if exact_rows
            else " None are identical across every loaded column because excluded fields differ."
        )
        findings.append(
            Finding(
                "warning",
                "Repeated observations",
                f"{repeated_rows:,} rows ({pct(repeated_rows, rows)}) repeat an earlier "
                f"observation across the substantive columns.{exact_note}{exclusion_note}",
            )
        )
    else:
        findings.append(
            Finding(
                "positive",
                "No repeated observations",
                "No repeated observations were found across the substantive columns."
                + exclusion_note,
            )
        )

    high_missing = [p for p in profiles if p["missing_rate"] >= 0.20]
    any_missing = sum(int(p["missing"]) for p in profiles)
    if high_missing:
        labels = ", ".join(
            f"{p['name']} ({p['missing_rate']:.1%})" for p in high_missing[:5]
        )
        findings.append(
            Finding(
                "warning",
                "Substantial missingness",
                f"{len(high_missing)} columns are at least 20% missing: {labels}.",
            )
        )
    elif any_missing:
        findings.append(
            Finding(
                "info",
                "Some values are missing",
                f"{any_missing:,} cells are missing across the dataset.",
            )
        )
    else:
        findings.append(
            Finding("positive", "No missing values", "Every loaded cell has a value.")
        )

    constants = [p["name"] for p in profiles if p["unique"] <= 1]
    if constants:
        findings.append(
            Finding(
                "warning",
                "Constant columns",
                f"{len(constants)} columns carry no variation: {', '.join(constants[:8])}.",
            )
        )

    identifiers = [
        p["name"]
        for p in profiles
        if p["role"] == "possible identifier" and p["name"] != target
    ]
    if identifiers:
        findings.append(
            Finding(
                "review",
                "Possible identifiers",
                "Review whether these high-cardinality fields are identifiers rather than "
                f"measurements: {', '.join(identifiers[:8])}.",
            )
        )

    pipeline_metadata = [
        p["name"] for p in profiles if p["role"] == "possible pipeline metadata"
    ]
    if pipeline_metadata:
        findings.append(
            Finding(
                "review",
                "Possible pipeline metadata",
                "These fields look derived from data preparation or evaluation rather "
                "than measured attributes: "
                + ", ".join(pipeline_metadata[:8])
                + ("." if len(pipeline_metadata) <= 8 else ", and others."),
            )
        )

    numeric_outliers = [
        (p["name"], p.get("numeric", {}).get("outliers_iqr", 0))
        for p in profiles
        if p.get("numeric", {}).get("outliers_iqr", 0)
    ]
    if numeric_outliers:
        numeric_outliers.sort(key=lambda item: (-item[1], item[0]))
        labels = ", ".join(
            f"{name} ({count:,})" for name, count in numeric_outliers[:5]
        )
        findings.append(
            Finding(
                "review",
                "Potential numeric outliers",
                f"IQR screening flagged unusual values in {labels}. Verify these against domain expectations.",
            )
        )

    if target:
        series = frame[target]
        present = series.dropna()
        unique = int(present.nunique())
        missing = int(series.isna().sum())
        if missing:
            findings.append(
                Finding(
                    "warning",
                    "Target values are missing",
                    f"{missing:,} rows ({pct(missing, rows)}) have no value for {target}.",
                )
            )
        if unique == 0:
            findings.append(
                Finding(
                    "warning", "Target is empty", f"{target} has no observed values."
                )
            )
        elif unique == 1:
            findings.append(
                Finding(
                    "warning",
                    "Target has one observed value",
                    f"{target} cannot distinguish outcomes in this dataset.",
                )
            )
        elif unique <= 20:
            counts = present.value_counts(dropna=False)
            largest = int(counts.iloc[0])
            smallest = int(counts.iloc[-1])
            ratio = largest / smallest if smallest else math.inf
            findings.append(
                Finding(
                    "warning" if ratio >= 3 else "info",
                    "Target class balance",
                    f"{target} has {unique} observed classes; the largest has {largest:,} "
                    f"rows and the smallest has {smallest:,} ({ratio:.1f}:1 ratio).",
                )
            )
        elif pd.api.types.is_numeric_dtype(series.dtype):
            findings.append(
                Finding(
                    "info",
                    "Numeric target",
                    f"{target} has {unique:,} distinct observed values; its distribution "
                    "and numeric relationships are shown below.",
                )
            )

        target_copy = []
        for name in frame.columns:
            if name == target:
                continue
            comparable = frame[[target, name]].dropna()
            if not comparable.empty and comparable[target].astype(str).equals(
                comparable[name].astype(str)
            ):
                target_copy.append(str(name))
        if target_copy:
            findings.append(
                Finding(
                    "warning",
                    "Columns duplicate the target",
                    f"These fields exactly match {target} where both are present: "
                    f"{', '.join(target_copy[:8])}.",
                )
            )

    if time_column:
        raw = frame[time_column]
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
        invalid = int(raw.notna().sum() - parsed.notna().sum())
        if parsed.notna().any():
            start = parsed.min().date().isoformat()
            end = parsed.max().date().isoformat()
            detail = f"{time_column} spans {start} to {end}."
            if invalid:
                detail += f" {invalid:,} non-missing values could not be parsed."
            findings.append(
                Finding("warning" if invalid else "info", "Time coverage", detail)
            )
        else:
            findings.append(
                Finding(
                    "warning",
                    "Time values are not parseable",
                    f"No non-missing value in {time_column} could be parsed as a timestamp.",
                )
            )

    if group_column:
        counts = frame[group_column].dropna().value_counts()
        groups = len(counts)
        singleton_groups = int((counts == 1).sum()) if groups else 0
        largest = int(counts.iloc[0]) if groups else 0
        findings.append(
            Finding(
                "info",
                "Group structure",
                f"{group_column} contains {groups:,} observed groups; "
                f"{singleton_groups:,} occur once and the largest contains {largest:,} rows.",
            )
        )
    return findings


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def truncate_middle(value: str, length: int = 68) -> str:
    if len(value) <= length:
        return value
    half = (length - 1) // 2
    return value[:half] + "…" + value[-half:]


def bar_chart_svg(
    items: Iterable[tuple[str, float, str]],
    *,
    aria_label: str,
    width: int = 760,
) -> str:
    data = list(items)
    if not data:
        return '<p class="empty">No chartable values.</p>'
    row_height = 33
    label_width = 190
    value_width = 95
    chart_width = width - label_width - value_width - 28
    height = 20 + row_height * len(data)
    maximum = max(float(value) for _, value, _ in data) or 1.0
    rows = []
    for index, (label, value, shown) in enumerate(data):
        y = 12 + index * row_height
        bar_width = max(1.0, chart_width * float(value) / maximum) if value else 0.0
        rows.append(
            f'<text x="{label_width - 10}" y="{y + 18}" text-anchor="end" '
            f'class="svg-label">{escape(label)}</text>'
            f'<rect x="{label_width}" y="{y + 5}" width="{bar_width:.2f}" height="18" '
            f'rx="3" class="bar" />'
            f'<text x="{label_width + chart_width + 10}" y="{y + 18}" '
            f'class="svg-value">{escape(shown)}</text>'
        )
    return (
        f'<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(aria_label)}">{"".join(rows)}</svg>'
    )


def histogram_svg(
    values: pd.Series,
    *,
    column: str,
    width: int = 520,
    height: int = 180,
) -> str:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna()
    if numeric.empty:
        return '<p class="empty">No finite numeric values.</p>'
    distinct = int(numeric.nunique())
    bins = min(18, max(4, int(math.sqrt(len(numeric)))))
    if distinct <= 1:
        counts = np.array([len(numeric)])
        edges = np.array([float(numeric.iloc[0]) - 0.5, float(numeric.iloc[0]) + 0.5])
    else:
        counts, edges = np.histogram(numeric.to_numpy(dtype=float), bins=bins)
    left, top, bottom, right = 42, 12, 34, 12
    chart_width = width - left - right
    chart_height = height - top - bottom
    maximum = int(counts.max()) or 1
    gap = 2
    bar_width = chart_width / len(counts)
    rects = []
    for index, count in enumerate(counts):
        rendered_height = chart_height * int(count) / maximum
        rects.append(
            f'<rect x="{left + index * bar_width + gap / 2:.2f}" '
            f'y="{top + chart_height - rendered_height:.2f}" '
            f'width="{max(0.5, bar_width - gap):.2f}" height="{rendered_height:.2f}" '
            f'class="hist-bar"><title>{int(count):,} rows</title></rect>'
        )
    minimum = display_number(float(edges[0]))
    maximum_label = display_number(float(edges[-1]))
    median = display_number(float(numeric.median()))
    return (
        f'<svg class="histogram" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Distribution of {escape(column)}">'
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" '
        f'y2="{top + chart_height}" class="axis" />'
        f"{''.join(rects)}"
        f'<text x="{left}" y="{height - 10}" class="svg-value">{escape(minimum)}</text>'
        f'<text x="{left + chart_width / 2}" y="{height - 10}" text-anchor="middle" '
        f'class="svg-value">median {escape(median)}</text>'
        f'<text x="{left + chart_width}" y="{height - 10}" text-anchor="end" '
        f'class="svg-value">{escape(maximum_label)}</text></svg>'
    )


def correlation_table(
    frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
    max_columns: int = 12,
) -> tuple[str, str]:
    eligible = [
        profile["name"]
        for profile in profiles
        if profile["type"] == "numeric"
        and profile["role"]
        not in {"possible identifier", "possible pipeline metadata", "target"}
    ]
    numeric = (
        frame[eligible]
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
    )
    varying = [
        name
        for name in numeric.columns
        if numeric[name].notna().sum() >= 3 and numeric[name].nunique(dropna=True) > 1
    ][:max_columns]
    if len(varying) < 2:
        return "", "Fewer than two varying numeric columns were available."
    corr = numeric[varying].corr(method="pearson")
    header = "".join(f"<th>{escape(name)}</th>" for name in varying)
    rows = []
    for row_name in varying:
        cells = []
        for column in varying:
            value = corr.loc[row_name, column]
            if pd.isna(value):
                cells.append('<td class="corr na">—</td>')
                continue
            number = float(value)
            alpha = 0.08 + 0.72 * abs(number)
            color = (
                f"rgba(37,99,235,{alpha:.3f})"
                if number >= 0
                else f"rgba(220,38,38,{alpha:.3f})"
            )
            foreground = "#ffffff" if abs(number) >= 0.55 else "#172033"
            cells.append(
                f'<td class="corr" style="background:{color};color:{foreground}" '
                f'title="{escape(row_name)} and {escape(column)}: {number:.3f}">'
                f"{number:.2f}</td>"
            )
        rows.append(f"<tr><th>{escape(row_name)}</th>{''.join(cells)}</tr>")
    table = (
        '<div class="table-scroll"><table class="correlation"><thead><tr><th></th>'
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    return table, (
        "Pearson correlations range from −1 to +1. Strong relationships can be "
        "informative or redundant; correlation alone does not establish causation."
    )


def profile_table(profiles: list[dict[str, Any]], rows: int) -> str:
    rendered = []
    for profile in profiles:
        numeric = profile.get("numeric", {})
        center = f"median {display_number(numeric.get('median'))}" if numeric else "—"
        rendered.append(
            "<tr>"
            f"<td><code>{escape(profile['name'])}</code></td>"
            f"<td>{escape(profile['type'])}</td>"
            f"<td>{escape(profile['role'])}</td>"
            f'<td>{profile["missing"]:,} <span class="muted">'
            f"({pct(profile['missing'], rows)})</span></td>"
            f"<td>{profile['unique']:,}</td>"
            f"<td>{escape(center)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        "<th>Column</th><th>Observed type</th><th>Possible role</th>"
        "<th>Missing</th><th>Distinct</th><th>Numeric centre</th>"
        f"</tr></thead><tbody>{''.join(rendered)}</tbody></table></div>"
    )


def eta_squared(groups: pd.Series, values: pd.Series) -> float:
    data = pd.DataFrame(
        {
            "group": groups,
            "value": pd.to_numeric(values, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    if len(data) < 3 or data["group"].nunique() < 2:
        return 0.0
    grand_mean = float(data["value"].mean())
    total = float(((data["value"] - grand_mean) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, values_for_group in data.groupby("group", observed=True)["value"]:
        between += (
            len(values_for_group) * (float(values_for_group.mean()) - grand_mean) ** 2
        )
    return max(0.0, min(1.0, between / total))


def cramers_v(first: pd.Series, second: pd.Series) -> float:
    data = pd.DataFrame({"first": first, "second": second}).dropna()
    if len(data) < 3:
        return 0.0
    table = pd.crosstab(data["first"], data["second"])
    if min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    total = float(observed.sum())
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    valid = expected > 0
    chi_squared = float((((observed - expected) ** 2)[valid] / expected[valid]).sum())
    denominator = total * min(observed.shape[0] - 1, observed.shape[1] - 1)
    return math.sqrt(chi_squared / denominator) if denominator > 0 else 0.0


def eligible_relationship_profiles(
    profiles: list[dict[str, Any]],
    target: str,
) -> list[dict[str, Any]]:
    excluded_roles = {"possible identifier", "possible pipeline metadata"}
    return [
        profile
        for profile in profiles
        if profile["name"] != target and profile["role"] not in excluded_roles
    ]


def classification_relationships(
    frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
    target: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = eligible_relationship_profiles(profiles, target)
    numeric: list[dict[str, Any]] = []
    categorical: list[dict[str, Any]] = []
    for profile in candidates:
        name = profile["name"]
        if profile["type"] == "numeric" and profile["unique"] > 1:
            data = frame[[target, name]].dropna()
            medians = data.groupby(target, observed=True)[name].median()
            numeric.append(
                {
                    "name": name,
                    "score": eta_squared(frame[target], frame[name]),
                    "conditional": "; ".join(
                        f"{normalized_label(label)}: {display_number(float(value))}"
                        for label, value in medians.items()
                    ),
                }
            )
        elif 1 < profile["unique"] <= 50:
            data = frame[[target, name]].dropna()
            level_rows = []
            top_levels = data[name].value_counts().head(4).index
            for level in top_levels:
                selected = data[data[name] == level][target]
                composition = selected.value_counts(normalize=True).head(5)
                mix = ", ".join(
                    f"{normalized_label(label)} {float(value):.1%}"
                    for label, value in composition.items()
                )
                level_rows.append(
                    {
                        "level": normalized_label(level),
                        "rows": len(selected),
                        "mix": mix,
                    }
                )
            categorical.append(
                {
                    "name": name,
                    "score": cramers_v(frame[name], frame[target]),
                    "levels": level_rows,
                }
            )
    numeric.sort(key=lambda item: (-item["score"], item["name"]))
    categorical.sort(key=lambda item: (-item["score"], item["name"]))
    return numeric, categorical


def numeric_target_relationships(
    frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
    target: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = eligible_relationship_profiles(profiles, target)
    numeric: list[dict[str, Any]] = []
    categorical: list[dict[str, Any]] = []
    target_values = pd.to_numeric(frame[target], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    for profile in candidates:
        name = profile["name"]
        if profile["type"] == "numeric" and profile["unique"] > 1:
            values = pd.to_numeric(frame[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            correlation = values.corr(target_values)
            if pd.notna(correlation):
                numeric.append(
                    {
                        "name": name,
                        "score": abs(float(correlation)),
                        "signed": float(correlation),
                    }
                )
        elif 1 < profile["unique"] <= 50:
            data = pd.DataFrame(
                {"feature": frame[name], "target": target_values}
            ).dropna()
            medians = data.groupby("feature", observed=True)["target"].agg(
                ["count", "median"]
            )
            medians = medians.sort_values("count", ascending=False).head(5)
            categorical.append(
                {
                    "name": name,
                    "score": eta_squared(frame[name], target_values),
                    "conditional": "; ".join(
                        f"{normalized_label(label)}: {display_number(float(row['median']))} "
                        f"(n={int(row['count']):,})"
                        for label, row in medians.iterrows()
                    ),
                }
            )
    numeric.sort(key=lambda item: (-item["score"], item["name"]))
    categorical.sort(key=lambda item: (-item["score"], item["name"]))
    return numeric, categorical


def render_relationship_table(
    headers: list[str],
    rows: list[list[Any]],
) -> str:
    header = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        f"{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def target_relationship_section(
    frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
    target: str | None,
) -> str:
    if not target:
        return ""
    series = frame[target]
    observed = series.dropna()
    categorical_target = (
        not pd.api.types.is_numeric_dtype(series.dtype) or observed.nunique() <= 20
    )
    parts = [
        '<section id="target-relationships">',
        "<h2>Target–feature behavior</h2>",
        (
            "<p>These are descriptive conditional patterns in the loaded data. They do "
            "not select features, estimate future performance, or establish "
            "causation.</p>"
        ),
    ]
    if categorical_target:
        numeric, categorical = classification_relationships(frame, profiles, target)
        parts.extend(
            [
                "<h3>Conditional numeric behavior</h3>",
                (
                    "<p>η² summarizes how much observed variation in each numeric field "
                    "aligns with target groups (0 means no separation; 1 means complete "
                    "separation). Class medians show the direction of the pattern.</p>"
                ),
            ]
        )
        if numeric:
            parts.append(
                bar_chart_svg(
                    [
                        (
                            item["name"],
                            float(item["score"]),
                            f"η² {item['score']:.3f}",
                        )
                        for item in numeric[:8]
                    ],
                    aria_label=f"Numeric relationships with {target}",
                )
            )
            parts.append(
                render_relationship_table(
                    ["Feature", "η²", f"Median by {target}"],
                    [
                        [
                            item["name"],
                            f"{item['score']:.3f}",
                            item["conditional"],
                        ]
                        for item in numeric[:8]
                    ],
                )
            )
        else:
            parts.append(
                '<p class="empty">No eligible varying numeric features were available.</p>'
            )

        parts.extend(
            [
                "<h3>Conditional categorical behavior</h3>",
                (
                    "<p>Cramér’s V summarizes association between category composition "
                    "and the target (0 means no observed association; 1 means complete "
                    "association). The table shows target mix within frequent levels.</p>"
                ),
            ]
        )
        if categorical:
            parts.append(
                bar_chart_svg(
                    [
                        (
                            item["name"],
                            float(item["score"]),
                            f"V {item['score']:.3f}",
                        )
                        for item in categorical[:8]
                    ],
                    aria_label=f"Categorical relationships with {target}",
                )
            )
            conditional_rows = []
            for item in categorical[:5]:
                for level in item["levels"]:
                    conditional_rows.append(
                        [
                            item["name"],
                            level["level"],
                            f"{level['rows']:,}",
                            level["mix"],
                        ]
                    )
            parts.append(
                render_relationship_table(
                    ["Feature", "Level", "Rows", f"{target} composition"],
                    conditional_rows,
                )
            )
        else:
            parts.append(
                '<p class="empty">No eligible low-cardinality categorical features were available.</p>'
            )
    else:
        numeric, categorical = numeric_target_relationships(frame, profiles, target)
        parts.extend(
            [
                "<h3>Numeric relationships</h3>",
                (
                    "<p>Pearson r describes linear co-movement with the target; its sign "
                    "gives direction and its magnitude gives strength. Nonlinear "
                    "patterns may not appear here.</p>"
                ),
            ]
        )
        if numeric:
            parts.append(
                bar_chart_svg(
                    [
                        (
                            item["name"],
                            float(item["score"]),
                            f"r {item['signed']:+.3f}",
                        )
                        for item in numeric[:8]
                    ],
                    aria_label=f"Numeric correlations with {target}",
                )
            )
        else:
            parts.append(
                '<p class="empty">No eligible varying numeric features were available.</p>'
            )
        parts.extend(
            [
                "<h3>Conditional categorical behavior</h3>",
                (
                    "<p>η² describes how much observed target variation aligns with "
                    "category groups. Frequent-level target medians show the conditional "
                    "pattern.</p>"
                ),
            ]
        )
        if categorical:
            parts.append(
                bar_chart_svg(
                    [
                        (
                            item["name"],
                            float(item["score"]),
                            f"η² {item['score']:.3f}",
                        )
                        for item in categorical[:8]
                    ],
                    aria_label=f"Categorical relationships with {target}",
                )
            )
            parts.append(
                render_relationship_table(
                    ["Feature", "η²", f"Median {target} by frequent level"],
                    [
                        [
                            item["name"],
                            f"{item['score']:.3f}",
                            item["conditional"],
                        ]
                        for item in categorical[:8]
                    ],
                )
            )
        else:
            parts.append(
                '<p class="empty">No eligible low-cardinality categorical features were available.</p>'
            )
    parts.append("</section>")
    return "".join(parts)


def target_section(frame: pd.DataFrame, target: str | None) -> str:
    if not target:
        return (
            '<section id="target"><h2>Target view</h2>'
            '<p class="empty">No target hint was supplied. This report remains a '
            "model-independent exploration.</p></section>"
        )
    series = frame[target]
    observed = series.dropna()
    if observed.nunique() <= 20:
        counts = observed.map(normalized_label).value_counts().head(20)
        chart = bar_chart_svg(
            [
                (str(label), float(count), f"{int(count):,}")
                for label, count in counts.items()
            ],
            aria_label=f"Class distribution for {target}",
        )
        explanation = (
            "Class counts reveal whether some outcomes are much less represented than "
            "others. Missing target values are excluded from these bars."
        )
    elif pd.api.types.is_numeric_dtype(series.dtype):
        chart = histogram_svg(series, column=target)
        explanation = (
            "The histogram describes the observed target range and concentration. It "
            "does not imply a modeling task or evaluation design."
        )
    else:
        counts = observed.map(normalized_label).value_counts().head(20)
        chart = bar_chart_svg(
            [
                (str(label), float(count), f"{int(count):,}")
                for label, count in counts.items()
            ],
            aria_label=f"Most frequent values for {target}",
        )
        explanation = "Only the 20 most frequent observed target values are shown."
    return (
        f'<section id="target"><h2>Target view: <code>{escape(target)}</code></h2>'
        f"<p>{escape(explanation)}</p>{chart}</section>"
    )


def time_section(frame: pd.DataFrame, time_column: str | None) -> str:
    if not time_column:
        return ""
    parsed = pd.to_datetime(frame[time_column], errors="coerce", utc=True).dropna()
    if parsed.empty:
        return (
            f'<section id="time"><h2>Time view: <code>{escape(time_column)}</code></h2>'
            '<p class="empty">No values could be parsed as timestamps.</p></section>'
        )
    date_span = max(0, (parsed.max() - parsed.min()).days)
    if date_span > 730:
        periods = parsed.dt.to_period("Y").astype(str)
        grain = "year"
    elif date_span > 90:
        periods = parsed.dt.to_period("M").astype(str)
        grain = "month"
    else:
        periods = parsed.dt.floor("D").dt.strftime("%Y-%m-%d")
        grain = "day"
    counts = periods.value_counts().sort_index()
    if len(counts) > 36:
        # Deterministic contiguous buckets preserve the overall timeline without
        # selecting arbitrary peaks.
        positions = np.array_split(np.arange(len(counts)), 36)
        compact = []
        for positions_for_bucket in positions:
            part = counts.iloc[positions_for_bucket]
            compact.append((str(part.index[0]), int(part.sum())))
        items = compact
    else:
        items = [(str(label), int(value)) for label, value in counts.items()]
    chart = bar_chart_svg(
        [(label, float(count), f"{count:,}") for label, count in items],
        aria_label=f"Rows over time for {time_column}",
    )
    return (
        f'<section id="time"><h2>Time view: <code>{escape(time_column)}</code></h2>'
        f"<p>Counts are grouped by {grain}. Gaps or abrupt volume changes may reflect "
        f"collection changes as well as real-world behavior.</p>{chart}</section>"
    )


def numeric_sections(
    plot_frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
    maximum: int,
) -> str:
    selected = [
        profile
        for profile in profiles
        if profile["type"] == "numeric"
        and profile["unique"] > 1
        and profile["role"]
        not in {"possible identifier", "possible pipeline metadata", "target"}
    ][:maximum]
    if not selected:
        return '<p class="empty">No varying numeric columns were available.</p>'
    cards = []
    for profile in selected:
        numeric = profile.get("numeric", {})
        note = (
            f"Full-data range {display_number(numeric.get('min'))} to "
            f"{display_number(numeric.get('max'))}; "
            f"{numeric.get('outliers_iqr', 0):,} IQR-screened outliers."
        )
        cards.append(
            '<article class="chart-card">'
            f"<h3><code>{escape(profile['name'])}</code></h3>"
            f"<p>{escape(note)}</p>"
            f"{histogram_svg(plot_frame[profile['name']], column=profile['name'])}"
            "</article>"
        )
    return f'<div class="chart-grid">{"".join(cards)}</div>'


def categorical_sections(
    plot_frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
    maximum: int,
    target: str | None,
) -> str:
    candidates = [
        profile
        for profile in profiles
        if profile["type"] != "numeric"
        and 1 < profile["unique"] <= 100
        and profile["name"] != target
        and profile["role"] not in {"possible identifier", "possible pipeline metadata"}
    ][:maximum]
    if not candidates:
        return '<p class="empty">No suitable low-cardinality categorical columns were available.</p>'
    cards = []
    for profile in candidates:
        series = plot_frame[profile["name"]].map(normalized_label)
        counts = series.value_counts(dropna=False).head(12)
        chart = bar_chart_svg(
            [
                (str(label), float(count), f"{int(count):,}")
                for label, count in counts.items()
            ],
            aria_label=f"Most frequent values for {profile['name']}",
        )
        note = (
            "Top values in the chart sample; long-tail values beyond the first 12 "
            "are not drawn."
        )
        cards.append(
            '<article class="chart-card wide">'
            f"<h3><code>{escape(profile['name'])}</code></h3>"
            f"<p>{escape(note)}</p>{chart}</article>"
        )
    return f'<div class="chart-grid single">{"".join(cards)}</div>'


def render_html(
    *,
    source: Path,
    source_hash: str,
    frame: pd.DataFrame,
    plot_frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
    findings: list[Finding],
    duplicates: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    rows, columns = frame.shape
    repeated_rows = int(duplicates["repeated"])
    missing_cells = int(frame.isna().sum().sum())
    numeric_columns = sum(p["type"] == "numeric" for p in profiles)
    categorical_columns = columns - numeric_columns
    finding_html = "".join(
        '<li class="finding">'
        f'<span class="severity {escape(finding.severity)}">{escape(finding.severity)}</span>'
        f"<div><strong>{escape(finding.title)}</strong>"
        f"<p>{escape(finding.detail)}</p></div></li>"
        for finding in findings
    )
    missing_items = [
        (
            profile["name"],
            float(profile["missing_rate"]),
            f"{profile['missing_rate']:.1%}",
        )
        for profile in profiles
        if profile["missing"]
    ]
    missing_items.sort(key=lambda item: (-item[1], item[0]))
    missing_chart = (
        bar_chart_svg(
            missing_items[:20],
            aria_label="Missing-value rate by column",
        )
        if missing_items
        else '<p class="empty">No missing values were detected.</p>'
    )
    correlation_html, correlation_explanation = correlation_table(plot_frame, profiles)
    hint_rows = [
        ("Target", args.target or "not supplied"),
        ("Time", args.time_column or "not supplied"),
        ("Group", args.group_column or "not supplied"),
    ]
    hints = "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in hint_rows
    )
    sampling_note = (
        f"Charts use all {rows:,} rows."
        if len(plot_frame) == rows
        else (
            f"Charts use a deterministic sample of {len(plot_frame):,} from "
            f"{rows:,} rows (seed {args.seed}). Full-data counts and summaries are "
            "not sampled."
        )
    )
    safe_filename = truncate_middle(source.name)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Exploratory data analysis — {escape(safe_filename)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#dbe2ea;
      --surface:#fff; --soft:#f5f7fb; --blue:#2563eb; --red:#dc2626;
      --amber:#a16207; --green:#15803d; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#eef2f7; color:var(--ink);
      font:15px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1160px,calc(100% - 32px)); margin:32px auto 64px; }}
    header,section {{ background:var(--surface); border:1px solid var(--line);
      border-radius:16px; box-shadow:0 8px 28px rgba(23,32,51,.06); }}
    header {{ padding:34px; background:linear-gradient(135deg,#0f2a55,#174b9a); color:#fff; }}
    section {{ margin-top:20px; padding:28px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(26px,5vw,42px); line-height:1.1; }}
    h2 {{ margin:0 0 8px; font-size:24px; }}
    h3 {{ margin:0 0 4px; font-size:17px; }}
    p {{ margin:6px 0 14px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }}
    .subtitle {{ color:#d9e8ff; max-width:780px; }}
    .metadata {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
      gap:12px; margin-top:24px; }}
    .metadata div {{ border:1px solid rgba(255,255,255,.2); border-radius:10px;
      padding:11px 13px; background:rgba(255,255,255,.07); }}
    .metadata span {{ display:block; color:#c6d9f8; font-size:12px; text-transform:uppercase;
      letter-spacing:.05em; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
      gap:12px; margin-top:20px; }}
    .metric {{ background:var(--soft); border:1px solid var(--line); border-radius:12px; padding:16px; }}
    .metric b {{ display:block; font-size:27px; }}
    .metric span,.muted,.empty {{ color:var(--muted); }}
    .findings {{ list-style:none; padding:0; margin:20px 0 0; display:grid; gap:12px; }}
    .finding {{ display:grid; grid-template-columns:88px 1fr; gap:14px;
      align-items:start; padding:14px; border:1px solid var(--line); border-radius:12px; }}
    .finding p {{ color:var(--muted); margin:3px 0 0; }}
    .severity {{ display:inline-block; border-radius:999px; padding:3px 8px;
      text-align:center; text-transform:uppercase; font-size:10px; font-weight:800; letter-spacing:.05em; }}
    .severity.warning {{ background:#fff1f2; color:#be123c; }}
    .severity.review {{ background:#fff7ed; color:#9a3412; }}
    .severity.info {{ background:#eff6ff; color:#1d4ed8; }}
    .severity.positive {{ background:#f0fdf4; color:#15803d; }}
    .table-scroll {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; white-space:nowrap; }}
    thead th {{ background:var(--soft); }}
    .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; margin-top:16px; }}
    .chart-grid.single {{ grid-template-columns:1fr; }}
    .chart-card {{ border:1px solid var(--line); border-radius:12px; padding:16px; min-width:0; }}
    .chart-card p {{ color:var(--muted); font-size:13px; }}
    svg {{ display:block; max-width:100%; height:auto; overflow:visible; }}
    .bar {{ fill:#2563eb; }}
    .hist-bar {{ fill:#3b82f6; }}
    .axis {{ stroke:#98a2b3; stroke-width:1; }}
    .svg-label,.svg-value {{ font:12px ui-sans-serif,system-ui,sans-serif; fill:#475467; }}
    .correlation th {{ max-width:150px; overflow:hidden; text-overflow:ellipsis; }}
    .correlation td {{ text-align:center; min-width:56px; }}
    .corr.na {{ background:#f2f4f7; color:#98a2b3; }}
    .note {{ border-left:4px solid var(--blue); padding:9px 14px; background:#eff6ff;
      border-radius:4px 10px 10px 4px; color:#1e3a5f; }}
    footer {{ color:var(--muted); text-align:center; margin-top:24px; font-size:13px; }}
    @media (max-width:640px) {{
      main {{ width:min(100% - 18px,1160px); margin-top:9px; }}
      header,section {{ padding:20px; border-radius:12px; }}
      .finding {{ grid-template-columns:1fr; }}
      .chart-grid {{ grid-template-columns:1fr; }}
    }}
    @media print {{
      body {{ background:#fff; }} main {{ width:100%; margin:0; }}
      header,section {{ box-shadow:none; break-inside:avoid; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Exploratory data analysis</h1>
    <p class="subtitle">A descriptive, model-independent view of structure, quality,
      distributions, and relationships. This report does not train models or define a
      modeling workflow.</p>
    <div class="metadata">
      <div><span>Source</span><code>{escape(safe_filename)}</code></div>
      <div><span>SHA-256</span><code>{source_hash}</code></div>
      <div><span>Chart sampling</span>{escape(sampling_note)}</div>
    </div>
  </header>

  <section id="overview">
    <h2>Overview</h2>
    <div class="cards">
      <div class="metric"><b>{rows:,}</b><span>rows</span></div>
      <div class="metric"><b>{columns:,}</b><span>columns</span></div>
      <div class="metric"><b>{numeric_columns:,}</b><span>numeric columns</span></div>
      <div class="metric"><b>{categorical_columns:,}</b><span>other columns</span></div>
      <div class="metric"><b>{missing_cells:,}</b><span>missing cells</span></div>
      <div class="metric"><b>{repeated_rows:,}</b><span>repeated observations</span></div>
    </div>
    <table><tbody>{hints}</tbody></table>
  </section>

  <section id="findings">
    <h2>Important findings</h2>
    <p>These observations identify where domain context or closer inspection is most useful.</p>
    <ul class="findings">{finding_html}</ul>
  </section>

  <section id="quality">
    <h2>Column quality and structure</h2>
    <p>Possible roles are heuristics based on observed values and names. They should be
      confirmed against the data-generating process.</p>
    {profile_table(profiles, rows)}
    <h3>Missing-value rates</h3>
    {missing_chart}
  </section>

  {target_section(frame, args.target)}
  {target_relationship_section(frame, profiles, args.target)}
  {time_section(frame, args.time_column)}

  <section id="numeric">
    <h2>Numeric distributions</h2>
    <p>{escape(sampling_note)} Histograms summarize shape and concentration; an IQR
      screen is only a prompt for review, not proof that a value is erroneous.</p>
    {numeric_sections(plot_frame, profiles, args.max_numeric_charts)}
  </section>

  <section id="categorical">
    <h2>Categorical distributions</h2>
    <p>{escape(sampling_note)} Categories are normalized only for display; source values
      are not changed.</p>
    {categorical_sections(plot_frame, profiles, args.max_categorical_charts, args.target)}
  </section>

  <section id="relationships">
    <h2>Numeric relationships</h2>
    <p>{escape(correlation_explanation)}</p>
    {correlation_html or '<p class="empty">No correlation matrix was produced.</p>'}
  </section>

  <section id="limits">
    <h2>Interpretation boundaries</h2>
    <div class="note">This is exploratory evidence, not a model specification. It does
      not choose features, split data, estimate future performance, or prove causal
      relationships. Any later modeling workflow must independently inspect the source
      data at prediction time.</div>
  </section>

  <footer>Self-contained report: no external stylesheets, scripts, images, or data files.</footer>
</main>
</body>
</html>
"""


def write_atomic(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_name = handle.name
        os.replace(temporary_name, output)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    source, output = validate_args(args)
    frame = load_dataframe(source, args.delimiter)
    if frame.empty:
        raise SystemExit("The dataset contains no rows.")
    if len(frame.columns) == 0:
        raise SystemExit("The dataset contains no columns.")
    duplicate_names = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicate_names:
        raise SystemExit(
            "The dataset contains duplicate column names: " + ", ".join(duplicate_names)
        )
    frame.columns = [str(column) for column in frame.columns]
    duplicate_names = [
        name
        for index, name in enumerate(frame.columns)
        if name in frame.columns[:index]
    ]
    if duplicate_names:
        raise SystemExit(
            "Column names collide after conversion to text: "
            + ", ".join(duplicate_names)
        )
    for argument_name, value in (
        ("--target", args.target),
        ("--time-column", args.time_column),
        ("--group-column", args.group_column),
    ):
        if value and value not in frame.columns:
            raise SystemExit(f"{argument_name} column was not found: {value}")
    profiles = build_column_profiles(
        frame,
        target=args.target,
        time_column=args.time_column,
        group_column=args.group_column,
    )
    duplicates = duplicate_summary(frame, profiles)
    findings = make_findings(
        frame,
        profiles,
        duplicates,
        args.target,
        args.time_column,
        args.group_column,
    )
    if len(frame) > args.max_plot_rows:
        plot_frame = frame.sample(
            n=args.max_plot_rows,
            random_state=args.seed,
            replace=False,
        ).sort_index()
    else:
        plot_frame = frame
    digest = source_sha256(source)
    report = render_html(
        source=source,
        source_hash=digest,
        frame=frame,
        plot_frame=plot_frame,
        profiles=profiles,
        findings=findings,
        duplicates=duplicates,
        args=args,
    )
    write_atomic(output, report)
    return {
        "status": "ok",
        "report": str(output),
        "source": {
            "file": source.name,
            "sha256": digest,
            "rows": len(frame),
            "columns": len(frame.columns),
        },
        "chart_sample": {
            "rows": len(plot_frame),
            "seed": int(args.seed),
        },
        "findings": [finding.as_dict() for finding in findings],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args)
    lines = ["## EDA findings", ""]
    for finding in result["findings"]:
        severity = str(finding["severity"]).capitalize()
        lines.append(f"- **{finding['title']}** ({severity}) — {finding['detail']}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
