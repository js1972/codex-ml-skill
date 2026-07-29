#!/usr/bin/env python3
"""Render one self-contained HTML report from a consolidated ML run."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Run directory containing run.json and results.md",
    )
    parser.add_argument("--run-json", type=Path, help="Override run.json path")
    parser.add_argument("--results", type=Path, help="Override results.md path")
    parser.add_argument("--output", type=Path, help="Override report.html path")
    return parser.parse_args(argv)


def esc(value: Any) -> str:
    if value is None:
        return "—"
    return html.escape(str(value))


def fmt_number(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.5g}"
    return esc(value)


def badge(value: Any) -> str:
    text = str(value or "unknown")
    key = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return f'<span class="badge badge-{esc(key)}">{esc(text)}</span>'


def definition_rows(items: list[tuple[str, Any]]) -> str:
    return "".join(
        f"<dt>{esc(label)}</dt><dd>{fmt_number(value)}</dd>" for label, value in items
    )


def compact(value: Any) -> str:
    if value in (None, "", [], {}):
        return "—"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {compact(item)}" for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(compact(item) for item in value)
    return str(value)


def markdown_summary(markdown: str) -> str:
    """Render deliberately small Markdown subset without external dependencies."""
    blocks: list[str] = []
    list_items: list[str] = []

    def flush_list() -> None:
        if list_items:
            blocks.append("<ul>" + "".join(list_items) + "</ul>")
            list_items.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            flush_list()
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush_list()
            level = min(len(heading.group(1)) + 1, 5)
            blocks.append(f"<h{level}>{esc(heading.group(2))}</h{level}>")
        elif line.startswith(("- ", "* ")):
            list_items.append(f"<li>{esc(line[2:].strip())}</li>")
        else:
            flush_list()
            blocks.append(f"<p>{esc(line)}</p>")
    flush_list()
    return "".join(blocks) or "<p>No narrative results were supplied.</p>"


def backend_rows(backends: dict[str, Any], metric_name: str) -> tuple[str, list[float]]:
    rows: list[str] = []
    scores: list[float] = []
    for name, backend in backends.items():
        evaluation = backend.get("evaluation") or {}
        score = evaluation.get("score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            scores.append(float(score))
        evidence = backend.get("evidence")
        if isinstance(evidence, list):
            evidence_text = "; ".join(str(item) for item in evidence)
        elif isinstance(evidence, dict):
            evidence_text = "; ".join(
                f"{key}: {value}" for key, value in evidence.items()
            )
        else:
            evidence_text = str(evidence or "—")
        rows.append(
            "<tr>"
            f"<td><strong>{esc(name)}</strong></td>"
            f"<td>{badge(backend.get('status'))}</td>"
            f"<td>{fmt_number(score)}</td>"
            f"<td>{esc(evidence_text)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append(
            f'<tr><td colspan="4">No backend results were recorded for {esc(metric_name)}.</td></tr>'
        )
    return "".join(rows), scores


def score_chart(backends: dict[str, Any], direction: str) -> str:
    scored: list[tuple[str, float]] = []
    for name, backend in backends.items():
        value = (backend.get("evaluation") or {}).get("score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            scored.append((name, float(value)))
    if not scored:
        return (
            '<p class="muted">No comparable numeric backend scores were recorded.</p>'
        )

    values = [value for _, value in scored]
    low, high = min(values), max(values)
    span = high - low
    maximize_baseline = min(0.0, low)
    maximize_span = max(high - maximize_baseline, span, 1e-12)
    row_height = 42
    width = 760
    height = 36 + row_height * len(scored)
    elements = [
        (
            f'<svg class="score-chart" viewBox="0 0 {width} {height}" '
            'role="img" aria-label="Backend primary metric comparison">'
        )
    ]
    for index, (name, value) in enumerate(scored):
        y = 24 + index * row_height
        if direction == "minimize":
            relative_quality = 1.0 if span == 0 else (high - value) / span
        else:
            relative_quality = (value - maximize_baseline) / maximize_span
        bar_width = max(4.0, 500.0 * relative_quality)
        elements.extend(
            [
                f'<text x="0" y="{y + 16}" class="chart-label">{esc(name)}</text>',
                f'<rect x="170" y="{y}" width="{bar_width:.2f}" height="22" rx="5" />',
                (
                    f'<text x="{min(690, 180 + bar_width):.2f}" y="{y + 16}" '
                    f'class="chart-value">{fmt_number(value)}</text>'
                ),
            ]
        )
    elements.append("</svg>")
    return "".join(elements)


def approval_cards(approval: dict[str, Any]) -> str:
    tracks = approval.get("tracks") or {}
    cards: list[str] = []
    for name in ("classical", "autogluon", "sap_rpt"):
        track = tracks.get(name) or {}
        budget = track.get("budget") or {}
        budget_text = (
            ", ".join(f"{key}: {value}" for key, value in budget.items()) or "none"
        )
        cards.append(
            '<article class="mini-card">'
            f"<h3>{esc(name.replace('_', ' ').title())}</h3>"
            f"<p>{badge(track.get('status', 'not recorded'))}</p>"
            f"<p><strong>Selected:</strong> {fmt_number(track.get('selected', False))}</p>"
            f"<p><strong>Budget:</strong> {esc(budget_text)}</p>"
            "</article>"
        )
    return "".join(cards)


def approval_scope_cards(approval: dict[str, Any]) -> str:
    scope = approval.get("scope") or {}
    labels = (
        ("target", "Target"),
        ("feature_contract", "Feature contract"),
        ("split_design", "Split design"),
        ("primary_metric", "Primary metric"),
    )
    return "".join(
        '<article class="mini-card">'
        f"<h3>{esc(label)}</h3>"
        f"<p>{badge('confirmed' if scope.get(key) is True else 'not confirmed')}</p>"
        "</article>"
        for key, label in labels
    )


def preflight_rows(findings: list[Any]) -> str:
    rows: list[str] = []
    for finding in findings:
        if isinstance(finding, dict):
            severity = finding.get("severity") or "review"
            code = finding.get("code") or "finding"
            message = finding.get("message") or finding.get("detail") or finding
        else:
            severity = "review"
            code = "finding"
            message = finding
        rows.append(
            "<tr>"
            f"<td>{badge(severity)}</td>"
            f"<td>{esc(code)}</td>"
            f"<td>{esc(message)}</td>"
            "</tr>"
        )
    return (
        "".join(rows) or '<tr><td colspan="3">No retained preflight findings.</td></tr>'
    )


def backend_detail_sections(backends: dict[str, Any]) -> str:
    sections: list[str] = []
    classical = backends.get("classical")
    if isinstance(classical, dict):
        candidates = classical.get("candidates") or []
        rows = (
            "".join(
                "<tr>"
                f"<td>{esc(candidate.get('name'))}</td>"
                f"<td>{esc(candidate.get('family'))}</td>"
                f"<td>{badge(candidate.get('status'))}</td>"
                f"<td>{fmt_number(candidate.get('score'))}</td>"
                f"<td>{esc(candidate.get('consideration_basis'))}</td>"
                "</tr>"
                for candidate in candidates
                if isinstance(candidate, dict)
            )
            or '<tr><td colspan="5">No classical candidate ledger was retained.</td></tr>'
        )
        sections.append(
            '<article class="backend-detail">'
            "<h3>Classical baselines and leaderboard</h3>"
            f"<p><strong>Preprocessing:</strong> {esc(compact(classical.get('preprocessing')))}</p>"
            f"<p><strong>Search:</strong> {esc(compact(classical.get('search')))}</p>"
            '<div class="table-wrap"><table><thead><tr>'
            "<th>Candidate</th><th>Family</th><th>Status</th><th>Score</th>"
            f"<th>Consideration basis</th></tr></thead><tbody>{rows}</tbody></table></div>"
            "</article>"
        )

    autogluon = backends.get("autogluon")
    if isinstance(autogluon, dict):
        build = autogluon.get("build") or {}
        sections.append(
            '<article class="backend-detail">'
            "<h3>AutoGluon settings and leaderboard</h3>"
            f"<dl>{
                definition_rows(
                    [
                        ('Status', autogluon.get('status')),
                        ('Preset', build.get('preset')),
                        ('Time limit (seconds)', build.get('time_limit_seconds')),
                        ('Predictor path', build.get('predictor_path')),
                        ('Data handling', compact(autogluon.get('data_handling'))),
                        ('Evidence', compact(autogluon.get('evidence'))),
                    ]
                )
            }</dl>"
            "</article>"
        )

    sap_rpt = backends.get("sap_rpt")
    if isinstance(sap_rpt, dict):
        sections.append(
            '<article class="backend-detail">'
            "<h3>SAP RPT context, access, and latency</h3>"
            f"<dl>{
                definition_rows(
                    [
                        ('Status', sap_rpt.get('status')),
                        ('Model', compact(sap_rpt.get('model'))),
                        ('Access route', compact(sap_rpt.get('access'))),
                        ('Context policy', compact(sap_rpt.get('context'))),
                        (
                            'Transfer confirmation',
                            compact(sap_rpt.get('transfer_confirmation')),
                        ),
                        ('Latency and evidence', compact(sap_rpt.get('evidence'))),
                    ]
                )
            }</dl>"
            "</article>"
        )
    return "".join(sections) or "<p>No backend-specific details were recorded.</p>"


def render(run: dict[str, Any], results_markdown: str) -> str:
    problem = run.get("problem") or {}
    feature_contract = problem.get("feature_contract") or {}
    data = run.get("data") or {}
    preflight = run.get("modeling_preflight") or {}
    evaluation = run.get("evaluation") or {}
    primary_metric = evaluation.get("primary_metric") or {}
    approval = run.get("approval") or {}
    backends = run.get("backends") or {}
    selection = run.get("selection") or {}
    inference = run.get("inference") or {}
    lineage = run.get("lineage") or {}
    metric_name = str(primary_metric.get("name") or "primary metric")
    direction = str(primary_metric.get("direction") or "maximize")
    comparison_rows, _ = backend_rows(backends, metric_name)
    findings = preflight.get("findings") or []
    finding_rows = preflight_rows(findings)
    backend_details = backend_detail_sections(backends)

    inference_commands = inference.get("backends") or {}
    command_rows = (
        "".join(
            f"<tr><td>{esc(name)}</td><td><code>{esc(command)}</code></td></tr>"
            for name, command in inference_commands.items()
        )
        or '<tr><td colspan="2">No inference commands recorded.</td></tr>'
    )
    included_features = feature_contract.get("included") or []
    excluded_features = feature_contract.get("excluded") or []
    generated_at = datetime.now(UTC).isoformat()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML experiment report — {esc(run.get("run_id"))}</title>
<style>
:root {{
  color-scheme: light;
  --ink: #14213d; --muted: #5d6b82; --line: #dce3ed; --paper: #ffffff;
  --wash: #f5f7fb; --accent: #006b5e; --accent-soft: #dff3ef;
  --warning: #8a5200; --danger: #9a2938; --success: #166534;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 15px/1.55 Inter, ui-sans-serif, system-ui, sans-serif;
  color: var(--ink); background: var(--wash); }}
main {{ width: min(1120px, calc(100% - 32px)); margin: 32px auto 64px; }}
header {{ padding: 34px; border-radius: 18px; color: white;
  background: linear-gradient(125deg, #102a43, #006b5e); box-shadow: 0 12px 32px #102a4326; }}
h1 {{ margin: 6px 0 8px; font-size: clamp(28px, 4vw, 46px); line-height: 1.06; }}
h2 {{ margin: 0 0 18px; font-size: 22px; }}
h3 {{ margin: 0 0 8px; font-size: 16px; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: 12px; opacity: .8; }}
.lede {{ max-width: 760px; margin: 0; font-size: 17px; }}
section {{ margin-top: 22px; padding: 26px; border: 1px solid var(--line);
  border-radius: 16px; background: var(--paper); box-shadow: 0 4px 16px #14213d0a; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
.mini-card {{ padding: 16px; border: 1px solid var(--line); border-radius: 12px; background: #fbfcfe; }}
.mini-card p {{ margin: 5px 0; }}
.backend-detail + .backend-detail {{ margin-top: 24px; padding-top: 22px; border-top: 1px solid var(--line); }}
dl {{ display: grid; grid-template-columns: minmax(160px, .35fr) 1fr; gap: 8px 20px; margin: 0; }}
dt {{ color: var(--muted); }} dd {{ margin: 0; overflow-wrap: anywhere; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 11px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
.table-wrap {{ overflow-x: auto; }}
.badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #e8edf5;
  font-size: 12px; font-weight: 700; }}
.badge-completed, .badge-passed, .badge-approved {{ color: var(--success); background: #dcfce7; }}
.badge-confirmed {{ color: var(--success); background: #dcfce7; }}
.badge-warning, .badge-review-required, .badge-failed {{ color: var(--warning); background: #fef3c7; }}
.badge-blocker, .badge-blocked {{ color: var(--danger); background: #ffe4e6; }}
.badge-declined {{ color: var(--muted); background: #eef2f7; }}
code {{ padding: 2px 5px; border-radius: 5px; background: #eef2f7; white-space: pre-wrap; overflow-wrap: anywhere; }}
.score-chart {{ width: 100%; min-height: 130px; }}
.score-chart rect {{ fill: var(--accent); }}
.score-chart text {{ font: 13px ui-sans-serif, system-ui, sans-serif; fill: var(--ink); }}
.score-chart .chart-value {{ font-weight: 700; }}
.muted, footer {{ color: var(--muted); }}
.narrative h3, .narrative h4, .narrative h5 {{ margin: 20px 0 8px; }}
.narrative p, .narrative ul {{ margin: 8px 0; }}
footer {{ margin-top: 20px; text-align: center; font-size: 12px; }}
@media (max-width: 640px) {{
  main {{ width: min(100% - 18px, 1120px); margin-top: 9px; }}
  header, section {{ padding: 20px; border-radius: 12px; }}
  dl {{ grid-template-columns: 1fr; gap: 2px; }} dd {{ margin-bottom: 10px; }}
}}
@media print {{ body {{ background: white; }} main {{ width: 100%; margin: 0; }} section {{ box-shadow: none; break-inside: avoid; }} }}
</style>
</head>
<body>
<main>
<header>
  <div class="eyebrow">Consolidated machine-learning experiment</div>
  <h1>{esc(problem.get("target"))}</h1>
  <p class="lede">Run {
        esc(run.get("run_id"))
    } compares every user-approved backend on one evaluation contract and records one deployable recommendation.</p>
</header>

<section>
  <h2>Decision summary</h2>
  <div class="grid">
    <article class="mini-card"><h3>Predictive winner</h3><p>{
        esc(selection.get("predictive_winner"))
    }</p></article>
    <article class="mini-card"><h3>Operational recommendation</h3><p>{
        esc(selection.get("operational_recommendation"))
    }</p></article>
    <article class="mini-card"><h3>Primary metric</h3><p>{esc(metric_name)} · {
        esc(direction)
    }</p></article>
    <article class="mini-card"><h3>Default inference backend</h3><p>{
        esc(inference.get("default_backend"))
    }</p></article>
  </div>
  <p>{esc(selection.get("rationale"))}</p>
</section>

<section>
  <h2>Problem and data contract</h2>
  <dl>{
        definition_rows(
            [
                ("Task", problem.get("task")),
                ("Target", problem.get("target")),
                ("Prediction moment", problem.get("prediction_moment")),
                ("Row grain", problem.get("row_grain")),
                ("Data source", data.get("source")),
                ("Data fingerprint", data.get("fingerprint")),
                ("Rows", data.get("row_count")),
                ("Included features", ", ".join(map(str, included_features)) or "—"),
                ("Excluded features", ", ".join(map(str, excluded_features)) or "—"),
            ]
        )
    }</dl>
</section>

<section>
  <h2>Modeling preflight</h2>
  <p>Status: {badge(preflight.get("status"))}</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Severity</th><th>Check</th><th>Finding</th></tr></thead>
    <tbody>{finding_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>User approval and budgets</h2>
  <p>Approved at {
        esc(approval.get("approved_at"))
    }. The modeling scope and execution choices below were confirmed before work began.</p>
  <div class="grid">{approval_scope_cards(approval)}</div>
  <h3>Execution tracks</h3>
  <div class="grid">{approval_cards(approval)}</div>
</section>

<section>
  <h2>Evaluation contract</h2>
  <dl>{
        definition_rows(
            [
                ("Design", evaluation.get("design")),
                ("Split fingerprint", evaluation.get("split_fingerprint")),
                (
                    "Evaluation rows fingerprint",
                    evaluation.get("evaluation_rows_fingerprint"),
                ),
                ("Metric", metric_name),
                ("Direction", direction),
            ]
        )
    }</dl>
</section>

<section>
  <h2>Backend comparison</h2>
  <p class="muted">Bar length represents primary-metric desirability after applying the declared {
        esc(direction)
    } direction; longer is better.</p>
  {score_chart(backends, direction)}
  <div class="table-wrap"><table>
    <thead><tr><th>Backend</th><th>Status</th><th>{
        esc(metric_name)
    }</th><th>Evidence</th></tr></thead>
    <tbody>{comparison_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>Backend details</h2>
  {backend_details}
</section>

<section>
  <h2>Inference</h2>
  <dl>{
        definition_rows(
            [
                ("Entrypoint", inference.get("entrypoint")),
                ("Default backend", inference.get("default_backend")),
                ("Input format", (inference.get("input") or {}).get("format")),
                (
                    "Required input columns",
                    ", ".join(
                        map(
                            str,
                            (inference.get("input") or {}).get("required_columns")
                            or [],
                        )
                    )
                    or "—",
                ),
                (
                    "Optional input columns",
                    ", ".join(
                        map(
                            str,
                            (inference.get("input") or {}).get("optional_columns")
                            or [],
                        )
                    )
                    or "—",
                ),
                ("Input dtypes", compact((inference.get("input") or {}).get("dtypes"))),
                (
                    "Missing-value policy",
                    (inference.get("input") or {}).get("missing_value_policy"),
                ),
                (
                    "Extra-column policy",
                    (inference.get("input") or {}).get("extra_column_policy"),
                ),
                (
                    "Target excluded from input",
                    (inference.get("input") or {}).get("target_column"),
                ),
                (
                    "Identifier columns",
                    ", ".join(
                        map(
                            str,
                            (inference.get("input") or {}).get("identifier_columns")
                            or [],
                        )
                    )
                    or "—",
                ),
                (
                    "Feature order",
                    ", ".join(
                        map(
                            str,
                            (inference.get("input") or {}).get("feature_order") or [],
                        )
                    )
                    or "—",
                ),
                ("Output format", (inference.get("output") or {}).get("format")),
                (
                    "Prediction column",
                    (inference.get("output") or {}).get("prediction_column"),
                ),
                (
                    "Row ID column",
                    (inference.get("output") or {}).get("row_id_column"),
                ),
                (
                    "Probability columns",
                    ", ".join(
                        map(
                            str,
                            (inference.get("output") or {}).get("probability_columns")
                            or [],
                        )
                    )
                    or "—",
                ),
                (
                    "Probability bounds",
                    compact((inference.get("output") or {}).get("probability_bounds")),
                ),
                (
                    "Finite values required",
                    (inference.get("output") or {}).get("finite_values"),
                ),
            ]
        )
    }</dl>
  <div class="table-wrap"><table>
    <thead><tr><th>Backend</th><th>Command</th></tr></thead>
    <tbody>{command_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>Uncertainty and limitations</h2>
  <dl>{
        definition_rows(
            [
                ("Uncertainty", compact(evaluation.get("uncertainty"))),
                (
                    "Limitations",
                    compact(selection.get("limitations") or lineage.get("notes")),
                ),
            ]
        )
    }</dl>
</section>

<section>
  <h2>Intended use, prohibited use, and monitoring</h2>
  <dl>{
        definition_rows(
            [
                ("Intended use", problem.get("intended_use")),
                ("Prohibited use", compact(problem.get("prohibited_uses"))),
                ("Monitoring", compact(selection.get("monitoring"))),
            ]
        )
    }</dl>
</section>

<section>
  <h2>Results narrative</h2>
  <div class="narrative">{markdown_summary(results_markdown)}</div>
</section>

<section>
  <h2>Lineage</h2>
  <dl>{
        definition_rows(
            [
                ("Source data fingerprint", lineage.get("source_data_fingerprint")),
                ("Parent run", lineage.get("parent_run_id")),
                ("Notes", "; ".join(map(str, lineage.get("notes") or [])) or "—"),
            ]
        )
    }</dl>
</section>

<footer>Generated {
        esc(generated_at)
    } · This file contains all styles and visuals inline.</footer>
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    run_json_path = (args.run_json or run_dir / "run.json").expanduser().resolve()
    results_path = (args.results or run_dir / "results.md").expanduser().resolve()
    output_path = (args.output or run_dir / "report.html").expanduser().resolve()
    try:
        run = json.loads(run_json_path.read_text(encoding="utf-8"))
        results_markdown = results_path.read_text(encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render(run, results_markdown), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Unable to render report: {exc}", file=sys.stderr)
        return 2
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
