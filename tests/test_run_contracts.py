from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
VALIDATOR = REPOSITORY / "skills/ml-model-builder/scripts/validate_run.py"
REPORT_RENDERER = REPOSITORY / "skills/ml-model-builder/scripts/render_report.py"
RUN_RELATIVE = Path("artefacts/runs/test-run")
FINGERPRINTS = {
    "data": "sha256:" + "a" * 64,
    "split": "sha256:" + "b" * 64,
    "rows": "sha256:" + "c" * 64,
    "context": "sha256:" + "d" * 64,
}
SCORES = {
    "classical": 0.72,
    "autogluon": 0.79,
    "sap_rpt": 0.83,
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_validator(
    project: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(project),
            "--artifacts-dir",
            RUN_RELATIVE.as_posix(),
            *extra,
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


def inference_script(*, fault: str | None = None) -> str:
    fault_literal = repr(fault)
    return f"""\
import argparse
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    required=True,
    choices=["classical", "autogluon", "sap_rpt"],
)
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.input, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    source_columns = set(reader.fieldnames or [])
    source_rows = list(reader)
required = {{"row_id", "fixed_acidity", "alcohol"}}
missing = sorted(required - source_columns)
if missing:
    print("missing required columns: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)

backend_probability = {{
    "classical": 0.72,
    "autogluon": 0.79,
    "sap_rpt": 0.83,
}}[args.backend]
rows = []
for row in source_rows:
    probability_good = backend_probability
    rows.append(
        {{
            "row_id": row["row_id"],
            "prediction": "good" if probability_good >= 0.5 else "bad",
            "probability_bad": 1.0 - probability_good,
            "probability_good": probability_good,
        }}
    )

fault = {fault_literal}
if fault == "misaligned_rows":
    rows.reverse()
elif fault == "nonfinite_probability" and rows:
    rows[0]["probability_good"] = "nan"
elif fault == "out_of_bounds_probability" and rows:
    rows[0]["probability_good"] = 1.2
elif fault == "nondeterministic_probability" and rows and "output-1." in args.output:
    rows[0]["probability_good"] = backend_probability - 0.01
elif fault == "omit_prediction":
    for row in rows:
        del row["prediction"]

fieldnames = [
    "row_id",
    "probability_bad",
    "probability_good",
]
if fault != "omit_prediction":
    fieldnames.insert(1, "prediction")
with open(args.output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
"""


def track_approval(backend: str, selected: bool) -> dict[str, Any]:
    if not selected:
        return {"selected": False, "status": "declined", "budget": None}
    shared = {
        "cpu_count": 4,
        "memory_gb": 8,
        "parallel_jobs": 2,
        "gpu_enabled": False,
    }
    budgets = {
        "classical": {
            **shared,
            "candidate_families": ["linear", "gradient_boosting"],
            "time_limit_seconds": 600,
            "optuna_trials": 12,
            "minimum_family_coverage": 2,
        },
        "autogluon": {
            **shared,
            "preset": "best_quality",
            "time_limit_seconds": 600,
            "disk_gb": 16,
        },
        "sap_rpt": {
            **shared,
            "max_requests": 20,
            "max_context_rows": 512,
            "max_query_rows": 320,
            "max_rows_per_request": 64,
            "max_retries": 2,
            "timeout_seconds": 120,
        },
    }
    return {
        "selected": True,
        "status": "approved",
        "budget": budgets[backend],
    }


def backend_evaluation(backend: str) -> dict[str, Any]:
    return {
        "split_fingerprint": FINGERPRINTS["split"],
        "evaluation_rows_fingerprint": FINGERPRINTS["rows"],
        "primary_metric": "macro_f1",
        "score": SCORES[backend],
    }


def create_backend(run_dir: Path, backend: str) -> dict[str, Any]:
    backend_dir = run_dir / "backends" / backend
    backend_dir.mkdir(parents=True)
    common = {
        "status": "completed",
        "retained": True,
        "evaluation": backend_evaluation(backend),
        "evidence": {
            "result_source": "shared_evaluation_rows",
            "rows_scored": 320,
        },
    }
    if backend == "classical":
        (backend_dir / "model.joblib").write_bytes(b"test model")
        return {
            **common,
            "preprocessing": {
                "scope": "fold_local",
                "steps": ["median_imputation", "standardization"],
            },
            "search": {
                "method": "optuna",
                "trials_budget": 12,
                "trials_completed": 12,
            },
            "candidates": [
                {
                    "name": "logistic_baseline",
                    "family": "linear",
                    "consideration_basis": "fixed interpretable baseline",
                    "status": "completed",
                    "score": 0.61,
                },
                {
                    "name": "xgboost",
                    "family": "gradient_boosting",
                    "consideration_basis": (
                        "nonlinear tabular candidate within the approved runtime"
                    ),
                    "status": "completed",
                    "score": 0.72,
                },
            ],
            "artifacts": {"model": "backends/classical/model.joblib"},
        }
    if backend == "autogluon":
        predictor = backend_dir / "predictor"
        predictor.mkdir()
        (predictor / "predictor.txt").write_text("placeholder", encoding="utf-8")
        return {
            **common,
            "build": {
                "preset": "best_quality",
                "time_limit_seconds": 600,
                "predictor_path": "backends/autogluon/predictor",
            },
            "data_handling": {
                "raw_tabular": True,
                "external_preprocessing": False,
                "external_optuna": False,
            },
        }
    write_json(
        backend_dir / "context_manifest.json",
        {
            "policy": "frozen labelled context from each evaluation training fold",
            "target": "quality_class",
        },
    )
    return {
        **common,
        "model": {
            "name": "SAP RPT",
            "version": "2026-07",
            "production_capable": True,
        },
        "access": {
            "route": "internal_cli",
            "client": "sap-rpt",
            "customer_production_route": "sap_ai_core",
        },
        "context": {
            "manifest": "backends/sap_rpt/context_manifest.json",
            "fingerprint": FINGERPRINTS["context"],
            "policy": "frozen labelled context reconstructed for inference",
        },
        "transfer_confirmation": {
            "schema_validated": True,
            "labels_validated": True,
            "query_rows_excluded_from_context": True,
        },
    }


def inference_case(backend: str, kind: str) -> dict[str, Any]:
    columns = ["row_id", "fixed_acidity", "alcohol"]
    rows = [
        {"row_id": "wine-1", "fixed_acidity": 7.1, "alcohol": 10.2},
        {"row_id": "wine-2", "fixed_acidity": 6.8, "alcohol": 11.1},
    ]
    if kind == "single_row":
        rows = rows[:1]
    elif kind == "empty_input":
        rows = []
    elif kind == "missing_required_column":
        columns = ["row_id", "fixed_acidity"]
        rows = [{"row_id": "wine-1", "fixed_acidity": 7.1}]
    successful = kind != "missing_required_column"
    value = {
        "name": f"{backend}-{kind}",
        "kind": kind,
        "backend": backend,
        "argv": [
            "{python}",
            "infer.py",
            "--backend",
            backend,
            "--input",
            "{input}",
            "--output",
            "{output}",
        ],
        "input": {
            "format": "csv",
            "columns": columns,
            "rows": rows,
        },
        "repeat_runs": 2 if kind in {"representative", "single_row"} else 1,
        "expect": (
            {
                "exit_code": 0,
                "output": {
                    "format": "csv",
                    "required_columns": [
                        "row_id",
                        "prediction",
                        "probability_bad",
                        "probability_good",
                    ],
                    "min_rows": len(rows),
                    "max_rows": len(rows),
                },
            }
            if successful
            else {
                "exit_code": 2,
                "stderr_contains": "missing required columns",
            }
        ),
    }
    return value


def handoff_body(selected: tuple[str, ...], winner: str) -> str:
    lines = [
        "Primary metric: macro F1.",
        "Predictive winner: " + winner.replace("_", " ") + ".",
        "Operational recommendation: " + winner.replace("_", " ") + ".",
        "Inference command: python infer.py --input new_wines.csv --output "
        + "predictions.csv.",
        "Intended use: offline wine quality classification experiments.",
        "Prohibited use: not for safety-critical or regulatory decisions.",
        "Limitations: this is a small public benchmark dataset.",
        "Uncertainty: repeated-fold variation accompanies every point estimate.",
        "Monitoring: monitor schema drift, feature drift, and outcome quality.",
    ]
    for backend in selected:
        lines.append(
            f"{backend.replace('_', ' ')} status: completed; "
            f"macro F1 score: {SCORES[backend]}."
        )
    if "classical" in selected:
        lines.append(
            "Classical baseline: logistic regression. "
            "Classical leaderboard: XGBoost ranked first."
        )
    if "autogluon" in selected:
        lines.append("AutoGluon preset: best quality.")
    if "sap_rpt" in selected:
        lines.append(
            "SAP RPT context: frozen labelled rows. "
            "Access: internal CLI. Latency: 120 ms per request."
        )
    return "\n".join(lines)


def build_project(
    project: Path,
    *,
    selected: tuple[str, ...] = ("classical",),
    winner: str | None = None,
    inference_fault: str | None = None,
) -> Path:
    run_dir = project / RUN_RELATIVE
    run_dir.mkdir(parents=True)
    backends = {backend: create_backend(run_dir, backend) for backend in selected}
    winner = winner or max(selected, key=SCORES.__getitem__)
    tracks = {
        backend: track_approval(backend, backend in selected)
        for backend in ("classical", "autogluon", "sap_rpt")
    }
    run_document = {
        "run_id": "test-run",
        "created_at": "2026-07-29T08:00:00+08:00",
        "problem": {
            "task": "classification",
            "target": "quality_class",
            "prediction_moment": "after laboratory measurements are available",
            "row_grain": "one row per wine sample",
            "intended_use": "offline wine quality classification experiments",
            "prohibited_uses": ["safety-critical or regulatory decisions"],
            "feature_contract": {
                "included": ["fixed_acidity", "alcohol"],
                "excluded": ["quality_class", "sample_id"],
            },
        },
        "data": {
            "source": "winequality-red.csv",
            "fingerprint": FINGERPRINTS["data"],
            "row_count": 1599,
        },
        "modeling_preflight": {
            "status": "passed",
            "target_validated": True,
            "row_grain_validated": True,
            "prediction_moment_validated": True,
            "leakage_reviewed": True,
            "feature_availability_reviewed": True,
            "split_suitable": True,
            "findings": [
                "Stratification preserves every observed quality class.",
            ],
        },
        "evaluation": {
            "design": "repeated_stratified_cross_validation",
            "split_fingerprint": FINGERPRINTS["split"],
            "evaluation_rows_fingerprint": FINGERPRINTS["rows"],
            "primary_metric": {
                "name": "macro_f1",
                "direction": "maximize",
            },
        },
        "approval": {
            "approved_at": "2026-07-29T08:05:00+08:00",
            "scope": {
                "target": True,
                "feature_contract": True,
                "split_design": True,
                "primary_metric": True,
            },
            "tracks": tracks,
        },
        "backends": backends,
        "selection": {
            "predictive_winner": winner,
            "operational_recommendation": winner,
            "rationale": "Highest macro F1 on the shared evaluation rows.",
            "primary_metric": "macro_f1",
        },
        "inference": {
            "entrypoint": "infer.py",
            "default_backend": winner,
            "input": {
                "format": "csv",
                "required_columns": ["row_id", "fixed_acidity", "alcohol"],
                "optional_columns": ["sulphates"],
                "dtypes": {
                    "row_id": "string",
                    "fixed_acidity": "float64",
                    "alcohol": "float64",
                    "sulphates": "float64",
                },
                "missing_value_policy": (
                    "reject missing required values; allow optional missing values"
                ),
                "extra_column_policy": "reject",
                "target_column": "quality_class",
                "identifier_columns": ["row_id"],
                "feature_order": ["fixed_acidity", "alcohol", "sulphates"],
            },
            "output": {
                "format": "csv",
                "prediction_column": "prediction",
                "probability_columns": [
                    "probability_bad",
                    "probability_good",
                ],
                "row_id_column": "row_id",
                "finite_values": True,
                "probability_bounds": [0, 1],
            },
            "backends": {
                backend: (
                    "python infer.py --backend "
                    f"{backend} --input new_wines.csv --output predictions.csv"
                )
                for backend in selected
            },
        },
        "lineage": {
            "source_data_fingerprint": FINGERPRINTS["data"],
            "parent_run_id": None,
            "notes": [],
        },
    }
    write_json(run_dir / "run.json", run_document)
    handoff = handoff_body(selected, winner)
    (run_dir / "report.html").write_text(
        f"""<!doctype html>
<html>
<head><meta charset="utf-8"><style>body {{ color: #222; }}</style></head>
<body>
<h1>Wine classifier report</h1>
<pre>{handoff}</pre>
<svg viewBox="0 0 10 10"><rect width="8" height="8"></rect></svg>
</body>
</html>
""",
        encoding="utf-8",
    )
    (run_dir / "results.md").write_text(
        "# Results\n\n" + handoff + "\n",
        encoding="utf-8",
    )
    (run_dir / "infer.py").write_text(
        inference_script(fault=inference_fault),
        encoding="utf-8",
    )
    (run_dir / "requirements.lock").write_text(
        "pandas==2.3.1\nscikit-learn==1.7.1\n",
        encoding="utf-8",
    )
    if any(backend in selected for backend in ("classical", "autogluon")):
        (run_dir / "train.py").write_text(
            "# Rebuilds only selected build-based backends.\n",
            encoding="utf-8",
        )
    write_json(
        run_dir / "validation.json",
        {
            "status": "passed",
            "validated_at": "2026-07-29T08:30:00+08:00",
            "inference_cases": [
                inference_case(backend, kind)
                for backend in selected
                for kind in (
                    "representative",
                    "single_row",
                    "empty_input",
                    "missing_required_column",
                )
            ],
        },
    )
    return run_dir


class ValidRunTests(unittest.TestCase):
    def test_valid_classical_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_project(project, selected=("classical",))
            completed = run_validator(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_valid_autogluon_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("autogluon",))
            completed = run_validator(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            run_document = read_json(run_dir / "run.json")
            self.assertNotIn("search", run_document["backends"]["autogluon"])
            self.assertNotIn("preprocessing", run_document["backends"]["autogluon"])

    def test_valid_rpt_only_run_has_no_train_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("sap_rpt",))
            self.assertFalse((run_dir / "train.py").exists())
            completed = run_validator(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_all_tracks_share_one_run_and_rpt_can_win(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_project(
                project,
                selected=("classical", "autogluon", "sap_rpt"),
                winner="sap_rpt",
            )
            completed = run_validator(project, "--run-inference-test")
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("temporary inputs/outputs", completed.stdout)

    def test_bundled_renderer_creates_a_valid_inclusive_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(
                project,
                selected=("classical", "autogluon", "sap_rpt"),
                winner="sap_rpt",
            )
            rendered = subprocess.run(
                [sys.executable, str(REPORT_RENDERER), str(run_dir)],
                cwd=REPOSITORY,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            completed = run_validator(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)


class ApprovalAndBackendSemanticsTests(unittest.TestCase):
    def test_missing_approval_timestamp_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            document = read_json(run_dir / "run.json")
            del document["approval"]["approved_at"]
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("approval.approved_at", completed.stdout)

    def test_approval_must_confirm_the_complete_execution_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            document = read_json(run_dir / "run.json")
            document["approval"]["scope"]["feature_contract"] = False
            del document["approval"]["scope"]["primary_metric"]
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "approval.scope.feature_contract must be true",
                completed.stdout,
            )
            self.assertIn(
                "approval.scope.primary_metric must be true",
                completed.stdout,
            )

    def test_unapproved_backend_execution_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("classical",))
            document = read_json(run_dir / "run.json")
            document["backends"]["autogluon"] = create_backend(
                run_dir,
                "autogluon",
            )
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("execution evidence exists for unapproved", completed.stdout)
            self.assertIn("directory exists for an unapproved", completed.stdout)

    def test_approved_backend_must_record_status_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("classical", "autogluon"))
            document = read_json(run_dir / "run.json")
            del document["backends"]["autogluon"]
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "approved tracks must record backend status/evidence",
                completed.stdout,
            )

    def test_classical_candidate_ledger_requires_unique_explained_entries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            document = read_json(run_dir / "run.json")
            candidates = document["backends"]["classical"]["candidates"]
            candidates[1]["name"] = candidates[0]["name"]
            candidates[0]["consideration_basis"] = ""
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("consideration_basis must be non-empty", completed.stdout)
            self.assertIn("candidate names must be unique", completed.stdout)

    def test_selected_track_budgets_require_resource_and_backend_controls(
        self,
    ) -> None:
        mutations = (
            ("classical", "cpu_count", "cpu_count must be a positive integer"),
            (
                "classical",
                "candidate_families",
                "candidate_families must be a non-empty unique string list",
            ),
            ("autogluon", "preset", "preset must be a non-empty string"),
            ("autogluon", "disk_gb", "disk_gb must be a positive finite number"),
            ("sap_rpt", "max_query_rows", "max_query_rows must be a positive integer"),
            (
                "sap_rpt",
                "timeout_seconds",
                "timeout_seconds must be a positive integer",
            ),
        )
        for backend, field, expected in mutations:
            with (
                self.subTest(backend=backend, field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                project = Path(directory)
                run_dir = build_project(project, selected=(backend,))
                document = read_json(run_dir / "run.json")
                del document["approval"]["tracks"][backend]["budget"][field]
                write_json(run_dir / "run.json", document)
                completed = run_validator(project)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected, completed.stdout)

    def test_autogluon_rejects_external_optuna_and_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("autogluon",))
            document = read_json(run_dir / "run.json")
            backend = document["backends"]["autogluon"]
            backend["data_handling"]["external_optuna"] = True
            backend["preprocessing"] = {"scope": "fold_local"}
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("external_optuna must be false", completed.stdout)
            self.assertIn("must not declare classical preprocessing", completed.stdout)

    def test_rpt_rejects_training_and_search_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("sap_rpt",))
            document = read_json(run_dir / "run.json")
            backend = document["backends"]["sap_rpt"]
            backend["training"] = {
                "fit": True,
                "hyperparameters": {"trials": 20},
            }
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("SAP RPT is pretrained", completed.stdout)

    def test_rpt_requires_context_access_and_transfer_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("sap_rpt",))
            document = read_json(run_dir / "run.json")
            backend = document["backends"]["sap_rpt"]
            del backend["access"]
            backend["transfer_confirmation"]["query_rows_excluded_from_context"] = False
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("backends.sap_rpt.access", completed.stdout)
            self.assertIn(
                "query_rows_excluded_from_context must be true",
                completed.stdout,
            )

    def test_completed_tracks_must_use_shared_folds_and_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(
                project,
                selected=("classical", "autogluon"),
            )
            document = read_json(run_dir / "run.json")
            evaluation = document["backends"]["autogluon"]["evaluation"]
            evaluation["split_fingerprint"] = "sha256:" + "f" * 64
            evaluation["primary_metric"] = "accuracy"
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("must match the shared evaluation contract", completed.stdout)
            self.assertIn("must match evaluation.primary_metric.name", completed.stdout)

    def test_default_inference_backend_is_operational_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(
                project,
                selected=("classical", "sap_rpt"),
                winner="sap_rpt",
            )
            document = read_json(run_dir / "run.json")
            document["inference"]["default_backend"] = "classical"
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "default_backend must match selection.operational_recommendation",
                completed.stdout,
            )

    def test_classical_optuna_trials_cannot_exceed_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            document = read_json(run_dir / "run.json")
            document["backends"]["classical"]["search"]["trials_completed"] = 13
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cannot exceed trials_budget", completed.stdout)

    def test_rpt_only_run_rejects_train_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("sap_rpt",))
            (run_dir / "train.py").write_text("pass\n", encoding="utf-8")
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("SAP RPT-only run", completed.stdout)

    def test_build_backend_requires_train_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("autogluon",))
            (run_dir / "train.py").unlink()
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing train.py", completed.stdout)


class MinimalArtifactTests(unittest.TestCase):
    def test_report_and_results_must_include_every_approved_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(
                project,
                selected=("classical", "autogluon", "sap_rpt"),
            )
            report_path = run_dir / "report.html"
            report_path.write_text(
                report_path.read_text(encoding="utf-8")
                .replace("SAP RPT", "omitted")
                .replace("sap rpt", "omitted"),
                encoding="utf-8",
            )
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "report.html: must include the approved sap rpt result or status",
                completed.stdout,
            )

    def test_report_and_results_require_operational_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            for filename in ("report.html", "results.md"):
                path = run_dir / filename
                path.write_text(
                    path.read_text(encoding="utf-8").replace(
                        "Monitoring: monitor schema drift, feature drift, "
                        "and outcome quality.",
                        "Operations are documented.",
                    ),
                    encoding="utf-8",
                )
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("report.html: must discuss monitoring", completed.stdout)
            self.assertIn("results.md: must discuss monitoring", completed.stdout)

    def test_track_specific_handoff_details_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(
                project,
                selected=("classical", "autogluon", "sap_rpt"),
            )
            replacements = {
                "Classical baseline": "Classical reference",
                "Classical leaderboard": "Classical comparison",
                "AutoGluon preset": "AutoGluon setting",
                "Latency": "Timing",
            }
            for filename in ("report.html", "results.md"):
                path = run_dir / filename
                source = path.read_text(encoding="utf-8")
                for original, replacement in replacements.items():
                    source = source.replace(original, replacement)
                path.write_text(source, encoding="utf-8")
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("classical baseline", completed.stdout)
            self.assertIn("classical leaderboard", completed.stdout)
            self.assertIn("AutoGluon preset", completed.stdout)
            self.assertIn("SAP RPT latency", completed.stdout)

    def test_report_must_not_load_external_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            report_path = run_dir / "report.html"
            report_path.write_text(
                report_path.read_text(encoding="utf-8").replace(
                    "</body>",
                    '<script src="charts.js"></script></body>',
                ),
                encoding="utf-8",
            )
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("must be self-contained", completed.stdout)
            self.assertIn("charts.js", completed.stdout)

    def test_report_accepts_embedded_data_uri_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            report_path = run_dir / "report.html"
            report_path.write_text(
                report_path.read_text(encoding="utf-8").replace(
                    "</body>",
                    '<img src="data:image/png;base64,AA=="></body>',
                ),
                encoding="utf-8",
            )
            completed = run_validator(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_forbidden_eda_and_generated_output_clutter_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            (run_dir / "figures").mkdir()
            (run_dir / "data_summary.md").write_text("EDA", encoding="utf-8")
            (run_dir / "inference_outputs").mkdir()
            (run_dir / "backends/classical/__pycache__").mkdir()
            (run_dir / "backends/classical/diagnostics").mkdir()
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "forbidden run artifact or directory: figures", completed.stdout
            )
            self.assertIn("data_summary.md", completed.stdout)
            self.assertIn("inference_outputs", completed.stdout)
            self.assertIn("__pycache__", completed.stdout)
            self.assertIn("diagnostics", completed.stdout)

    def test_parent_duplication_declaration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            document = read_json(run_dir / "run.json")
            document["lineage"]["duplicated_parent_files"] = ["model.joblib"]
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("parent artifacts must not be copied", completed.stdout)

    def test_requirements_must_be_exactly_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            (run_dir / "requirements.lock").write_text(
                "pandas>=2\n",
                encoding="utf-8",
            )
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exact 'package==version' pin", completed.stdout)

    def test_standard_library_only_run_can_declare_no_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("sap_rpt",))
            (run_dir / "requirements.lock").write_text(
                "# No third-party dependencies\n",
                encoding="utf-8",
            )
            completed = run_validator(project)
            self.assertEqual(completed.returncode, 0, completed.stdout)


class InferenceExecutionTests(unittest.TestCase):
    def test_declared_inference_runs_with_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("sap_rpt",))
            completed = run_validator(project, "--run-inference-test")
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertFalse((run_dir / "inference_outputs").exists())
            self.assertEqual(
                sorted(path.name for path in run_dir.iterdir()),
                [
                    "backends",
                    "infer.py",
                    "report.html",
                    "requirements.lock",
                    "results.md",
                    "run.json",
                    "validation.json",
                ],
            )

    def test_actual_inference_output_columns_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_project(
                project,
                selected=("classical",),
                inference_fault="omit_prediction",
            )
            completed = run_validator(project, "--run-inference-test")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("output is missing columns: prediction", completed.stdout)

    def test_case_argv_must_dispatch_to_its_declared_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(
                project,
                selected=("classical", "autogluon"),
            )
            validation = read_json(run_dir / "validation.json")
            autogluon_case = next(
                case
                for case in validation["inference_cases"]
                if case["backend"] == "autogluon"
            )
            backend_position = autogluon_case["argv"].index("--backend") + 1
            autogluon_case["argv"][backend_position] = "classical"
            write_json(run_dir / "validation.json", validation)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "must select its declared backend with '--backend autogluon'",
                completed.stdout,
            )

    def test_each_backend_requires_all_edge_case_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project, selected=("sap_rpt",))
            validation = read_json(run_dir / "validation.json")
            validation["inference_cases"] = [
                case
                for case in validation["inference_cases"]
                if case["kind"] != "empty_input"
            ]
            write_json(run_dir / "validation.json", validation)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "missing required inference case kinds: empty_input",
                completed.stdout,
            )

    def test_rich_inference_contract_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            document = read_json(run_dir / "run.json")
            contract = document["inference"]
            del contract["input"]["dtypes"]["alcohol"]
            contract["input"]["required_columns"].append("quality_class")
            contract["output"]["probability_bounds"] = [-0.1, 1]
            write_json(run_dir / "run.json", document)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "dtypes must define a non-empty dtype for every",
                completed.stdout,
            )
            self.assertIn("inference input must exclude the target", completed.stdout)
            self.assertIn("probability_bounds must be [0, 1]", completed.stdout)

    def test_actual_inference_preserves_row_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_project(
                project,
                selected=("classical",),
                inference_fault="misaligned_rows",
            )
            completed = run_validator(project, "--run-inference-test")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "did not preserve input row identifier alignment", completed.stdout
            )

    def test_actual_probabilities_are_finite_and_bounded(self) -> None:
        mutations = (
            ("nonfinite_probability", "is not finite"),
            ("out_of_bounds_probability", "is outside [0, 1]"),
        )
        for fault, expected in mutations:
            with (
                self.subTest(fault=fault),
                tempfile.TemporaryDirectory() as directory,
            ):
                project = Path(directory)
                build_project(
                    project,
                    selected=("autogluon",),
                    inference_fault=fault,
                )
                completed = run_validator(project, "--run-inference-test")
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected, completed.stdout)

    def test_repeated_inference_must_be_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            build_project(
                project,
                selected=("sap_rpt",),
                inference_fault="nondeterministic_probability",
            )
            completed = run_validator(project, "--run-inference-test")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("is not deterministic across 2 repeat runs", completed.stdout)

    def test_inference_case_must_embed_input_and_use_temp_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run_dir = build_project(project)
            validation = read_json(run_dir / "validation.json")
            case = validation["inference_cases"][0]
            case["argv"] = [sys.executable, "infer.py", "--input", "fixture.csv"]
            del case["input"]
            write_json(run_dir / "validation.json", validation)
            completed = run_validator(project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("temporary {input} placeholder", completed.stdout)
            self.assertIn("temporary {output} placeholder", completed.stdout)
            self.assertIn("input must define inline data", completed.stdout)


if __name__ == "__main__":
    unittest.main()
