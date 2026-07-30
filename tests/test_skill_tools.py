from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "skills" / "ml-model-builder" / "scripts"
INSPECT_SCRIPT = SCRIPTS / "inspect_model_data.py"
REPORT_SCRIPT = SCRIPTS / "render_report.py"


class ApprovalGuidanceTests(unittest.TestCase):
    def test_rpt_uses_one_structured_approval_without_magic_words(self) -> None:
        skill = " ".join(
            (REPO_ROOT / "skills/ml-model-builder/SKILL.md")
            .read_text(encoding="utf-8")
            .lower()
            .split()
        )
        governance = " ".join(
            (REPO_ROOT / "skills/ml-model-builder/references/governance.md")
            .read_text(encoding="utf-8")
            .lower()
            .split()
        )
        sap_rpt = " ".join(
            (REPO_ROOT / "skills/ml-model-builder/references/sap-rpt.md")
            .read_text(encoding="utf-8")
            .lower()
            .split()
        )

        self.assertIn("native structured question tool", skill)
        self.assertIn("never require the user to type an exact sentence", skill)
        self.assertIn("do not ask for a second rpt confirmation", skill)
        self.assertIn(
            "collect every foreseeable blocking decision into this one "
            "structured question invocation",
            skill,
        )
        self.assertIn("default autogluon to `run_to_completion`", skill)
        self.assertIn("without routine follow-up questions", skill)
        self.assertIn("do not obtain a second confirmation", governance)
        self.assertIn("one structured question invocation", governance)
        self.assertIn("do not ask a second rpt-specific confirmation", sap_rpt)
        self.assertNotIn("obtain a second explicit confirmation", governance)


class ModelingPreflightTests(unittest.TestCase):
    def run_inspection(
        self, dataset: Path, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(INSPECT_SCRIPT),
                str(dataset),
                "--target",
                "quality",
                "--task",
                "classification",
                "--row-grain",
                "one tested wine batch",
                "--prediction-moment",
                "after laboratory measurements are available",
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_preflight_prints_modeling_assessment_without_creating_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "wine.csv"
            pd.DataFrame(
                {
                    "fixed_acidity": [7.0, 7.4, 7.8, 6.9, 7.2, 7.6],
                    "volatile_acidity": [0.3, 0.5, 0.4, 0.2, 0.6, 0.35],
                    "quality": [5, 6, 5, 7, 6, 7],
                }
            ).to_csv(dataset, index=False)
            before = sorted(path.name for path in root.iterdir())

            result = self.run_inspection(dataset)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["problem"]["target"], "quality")
            self.assertEqual(
                payload["problem"]["prediction_moment"],
                "after laboratory measurements are available",
            )
            self.assertTrue(payload["source"]["fingerprint"].startswith("sha256:"))
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                before,
                "Modeling preflight must not create EDA or run artifacts",
            )
            serialized = json.dumps(payload).lower()
            self.assertNotIn("chart", serialized)
            self.assertNotIn("figure", serialized)
            self.assertNotIn("html", serialized)

    def test_preflight_blocks_a_direct_target_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "leaky.csv"
            pd.DataFrame(
                {
                    "feature": [10, 20, 30, 40],
                    "quality_copy": [5, 6, 5, 6],
                    "quality": [5, 6, 5, 6],
                }
            ).to_csv(dataset, index=False)

            result = self.run_inspection(dataset)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["modeling_preflight"]["status"], "blocked")
            codes = {
                finding["code"] for finding in payload["modeling_preflight"]["findings"]
            }
            self.assertIn("direct_target_copy", codes)

    def test_preflight_rejects_unknown_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "data.csv"
            pd.DataFrame({"feature": [1, 2, 3]}).to_csv(dataset, index=False)

            result = self.run_inspection(dataset)

            self.assertEqual(result.returncode, 2)
            self.assertIn("Target column not found", result.stderr)

    def test_missing_runtime_dependencies_fail_cleanly_without_installing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "data.csv"
            dataset.write_text("feature,quality\n1,0\n2,1\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "-S", str(INSPECT_SCRIPT), str(dataset)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn(
                "interpreter where pandas and numpy already exist", result.stderr
            )
            self.assertIn("do not install dependencies", result.stderr)

    def test_unique_continuous_measure_is_not_excluded_as_an_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "continuous.csv"
            pd.DataFrame(
                {
                    "fixed_acidity": [
                        6.0 + index * 0.137 + (index % 3) * 0.011 for index in range(24)
                    ],
                    "quality": [5, 6] * 12,
                }
            ).to_csv(dataset, index=False)

            result = self.run_inspection(dataset)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            contract = payload["problem"]["feature_contract"]
            self.assertIn("fixed_acidity", contract["included"])
            self.assertNotIn("fixed_acidity", contract["auto_excluded_identifiers"])

    def test_preflight_finds_duplicates_and_group_leakage_hidden_by_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "wine_pipeline.csv"
            rows = [
                {
                    "sample_id": f"wine-{fold}-{quality}-{replicate}",
                    "fold": fold,
                    "duplicate_group": f"quality-{quality}",
                    "fixed_acidity": acidity,
                    "volatile_acidity": volatile,
                    "quality": quality,
                }
                for fold in (0, 1)
                for quality, acidity, volatile in (
                    (5, 7.0, 0.30),
                    (6, 7.0, 0.30),
                )
                for replicate in (0, 1)
            ]
            pd.DataFrame(rows).to_csv(dataset, index=False)

            result = self.run_inspection(dataset, "--group-column", "duplicate_group")

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            duplicates = payload["dataset"]["duplicates"]
            self.assertEqual(duplicates["exact_rows"], 0)
            self.assertGreater(duplicates["substantive_rows"], 0)
            self.assertIn("sample_id", duplicates["excluded_comparison_columns"])
            self.assertIn("fold", duplicates["excluded_comparison_columns"])
            self.assertIn("duplicate_group", duplicates["excluded_comparison_columns"])

            contract = payload["problem"]["feature_contract"]
            self.assertEqual(
                contract["included"], ["fixed_acidity", "volatile_acidity"]
            )
            self.assertIn("sample_id", contract["auto_excluded_identifiers"])
            self.assertIn("fold", contract["auto_excluded_pipeline_metadata"])

            group = payload["split_context"]["group"]
            self.assertTrue(group["target_is_function_of_group"])
            self.assertGreater(group["identical_feature_signatures_across_groups"], 0)
            self.assertGreater(group["conflicting_label_signatures_across_groups"], 0)
            signature_grouping = payload["split_context"][
                "exact_feature_signature_grouping"
            ]
            self.assertFalse(signature_grouping["fallback_recommended"])

            fold_audit = payload["split_context"]["fold_metadata"]
            self.assertEqual(len(fold_audit), 1)
            self.assertEqual(
                {partition["classes"] for partition in fold_audit[0]["partitions"]},
                {2},
            )
            self.assertEqual(
                {partition["rows"] for partition in fold_audit[0]["partitions"]},
                {4},
            )
            self.assertGreater(
                fold_audit[0]["group_overlap"]["duplicate_group"][
                    "groups_crossing_partitions"
                ],
                0,
            )

            self.assertEqual(payload["modeling_preflight"]["status"], "blocked")
            codes = {
                finding["code"] for finding in payload["modeling_preflight"]["findings"]
            }
            self.assertTrue(
                {
                    "pipeline_metadata_excluded",
                    "substantive_duplicate_observations",
                    "conflicting_labels_for_identical_features",
                    "feature_signatures_cross_groups",
                    "conflicting_labels_cross_groups",
                    "target_dependent_group_assignment",
                    "group_ids_cross_folds",
                }.issubset(codes)
            )

    def test_preflight_recommends_exact_feature_signature_grouping_without_ids(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "repeated_wines.csv"
            rows = [
                {
                    "fixed_acidity": 6.0 + signature,
                    "alcohol": 9.0 + signature / 10,
                    "quality": signature % 2,
                }
                for signature in range(6)
                for _ in range(2)
            ]
            pd.DataFrame(rows).to_csv(dataset, index=False)

            result = self.run_inspection(dataset)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            grouping = payload["split_context"]["exact_feature_signature_grouping"]
            self.assertTrue(grouping["fallback_recommended"])
            self.assertEqual(grouping["repeated_signatures"], 6)
            self.assertEqual(grouping["rows_in_repeated_signatures"], 12)
            codes = {
                finding["code"] for finding in payload["modeling_preflight"]["findings"]
            }
            self.assertIn("exact_feature_signature_grouping_fallback", codes)


def representative_run() -> dict:
    data_fingerprint = "sha256:" + "a" * 64
    split_fingerprint = "sha256:" + "b" * 64
    evaluation_rows_fingerprint = "sha256:" + "c" * 64
    return {
        "run_id": "wine-quality-comparison",
        "created_at": "2026-07-29T10:00:00+00:00",
        "problem": {
            "task": "classification",
            "target": "quality",
            "prediction_moment": "after laboratory measurements",
            "row_grain": "one wine batch",
            "intended_use": "offline wine-quality classification",
            "prohibited_uses": ["automated safety or pricing decisions"],
            "feature_contract": {
                "included": ["fixed_acidity", "volatile_acidity"],
                "excluded": ["sample_id"],
            },
        },
        "data": {
            "source": "wine.csv",
            "fingerprint": data_fingerprint,
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
            "findings": ["Feature availability was confirmed with the user."],
        },
        "evaluation": {
            "design": "stratified holdout",
            "split_fingerprint": split_fingerprint,
            "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
            "primary_metric": {"name": "macro_f1", "direction": "maximize"},
            "uncertainty": {"method": "group bootstrap", "confidence": 0.95},
        },
        "approval": {
            "approved_at": "2026-07-29T10:02:00+00:00",
            "scope": {
                "target": True,
                "feature_contract": True,
                "split_design": True,
                "primary_metric": True,
            },
            "tracks": {
                "classical": {
                    "selected": True,
                    "status": "approved",
                    "budget": {
                        "cpu_count": 4,
                        "parallel_jobs": 1,
                        "memory_gb": 8,
                        "gpu_enabled": False,
                        "candidate_families": ["tree_ensemble"],
                        "time_limit_seconds": 900,
                        "optuna_trials": 30,
                        "minimum_family_coverage": 1,
                    },
                },
                "autogluon": {
                    "selected": True,
                    "status": "approved",
                    "budget": {
                        "cpu_count": 4,
                        "parallel_jobs": 1,
                        "memory_gb": 8,
                        "gpu_enabled": False,
                        "preset": "medium_quality",
                        "run_mode": "run_to_completion",
                        "time_limit_seconds": None,
                        "runtime_estimate": {
                            "lower_seconds": 300,
                            "upper_seconds": 3600,
                            "basis": (
                                "small tabular dataset on approved CPU resources"
                            ),
                        },
                        "disk_gb": 10,
                    },
                },
                "sap_rpt": {
                    "selected": True,
                    "status": "approved",
                    "budget": {
                        "cpu_count": 2,
                        "parallel_jobs": 1,
                        "memory_gb": 4,
                        "gpu_enabled": False,
                        "max_requests": 20,
                        "max_context_rows": 1000,
                        "max_request_rows": 1200,
                        "max_query_batch_rows": 100,
                        "max_columns": 20,
                        "max_retries": 2,
                        "timeout_seconds": 120,
                    },
                },
            },
            "amendments": [
                {
                    "id": "rpt-capacity",
                    "approved_at": "2026-07-29T10:03:00+00:00",
                    "reason": "Confirmed deployed endpoint capacity.",
                    "changes": [
                        {
                            "path": "tracks.sap_rpt.budget.max_context_rows",
                            "before": 512,
                            "after": 1000,
                        }
                    ],
                }
            ],
            "remote_transfers": [
                {
                    "id": "rpt-transfer-1",
                    "approved_at": "2026-07-29T10:04:00+00:00",
                    "backend": "sap_rpt",
                    "destination": "SAP internal managed RPT endpoint",
                    "purpose": "shared evaluation",
                    "data_scope": {
                        "features": ["fixed_acidity", "volatile_acidity"],
                        "labels": True,
                        "query_rows": True,
                        "identifiers": ["row_id"],
                    },
                }
            ],
        },
        "backends": {
            "classical": {
                "status": "completed",
                "retained": True,
                "preprocessing": {"scope": "fold_local"},
                "search": {
                    "method": "optuna",
                    "trials_budget": 30,
                    "trials_completed": 30,
                },
                "candidates": [
                    {
                        "name": "random_forest",
                        "family": "tree_ensemble",
                        "status": "completed",
                        "score": 0.64,
                        "consideration_basis": "nonlinear tabular baseline",
                    }
                ],
                "evaluation": {
                    "split_fingerprint": split_fingerprint,
                    "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
                    "primary_metric": "macro_f1",
                    "score": 0.64,
                },
                "evidence": {"best_candidate": "random_forest"},
                "artifacts": {"model": "backends/classical/model.joblib"},
            },
            "autogluon": {
                "status": "completed",
                "retained": True,
                "build": {
                    "preset": "medium_quality",
                    "run_mode": "run_to_completion",
                    "time_limit_seconds": None,
                    "predictor_path": "backends/autogluon/predictor",
                    "fold_fitting_strategy": "sequential_local",
                    "fold_fitting_strategy_reason": (
                        "parallel_jobs=1 and bounded local execution"
                    ),
                    "training_diagnostics": {
                        "fit_summary_captured": True,
                        "elapsed_seconds": 612.7,
                        "completion_status": "completed_configuration",
                        "stop_reason": "configured model roster completed",
                    },
                    "packaging": {
                        "method": "clone_for_deployment",
                        "model": "best",
                        "diagnostics_captured_before_clone": True,
                        "prediction_equivalence": {
                            "validated": True,
                            "rows": 8,
                            "absolute_tolerance": 1e-12,
                        },
                        "training_predictor_retained": False,
                        "training_predictor_path": None,
                        "retention_reason": None,
                        "deployment_predictor_bytes": 1000000,
                        "peak_packaging_disk_bytes": 4000000,
                    },
                },
                "data_handling": {
                    "raw_tabular": True,
                    "external_preprocessing": False,
                    "external_optuna": False,
                },
                "runtime": {
                    "cold_start_subprocess": True,
                    "limits_set_before_imports": True,
                    "native_thread_limits": {
                        "OMP_NUM_THREADS": 1,
                        "MKL_NUM_THREADS": 1,
                        "OPENBLAS_NUM_THREADS": 1,
                        "VECLIB_MAXIMUM_THREADS": 1,
                    },
                },
                "native_leaderboard": [
                    {"model": "WeightedEnsemble_L2", "score_val": 0.67}
                ],
                "internal_failures": [
                    {
                        "component": "NeuralNetFastAI",
                        "stage": "fit",
                        "status": "failed",
                        "reason": "Optional dependency incompatibility.",
                        "track_impact": "track completed without this component",
                    }
                ],
                "evaluation": {
                    "split_fingerprint": split_fingerprint,
                    "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
                    "primary_metric": "macro_f1",
                    "score": 0.67,
                },
                "evidence": {"leaderboard": "native summary retained"},
            },
            "sap_rpt": {
                "status": "completed",
                "retained": True,
                "model": {
                    "name": "sap-rpt",
                    "version": "production",
                    "production_capable": True,
                },
                "access": {
                    "route": "internal_cli",
                    "client": "sap-rpt",
                    "customer_production_route": "sap_ai_core",
                },
                "context": {
                    "manifest": "backends/sap_rpt/context.json",
                    "fingerprint": "sha256:context",
                    "policy": "train rows only",
                },
                "transfer_confirmation": {
                    "approval_id": "rpt-transfer-1",
                    "schema_validated": True,
                    "labels_validated": True,
                    "query_rows_excluded_from_context": True,
                },
                "evaluation": {
                    "split_fingerprint": split_fingerprint,
                    "evaluation_rows_fingerprint": evaluation_rows_fingerprint,
                    "primary_metric": "macro_f1",
                    "score": 0.69,
                },
                "evidence": {
                    "request_ids_retained": True,
                    "latency_ms": 420,
                },
            },
        },
        "selection": {
            "predictive_winner": "sap_rpt",
            "operational_recommendation": "sap_rpt",
            "rationale": "Highest macro F1 on the shared evaluation rows.",
            "primary_metric": "macro_f1",
            "limitations": ["The minority class is relatively small."],
            "monitoring": {"metric": "macro_f1", "cadence": "monthly"},
        },
        "inference": {
            "entrypoint": "infer.py",
            "default_backend": "sap_rpt",
            "input": {
                "format": "csv",
                "required_columns": [
                    "row_id",
                    "fixed_acidity",
                    "volatile_acidity",
                ],
                "optional_columns": [],
                "dtypes": {
                    "row_id": "string",
                    "fixed_acidity": "float",
                    "volatile_acidity": "float",
                },
                "missing_value_policy": "reject missing required values",
                "extra_column_policy": "reject",
                "target_column": "quality",
                "identifier_columns": ["row_id"],
                "feature_order": ["fixed_acidity", "volatile_acidity"],
            },
            "output": {
                "format": "csv",
                "prediction_column": "prediction",
                "probability_columns": [],
                "probability_bounds": None,
                "row_id_column": "row_id",
                "finite_values": True,
            },
            "backends": {
                "classical": "python infer.py --backend classical --input new.csv --output predictions.csv",
                "autogluon": "python infer.py --backend autogluon --input new.csv --output predictions.csv",
                "sap_rpt": "python infer.py --backend sap_rpt --input new.csv --output predictions.csv",
            },
        },
        "lineage": {
            "source_data_fingerprint": data_fingerprint,
            "parent_run_id": None,
            "notes": ["All backends used identical evaluation rows."],
        },
    }


class ConsolidatedReportTests(unittest.TestCase):
    def test_report_is_one_self_contained_backend_inclusive_html_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(
                json.dumps(representative_run()), encoding="utf-8"
            )
            (root / "results.md").write_text(
                "# Findings\n\n- SAP RPT achieved the highest macro F1.\n"
                "- AutoGluon ranked second.\n\n## Limitations\n\nSmall minority classes.",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(REPORT_SCRIPT), str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = root / "report.html"
            self.assertTrue(report.is_file())
            text = report.read_text(encoding="utf-8")
            lower = text.lower()
            self.assertIn("<!doctype html>", lower)
            self.assertIn("<style>", lower)
            self.assertIn("<svg", lower)
            self.assertIn("classical", lower)
            self.assertIn("autogluon", lower)
            self.assertIn("sap_rpt", lower)
            self.assertIn("predictive winner", lower)
            self.assertIn("inference", lower)
            self.assertIn("0.69", lower)
            self.assertIn("feature contract", lower)
            self.assertIn("confirmed", lower)
            self.assertIn("classical baselines and leaderboard", lower)
            self.assertIn("autogluon settings", lower)
            self.assertIn("medium_quality", lower)
            self.assertIn("run_to_completion", lower)
            self.assertIn("deployment clone", lower)
            self.assertIn("internal failure ledger", lower)
            self.assertIn("neuralnetfastai", lower)
            self.assertIn("sap rpt context, access, and latency", lower)
            self.assertIn("approved amendments", lower)
            self.assertIn("approved remote transfers", lower)
            self.assertIn("uncertainty and limitations", lower)
            self.assertIn("intended use, prohibited use, and monitoring", lower)
            self.assertNotIn("<link", lower)
            self.assertNotIn("<script", lower)
            self.assertNotIn("<img", lower)
            self.assertNotIn('src="', lower)
            self.assertNotIn('href="http', lower)
            self.assertEqual(list(root.glob("*.html")), [report])
            self.assertFalse((root / "figures").exists())

    def test_report_renderer_requires_run_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(REPORT_SCRIPT), tmp],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Unable to render report", result.stderr)


if __name__ == "__main__":
    unittest.main()
