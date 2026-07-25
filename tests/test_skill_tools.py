from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PROFILE = REPOSITORY / "skills/ml-model-builder/scripts/profile_dataset.py"
VALIDATE = REPOSITORY / "skills/ml-model-builder/scripts/validate_run.py"
PACKAGE = REPOSITORY / "scripts/package_skill.py"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


class ProfileDatasetTests(unittest.TestCase):
    def rows(self) -> list[dict]:
        values = []
        for index in range(80):
            values.append(
                {
                    "age": 20 + index % 40,
                    "income": 30000 + index * 700,
                    "segment": ["north", "south", "west"][index % 3],
                    "email": f"person{index}@example.test",
                    "event_time": f"2025-01-{index % 28 + 1:02d}",
                    "churned": index % 2,
                }
            )
        return values

    def test_analysis_only_writes_labeled_deterministic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            output = root / "artefacts"
            write_csv(source, self.rows())
            completed = run(
                PROFILE,
                "--input",
                source,
                "--output-dir",
                output,
                "--mode",
                "analysis-only",
                "--task",
                "classification",
                "--target",
                "churned",
                "--time-column",
                "event_time",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for filename in [
                "data_profile.json",
                "data_report.html",
                "data_summary.md",
                "data_fingerprint.json",
                "schema.json",
                "config.json",
            ]:
                self.assertTrue((output / filename).is_file())
            report = json.loads((output / "data_profile.json").read_text())
            self.assertEqual(report["schema_version"], "2.0")
            self.assertGreaterEqual(len(report["figures"]), 5)
            report_html = (output / "data_report.html").read_text()
            self.assertIn("observed labels", report_html)
            validated = run(VALIDATE, root)
            self.assertEqual(validated.returncode, 0, validated.stdout)

    def test_model_mode_requires_persisted_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            write_csv(source, self.rows())
            completed = run(
                PROFILE,
                "--input",
                source,
                "--output-dir",
                root / "artefacts",
                "--mode",
                "model",
                "--task",
                "classification",
                "--target",
                "churned",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("persisted partition", completed.stderr)

    def test_target_profile_uses_training_rows_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.rows()
            for index, row in enumerate(rows):
                row["_ml_partition"] = "train" if index < 60 else "holdout"
                if index >= 60:
                    row["churned"] = "HOLDOUT_SECRET"
            source = root / "data.csv"
            output = root / "artefacts"
            write_csv(source, rows)
            completed = run(
                PROFILE,
                "--input",
                source,
                "--output-dir",
                output,
                "--mode",
                "model",
                "--task",
                "classification",
                "--target",
                "churned",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output / "data_profile.json").read_text())
            self.assertEqual(report["analysis_population"]["rows"], 60)
            self.assertEqual(report["columns"]["churned"]["unique_count"], 2)
            self.assertNotIn(
                "HOLDOUT_SECRET", (output / "data_report.html").read_text()
            )

    def test_single_class_training_target_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.rows()
            for row in rows:
                row["_ml_partition"] = "train"
                row["churned"] = 1
            source = root / "data.csv"
            output = root / "artefacts"
            write_csv(source, rows)
            completed = run(
                PROFILE,
                "--input",
                source,
                "--output-dir",
                output,
                "--mode",
                "model",
                "--task",
                "classification",
                "--target",
                "churned",
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            report = json.loads((output / "data_profile.json").read_text())
            self.assertIn(
                "single_class_target",
                {finding["code"] for finding in report["findings"]},
            )


class ValidateRunTests(unittest.TestCase):
    def test_valid_analysis_only_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = project / "artefacts"
            (artifacts / "figures").mkdir(parents=True)
            (artifacts / "figures/chart.png").write_bytes(b"png")
            (artifacts / "config.json").write_text(
                json.dumps({"schema_version": "2.0", "mode": "analysis-only"})
            )
            (artifacts / "data_profile.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "mode": "analysis-only",
                    }
                )
            )
            for filename in ["data_fingerprint.json", "schema.json"]:
                (artifacts / filename).write_text(json.dumps({"schema_version": "2.0"}))
            (artifacts / "data_report.html").write_text("<html></html>")
            (artifacts / "data_summary.md").write_text("# Summary")
            completed = run(VALIDATE, project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_incomplete_model_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = project / "artefacts"
            artifacts.mkdir()
            (artifacts / "config.json").write_text(
                json.dumps({"schema_version": "2.0", "mode": "model-building"})
            )
            completed = run(VALIDATE, project)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing artefacts/model.joblib", completed.stdout)

    def test_complete_model_contract_passes_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = project / "artefacts"
            (artifacts / "figures").mkdir(parents=True)
            (artifacts / "figures/chart.png").write_bytes(b"png")
            model_bytes = b"trusted-test-model-placeholder"
            (artifacts / "model.joblib").write_bytes(model_bytes)
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            documents = {
                "config.json": {
                    "schema_version": "2.0",
                    "mode": "model-building",
                    "problem": {
                        "task": "classification",
                        "prediction_moment": "application time",
                    },
                    "split": {
                        "assignment_column": "_ml_partition",
                        "holdout_target_sealed": True,
                    },
                    "analysis": {"target_aware_partition": "train"},
                },
                "data_profile.json": {
                    "schema_version": "2.0",
                    "mode": "model",
                },
                "data_fingerprint.json": {"schema_version": "2.0"},
                "schema.json": {
                    "schema_version": "2.0",
                    "partition_column": "_ml_partition",
                },
                "feature_manifest.json": {
                    "schema_version": "2.0",
                    "raw_input_features": ["age"],
                },
                "metrics.json": {
                    "schema_version": "2.0",
                    "final": {
                        "eval_set": "holdout_test",
                        "score": 0.7,
                    },
                },
                "inference_test.json": {
                    "command": "python artefacts/infer.py --input test.csv",
                    "status": "passed",
                    "row_count": 1,
                    "trusted_model_sha256": model_hash,
                },
            }
            for filename, document in documents.items():
                (artifacts / filename).write_text(json.dumps(document))
            for filename, contents in {
                "data_report.html": "<html></html>",
                "data_summary.md": "# Summary",
                "train.py": "pass\n",
                "infer.py": "pass\n",
                "model_card.md": "# Model card",
                "requirements.lock": "scikit-learn==1.7.1\n",
            }.items():
                (artifacts / filename).write_text(contents)
            (project / "results.md").write_text(
                "# Results\n\nPrediction moment: application time."
            )
            completed = run(VALIDATE, project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_unlabeled_anomaly_contract_does_not_require_fake_holdout_score(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = project / "artefacts"
            (artifacts / "figures").mkdir(parents=True)
            (artifacts / "figures/chart.png").write_bytes(b"png")
            model_bytes = b"trusted-anomaly-scorer"
            (artifacts / "model.joblib").write_bytes(model_bytes)
            documents = {
                "config.json": {
                    "schema_version": "2.0",
                    "mode": "model-building",
                    "problem": {
                        "task": "anomaly",
                        "labels_available": False,
                        "prediction_moment": "UTC daily cutoff",
                    },
                    "split": {"assignment_column": "_ml_partition"},
                    "analysis": {"population_partition": "reference"},
                },
                "data_profile.json": {
                    "schema_version": "2.0",
                    "mode": "model",
                },
                "data_fingerprint.json": {"schema_version": "2.0"},
                "schema.json": {
                    "schema_version": "2.0",
                    "partition_column": "_ml_partition",
                },
                "feature_manifest.json": {
                    "schema_version": "2.0",
                    "raw_input_features": ["amount"],
                },
                "metrics.json": {
                    "schema_version": "2.0",
                    "final": {
                        "eval_set": "future_scoring_window",
                        "score": None,
                        "predictive_performance_available": False,
                    },
                    "anomaly_evaluation": {
                        "review_capacity": 200,
                        "unreviewed_rows_treated_as_negative": False,
                    },
                },
                "inference_test.json": {
                    "command": "python artefacts/infer.py --input batch.csv",
                    "status": "passed",
                    "row_count": 250,
                    "trusted_model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                },
            }
            for filename, document in documents.items():
                (artifacts / filename).write_text(json.dumps(document))
            for filename, contents in {
                "data_report.html": "<html></html>",
                "data_summary.md": "# Summary",
                "train.py": "pass\n",
                "infer.py": "pass\n",
                "model_card.md": "# Model card",
                "requirements.lock": "scikit-learn==1.7.1\n",
            }.items():
                (artifacts / filename).write_text(contents)
            (project / "results.md").write_text(
                "# Results\n\nPrediction moment: UTC daily cutoff."
            )
            completed = run(VALIDATE, project)
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_legacy_v1_model_run_passes_with_explicit_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifacts = project / "artefacts"
            artifacts.mkdir()
            (artifacts / "config.json").write_text(
                json.dumps({"mode": "model-building"})
            )
            (artifacts / "metrics.json").write_text(
                json.dumps(
                    {
                        "final": {
                            "eval_set": "holdout_test",
                            "score": 0.75,
                        }
                    }
                )
            )
            for filename, contents in {
                "train.py": "pass\n",
                "infer.py": "pass\n",
                "model.joblib": "legacy-placeholder",
            }.items():
                (artifacts / filename).write_text(contents)
            completed = run(VALIDATE, project)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("legacy v1", completed.stdout)


class PackageSkillTests(unittest.TestCase):
    def test_archive_is_deterministic_and_matches_source(self):
        output = REPOSITORY / "dist/ml-model-builder.skill"
        first = run(PACKAGE)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        first_hash = hashlib.sha256(output.read_bytes()).hexdigest()
        second = run(PACKAGE)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(first_hash, hashlib.sha256(output.read_bytes()).hexdigest())
        with zipfile.ZipFile(output) as archive:
            skill_name = "ml-model-builder/SKILL.md"
            self.assertIn(skill_name, archive.namelist())
            self.assertEqual(
                archive.read(skill_name),
                (REPOSITORY / "skills/ml-model-builder/SKILL.md").read_bytes(),
            )
            self.assertIn(
                "ml-model-builder/scripts/profile_dataset.py",
                archive.namelist(),
            )


if __name__ == "__main__":
    unittest.main()
