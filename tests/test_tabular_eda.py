from __future__ import annotations

import csv
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ANALYZE = REPOSITORY / "skills/tabular-eda/scripts/analyze_tabular.py"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analyzer(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ANALYZE), *map(str, arguments)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )


class TabularEdaTests(unittest.TestCase):
    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(72):
            rows.append(
                {
                    "record_id": f"wine-{index:03d}",
                    "acidity": 5.2 + (index % 12) * 0.3,
                    "sulphates": ""
                    if index in {2, 19, 53}
                    else 0.3 + (index % 9) * 0.08,
                    "region": ["north", "south", "west"][index % 3],
                    "observed_at": f"2025-{index % 6 + 1:02d}-{index % 27 + 1:02d}",
                    "producer": f"group-{index % 8}",
                    "quality": [5, 5, 5, 6, 6, 7][index % 6],
                    "constant_note": "measured",
                }
            )
        rows.append(dict(rows[0]))
        rows[-1]["record_id"] = rows[0]["record_id"]
        rows[20]["acidity"] = 200.0
        return rows

    def test_csv_writes_only_requested_self_contained_html_and_stdout_findings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wine.csv"
            destination = root / "deliverables" / "eda_report.html"
            write_csv(source, self.rows())

            completed = run_analyzer(
                "--input",
                source,
                "--output",
                destination,
                "--target",
                "quality",
                "--time-column",
                "observed_at",
                "--group-column",
                "producer",
                "--max-plot-rows",
                31,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(completed.stdout.startswith("## EDA findings\n\n"))
            finding_lines = completed.stdout.splitlines()[2:]
            self.assertGreaterEqual(len(finding_lines), 6)
            self.assertTrue(
                all(line.startswith("- **") and " — " in line for line in finding_lines)
            )
            self.assertIn(
                f"The dataset contains {len(self.rows()):,} rows and 8 columns.",
                completed.stdout,
            )

            files = [path for path in destination.parent.rglob("*") if path.is_file()]
            self.assertEqual(files, [destination])
            report = destination.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", report)
            self.assertIn("<style>", report)
            self.assertIn("<svg", report)
            self.assertIn("Important findings", report)
            self.assertIn("Interpretation boundaries", report)
            self.assertIn("deterministic sample", report)
            self.assertIn("model-independent", report)
            self.assertNotIn("target-independent", report)
            self.assertIn("Target–feature behavior", report)
            self.assertIn("Conditional numeric behavior", report)
            self.assertIn("Conditional categorical behavior", report)
            self.assertIn("Median by quality", report)
            self.assertRegex(
                report,
                r"<code>quality</code></td><td>numeric</td><td>target</td>",
            )
            self.assertNotRegex(report, r"<script\b[^>]*\bsrc\s*=")
            self.assertNotRegex(report, r"<link\b[^>]*\bhref\s*=")
            self.assertNotRegex(report, r"<(?:img|iframe)\b[^>]*\bsrc\s*=")
            self.assertNotRegex(report, r"url\(\s*['\"]?(?:https?:|/|\.{1,2}/)")
            self.assertNotIn("http://", report)
            self.assertNotIn("https://", report)

    def test_report_is_deterministic_and_sampling_seed_changes_only_sampled_charts(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            write_csv(source, self.rows())
            first = root / "one" / "eda_report.html"
            second = root / "two" / "eda_report.html"
            third = root / "three" / "eda_report.html"

            first_run = run_analyzer(
                "--input",
                source,
                "--output",
                first,
                "--target",
                "quality",
                "--max-plot-rows",
                20,
                "--seed",
                7,
            )
            second_run = run_analyzer(
                "--input",
                source,
                "--output",
                second,
                "--target",
                "quality",
                "--max-plot-rows",
                20,
                "--seed",
                7,
            )
            third_run = run_analyzer(
                "--input",
                source,
                "--output",
                third,
                "--target",
                "quality",
                "--max-plot-rows",
                20,
                "--seed",
                8,
            )

            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertEqual(third_run.returncode, 0, third_run.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotEqual(first.read_bytes(), third.read_bytes())

            self.assertEqual(first_run.stdout, second_run.stdout)
            self.assertEqual(first_run.stdout, third_run.stdout)

    def test_common_data_issues_are_explained_in_stdout_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "issues.csv"
            destination = root / "report.html"
            rows = self.rows()
            for index in range(25):
                rows[index]["sulphates"] = ""
            for row in rows:
                row["quality_copy"] = row["quality"]
            rows[-1] = dict(rows[0])
            write_csv(source, rows)

            completed = run_analyzer(
                "--input",
                source,
                "--output",
                destination,
                "--target",
                "quality",
                "--time-column",
                "observed_at",
                "--group-column",
                "producer",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            titles = {
                "Repeated observations",
                "Substantial missingness",
                "Constant columns",
                "Possible identifiers",
                "Potential numeric outliers",
                "Target class balance",
                "Columns duplicate the target",
                "Time coverage",
                "Group structure",
            }
            for title in titles:
                self.assertIn(f"**{title}**", completed.stdout)

            report = destination.read_text(encoding="utf-8")
            for title in titles:
                self.assertIn(title, report)

    def test_duplicates_ignore_ids_and_pipeline_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pipeline.csv"
            output = root / "eda_report.html"
            rows = [
                {
                    "_row_id": "row-a",
                    "_outer_fold": 1,
                    "_duplicate_group": "group-a",
                    "_ml_partition": "train",
                    "train_split": "development",
                    "measurement": 10.0,
                    "kind": "red",
                    "quality": 0,
                },
                {
                    "_row_id": "row-b",
                    "_outer_fold": 2,
                    "_duplicate_group": "group-a",
                    "_ml_partition": "holdout",
                    "train_split": "evaluation",
                    "measurement": 10.0,
                    "kind": "red",
                    "quality": 0,
                },
                {
                    "_row_id": "row-c",
                    "_outer_fold": 1,
                    "_duplicate_group": "group-b",
                    "_ml_partition": "train",
                    "train_split": "development",
                    "measurement": 15.0,
                    "kind": "white",
                    "quality": 1,
                },
                {
                    "_row_id": "row-d",
                    "_outer_fold": 2,
                    "_duplicate_group": "group-c",
                    "_ml_partition": "holdout",
                    "train_split": "evaluation",
                    "measurement": 18.0,
                    "kind": "white",
                    "quality": 1,
                },
            ]
            write_csv(source, rows)

            completed = run_analyzer(
                "--input",
                source,
                "--output",
                output,
                "--target",
                "quality",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("**Repeated observations**", completed.stdout)
            self.assertIn(
                "1 rows (25.0%) repeat an earlier observation", completed.stdout
            )
            self.assertIn("_row_id", completed.stdout)
            self.assertIn("**Possible pipeline metadata**", completed.stdout)
            self.assertIn("_outer_fold", completed.stdout)
            self.assertIn("_duplicate_group", completed.stdout)
            self.assertIn("_ml_partition", completed.stdout)
            self.assertIn("train_split", completed.stdout)
            self.assertNotIn("No repeated observations", completed.stdout)

            report = output.read_text(encoding="utf-8")
            self.assertRegex(
                report,
                r"<code>_outer_fold</code></td><td>numeric</td>"
                r"<td>possible pipeline metadata</td>",
            )
            self.assertRegex(
                report,
                r"<code>_duplicate_group</code></td><td>categorical/text</td>"
                r"<td>possible pipeline metadata</td>",
            )
            self.assertRegex(
                report,
                r"<code>_ml_partition</code></td><td>categorical/text</td>"
                r"<td>possible pipeline metadata</td>",
            )
            self.assertRegex(
                report,
                r"<code>train_split</code></td><td>categorical/text</td>"
                r"<td>possible pipeline metadata</td>",
            )
            self.assertRegex(
                report,
                r"<code>quality</code></td><td>numeric</td><td>target</td>",
            )
            relationships = report.split(
                '<section id="target-relationships">', maxsplit=1
            )[1].split("</section>", maxsplit=1)[0]
            self.assertIn("measurement", relationships)
            self.assertIn("kind", relationships)
            self.assertNotIn("_outer_fold", relationships)
            self.assertNotIn("_duplicate_group", relationships)
            self.assertNotIn("_row_id", relationships)
            numeric_distributions = report.split('<section id="numeric">', maxsplit=1)[
                1
            ].split("</section>", maxsplit=1)[0]
            numeric_relationships = report.split(
                '<section id="relationships">', maxsplit=1
            )[1].split("</section>", maxsplit=1)[0]
            self.assertNotIn("_outer_fold", numeric_distributions)
            self.assertNotIn("_outer_fold", numeric_relationships)

    def test_unique_continuous_measure_is_not_mistaken_for_an_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "continuous.csv"
            output = root / "eda_report.html"
            write_csv(
                source,
                [
                    {
                        "record_id": f"row-{index}",
                        "continuous_measure": index + index / 100,
                        "target": index % 2,
                    }
                    for index in range(24)
                ],
            )

            completed = run_analyzer(
                "--input",
                source,
                "--output",
                output,
                "--target",
                "target",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = output.read_text(encoding="utf-8")
            self.assertRegex(
                report,
                r"<code>continuous_measure</code></td><td>numeric</td>"
                r"<td>numeric measure</td>",
            )
            self.assertRegex(
                report,
                r"<code>record_id</code></td><td>categorical/text</td>"
                r"<td>possible identifier</td>",
            )

    def test_missing_hint_column_fails_without_writing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            output = root / "output" / "eda_report.html"
            write_csv(source, self.rows())

            completed = run_analyzer(
                "--input",
                source,
                "--output",
                output,
                "--target",
                "does_not_exist",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("was not found", completed.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

    @unittest.skipUnless(
        importlib.util.find_spec("pandas")
        and (
            importlib.util.find_spec("pyarrow")
            or importlib.util.find_spec("fastparquet")
        ),
        "Parquet engine is not installed",
    )
    def test_parquet_input(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wine.parquet"
            output = root / "eda_report.html"
            pd.DataFrame(self.rows()).to_parquet(source, index=False)

            completed = run_analyzer(
                "--input",
                source,
                "--output",
                output,
                "--target",
                "quality",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                f"The dataset contains {len(self.rows()):,} rows and 8 columns.",
                completed.stdout,
            )
            self.assertTrue(output.is_file())
            self.assertNotRegex(
                output.read_text(encoding="utf-8"),
                re.compile(r"<(?:script|img|iframe)\b[^>]*\bsrc\s*=", re.IGNORECASE),
            )


if __name__ == "__main__":
    unittest.main()
