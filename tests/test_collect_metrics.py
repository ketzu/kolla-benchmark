import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_metrics import (
    COLUMNS,
    collect_rows,
    prompt_matrix,
    write_report,
)


PROMPTS = [
    {"name": "simple", "user": "Correct it. {sentence}"},
    {"name": "simple", "system": "Correct it.", "user": "{sentence}"},
    {"name": "korean", "user": "고쳐라.\n문장: {sentence}"},
]


class CollectMetricsTests(unittest.TestCase):
    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_collects_and_flattens_every_nested_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            cost_file = results_dir / "cost.csv"
            cost_file.parent.mkdir()
            cost_file.write_text(
                "API Key,Model,Value,% of Total\n"
                "eval,provider/has-cost,1.234567,100\n",
                encoding="utf-8",
            )
            payload = {
                "provenance": {"model": "provider/has-cost"},
                "metrics": {"precision": 0.5, "recall": 0.4, "f05": 0.45},
                "counts": {"tp": 10, "fp": 2, "fn": 3},
                "summary": {
                    "sentences": 20,
                    "exact_match": 4,
                    "perfect": 5,
                    "unchanged": 6,
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                },
                "failures": [{"error": "first"}, {"error": "second"}],
            }
            self.write_json(results_dir / "batch-a" / "first.json", payload)
            other_payload = dict(payload)
            other_payload["provenance"] = {"model": "provider/no-cost"}
            self.write_json(results_dir / "batch-b" / "nested" / "second.json", other_payload)

            rows = collect_rows(results_dir, cost_file)

            self.assertEqual(
                [row["model"] for row in rows],
                ["provider/has-cost", "provider/no-cost"],
            )
            self.assertEqual(rows[0]["precision"], 0.5)
            self.assertEqual(rows[0]["tp"], 10)
            self.assertEqual(rows[0]["sentences"], 20)
            self.assertEqual(rows[0]["failure_count"], 2)
            self.assertEqual(rows[0]["prompt_tokens"], 100)
            self.assertEqual(rows[0]["completion_tokens"], 200)
            self.assertEqual(rows[0]["total_tokens"], 300)
            self.assertEqual(rows[0]["exact_match"], 4)
            self.assertEqual(rows[0]["perfect"], 5)
            self.assertEqual(rows[0]["unchanged"], 6)
            self.assertEqual(rows[0]["cost"], "1.234567")
            self.assertEqual(rows[0]["file"], "batch-a/first.json")
            self.assertEqual(rows[1]["cost"], "0")

    def test_names_runs_by_the_prompt_they_sent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            prompt_file = root / "prompts.json"
            prompt_file.write_text(json.dumps(PROMPTS), encoding="utf-8")
            runs = {
                "user.json": {"prompt": "Correct it. {sentence}"},
                "system.json": {"system": "Correct it.", "prompt": "{sentence}"},
                # Line endings as a Windows shell might have passed them.
                "crlf.json": {"system": None, "prompt": "고쳐라.\r\n문장: {sentence}"},
                # Written before templates: the prompt was sent as the system message.
                "legacy.json": {"prompt": "Correct it."},
                "custom.json": {"system": "Something else.", "prompt": "{sentence}"},
            }
            for name, prompt in runs.items():
                self.write_json(
                    results_dir / name, {"provenance": {"model": "m", **prompt}}
                )

            rows = collect_rows(results_dir, results_dir / "cost.csv", prompt_file)

            variants = {row["file"]: (row["prompt_name"], row["prompt_type"]) for row in rows}
            self.assertEqual(
                variants,
                {
                    "user.json": ("simple", "user"),
                    "system.json": ("simple", "system+user"),
                    "crlf.json": ("korean", "user"),
                    "legacy.json": ("simple", "system+user"),
                    "custom.json": ("custom", "system+user"),
                },
            )

    def test_prompt_matrix_keeps_the_latest_run_per_model_and_variant(self):
        def row(model, name, kind, f05, started):
            return {
                "model": model,
                "prompt_name": name,
                "prompt_type": kind,
                "f05": f05,
                "started": started,
            }

        columns, matrix = prompt_matrix(
            [
                row("b", "simple", "user", 0.1, "2026-09-14T10:00:00Z"),
                row("b", "simple", "user", 0.2, "2026-09-14T12:00:00Z"),
                row("b", "simple", "user", 0.3, "2026-09-14T11:00:00Z"),
                row("a", "korean", "system+user", 0.4, "2026-09-14T10:00:00Z"),
            ]
        )

        self.assertEqual(columns, ["model", "korean/system+user", "simple/user"])
        self.assertEqual(
            matrix,
            [
                {"model": "a", "korean/system+user": 0.4},
                {"model": "b", "simple/user": 0.2},
            ],
        )

    def test_writes_requested_columns_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "metrics.csv"
            write_report(
                output,
                [
                    {
                        "model": "provider/model",
                        "prompt_name": "simple",
                        "prompt_type": "user",
                        "precision": 1,
                        "recall": 2,
                        "f05": 3,
                        "tp": 4,
                        "fp": 5,
                        "fn": 6,
                        "sentences": 7,
                        "failure_count": 8,
                        "prompt_tokens": 9,
                        "completion_tokens": 10,
                        "total_tokens": 19,
                        "exact_match": 11,
                        "perfect": 12,
                        "unchanged": 13,
                        "cost": "0",
                        "started": "2026-09-14T10:00:00Z",
                        "file": "batch/model.json",
                    }
                ],
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

            self.assertEqual(rows[0], COLUMNS)
            self.assertEqual(rows[1][:3], ["provider/model", "simple", "user"])
            self.assertEqual(rows[1][-1], "batch/model.json")


if __name__ == "__main__":
    unittest.main()
