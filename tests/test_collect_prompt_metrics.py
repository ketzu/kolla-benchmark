import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_prompt_metrics import (
    COLUMNS,
    collect_rows,
    prompt_matrix,
    write_csv,
)


PROMPTS = [
    {"name": "simple", "user": "Correct it. {sentence}"},
    {"name": "simple", "system": "Correct it.", "user": "{sentence}"},
    {"name": "korean", "user": "고쳐라.\n문장: {sentence}"},
]


class CollectPromptMetricsTests(unittest.TestCase):
    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_flattens_a_run_with_its_prompt_variant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt_file = root / "prompts.json"
            prompt_file.write_text(json.dumps(PROMPTS), encoding="utf-8")
            self.write_json(
                root / "results" / "batch" / "provider_model" / "p1.json",
                {
                    "provenance": {
                        "model": "provider/model",
                        "endpoint": "https://openrouter.ai/api/v1",
                        "prompt": "Correct it. {sentence}",
                        "started": "2026-09-14T10:00:00Z",
                    },
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
                    "failures": [{"error": "first"}],
                },
            )

            model_info_file = root / "model-info.csv"
            model_info_file.write_text(
                "model,company,params,quantization,file_bytes\nprovider/model,Maker,3B,Q8_0,1234\n",
                encoding="utf-8",
            )

            (row,) = collect_rows(root / "results", prompt_file, model_info_file)

            self.assertEqual(
                row,
                {
                    "model": "provider/model",
                    "provider": "openrouter",
                    "company": "Maker",
                    "params": "3B",
                    "quantization": "Q8_0",
                    "file_bytes": "1234",
                    "prompt_name": "simple",
                    "prompt_type": "user",
                    "precision": 0.5,
                    "recall": 0.4,
                    "f05": 0.45,
                    "tp": 10,
                    "fp": 2,
                    "fn": 3,
                    "sentences": 20,
                    "failure_count": 1,
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                    "exact_match": 4,
                    "perfect": 5,
                    "unchanged": 6,
                    "started": "2026-09-14T10:00:00Z",
                    "file": "batch/provider_model/p1.json",
                },
            )
            self.assertEqual(set(row), set(COLUMNS))

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

            rows = collect_rows(results_dir, prompt_file, root / "model-info.csv")

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

    def test_writes_the_matrix_with_empty_cells_for_missing_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prompt-matrix.csv"

            write_csv(
                output,
                ["model", "korean/user", "simple/user"],
                [{"model": "a", "simple/user": 0.5}],
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

            self.assertEqual(rows, [["model", "korean/user", "simple/user"], ["a", "", "0.5"]])


if __name__ == "__main__":
    unittest.main()
