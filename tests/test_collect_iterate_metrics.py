import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_iterate_metrics import COLUMNS, collect_rows
from scripts.collect_prompt_metrics import write_csv


PROMPTS = [
    {
        "name": "extended",
        "user": "Correct the following korean sentence. Only correct actual errors. Reply with only the corrected sentence. {sentence}",
    },
]

REQUESTS = {
    "converged": 18,
    "total": 41,
    "p25": 1.0,
    "median": 2.0,
    "mean": 2.05,
    "p75": 2.25,
    "p90": 4.1,
    "max": 10,
}


class CollectIterateMetricsTests(unittest.TestCase):
    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def run_payload(self, **provenance):
        return {
            "provenance": {
                "model": "provider/model",
                "endpoint": "http://localhost:1234/v1",
                "system": None,
                "prompt": PROMPTS[0]["user"],
                "max_iterations": 10,
                "started": "2026-09-14T10:00:00Z",
                **provenance,
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
                "requests": REQUESTS,
            },
            "failures": [{"original": "a", "error": "boom", "rounds": [{"answer": "b"}]}],
            "results": [],
        }

    def test_flattens_an_iterated_run_with_its_request_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt_file = root / "prompts.json"
            prompt_file.write_text(json.dumps(PROMPTS), encoding="utf-8")
            self.write_json(root / "results" / "batch" / "provider_model.json", self.run_payload())

            (row,) = collect_rows(root / "results", prompt_file, root / "missing.csv")

            self.assertEqual(
                row,
                {
                    "model": "provider/model",
                    "provider": "lmstudio",
                    "company": "",
                    "params": "",
                    "quantization": "",
                    "file_bytes": "",
                    "prompt_name": "extended",
                    "prompt_type": "user",
                    "max_iterations": 10,
                    "precision": 0.5,
                    "recall": 0.4,
                    "f05": 0.45,
                    "tp": 10,
                    "fp": 2,
                    "fn": 3,
                    "sentences": 20,
                    "failure_count": 1,
                    "converged": 18,
                    "requests_total": 41,
                    "requests_p25": 1.0,
                    "requests_median": 2.0,
                    "requests_mean": 2.05,
                    "requests_p75": 2.25,
                    "requests_p90": 4.1,
                    "requests_max": 10,
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                    "exact_match": 4,
                    "perfect": 5,
                    "unchanged": 6,
                    "started": "2026-09-14T10:00:00Z",
                    "file": "batch/provider_model.json",
                },
            )
            self.assertEqual(list(row), COLUMNS)

    def test_rejects_a_run_that_was_not_iterated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_json(root / "results" / "run.json", self.run_payload(max_iterations=None))

            with self.assertRaisesRegex(ValueError, "not an iterated run"):
                collect_rows(root / "results", root / "prompts.json", root / "missing.csv")

    def test_writes_every_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_json(root / "results" / "run.json", self.run_payload())
            rows = collect_rows(root / "results", root / "prompts.json", root / "missing.csv")
            output = root / "iterate-metrics.csv"

            write_csv(output, COLUMNS, rows)

            with output.open(newline="", encoding="utf-8") as handle:
                (written,) = list(csv.DictReader(handle))
            self.assertEqual(list(written), COLUMNS)
            self.assertEqual(written["prompt_name"], "custom")
            self.assertEqual(written["requests_p90"], "4.1")


if __name__ == "__main__":
    unittest.main()
