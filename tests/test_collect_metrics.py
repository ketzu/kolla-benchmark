import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_metrics import (
    COLUMNS,
    DETAIL_COLUMNS,
    collect_details,
    collect_rows,
    write_report,
)


def scored(fp, fn):
    return {"score": {"counts": {"tp": 1, "fp": fp, "fn": fn}}}


class CollectMetricsTests(unittest.TestCase):
    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_describes_runs_from_results_and_model_info(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            model_info = root / "model-info.csv"
            model_info.write_text(
                "model,display_name,company,params,quantization,file_bytes\n"
                "local/known,Known Model,Known Inc,3B,Q8_0,1234\n",
                encoding="utf-8",
            )
            self.write_json(
                results_dir / "batch-a" / "known.json",
                {
                    "provenance": {
                        "model": "local/known",
                        "endpoint": "http://localhost:1234/v1",
                        "started": "2026-09-11T15:01:34Z",
                        "rescored": None,
                    },
                    "results": [
                        scored(0, 0),
                        scored(0, 0),
                        scored(2, 0),
                        scored(0, 1),
                        scored(1, 1),
                    ],
                },
            )
            self.write_json(
                results_dir / "batch-b" / "unknown.json",
                {
                    "provenance": {
                        "model": "provider/unknown",
                        "endpoint": "https://openrouter.ai/api/v1",
                        "started": "2026-09-12T06:07:46Z",
                        "rescored": "2026-09-12T09:48:53Z",
                    },
                    "results": [],
                },
            )

            known, unknown = collect_details(results_dir, model_info)

            self.assertEqual(
                known,
                {
                    "model": "local/known",
                    "display_name": "Known Model",
                    "company": "Known Inc",
                    "provider": "Local",
                    "endpoint": "http://localhost:1234/v1",
                    "started_at": "2026-09-11T15:01:34Z",
                    "rescored_at": "",
                    "perfect": 2,
                    "over_corrected_only": 1,
                    "under_corrected_only": 1,
                    "mixed_error": 1,
                    "params": "3B",
                    "quantization": "Q8_0",
                    "file_bytes": "1234",
                },
            )
            self.assertEqual(unknown["provider"], "OpenRouter")
            self.assertEqual(unknown["rescored_at"], "2026-09-12T09:48:53Z")
            for column in ["display_name", "company", "params", "quantization", "file_bytes"]:
                self.assertEqual(unknown[column], "")
            self.assertEqual(set(known), set(DETAIL_COLUMNS))

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
            model_info_file = root / "model-info.csv"
            model_info_file.write_text(
                "model,company,params,quantization,file_bytes\n"
                "provider/has-cost,Maker,3B,Q8_0,1234\n",
                encoding="utf-8",
            )
            payload = {
                "provenance": {
                    "model": "provider/has-cost",
                    "endpoint": "http://localhost:1234/v1",
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
                "failures": [{"error": "first"}, {"error": "second"}],
            }
            self.write_json(results_dir / "batch-a" / "first.json", payload)
            other_payload = dict(payload)
            other_payload["provenance"] = {"model": "provider/no-cost"}
            self.write_json(results_dir / "batch-b" / "nested" / "second.json", other_payload)

            rows = collect_rows(results_dir, cost_file, model_info_file)

            self.assertEqual(
                [row["model"] for row in rows],
                ["provider/has-cost", "provider/no-cost"],
            )
            self.assertEqual(rows[0]["provider"], "Local")
            self.assertEqual(rows[0]["company"], "Maker")
            self.assertEqual(rows[0]["quantization"], "Q8_0")
            self.assertEqual(rows[1]["provider"], "")
            self.assertEqual(rows[1]["company"], "")
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
            self.assertEqual(rows[1]["cost"], "0")

    def test_reads_vast_sidecars_for_provider_and_cost_without_counting_them_as_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            payload = {
                "provenance": {"model": "Qwen/remote", "endpoint": "http://vast.local:8000/v1"},
                "results": [],
            }
            self.write_json(results_dir / "batch" / "Qwen_remote.json", payload)
            self.write_json(results_dir / "batch" / "Qwen_remote.vast.json", {"cost_usd": 0.1})
            other = {"provenance": {"model": "Qwen/priced", "endpoint": "http://vast.local:8000/v1"}}
            self.write_json(results_dir / "batch" / "Qwen_priced.json", other)
            self.write_json(results_dir / "batch" / "Qwen_priced.vast.json", {"cost_usd": 0.2})
            cost_file = root / "cost.csv"
            cost_file.write_text("API Key,Model,Value\neval,Qwen/priced,9.5\n", encoding="utf-8")

            rows = collect_rows(results_dir, cost_file, root / "missing.csv")
            details = collect_details(results_dir, root / "missing.csv")

            self.assertEqual(
                [(row["model"], row["cost"]) for row in rows],
                [("Qwen/priced", "9.5"), ("Qwen/remote", "0.1")],
            )
            self.assertEqual([row["provider"] for row in details], ["Vast.ai", "Vast.ai"])

    def test_writes_requested_columns_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "metrics.csv"
            write_report(
                output,
                [
                    {
                        "model": "provider/model",
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
                    }
                ],
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

            self.assertEqual(rows[0], COLUMNS)
            self.assertEqual(rows[1][0], "provider/model")
            self.assertEqual(rows[1][-1], "0")


if __name__ == "__main__":
    unittest.main()
