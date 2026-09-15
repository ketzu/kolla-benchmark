import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vast"))

import common  # noqa: E402
from common import ModelEntry  # noqa: E402


class ModelListTests(unittest.TestCase):
    def test_reads_repos_flags_and_skips_comments(self):
        text = "# vllm models\r\nQwen/Qwen3-0.6B\r\n\r\n  #skipped/model\r\nQwen/Qwen3-8B-FP8 --max-model-len 8192  # short\r\n"

        self.assertEqual(
            common.parse_model_list(text),
            [
                ModelEntry("Qwen/Qwen3-0.6B"),
                ModelEntry("Qwen/Qwen3-8B-FP8", ("--max-model-len", "8192")),
            ],
        )

    def test_manifest_round_trip(self):
        entries = [ModelEntry("a/one"), ModelEntry("b/two", ("--dtype", "half"))]

        self.assertEqual(common.manifest_entries({"models": common.manifest_models(entries)}), entries)


class VllmCommandTests(unittest.TestCase):
    def test_adds_defaults_for_a_single_gpu(self):
        command = common.vllm_command(ModelEntry("a/one"), gpu_count=1)

        self.assertEqual(command[:3], ["vllm", "serve", "a/one"])
        self.assertIn("--served-model-name", command)
        self.assertEqual(command[command.index("--max-model-len") + 1], "16384")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "128")
        self.assertIn("--language-model-only", command)
        self.assertNotIn("--tensor-parallel-size", command)

    def test_leaves_multimodal_limits_to_a_line_that_sets_them(self):
        limited = common.vllm_command(ModelEntry("a/one", ("--limit-mm-per-prompt", '{"image": 1}')), gpu_count=1)
        explicit = common.vllm_command(ModelEntry("a/one", ("--language-model-only",)), gpu_count=1)

        self.assertNotIn("--language-model-only", limited)
        self.assertEqual(explicit.count("--language-model-only"), 1)

    def test_keeps_the_line_s_own_sequence_limit(self):
        command = common.vllm_command(ModelEntry("a/one", ("--max-num-seqs=48",)), gpu_count=1)

        self.assertNotIn("--max-num-seqs", command)
        self.assertIn("--max-num-seqs=48", command)

    def test_splits_across_gpus_unless_the_line_decides(self):
        default = common.vllm_command(ModelEntry("a/one"), gpu_count=4)
        chosen = common.vllm_command(ModelEntry("a/one", ("-tp", "2", "--max-model-len=4096")), gpu_count=4)

        self.assertEqual(default[default.index("--tensor-parallel-size") + 1], "4")
        self.assertNotIn("--tensor-parallel-size", chosen)
        self.assertNotIn("--max-model-len", chosen)
        self.assertIn("--max-model-len=4096", chosen)


class BenchmarkArgsTests(unittest.TestCase):
    def test_refuses_flags_the_runner_owns(self):
        for args in (["--model", "x"], ["-mx"], ["--output=run.json"], ["--url", "u"], ["--api-key", "k"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                common.check_benchmark_args(args)

    def test_passes_everything_else(self):
        common.check_benchmark_args(["--limit", "0", "--concurrency", "32", "--prompt", "Correct it"])


class ReasoningLeakTests(unittest.TestCase):
    def test_finds_reasoning_markers(self):
        for answer, marker in (
            ("<think>\nOkay, the user wants...\n</think>\n\n목요일이었습니다.", "<think>"),
            ("still thinking</think>목요일이었습니다.", "</think>"),
            ("<thought>hmm</thought>목요일", "<thought>"),
            ("[THINK]hmm[/THINK]목요일", "[THINK]"),
            ("<|channel|>analysis<|message|>hmm", "<|channel|>"),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(common.leaked_reasoning(answer), marker)

    def test_a_plain_answer_is_clean(self):
        self.assertIsNone(common.leaked_reasoning("목요일이었습니다."))

    def test_prompt_follows_the_benchmark_arguments(self):
        self.assertEqual(common.benchmark_prompt(["--limit", "0"]), common.DEFAULT_PROMPT)
        self.assertEqual(common.benchmark_prompt(["--prompt", "Fix it."]), "Fix it.")
        self.assertEqual(common.benchmark_prompt(["--prompt=Fix it."]), "Fix it.")


class HelperTests(unittest.TestCase):
    def test_safe_name_matches_run_models(self):
        self.assertEqual(common.safe_name("Qwen/Qwen3-8B-FP8"), "Qwen_Qwen3-8B-FP8")
        self.assertEqual(common.safe_name("a:b c.d"), "a_b_c.d")

    def test_durations(self):
        self.assertEqual(common.parse_duration("90"), 90)
        self.assertEqual(common.parse_duration("30m"), 1800)
        self.assertEqual(common.parse_duration("12h"), 43200)
        with self.assertRaises(ValueError):
            common.parse_duration("1d")

    def test_cost_and_batch_id(self):
        self.assertEqual(common.cost_usd(1800, 2.0), 1.0)
        self.assertEqual(
            common.new_batch_id(datetime(2026, 9, 15, 12, 3, 4, tzinfo=timezone.utc)),
            "20260915-120304",
        )
        self.assertEqual(common.key("B", "results", "x.json"), "kolla/B/results/x.json")


if __name__ == "__main__":
    unittest.main()
