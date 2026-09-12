import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_m2 import main


class VerifyM2Tests(unittest.TestCase):
    def write_reference(
        self, directory: Path, result: tuple[float, float, float]
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "levenshtein.py").write_text(
            """
def batch_multi_pre_rec_f1(
    candidates,
    sources,
    gold_edits,
    max_unchanged_words=2,
    beta=0.5,
    ignore_whitespace_casing=False,
    verbose=False,
    very_verbose=False,
):
    assert candidates == ["a c"]
    assert sources == ["a b"]
    assert gold_edits == [{0: [(1, 2, "b", ["c"])]}]
    if verbose:
        print("CORRECT EDITS  : 1")
        print("PROPOSED EDITS : 2")
        print("GOLD EDITS     : 1")
    return %r
"""
            % (result,),
            encoding="utf-8",
        )

    def write_result(self, path: Path, *, counts=None, metrics=None) -> None:
        payload = {
            "counts": counts or {"tp": 1, "fp": 1, "fn": 0},
            "metrics": metrics or {"precision": 0.5, "recall": 1.0, "f05": 5 / 9},
            "results": [
                {
                    "base": {
                        "tokens": ["a", "b"],
                        "references": [
                            {
                                "annotator": 0,
                                "edits": [
                                    {
                                        "start": 1,
                                        "end": 2,
                                        "replacements": ["c"],
                                    }
                                ],
                            }
                        ],
                    },
                    "answer_tokens": ["a", "c"],
                    "score": {"counts": {"tp": 1, "fp": 1, "fn": 0}},
                }
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def write_multi_reference_result(self, path: Path) -> None:
        payload = {
            "counts": {"tp": 1, "fp": 0, "fn": 0},
            "metrics": {"precision": 1.0, "recall": 1.0, "f05": 1.0},
            "results": [
                {
                    "base": {
                        "tokens": ["a", "b"],
                        "references": [
                            {
                                "annotator": 0,
                                "edits": [
                                    {
                                        "start": 1,
                                        "end": 2,
                                        "replacements": ["x"],
                                    }
                                ],
                            },
                            {
                                "annotator": 1,
                                "edits": [
                                    {
                                        "start": 1,
                                        "end": 2,
                                        "replacements": ["c"],
                                    }
                                ],
                            },
                        ],
                    },
                    "answer_tokens": ["a", "c"],
                    "score": {"counts": {"tp": 1, "fp": 0, "fn": 0}},
                }
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def write_best_candidate_reference(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "levenshtein.py").write_text(
            """
def batch_multi_pre_rec_f1(
    candidates,
    sources,
    gold_edits,
    max_unchanged_words=2,
    beta=0.5,
    ignore_whitespace_casing=False,
    verbose=False,
    very_verbose=False,
):
    assert candidates == ["a c"]
    assert sources == ["a b"]
    assert len(gold_edits) == 1
    annotator = next(iter(gold_edits[0]))
    if annotator == 0:
        correct, proposed, gold, metrics = 0, 1, 1, (0.0, 0.0, 0.0)
    else:
        assert annotator == 1
        correct, proposed, gold, metrics = 1, 1, 1, (1.0, 1.0, 1.0)
    if verbose:
        print("CORRECT EDITS  : %d" % correct)
        print("PROPOSED EDITS : %d" % proposed)
        print("GOLD EDITS     : %d" % gold)
    return metrics
""",
            encoding="utf-8",
        )

    def run_main(self, result: Path, reference: Path) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = main([str(result), "--reference-dir", str(reference)])
        return status, output.getvalue()

    def test_matching_result_is_verified_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = root / "run.json"
            reference = root / "m2scorer" / "scripts"
            self.write_result(result)
            self.write_reference(reference, (0.5, 1.0, 5 / 9))
            before = result.read_bytes()

            status, output = self.run_main(result, reference)

            self.assertEqual(status, 0)
            self.assertIn("verified", output)
            self.assertEqual(result.read_bytes(), before)

    def test_mismatch_is_reported_and_returns_nonzero_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = root / "run.json"
            reference = root / "m2scorer" / "scripts"
            self.write_result(result, counts={"tp": 0, "fp": 1, "fn": 1})
            self.write_reference(reference, (0.5, 1.0, 5 / 9))
            before = result.read_bytes()

            status, output = self.run_main(result, reference)

            self.assertEqual(status, 1)
            self.assertIn("counts.tp", output)
            self.assertIn("Rust: 0", output)
            self.assertIn("reference: 1", output)
            self.assertEqual(result.read_bytes(), before)

    def test_malformed_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = root / "run.json"
            reference = root / "m2scorer" / "scripts"
            result.write_text(
                json.dumps({"counts": {}, "metrics": {}}), encoding="utf-8"
            )
            self.write_reference(reference, (0.5, 1.0, 5 / 9))

            status, output = self.run_main(result, reference)

            self.assertEqual(status, 2)
            self.assertIn("invalid result file", output)

    def test_best_candidate_is_selected_per_sentence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = root / "run.json"
            reference = root / "m2scorer" / "scripts"
            self.write_multi_reference_result(result)
            self.write_best_candidate_reference(reference)

            status, output = self.run_main(result, reference)

            self.assertEqual(status, 0)
            self.assertIn("TP 1, FP 0, FN 0", output)


if __name__ == "__main__":
    unittest.main()
