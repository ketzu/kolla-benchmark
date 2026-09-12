#!/usr/bin/env python3
"""Verify stored Rust M2 scores against the Python 3 reference scorer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any


COUNT_FIELDS = ("tp", "fp", "fn")
METRIC_FIELDS = ("precision", "recall", "f05")
REFERENCE_TOTAL_LABELS = {
    "CORRECT EDITS  :": "correct",
    "PROPOSED EDITS :": "proposed",
    "GOLD EDITS     :": "gold",
}


class ReferenceOutput:
    """Keep only the final totals printed by the verbose reference scorer."""

    def __init__(self) -> None:
        self._pending = ""
        self.values: dict[str, int] = {}

    def write(self, text: str) -> int:
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._consume(line.rstrip("\r"))
        return len(text)

    def flush(self) -> None:
        if self._pending:
            self._consume(self._pending)
            self._pending = ""

    def _consume(self, line: str) -> None:
        for label, name in REFERENCE_TOTAL_LABELS.items():
            if line.startswith(label):
                try:
                    self.values[name] = int(line[len(label) :].strip())
                except ValueError as error:
                    raise ValueError(
                        f"reference printed a non-integer {name} total: {line!r}"
                    ) from error


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    return float(value)


def _strings(value: Any, path: str) -> list[str]:
    return [
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    ]


def _load_payload(result_file: Path) -> Mapping[str, Any]:
    try:
        with result_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error}") from error
    return _mapping(payload, "result file")


def _stored_values(
    payload: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, float]]:
    counts = _mapping(payload.get("counts"), "counts")
    stored_counts = {
        field: _integer(counts.get(field), f"counts.{field}") for field in COUNT_FIELDS
    }
    metrics = _mapping(payload.get("metrics"), "metrics")
    stored_metrics = {
        field: _number(metrics.get(field), f"metrics.{field}")
        for field in METRIC_FIELDS
    }
    return stored_counts, stored_metrics


def _reference_inputs(
    payload: Mapping[str, Any],
) -> tuple[
    list[str], list[str], list[dict[int, list[tuple[int, int, str, list[str]]]]]
]:
    results = _sequence(payload.get("results"), "results")
    candidates: list[str] = []
    sources: list[str] = []
    gold_edits: list[dict[int, list[tuple[int, int, str, list[str]]]]] = []

    for result_index, value in enumerate(results):
        result_path = f"results[{result_index}]"
        result = _mapping(value, result_path)
        base = _mapping(result.get("base"), f"{result_path}.base")
        source_tokens = _strings(base.get("tokens"), f"{result_path}.base.tokens")
        answer_tokens = _strings(
            result.get("answer_tokens"), f"{result_path}.answer_tokens"
        )
        references = _sequence(base.get("references"), f"{result_path}.base.references")

        sentence_gold: dict[int, list[tuple[int, int, str, list[str]]]] = {}
        for reference_index, reference_value in enumerate(references):
            reference_path = f"{result_path}.base.references[{reference_index}]"
            reference = _mapping(reference_value, reference_path)
            annotator = _integer(
                reference.get("annotator"), f"{reference_path}.annotator"
            )
            if annotator in sentence_gold:
                raise ValueError(f"duplicate annotator {annotator} in {result_path}")

            edits = _sequence(reference.get("edits"), f"{reference_path}.edits")
            parsed_edits: list[tuple[int, int, str, list[str]]] = []
            for edit_index, edit_value in enumerate(edits):
                edit_path = f"{reference_path}.edits[{edit_index}]"
                edit = _mapping(edit_value, edit_path)
                start = _integer(edit.get("start"), f"{edit_path}.start")
                end = _integer(edit.get("end"), f"{edit_path}.end")
                if start < 0 or start > end or end > len(source_tokens):
                    raise ValueError(
                        f"{edit_path} span {start}..{end} is outside the source"
                    )
                replacements = _strings(
                    edit.get("replacements"), f"{edit_path}.replacements"
                )
                parsed_edits.append(
                    (start, end, " ".join(source_tokens[start:end]), replacements)
                )
            sentence_gold[annotator] = parsed_edits

        if not sentence_gold:
            sentence_gold[0] = []

        sources.append(" ".join(source_tokens))
        candidates.append(" ".join(answer_tokens))
        gold_edits.append(sentence_gold)

    return candidates, sources, gold_edits


def _reference_module(reference_dir: Path) -> ModuleType:
    candidates = [reference_dir / "levenshtein.py"]
    if reference_dir.is_dir():
        candidates.append(reference_dir / "scripts" / "levenshtein.py")
    module_path = next((path for path in candidates if path.is_file()), None)
    if module_path is None:
        raise ValueError(
            f"could not find levenshtein.py under {reference_dir}; "
            "use --reference-dir with the Python 3 m2scorer scripts directory"
        )

    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(
        "kolla_m2_reference_levenshtein", module_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load reference module {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ValueError(
            f"could not import reference module {module_path}: {error}"
        ) from error
    if not hasattr(module, "batch_multi_pre_rec_f1"):
        raise ValueError(
            f"reference module {module_path} has no batch_multi_pre_rec_f1"
        )
    return module


def _score_one_reference(
    reference: ModuleType,
    candidate: str,
    source: str,
    gold_edits: dict[int, list[tuple[int, int, str, list[str]]]],
) -> tuple[dict[str, int], dict[str, float]]:
    output = ReferenceOutput()
    with redirect_stdout(output):
        metrics = reference.batch_multi_pre_rec_f1(
            [candidate],
            [source],
            [gold_edits],
            max_unchanged_words=2,
            beta=0.5,
            ignore_whitespace_casing=False,
            verbose=True,
            very_verbose=False,
        )
    output.flush()

    if not isinstance(metrics, Sequence) or len(metrics) != 3:
        raise ValueError(f"reference returned invalid metrics: {metrics!r}")
    reference_metrics = {
        field: float(value) for field, value in zip(METRIC_FIELDS, metrics)
    }
    if set(output.values) != set(REFERENCE_TOTAL_LABELS.values()):
        missing = sorted(set(REFERENCE_TOTAL_LABELS.values()) - set(output.values))
        raise ValueError(f"reference did not print final totals: {', '.join(missing)}")

    correct = output.values["correct"]
    proposed = output.values["proposed"]
    gold = output.values["gold"]
    if correct > proposed or correct > gold:
        raise ValueError(
            f"reference returned impossible totals: correct={correct}, "
            f"proposed={proposed}, gold={gold}"
        )
    reference_counts = {
        "tp": correct,
        "fp": proposed - correct,
        "fn": gold - correct,
    }
    return reference_counts, reference_metrics


def _metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    correct = counts["tp"]
    proposed = correct + counts["fp"]
    gold = correct + counts["fn"]
    precision = correct / proposed if proposed else 1.0
    recall = correct / gold if gold else 1.0
    denominator = 0.25 * precision + recall
    f05 = 1.25 * precision * recall / denominator if denominator else 0.0
    return {"precision": precision, "recall": recall, "f05": f05}


def _score_with_reference(
    reference: ModuleType,
    candidates: list[str],
    sources: list[str],
    gold_edits: list[dict[int, list[tuple[int, int, str, list[str]]]]],
) -> tuple[dict[str, int], dict[str, float]]:
    total_counts = {field: 0 for field in COUNT_FIELDS}
    for candidate, source, sentence_golds in zip(candidates, sources, gold_edits):
        best_key: tuple[float, int, int, int] | None = None
        best_counts: dict[str, int] | None = None
        for reference_order, (annotator, gold) in enumerate(sentence_golds.items()):
            counts, metrics = _score_one_reference(
                reference, candidate, source, {annotator: gold}
            )
            key = (
                metrics["f05"],
                counts["tp"],
                -(counts["fp"] + counts["fn"]),
                -reference_order,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_counts = counts
        if best_counts is None:
            raise ValueError("reference set is empty")
        for field in COUNT_FIELDS:
            total_counts[field] += best_counts[field]

    return total_counts, _metrics_from_counts(total_counts)


def _differences(
    stored_counts: Mapping[str, int],
    stored_metrics: Mapping[str, float],
    reference_counts: Mapping[str, int],
    reference_metrics: Mapping[str, float],
    tolerance: float,
) -> list[str]:
    differences = [
        f"counts.{field}: Rust: {stored_counts[field]}, reference: {reference_counts[field]}"
        for field in COUNT_FIELDS
        if stored_counts[field] != reference_counts[field]
    ]
    differences.extend(
        f"metrics.{field}: Rust: {stored_metrics[field]:.12g}, "
        f"reference: {reference_metrics[field]:.12g}"
        for field in METRIC_FIELDS
        if not math.isclose(
            stored_metrics[field],
            reference_metrics[field],
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    )
    return differences


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Verify one stored benchmark result against the Python 3 M2 scorer."
    )
    parser.add_argument("result_file", type=Path, help="Rust result JSON to verify")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=repository_root / "m2scorer" / "scripts",
        help="m2scorer Python 3 scripts directory (default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="absolute/relative tolerance for metric comparisons (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tolerance < 0:
        print("error: --tolerance must be non-negative", file=sys.stderr)
        return 2

    try:
        payload = _load_payload(args.result_file)
        stored_counts, stored_metrics = _stored_values(payload)
        candidates, sources, gold_edits = _reference_inputs(payload)
        reference = _reference_module(args.reference_dir)
        reference_counts, reference_metrics = _score_with_reference(
            reference, candidates, sources, gold_edits
        )
    except (ImportError, OSError, ValueError, TypeError) as error:
        print(f"invalid result file or reference: {error}", file=sys.stderr)
        return 2

    differences = _differences(
        stored_counts,
        stored_metrics,
        reference_counts,
        reference_metrics,
        args.tolerance,
    )
    if differences:
        print(f"differences found in {args.result_file}:")
        for difference in differences:
            print(f"  {difference}")
        return 1

    print(
        f"verified {args.result_file}: "
        f"TP {stored_counts['tp']}, FP {stored_counts['fp']}, FN {stored_counts['fn']}; "
        f"P {stored_metrics['precision']:.4f}, "
        f"R {stored_metrics['recall']:.4f}, F0.5 {stored_metrics['f05']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
