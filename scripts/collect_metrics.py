#!/usr/bin/env python3
"""Collect flattened metrics from every benchmark result under a results directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


COLUMNS = [
    "model",
    "precision",
    "recall",
    "f05",
    "tp",
    "fp",
    "fn",
    "sentences",
    "failure_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "exact_match",
    "perfect",
    "unchanged",
    "cost",
]


def _section_value(payload: Mapping[str, Any], section_name: str, key: str) -> Any:
    section = payload.get(section_name)
    if not isinstance(section, Mapping):
        return 0
    return section.get(key, 0)


def _load_costs(cost_file: Path) -> dict[str, str]:
    if not cost_file.is_file():
        return {}

    with cost_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["Model"]: row.get("Value") or "0"
            for row in reader
            if row.get("Model")
        }


def collect_rows(results_dir: Path, cost_file: Path) -> list[dict[str, Any]]:
    """Return one flattened row for every JSON result below ``results_dir``."""
    costs = _load_costs(cost_file)
    rows: list[dict[str, Any]] = []

    for result_file in sorted(results_dir.rglob("*.json")):
        with result_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance.get("model"):
            raise ValueError(f"{result_file}: missing provenance.model")

        summary = payload.get("summary")
        if not isinstance(summary, Mapping):
            summary = {}
        prompt_tokens = summary.get("prompt_tokens", 0)
        completion_tokens = summary.get("completion_tokens", 0)
        failures = payload.get("failures")
        failure_count = len(failures) if isinstance(failures, list) else 0
        model = provenance["model"]

        rows.append(
            {
                "model": model,
                "precision": _section_value(payload, "metrics", "precision"),
                "recall": _section_value(payload, "metrics", "recall"),
                "f05": _section_value(payload, "metrics", "f05"),
                "tp": _section_value(payload, "counts", "tp"),
                "fp": _section_value(payload, "counts", "fp"),
                "fn": _section_value(payload, "counts", "fn"),
                "sentences": summary.get("sentences", 0),
                "failure_count": failure_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "exact_match": summary.get("exact_match", 0),
                "perfect": summary.get("perfect", 0),
                "unchanged": summary.get("unchanged", 0),
                "cost": costs.get(model, "0"),
            }
        )

    return rows


def write_report(output: Path, rows: list[Mapping[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, 0) for column in COLUMNS} for row in rows)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    default_results_dir = repository_root / "results"
    parser = argparse.ArgumentParser(
        description="Collect benchmark metrics from all result JSON files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"root directory to scan (default: {default_results_dir})",
    )
    parser.add_argument(
        "--cost-file",
        type=Path,
        help="CSV cost file; defaults to <results-dir>/cost.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output CSV; defaults to <results-dir>/metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cost_file = args.cost_file or args.results_dir / "cost.csv"
    output = args.output or args.results_dir / "metrics.csv"
    rows = collect_rows(args.results_dir, cost_file)
    write_report(output, rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
