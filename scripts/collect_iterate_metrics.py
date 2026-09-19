#!/usr/bin/env python3
"""Collect the metrics of an iterated correction experiment, one row per run.

Every sentence of such a run was sent back to the model answer after answer until an answer came
back unchanged or the run's ``max_iterations`` was reached; only the last answer is scored. Besides
the usual metrics, a row describes how many requests the scored sentences took. Runs are labelled
with their prompt like ``collect_prompt_metrics.py`` does.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_prompt_metrics as prompts  # noqa: E402
import model_info  # noqa: E402


REQUEST_COLUMNS = ["p25", "median", "mean", "p75", "p90", "max"]

# No cost column: the cost file holds one total per model, which cannot be split by experiment.
COLUMNS = [
    "model",
    *model_info.COLUMNS,
    "prompt_name",
    "prompt_type",
    "max_iterations",
    "precision",
    "recall",
    "f05",
    "tp",
    "fp",
    "fn",
    "sentences",
    "failure_count",
    "converged",
    "requests_total",
    *(f"requests_{column}" for column in REQUEST_COLUMNS),
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "exact_match",
    "perfect",
    "unchanged",
    "started",
    "file",
]


def _section_value(payload: Mapping[str, Any], section_name: str, key: str) -> Any:
    section = payload.get(section_name)
    if not isinstance(section, Mapping):
        return 0
    return section.get(key, 0)


def collect_rows(
    results_dir: Path, prompt_file: Path, model_info_file: Path
) -> list[dict[str, Any]]:
    """Return one row for every iterated run below ``results_dir``.

    A run that was not iterated has no request statistics, so it is rejected rather than
    mixed in.
    """
    known_prompts = prompts.load_prompts(prompt_file)
    models = model_info.load(model_info_file)
    rows: list[dict[str, Any]] = []

    for result_file in model_info.run_files(results_dir):
        with result_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance.get("model"):
            raise ValueError(f"{result_file}: missing provenance.model")
        if provenance.get("max_iterations") is None:
            raise ValueError(f"{result_file}: not an iterated run (no provenance.max_iterations)")

        summary = payload.get("summary")
        if not isinstance(summary, Mapping):
            summary = {}
        requests = summary.get("requests")
        if not isinstance(requests, Mapping):
            requests = {}
        prompt_tokens = summary.get("prompt_tokens", 0)
        completion_tokens = summary.get("completion_tokens", 0)
        failures = payload.get("failures")
        system, user = prompts.sent_prompt(provenance)

        rows.append(
            {
                "model": provenance["model"],
                **model_info.describe(provenance, models),
                "prompt_name": known_prompts.get((system, user), prompts.UNKNOWN_PROMPT),
                "prompt_type": prompts.prompt_type(system),
                "max_iterations": provenance["max_iterations"],
                "precision": _section_value(payload, "metrics", "precision"),
                "recall": _section_value(payload, "metrics", "recall"),
                "f05": _section_value(payload, "metrics", "f05"),
                "tp": _section_value(payload, "counts", "tp"),
                "fp": _section_value(payload, "counts", "fp"),
                "fn": _section_value(payload, "counts", "fn"),
                "sentences": summary.get("sentences", 0),
                "failure_count": len(failures) if isinstance(failures, list) else 0,
                "converged": requests.get("converged", 0),
                "requests_total": requests.get("total", 0),
                **{f"requests_{column}": requests.get(column, 0) for column in REQUEST_COLUMNS},
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "exact_match": summary.get("exact_match", 0),
                "perfect": summary.get("perfect", 0),
                "unchanged": summary.get("unchanged", 0),
                "started": provenance.get("started", ""),
                "file": result_file.relative_to(results_dir).as_posix(),
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    default_results_dir = repository_root / "iterate-results"
    default_prompt_file = repository_root / "scripts" / "prompts.json"
    default_model_info_file = repository_root / "scripts" / "model-info.csv"
    parser = argparse.ArgumentParser(
        description="Collect the metrics of an iterated correction experiment."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"root directory to scan (default: {default_results_dir})",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=default_prompt_file,
        help=f"prompt file naming the prompts runs are matched against (default: {default_prompt_file})",
    )
    parser.add_argument(
        "--model-info",
        type=Path,
        default=default_model_info_file,
        help=f"CSV describing each model (default: {default_model_info_file})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="one row per run; defaults to <results-dir>/iterate-metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.results_dir / "iterate-metrics.csv"
    rows = collect_rows(args.results_dir, args.prompts, args.model_info)
    prompts.write_csv(output, COLUMNS, rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
