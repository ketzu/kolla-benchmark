#!/usr/bin/env python3
"""Collect the metrics of a multi-prompt experiment, split by prompt name and message layout.

Every run is matched against the prompt file by the system prompt and user template it sent, so
renaming a prompt there relabels old runs too. The primary benchmark is collected by
``collect_metrics.py`` instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_info  # noqa: E402


# No cost column: the cost file holds one total per model, which a prompt split cannot divide.
COLUMNS = [
    "model",
    *model_info.COLUMNS,
    "prompt_name",
    "prompt_type",
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
    "started",
    "file",
]

SENTENCE_PLACEHOLDER = "{sentence}"
# A run whose prompt matches no entry of the prompt file.
UNKNOWN_PROMPT = "custom"


def prompt_type(system: str | None) -> str:
    """The message layout a prompt was sent with."""
    return "user" if system is None else "system+user"


def _normalize(text: str | None) -> str | None:
    return None if text is None else text.replace("\r\n", "\n").strip()


def sent_prompt(provenance: Mapping[str, Any]) -> tuple[str | None, str]:
    """The (system, user template) pair a run actually sent.

    Runs written before the prompt became a template stored the system prompt as ``prompt``
    and sent the bare sentence as the user message; no current prompt lacks the placeholder.
    """
    system = provenance.get("system")
    user = provenance.get("prompt") or ""
    if system is None and user and SENTENCE_PLACEHOLDER not in user:
        return _normalize(user), SENTENCE_PLACEHOLDER
    return _normalize(system), _normalize(user)


def load_prompts(prompt_file: Path) -> dict[tuple[str | None, str], str]:
    """Map every (system, user template) of the prompt file to its name."""
    if not prompt_file.is_file():
        return {}
    with prompt_file.open(encoding="utf-8") as handle:
        prompts = json.load(handle)
    return {
        (_normalize(prompt.get("system")), _normalize(prompt["user"])): prompt.get(
            "name", UNKNOWN_PROMPT
        )
        for prompt in prompts
    }


def _section_value(payload: Mapping[str, Any], section_name: str, key: str) -> Any:
    section = payload.get(section_name)
    if not isinstance(section, Mapping):
        return 0
    return section.get(key, 0)


def collect_rows(
    results_dir: Path, prompt_file: Path, model_info_file: Path
) -> list[dict[str, Any]]:
    """Return one row, labelled with its prompt variant, for every JSON result below ``results_dir``."""
    prompts = load_prompts(prompt_file)
    models = model_info.load(model_info_file)
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
        system, user = sent_prompt(provenance)

        rows.append(
            {
                "model": provenance["model"],
                **model_info.describe(provenance, models),
                "prompt_name": prompts.get((system, user), UNKNOWN_PROMPT),
                "prompt_type": prompt_type(system),
                "precision": _section_value(payload, "metrics", "precision"),
                "recall": _section_value(payload, "metrics", "recall"),
                "f05": _section_value(payload, "metrics", "f05"),
                "tp": _section_value(payload, "counts", "tp"),
                "fp": _section_value(payload, "counts", "fp"),
                "fn": _section_value(payload, "counts", "fn"),
                "sentences": summary.get("sentences", 0),
                "failure_count": len(failures) if isinstance(failures, list) else 0,
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


def prompt_matrix(rows: list[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """F0.5 per model (rows) and prompt variant (columns), named ``<name>/<type>``.

    When a model ran a variant more than once, the most recently started run wins.
    """
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (row["model"], f"{row['prompt_name']}/{row['prompt_type']}")
        if key not in latest or str(row["started"]) >= str(latest[key]["started"]):
            latest[key] = row

    variants = sorted({variant for _, variant in latest})
    models = sorted({model for model, _ in latest})
    matrix = []
    for model in models:
        line: dict[str, Any] = {"model": model}
        for variant in variants:
            if (model, variant) in latest:
                line[variant] = latest[(model, variant)]["f05"]
        matrix.append(line)
    return ["model", *variants], matrix


def write_csv(output: Path, columns: list[str], rows: list[Mapping[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    default_results_dir = repository_root / "results"
    default_prompt_file = repository_root / "scripts" / "prompts.json"
    default_model_info_file = repository_root / "scripts" / "model-info.csv"
    parser = argparse.ArgumentParser(
        description="Collect the metrics of a multi-prompt experiment, split by prompt variant."
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
        help="one row per run; defaults to <results-dir>/prompt-metrics.csv",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        help="F0.5 per model and prompt variant; defaults to <results-dir>/prompt-matrix.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.results_dir / "prompt-metrics.csv"
    matrix_output = args.matrix_output or args.results_dir / "prompt-matrix.csv"

    rows = collect_rows(args.results_dir, args.prompts, args.model_info)
    write_csv(output, COLUMNS, rows)
    print(f"wrote {len(rows)} rows to {output}")

    columns, matrix = prompt_matrix(rows)
    write_csv(matrix_output, columns, matrix)
    print(f"wrote {len(matrix)} models x {len(columns) - 1} prompt variants to {matrix_output}")


if __name__ == "__main__":
    main()
