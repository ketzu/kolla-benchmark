#!/usr/bin/env python3
"""Collect flattened metrics and run details from every benchmark result under a results directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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

DETAIL_COLUMNS = [
    "model",
    "display_name",
    "company",
    "provider",
    "endpoint",
    "started_at",
    "rescored_at",
    "perfect",
    "over_corrected_only",
    "under_corrected_only",
    "mixed_error",
    "params",
    "quantization",
    "file_bytes",
]

# Columns that cannot be read from a result and come from the hand maintained model list.
MODEL_INFO_COLUMNS = ["display_name", "company", "params", "quantization", "file_bytes"]

PROVIDERS = {
    "openrouter.ai": "OpenRouter",
    "localhost": "Local",
    "127.0.0.1": "Local",
    "::1": "Local",
    "vast.local": "Vast.ai",
}

# Written by the vast.ai runner next to a run: GPU, timings and cost of that run.
SIDECAR_SUFFIX = ".vast.json"


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


def _load_model_info(model_info_file: Path) -> dict[str, dict[str, str]]:
    if not model_info_file.is_file():
        return {}

    with model_info_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["model"]: row for row in reader if row.get("model")}


def _results(results_dir: Path) -> Iterator[tuple[Path, Mapping[str, Any], Mapping[str, Any]]]:
    """Yield ``(path, payload, provenance)`` for every JSON result below ``results_dir``."""
    for result_file in sorted(results_dir.rglob("*.json")):
        if result_file.name.endswith(SIDECAR_SUFFIX):
            continue
        with result_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance.get("model"):
            raise ValueError(f"{result_file}: missing provenance.model")
        yield result_file, payload, provenance


def _sidecar_cost(result_file: Path) -> str | None:
    """The GPU cost of a vast.ai run, from the sidecar next to it."""
    sidecar = result_file.with_name(result_file.name.removesuffix(".json") + SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return None
    with sidecar.open(encoding="utf-8") as handle:
        cost = json.load(handle).get("cost_usd")
    return None if cost is None else str(cost)


def _provider(endpoint: str) -> str:
    host = urlparse(endpoint).hostname or ""
    return PROVIDERS.get(host, host)


def _outcomes(payload: Mapping[str, Any]) -> dict[str, int]:
    """Sort every scored sentence by the kind of mistake its answer made."""
    outcomes = {
        "perfect": 0,
        "over_corrected_only": 0,
        "under_corrected_only": 0,
        "mixed_error": 0,
    }
    results = payload.get("results")
    for result in results if isinstance(results, list) else []:
        score = result.get("score") if isinstance(result, Mapping) else None
        counts = score.get("counts") if isinstance(score, Mapping) else None
        if not isinstance(counts, Mapping):
            continue
        fp = counts.get("fp", 0)
        fn = counts.get("fn", 0)
        if fp and fn:
            outcomes["mixed_error"] += 1
        elif fp:
            outcomes["over_corrected_only"] += 1
        elif fn:
            outcomes["under_corrected_only"] += 1
        else:
            outcomes["perfect"] += 1
    return outcomes


def collect_rows(results_dir: Path, cost_file: Path) -> list[dict[str, Any]]:
    """Return one flattened row for every JSON result below ``results_dir``."""
    costs = _load_costs(cost_file)
    rows: list[dict[str, Any]] = []

    for result_file, payload, provenance in _results(results_dir):
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
                "cost": costs[model] if model in costs else _sidecar_cost(result_file) or "0",
            }
        )

    return rows


def collect_details(results_dir: Path, model_info_file: Path) -> list[dict[str, Any]]:
    """Return one run description for every JSON result below ``results_dir``.

    Models missing from ``model_info_file`` keep their descriptive columns empty.
    """
    model_info = _load_model_info(model_info_file)
    rows: list[dict[str, Any]] = []

    for result_file, payload, provenance in _results(results_dir):
        model = provenance["model"]
        endpoint = provenance.get("endpoint") or ""
        info = model_info.get(model, {})

        rows.append(
            {
                "model": model,
                **{column: info.get(column) or "" for column in MODEL_INFO_COLUMNS},
                "provider": _provider(endpoint),
                "endpoint": endpoint,
                "started_at": provenance.get("started") or "",
                "rescored_at": provenance.get("rescored") or "",
                **_outcomes(payload),
            }
        )

    return rows


def write_report(
    output: Path,
    rows: list[Mapping[str, Any]],
    columns: Sequence[str] = COLUMNS,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, 0) for column in columns} for row in rows)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    default_results_dir = repository_root / "results"
    default_model_info = repository_root / "scripts" / "model-info.csv"
    parser = argparse.ArgumentParser(
        description="Collect benchmark metrics and run details from all result JSON files."
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
        "--model-info",
        type=Path,
        default=default_model_info,
        help=f"CSV with display name, company and weights per model (default: {default_model_info})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output metrics CSV; defaults to <results-dir>/metrics.csv",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        help="output run details CSV; defaults to <results-dir>/run-details.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cost_file = args.cost_file or args.results_dir / "cost.csv"
    output = args.output or args.results_dir / "metrics.csv"
    details_output = args.details_output or args.results_dir / "run-details.csv"

    rows = collect_rows(args.results_dir, cost_file)
    write_report(output, rows)
    print(f"wrote {len(rows)} rows to {output}")

    details = collect_details(args.results_dir, args.model_info)
    write_report(details_output, details, DETAIL_COLUMNS)
    print(f"wrote {len(details)} rows to {details_output}")
    for model in sorted({row["model"] for row in details if not row["display_name"]}):
        print(f"  {model} has no entry in {args.model_info}")


if __name__ == "__main__":
    main()
