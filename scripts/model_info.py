"""Describe the model behind a run: who served it, who made it, and in what form it ran.

Shared by ``collect_metrics.py``, ``collect_prompt_metrics.py`` and ``collect_iterate_metrics.py``.
A result only records the model id and endpoint, so everything but the provider comes from a
hand-maintained model info file.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


# Columns the collectors add after the model.
COLUMNS = ["provider", "company", "params", "quantization", "file_bytes"]
INFO_COLUMNS = COLUMNS[1:]
# Everything the model info file describes, including the name shown on the presentation page.
FILE_COLUMNS = ["display_name", *INFO_COLUMNS]

# Written by the vast.ai runner next to a run: GPU, timings and cost of that run, not a run itself.
SIDECAR_SUFFIX = ".vast.json"

# Providers by the host of their endpoint; any other endpoint is named by its host.
KNOWN_HOSTS = {
    "openrouter.ai": "OpenRouter",
    "localhost": "Local",
    "127.0.0.1": "Local",
    "::1": "Local",
    # The name the vast.ai runner gives the vLLM server on its instance, see scripts/vast.
    "vast.local": "Vast.ai",
}


def run_files(results_dir: Path) -> Iterator[Path]:
    """Every run below ``results_dir``, in a stable order, without the vast.ai sidecars."""
    for path in sorted(results_dir.rglob("*.json")):
        if not path.name.endswith(SIDECAR_SUFFIX):
            yield path


def provider(endpoint: str | None) -> str:
    """Who served a run, named after the endpoint it was sent to."""
    if not endpoint:
        return ""
    host = urlsplit(endpoint).hostname or ""
    return KNOWN_HOSTS.get(host, host)


def load(info_file: Path) -> dict[str, dict[str, str]]:
    """Map every model id of the model info file to its described columns."""
    if not info_file.is_file():
        return {}
    with info_file.open(newline="", encoding="utf-8") as handle:
        return {
            row["model"]: {column: row.get(column) or "" for column in FILE_COLUMNS}
            for row in csv.DictReader(handle)
            if row.get("model")
        }


def describe(provenance: Mapping[str, Any], models: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    """The model columns of a run; a model missing from the info file leaves its cells empty."""
    known = models.get(provenance["model"], {})
    return {
        "provider": provider(provenance.get("endpoint")),
        **{column: known.get(column, "") for column in INFO_COLUMNS},
    }
