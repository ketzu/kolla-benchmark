"""Shared by the vast.ai launcher on this machine and the runner on the rented instance.

Only the standard library is imported at module level: the instance fetches this file
before anything is installed, and the tests run without boto3.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Every batch lives below <PREFIX>/<batch>/ in the bucket.
PREFIX = "kolla"
# The CUDA 12.9 build runs on more hosts than the default CUDA 13 one.
IMAGE = "vllm/vllm-openai:v0.29.0-cu129"
MIN_CUDA = 12.9
VAST_API = "https://console.vast.ai/api/v0"
# Where the instance keeps the batch: scripts, source, results, logs.
WORK = "/workspace/kolla"
VLLM_PORT = 8000
# Resolves to 127.0.0.1 on the instance; the host name marks the runs as vast.ai runs.
ENDPOINT = f"http://vast.local:{VLLM_PORT}/v1"
# Without a limit vLLM reserves KV cache for the full context and small cards fail to load.
DEFAULT_MAX_MODEL_LEN = 16384
# vLLM allows 1024 sequences by default. Hybrid models (Qwen3.5 and later, Nemotron-H) need a
# Mamba cache block per sequence and refuse to start when that many do not fit; the benchmark
# only sends --concurrency requests at a time.
DEFAULT_MAX_NUM_SEQS = 128
# Uploaded next to the source bundle; the instance downloads them first.
RUNNER_FILES = ("onstart.sh", "runner.py", "common.py")
# The runner sets these for every model, a batch may not.
RUNNER_OWNED_FLAGS = ("--model", "-m", "--url", "-u", "--api-key", "--output", "-o")
SIDECAR_SUFFIX = ".vast.json"
# The benchmark's --prompt default, see src/openai.rs.
DEFAULT_PROMPT = "Correct the Korean sentence. Reply with the corrected sentence only."
# Only found in an answer when the server did not split the reasoning off.
REASONING_MARKERS = (
    "<think>", "</think>",
    "<thought>", "</thought>",
    "[THINK]", "[/THINK]",
    "<|channel|>",
    "<|START_THINKING|>", "<|END_THINKING|>",
)  # fmt: skip


@dataclass(frozen=True)
class ModelEntry:
    """One line of a model list: the Hugging Face repo and extra ``vllm serve`` flags."""

    model: str
    vllm_args: tuple[str, ...] = ()


def parse_model_list(text: str) -> list[ModelEntry]:
    """Read a model list: one repo per line, flags after it, ``#`` starts a comment."""
    entries = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            model, *args = shlex.split(line)
            entries.append(ModelEntry(model, tuple(args)))
    return entries


def manifest_models(entries: Iterable[ModelEntry]) -> list[dict[str, Any]]:
    return [{"model": entry.model, "vllm_args": list(entry.vllm_args)} for entry in entries]


def manifest_entries(manifest: Mapping[str, Any]) -> list[ModelEntry]:
    return [ModelEntry(item["model"], tuple(item["vllm_args"])) for item in manifest["models"]]


def _has_flag(args: Iterable[str], *flags: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in args for flag in flags)


def vllm_command(entry: ModelEntry, gpu_count: int) -> list[str]:
    """The ``vllm serve`` command for a model, with the defaults its line does not set."""
    args = list(entry.vllm_args)
    if not _has_flag(args, "--max-model-len"):
        args += ["--max-model-len", str(DEFAULT_MAX_MODEL_LEN)]
    if not _has_flag(args, "--max-num-seqs"):
        args += ["--max-num-seqs", str(DEFAULT_MAX_NUM_SEQS)]
    # The benchmark only sends text. Without image and video inputs vLLM neither loads nor
    # profiles the vision encoder, which saves memory and skips encoder bugs such as
    # EXAONE 4.5's missing input_norm in vLLM 0.29.
    if not _has_flag(args, "--language-model-only", "--limit-mm-per-prompt"):
        args.append("--language-model-only")
    if gpu_count > 1 and not _has_flag(args, "--tensor-parallel-size", "-tp"):
        args += ["--tensor-parallel-size", str(gpu_count)]
    return [
        "vllm", "serve", entry.model,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--served-model-name", entry.model,
        *args,
    ]  # fmt: skip


def check_benchmark_args(args: Iterable[str]) -> None:
    """Refuse benchmark flags the runner sets itself."""
    for arg in args:
        for flag in RUNNER_OWNED_FLAGS:
            long_form = flag.startswith("--")
            if arg == flag or arg.startswith(flag + "=") or (not long_form and arg.startswith(flag)):
                raise ValueError(f"{arg} is set by the runner for every model")


def benchmark_prompt(args: Iterable[str]) -> str:
    """The prompt a run uses: --prompt among the benchmark arguments, or the default."""
    args = list(args)
    for index, arg in enumerate(args):
        if arg == "--prompt" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--prompt="):
            return arg.split("=", 1)[1]
    return DEFAULT_PROMPT


def leaked_reasoning(answer: str) -> str | None:
    """The first reasoning marker in an answer, None when the answer is clean."""
    return next((marker for marker in REASONING_MARKERS if marker in answer), None)


def safe_name(model: str) -> str:
    """The file name of a model's run, the same rule scripts/run-models.sh uses."""
    return re.sub(r"[^a-zA-Z0-9.-]", "_", model)


def key(batch: str, *parts: str) -> str:
    return "/".join((PREFIX, batch, *parts))


def parse_duration(text: str) -> int:
    """Seconds in ``90``, ``90s``, ``30m`` or ``12h``."""
    match = re.fullmatch(r"(\d+)([smh]?)", text.strip())
    if not match:
        raise ValueError(f"cannot read duration {text!r}, use e.g. 90s, 30m or 12h")
    return int(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def new_batch_id(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")


def cost_usd(seconds: float, dph: float) -> float:
    return round(seconds / 3600 * dph, 4)


def new_status(batch: str, models: Iterable[str]) -> dict[str, Any]:
    return {
        "batch": batch,
        "instance_id": None,
        "gpu_name": None,
        "num_gpus": None,
        "dph": None,
        "started_at": None,
        "heartbeat_at": None,
        "phase": "booting",
        "models": [
            {
                "model": model,
                "state": "pending",
                "attempts": 0,
                "started_at": None,
                "finished_at": None,
                "f05": None,
                "error": None,
            }
            for model in models
        ],
    }


def read_json(storage: Storage, object_key: str) -> dict[str, Any] | None:
    raw = storage.get_bytes(object_key)
    return json.loads(raw) if raw is not None else None


def write_status(storage: Storage, status: dict[str, Any]) -> None:
    status["heartbeat_at"] = utc_now()
    storage.put_bytes(key(status["batch"], "status.json"), json.dumps(status, indent=2).encode())


class Storage:
    """The S3-compatible bucket, configured from S3_* environment variables."""

    def __init__(self, env: Mapping[str, str]) -> None:
        import boto3
        from botocore.config import Config

        endpoint = env.get("S3_ENDPOINT") or None
        self.bucket = env["S3_BUCKET"]
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=env.get("S3_REGION") or ("auto" if endpoint else "us-east-1"),
            aws_access_key_id=env["S3_ACCESS_KEY_ID"],
            aws_secret_access_key=env["S3_SECRET_ACCESS_KEY"],
            # Retries with backoff cover the flaky uploads of a long batch.
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
        )

    def put_bytes(self, object_key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=object_key, Body=data)

    def put_file(self, object_key: str, path: Path) -> None:
        self.client.upload_file(str(path), self.bucket, object_key)

    def get_bytes(self, object_key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=object_key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def download(self, object_key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, object_key, str(path))

    def list(self, prefix: str) -> list[tuple[str, int]]:
        """``(key, size)`` of every object below ``prefix``."""
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
        return [(item["Key"], item["Size"]) for page in pages for item in page.get("Contents", [])]

    def list_batches(self) -> list[str]:
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=f"{PREFIX}/", Delimiter="/"
        )
        prefixes = (item["Prefix"] for page in pages for item in page.get("CommonPrefixes", []))
        return sorted(prefix[len(PREFIX) + 1 :].rstrip("/") for prefix in prefixes)

    def presign_get(self, object_key: str, expires_seconds: int) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": object_key}, ExpiresIn=expires_seconds
        )
