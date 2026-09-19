# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
"""Benchmark self-hosted models on rented vast.ai GPUs.

    uv run scripts/vast/vast_bench.py launch --models scripts/vllm.txt --gpu 'gpu_ram>=80 num_gpus=1' -- --limit 0 --concurrency 32
    uv run scripts/vast/vast_bench.py status [BATCH]
    uv run scripts/vast/vast_bench.py pull [BATCH] [--logs]
    uv run scripts/vast/vast_bench.py destroy [BATCH]

launch rents one instance for the batch and returns once it has booted. The instance serves
every model with vLLM, benchmarks it, uploads the run to the S3 bucket and destroys itself
when the list is done or the deadline passes; this machine can go offline in the meantime.
Everything after -- is passed on to the benchmark.

Credentials come from the environment or .env in the repository root: VAST_API_KEY,
S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_ENDPOINT (unless AWS), optional
S3_REGION and HF_TOKEN (gated models).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    IMAGE,
    MIN_CUDA,
    RUNNER_FILES,
    VAST_API,
    WORK,
    Storage,
    check_benchmark_args,
    key,
    manifest_models,
    new_batch_id,
    parse_duration,
    parse_model_list,
    parse_utc,
    read_json,
)

VAST_API_V1 = "https://console.vast.ai/api/v1"
HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
ONSTART_LIMIT = 4048
# S3 refuses presigned URLs that live longer than a week.
MAX_PRESIGN_SECONDS = 7 * 24 * 3600
BOOT_POLL_SECONDS = 15
STALE_HEARTBEAT_SECONDS = 600
OFFER_LIMIT = 20
S3_VARIABLES = ("S3_ENDPOINT", "S3_BUCKET", "S3_REGION", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")

OPERATORS = {">=": "gte", "<=": "lte", "!=": "neq", ">": "gt", "<": "lt", "=": "eq"}
# The vast.ai CLI takes these in GB, the API wants MB; accept them the CLI way.
MEGABYTE_FIELDS = {"gpu_ram", "gpu_total_ram"}


def parse_offer_filter(text: str) -> dict[str, dict[str, Any]]:
    """Read ``gpu_ram>=80 num_gpus=1 gpu_name=H100_SXM,H100_NVL`` into an API query."""
    query: dict[str, dict[str, Any]] = {}
    for term in text.split():
        match = re.fullmatch(r"([a-z_]+)(>=|<=|!=|>|<|=)(.+)", term)
        if not match:
            raise ValueError(f"cannot read offer filter {term!r}, use e.g. gpu_ram>=80")
        field, operator, raw = match.groups()
        values = [_filter_value(field, value) for value in raw.split(",")]
        if operator == "=" and len(values) > 1:
            query[field] = {"in": values}
        else:
            query.setdefault(field, {})[OPERATORS[operator]] = values[0]
    return query


def _filter_value(field: str, raw: str) -> Any:
    if raw in ("true", "false"):
        return raw == "true"
    try:
        number = float(raw)
    except ValueError:
        # GPU names carry spaces, the command line does not: RTX_4090 is "RTX 4090".
        return raw.replace("_", " ")
    if field in MEGABYTE_FIELDS:
        number *= 1000
    return int(number) if number.is_integer() else number


def offer_query(gpu_filter: str, disk: int, max_dph: float | None) -> dict[str, Any]:
    query: dict[str, Any] = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "disk_space": {"gte": disk},
        "reliability": {"gte": 0.98},
        "inet_down": {"gte": 500},
        "cuda_max_good": {"gte": MIN_CUDA},
    }
    if max_dph is not None:
        query["dph_total"] = {"lte": max_dph}
    query.update(parse_offer_filter(gpu_filter))
    query.update(type="ondemand", order=[["dph_total", "asc"]], limit=OFFER_LIMIT)
    return query


class VastClient:
    """The few vast.ai REST calls a batch needs."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def _request(self, method: str, path: str, body: Any = None, base: str = VAST_API) -> Any:
        request = urllib.request.Request(
            f"{base}{path}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise SystemExit(f"vast.ai {method} {path} failed with {error.code}: {detail}") from error

    def search_offers(self, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        return self._request("POST", "/bundles/", query)["offers"]

    def create_instance(self, offer_id: int, body: Mapping[str, Any]) -> int:
        response = self._request("PUT", f"/asks/{offer_id}/", body)
        if not response.get("success"):
            raise SystemExit(f"vast.ai refused offer {offer_id}: {response}")
        return int(response["new_contract"])

    def instances(self) -> list[dict[str, Any]]:
        # The v0 listing is retired; v1 pages its results.
        found: list[dict[str, Any]] = []
        token = None
        while True:
            query = "?" + urllib.parse.urlencode({"next_token": token}) if token else ""
            page = self._request("GET", f"/instances/{query}", base=VAST_API_V1)
            found += page.get("instances") or []
            token = page.get("next_token")
            if not token:
                return found

    def instance(self, instance_id: int) -> dict[str, Any] | None:
        return self._request("GET", f"/instances/{instance_id}/").get("instances")

    def destroy_instance(self, instance_id: int) -> None:
        self._request("DELETE", f"/instances/{instance_id}/")


def load_dotenv(path: Path, env: dict[str, str]) -> None:
    """Fill ``env`` from a KEY=VALUE file without overriding what is already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        env.setdefault(name.strip().removeprefix("export "), value.strip().strip("'\""))


def require(env: Mapping[str, str], *names: str) -> None:
    missing = [name for name in names if not env.get(name)]
    if missing:
        raise SystemExit(f"missing configuration: {', '.join(missing)} (environment or .env)")


def source_bundle(repository: Path) -> tuple[str, bool, bytes]:
    """``(commit, dirty, tar.gz)`` of the working tree; tracked uncommitted changes included."""

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True).stdout

    commit = git("rev-parse", "HEAD").decode().strip()
    stash = git("stash", "create").decode().strip()
    return commit, bool(stash), git("archive", "--format=tar.gz", stash or "HEAD")


def onstart_command(storage: Storage, batch: str, expires_seconds: int) -> str:
    """Download the instance scripts with presigned URLs, then start onstart.sh."""
    fetch = "import sys,urllib.request as r;a=sys.argv[1:];[r.urlretrieve(a[i],a[i+1]) for i in range(0,len(a),2)]"
    downloads = " ".join(
        f"'{storage.presign_get(key(batch, name), expires_seconds)}' {name}" for name in RUNNER_FILES
    )
    command = f"mkdir -p {WORK} && cd {WORK} && python3 -c '{fetch}' {downloads} && bash onstart.sh"
    if len(command) > ONSTART_LIMIT:
        raise SystemExit(f"onstart command is {len(command)} characters, vast.ai takes {ONSTART_LIMIT}")
    return command


def instance_request(
    batch: str,
    offer: Mapping[str, Any],
    env: Mapping[str, str],
    onstart: str,
    deadline_at: int,
    disk: int,
    image: str,
) -> dict[str, Any]:
    variables = {
        "KOLLA_BATCH": batch,
        "KOLLA_DEADLINE_AT": str(deadline_at),
        "KOLLA_GPU_NAME": str(offer.get("gpu_name") or ""),
        "KOLLA_DPH": str(offer.get("dph_total") or 0),
        **{name: env[name] for name in (*S3_VARIABLES, "HF_TOKEN") if env.get(name)},
    }
    return {
        "client_id": "me",
        "image": image,
        "env": variables,
        "disk": disk,
        "label": f"kolla-{batch}",
        "onstart": onstart,
        "runtype": "ssh",
        "cancel_unavail": True,
    }


def wait_for_boot(
    storage: Storage,
    vast: VastClient,
    batch: str,
    instance_id: int,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """True once the instance wrote its first heartbeat, False when it will not."""
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        status = read_json(storage, key(batch, "status.json"))
        if status and status.get("instance_id") == instance_id:
            return True
        instance = vast.instance(instance_id) or {}
        state = instance.get("actual_status")
        if state in ("exited", "offline"):
            print(f"  instance {instance_id} is {state}: {instance.get('status_msg') or ''}")
            return False
        print(f"  instance {instance_id}: {state or 'scheduling'} {instance.get('status_msg') or ''}".rstrip())
        sleep(BOOT_POLL_SECONDS)
    print(f"  instance {instance_id} did not report within {timeout_seconds / 60:.0f} minutes")
    return False


def launch(
    options: argparse.Namespace,
    benchmark_args: list[str],
    env: Mapping[str, str],
    storage: Storage,
    vast: VastClient,
    bundle: Callable[[], tuple[str, bool, bytes]],
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    wait: Callable[..., bool] = wait_for_boot,
) -> str:
    entries = parse_model_list(Path(options.models).read_text(encoding="utf-8"))
    if not entries:
        raise SystemExit(f"no models listed in {options.models}")
    try:
        check_benchmark_args(benchmark_args)
        deadline_seconds = parse_duration(options.deadline)
        boot_seconds = parse_duration(options.boot_timeout)
        query = offer_query(options.gpu, options.disk, options.max_dph)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    offers = vast.search_offers(query)[: options.offers]
    if not offers:
        raise SystemExit("no offer matches the GPU filter, disk and price")

    started = now()
    batch = new_batch_id(started)
    commit, dirty, source = bundle()
    storage.put_bytes(key(batch, "source.tar.gz"), source)
    for name in RUNNER_FILES:
        storage.put_file(key(batch, name), HERE / name)
    manifest = {
        "batch": batch,
        "created_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit,
        "dirty": dirty,
        "image": options.image,
        "gpu_filter": options.gpu,
        "deadline_seconds": deadline_seconds,
        "models": manifest_models(entries),
        "benchmark_args": benchmark_args,
        "pip": list(options.pip),
    }
    storage.put_bytes(key(batch, "batch.json"), json.dumps(manifest, indent=2).encode())
    print(f"batch {batch}: {len(entries)} models, commit {commit[:12]}{' with local changes' if dirty else ''}")

    onstart = onstart_command(storage, batch, min(deadline_seconds + 3600, MAX_PRESIGN_SECONDS))
    for offer in offers:
        deadline_at = int(started.timestamp()) + deadline_seconds
        body = instance_request(batch, offer, env, onstart, deadline_at, options.disk, options.image)
        instance_id = vast.create_instance(offer["id"], body)
        print(
            f"rented instance {instance_id}: {offer.get('num_gpus')}x {offer.get('gpu_name')} "
            f"at ${offer.get('dph_total', 0):.3f}/h"
        )
        if wait(storage, vast, batch, instance_id, boot_seconds):
            print(f"instance {instance_id} is running the batch; follow it with: status {batch}")
            return batch
        print(f"  destroying instance {instance_id}, trying the next offer")
        vast.destroy_instance(instance_id)
    raise SystemExit(f"none of {len(offers)} offers came up; batch {batch} was not started")


def show_status(storage: Storage, batch: str, now: datetime) -> None:
    status = read_json(storage, key(batch, "status.json"))
    if status is None:
        print(f"batch {batch}: no instance has reported yet")
        return
    dph = status.get("dph") or 0
    print(f"batch {batch}: {status['phase']}")
    print(f"instance {status.get('instance_id')}: {status.get('num_gpus')}x {status.get('gpu_name')} at ${dph:.3f}/h")
    if status.get("started_at"):
        elapsed = (now - parse_utc(status["started_at"])).total_seconds()
        print(f"running {elapsed / 3600:.2f} h, about ${elapsed / 3600 * dph:.2f}")
    if status.get("heartbeat_at"):
        age = (now - parse_utc(status["heartbeat_at"])).total_seconds()
        stale = status["phase"] not in ("finished", "failed") and age > STALE_HEARTBEAT_SECONDS
        print(f"last heartbeat {age / 60:.0f} min ago{'  <- stale, check the vast.ai console' if stale else ''}")
    print()
    print(f"{'model':<50} {'state':<13} {'F0.5':>7}  error")
    for record in status["models"]:
        f05 = f"{record['f05']:.4f}" if record.get("f05") is not None else ""
        error = (record.get("error") or "").splitlines()
        print(f"{record['model']:<50} {record['state']:<13} {f05:>7}  {error[0] if error else ''}")


def pull(storage: Storage, batch: str, destination: Path, logs: bool) -> int:
    """Download the results (and logs) of a batch; return how many files were fetched."""
    fetched = 0
    folders = [("results", destination / batch)] + ([("logs", destination / batch / "logs")] if logs else [])
    for folder, target_dir in folders:
        prefix = key(batch, folder) + "/"
        for object_key, size in storage.list(prefix):
            target = target_dir / object_key[len(prefix) :]
            if target.is_file() and target.stat().st_size == size:
                continue
            storage.download(object_key, target)
            fetched += 1
    return fetched


def destroy(vast: VastClient, batch: str) -> int:
    label = f"kolla-{batch}"
    instances = [item for item in vast.instances() if item.get("label") == label]
    for item in instances:
        vast.destroy_instance(item["id"])
        print(f"destroyed instance {item['id']}")
    if not instances:
        print(f"no instance labelled {label}")
    return len(instances)


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    argv = list(argv)
    benchmark_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, benchmark_args = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("launch", help="rent an instance and start a batch")
    start.add_argument("--models", required=True, help="model list: a Hugging Face repo per line, vllm flags after it")
    start.add_argument("--gpu", default="", help="vast.ai offer filter, e.g. 'gpu_ram>=80 num_gpus=1'")
    start.add_argument("--max-dph", type=float, help="most $/hour to pay for the instance")
    start.add_argument("--disk", type=int, default=200, help="disk in GB, room for the largest model (default 200)")
    start.add_argument("--deadline", default="12h", help="destroy the instance after this long (default 12h)")
    start.add_argument("--boot-timeout", default="30m", help="give up on an offer after this long (default 30m)")
    start.add_argument("--offers", type=int, default=3, help="offers to try before giving up (default 3)")
    start.add_argument("--image", default=IMAGE, help=f"vLLM image (default {IMAGE})")
    start.add_argument(
        "--pip",
        action="append",
        default=[],
        metavar="PACKAGE",
        help="Python package to install on the instance before serving, repeatable, "
        "e.g. 'cohere-melody>=0.11.1' for Cohere's reasoning parser",
    )
    for name, text in (("status", "show the progress of a batch"), ("destroy", "destroy a batch's instance")):
        commands.add_parser(name, help=text).add_argument("batch", nargs="?")
    fetch = commands.add_parser("pull", help="download the results of a batch into results/<batch>/")
    fetch.add_argument("batch", nargs="?")
    fetch.add_argument("--logs", action="store_true", help="also download the logs")
    fetch.add_argument("--results-dir", type=Path, default=REPOSITORY / "results")

    options = parser.parse_args(argv)
    if benchmark_args and options.command != "launch":
        parser.error("benchmark arguments after -- only apply to launch")
    return options, benchmark_args


def main(argv: Sequence[str] | None = None) -> int:
    options, benchmark_args = parse_args(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ)
    load_dotenv(REPOSITORY / ".env", env)

    vast = None
    if options.command in ("launch", "destroy"):
        require(env, "VAST_API_KEY")
        vast = VastClient(env["VAST_API_KEY"])
    require(env, "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")
    storage = Storage(env)

    if options.command == "launch":
        launch(options, benchmark_args, env, storage, vast, lambda: source_bundle(REPOSITORY))
        return 0

    batch = options.batch
    if batch is None:
        batches = storage.list_batches()
        if not batches:
            raise SystemExit(f"no batches in bucket {env['S3_BUCKET']}")
        batch = batches[-1]

    if options.command == "status":
        show_status(storage, batch, datetime.now(timezone.utc))
    elif options.command == "pull":
        fetched = pull(storage, batch, options.results_dir, options.logs)
        print(f"fetched {fetched} files into {options.results_dir / batch}")
    elif options.command == "destroy":
        destroy(vast, batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
