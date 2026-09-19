#!/usr/bin/env python3
"""Run a benchmark batch on a rented vast.ai instance.

    runner.py phase booting|building|failed   record the phase with a heartbeat
    runner.py install                          install the batch's extra pip packages
    runner.py fetch                            download and unpack the benchmark source
    runner.py run                              serve and benchmark every model of the batch
    runner.py upload-log                       upload logs/runner.log
    runner.py destroy                          destroy this instance

onstart.sh calls these in order. Everything is configured by the container environment:
KOLLA_BATCH, KOLLA_GPU_NAME, KOLLA_DPH and the S3_* variables set by the launcher, and
CONTAINER_ID, CONTAINER_API_KEY and GPU_COUNT set by vast.ai.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from common import (
    ENDPOINT,
    SIDECAR_SUFFIX,
    VAST_API,
    VLLM_PORT,
    WORK,
    ModelEntry,
    Storage,
    benchmark_messages,
    cost_usd,
    key,
    leaked_reasoning,
    manifest_entries,
    new_status,
    read_json,
    safe_name,
    utc_now,
    vllm_command,
    write_status,
)

STARTUP_TIMEOUT_SECONDS = 45 * 60
READY_POLL_SECONDS = 10
HEARTBEAT_SECONDS = 60
# A model interrupted by a container restart is tried once more, then given up.
MAX_ATTEMPTS = 2
LOG_TAIL_LINES = 20
# One sentence asked before the benchmark, to catch reasoning that ends up in the answer.
PROBE_SENTENCE = "목요일이었습니다 ."
# Reasoning models can think for minutes about a single sentence.
PROBE_TIMEOUT_SECONDS = 600


class ModelFailed(Exception):
    """A model could not be served or benchmarked; the message explains why."""


def tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def say(message: str) -> None:
    """Print to the runner log; a console that cannot show Korean must not fail a model."""
    encoding = sys.stdout.encoding or "utf-8"
    print(message.encode(encoding, "backslashreplace").decode(encoding), flush=True)


class System:
    """Processes, the vLLM port, the clock and the model cache of the instance."""

    def start(self, command: Sequence[str], log: Path) -> subprocess.Popen:
        with log.open("ab") as handle:
            # Own session, so stopping it also stops the engine workers vLLM spawns.
            return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)

    def stop(self, process: subprocess.Popen) -> None:
        import signal

        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        # Leftover workers can hold GPU memory a little longer than the server lives.
        for _ in range(30):
            if not self.ready():
                break
            time.sleep(2)

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=5) as response:
                return response.status == 200
        except OSError:
            return False

    def probe(self, model: str, messages: list[dict[str, str]]) -> str:
        """The answer the served model gives to one request."""
        body = {"model": model, "messages": messages}
        request = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
            return json.load(response)["choices"][0]["message"].get("content") or ""

    def run(self, command: Sequence[str], log: Path, cwd: Path) -> int:
        with log.open("ab") as handle:
            return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=cwd).returncode

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _model_cache(self, model: str) -> Path:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
        except ImportError:
            HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
        return Path(HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))

    def weights_bytes(self, model: str) -> int:
        # Snapshots link into blobs; counting the blobs counts every file once.
        cache = self._model_cache(model)
        return sum(path.stat().st_size for path in (cache / "blobs").glob("*") if path.is_file())

    def remove_model_cache(self, model: str) -> None:
        shutil.rmtree(self._model_cache(model), ignore_errors=True)

    def vllm_version(self) -> str:
        try:
            return metadata.version("vllm")
        except metadata.PackageNotFoundError:
            return ""


class Runner:
    """Serves and benchmarks the models of a batch one after the other."""

    def __init__(
        self,
        storage: Storage,
        system: System,
        status: dict[str, Any],
        manifest: Mapping[str, Any],
        work: Path,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
    ) -> None:
        self.storage = storage
        self.system = system
        self.status = status
        self.manifest = manifest
        self.batch = manifest["batch"]
        self.work = work
        self.binary = work / "source" / "target" / "release" / "kolla-benchmark"
        self.heartbeat_seconds = heartbeat_seconds
        self.lock = threading.Lock()

    def save(self) -> None:
        with self.lock:
            write_status(self.storage, self.status)

    def run(self) -> str:
        """Run every model not finished yet; return the final phase of the batch."""
        self.status["phase"] = "running"
        self.save()
        stop = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat, args=(stop,), daemon=True)
        heartbeat.start()
        try:
            for entry in manifest_entries(self.manifest):
                record = self._record(entry.model)
                if record["state"] in ("done", "failed"):
                    continue
                if record["attempts"] >= MAX_ATTEMPTS:
                    self._update(record, state="failed", error=f"interrupted {record['attempts']} times")
                    continue
                self.run_model(entry, record)
        finally:
            stop.set()
            heartbeat.join()

        states = [record["state"] for record in self.status["models"]]
        self.status["phase"] = "failed" if states and all(s == "failed" for s in states) else "finished"
        self.save()
        return self.status["phase"]

    def run_model(self, entry: ModelEntry, record: dict[str, Any]) -> None:
        model = entry.model
        safe = safe_name(model)
        logs = self.work / "logs"
        results = self.work / "results"
        logs.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        vllm_log = logs / f"{safe}.vllm.log"
        bench_log = logs / f"{safe}.bench.log"
        result = results / f"{safe}.json"
        sidecar = results / f"{safe}{SIDECAR_SUFFIX}"
        rejected = logs / f"{safe}.rejected.json"
        result.unlink(missing_ok=True)
        rejected.unlink(missing_ok=True)

        say(f"=== {model} ===")
        self._update(
            record,
            state="serving",
            attempts=record["attempts"] + 1,
            started_at=utc_now(),
            finished_at=None,
            f05=None,
            error=None,
        )
        command = vllm_command(entry, self.status.get("num_gpus") or 1)
        server = None
        try:
            started = self.system.monotonic()
            server = self.system.start(command, vllm_log)
            problem = self._wait_ready(server)
            if problem:
                raise ModelFailed(f"{problem}\n{tail(vllm_log)}")
            startup_seconds = self.system.monotonic() - started

            # Without the right --reasoning-parser the think block is scored as the correction.
            messages = benchmark_messages(self.manifest["benchmark_args"], PROBE_SENTENCE)
            answer = self.system.probe(model, messages)
            marker = leaked_reasoning(answer)
            if marker:
                raise ModelFailed(
                    f"the answer contains {marker}: vLLM did not separate the reasoning, "
                    f"set --reasoning-parser for this model\n{answer[:500]}"
                )

            self._update(record, state="benchmarking")
            started = self.system.monotonic()
            code = self.system.run(self._benchmark_command(model, result), bench_log, self.work / "source")
            benchmark_seconds = self.system.monotonic() - started
            if code != 0 or not result.is_file():
                raise ModelFailed(f"benchmark exited with {code}\n{tail(bench_log)}")

            run = json.loads(result.read_text(encoding="utf-8"))
            answers = [item.get("answer") or "" for item in run.get("results", [])]
            leaked = [marker for marker in map(leaked_reasoning, answers) if marker]
            if leaked:
                # Kept for a look, but out of results/ where collect_metrics would count it.
                result.replace(rejected)
                raise ModelFailed(
                    f"{len(leaked)} of {len(answers)} answers contain {leaked[0]}: "
                    f"set --reasoning-parser for this model; the run is kept as logs/{rejected.name}"
                )
            f05 = run["metrics"]["f05"]
            facts = self._sidecar(model, command, startup_seconds, benchmark_seconds)
            sidecar.write_text(json.dumps(facts, indent=2), encoding="utf-8")
            for path in (result, sidecar):
                self.storage.put_file(key(self.batch, "results", path.name), path)
            self._update(record, state="done", f05=f05, finished_at=utc_now())
            say(f"{model}: F0.5 {f05:.4f}")
        except Exception as error:  # One model failing must not stop the batch.
            message = str(error) if isinstance(error, ModelFailed) else f"{type(error).__name__}: {error}"
            say(f"{model} failed: {message}")
            self._update(record, state="failed", error=message, finished_at=utc_now())
        finally:
            if server is not None:
                self.system.stop(server)
            self.system.remove_model_cache(model)
            self._upload_logs(vllm_log, bench_log, rejected)

    def _benchmark_command(self, model: str, result: Path) -> list[str]:
        return [
            str(self.binary),
            "--url", ENDPOINT,
            "--api-key", "vast",
            "--model", model,
            "--output", str(result),
            *self.manifest["benchmark_args"],
        ]  # fmt: skip

    def _wait_ready(self, server: Any) -> str | None:
        deadline = self.system.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            code = server.poll()
            if code is not None:
                return f"vllm exited with {code} before it was ready"
            if self.system.ready():
                return None
            if self.system.monotonic() >= deadline:
                return f"vllm not ready after {STARTUP_TIMEOUT_SECONDS // 60} minutes"
            self.system.sleep(READY_POLL_SECONDS)

    def _sidecar(self, model: str, command: list[str], startup: float, benchmark: float) -> dict[str, Any]:
        dph = self.status.get("dph") or 0
        return {
            "model": model,
            "batch": self.batch,
            "instance_id": self.status.get("instance_id"),
            "gpu_name": self.status.get("gpu_name"),
            "num_gpus": self.status.get("num_gpus"),
            "dph": dph,
            "image": self.manifest["image"],
            "vllm_version": self.system.vllm_version(),
            "vllm_command": command,
            "weights_bytes": self.system.weights_bytes(model),
            "startup_seconds": round(startup, 1),
            "benchmark_seconds": round(benchmark, 1),
            "cost_usd": cost_usd(startup + benchmark, dph),
        }

    def _record(self, model: str) -> dict[str, Any]:
        for record in self.status["models"]:
            if record["model"] == model:
                return record
        record = new_status(self.batch, [model])["models"][0]
        self.status["models"].append(record)
        return record

    def _update(self, record: dict[str, Any], **changes: Any) -> None:
        with self.lock:
            record.update(changes)
        self.save()

    def _upload_logs(self, *paths: Path) -> None:
        for path in paths:
            if not path.is_file():
                continue
            try:
                self.storage.put_file(key(self.batch, "logs", path.name), path)
            except Exception as error:  # A missing log must not fail the model.
                say(f"could not upload {path.name}: {error}")

    def _heartbeat(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            try:
                self.save()
            except Exception as error:  # The next beat tries again.
                say(f"heartbeat failed: {error}")


def current_status(storage: Storage, env: Mapping[str, str], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The batch status as stored, or a fresh one, describing this instance."""
    batch = manifest["batch"]
    status = read_json(storage, key(batch, "status.json"))
    if status is None:
        status = new_status(batch, [entry.model for entry in manifest_entries(manifest)])
    instance_id = int(env["CONTAINER_ID"]) if env.get("CONTAINER_ID") else None
    if status.get("instance_id") != instance_id:
        status.update(instance_id=instance_id, started_at=utc_now())
    status.update(
        gpu_name=env.get("KOLLA_GPU_NAME"),
        num_gpus=int(env.get("GPU_COUNT") or 1),
        dph=float(env.get("KOLLA_DPH") or 0),
    )
    return status


def install_packages(manifest: Mapping[str, Any], run: Callable[[list[str]], int]) -> bool:
    """Install the packages the batch asked for with launch --pip; True when there are none."""
    packages = list(manifest.get("pip") or [])
    if not packages:
        return True
    # The image's Python may be marked externally managed, hence the fallbacks.
    attempts = (
        [sys.executable, "-m", "pip", "install", "--quiet"],
        [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages"],
        ["uv", "pip", "install", "--system", "--break-system-packages", "--quiet"],
    )
    for command in attempts:
        try:
            if run([*command, *packages]) == 0:
                return True
        except OSError:  # uv is not in every image
            continue
    return False


def destroy_instance(env: Mapping[str, str], attempts: int = 5) -> None:
    request = urllib.request.Request(
        f"{VAST_API}/instances/{env['CONTAINER_ID']}/",
        method="DELETE",
        headers={"Authorization": f"Bearer {env['CONTAINER_API_KEY']}"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                print(f"destroy requested: {response.status}", flush=True)
                return
        except OSError as error:
            print(f"destroy attempt {attempt} failed: {error}", flush=True)
            time.sleep(10)
    raise SystemExit("could not destroy the instance")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    phase = commands.add_parser("phase")
    phase.add_argument("phase", choices=["booting", "building", "failed"])
    for name in ("install", "fetch", "run", "upload-log", "destroy"):
        commands.add_parser(name)
    args = parser.parse_args(argv)

    env = os.environ
    work = Path(env.get("KOLLA_WORK", WORK))
    if args.command == "destroy":
        destroy_instance(env)
        return 0

    storage = Storage(env)
    batch = env["KOLLA_BATCH"]
    if args.command == "upload-log":
        storage.put_file(key(batch, "logs", "runner.log"), work / "logs" / "runner.log")
        return 0
    if args.command == "fetch":
        archive = work / "source.tar.gz"
        storage.download(key(batch, "source.tar.gz"), archive)
        with tarfile.open(archive) as bundle:
            bundle.extractall(work / "source", filter="data")
        return 0

    manifest = read_json(storage, key(batch, "batch.json"))
    if manifest is None:
        raise SystemExit(f"no batch.json for batch {batch}")
    if args.command == "install":
        installed = install_packages(manifest, lambda command: subprocess.run(command).returncode)
        return 0 if installed else 1
    status = current_status(storage, env, manifest)
    if args.command == "phase":
        status["phase"] = args.phase
        write_status(storage, status)
        return 0

    phase = Runner(storage, System(), status, manifest, work).run()
    print(f"batch {batch} {phase}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
