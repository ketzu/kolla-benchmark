import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vast"))

import common  # noqa: E402
import runner  # noqa: E402


class FakeStorage:
    def __init__(self, failing_prefix=None):
        self.objects = {}
        self.failing_prefix = failing_prefix

    def put_bytes(self, key, data):
        self.objects[key] = data

    def put_file(self, key, path):
        if self.failing_prefix and key.startswith(self.failing_prefix):
            raise ConnectionError("bucket unreachable")
        self.objects[key] = Path(path).read_bytes()

    def get_bytes(self, key):
        return self.objects.get(key)


class FakeProcess:
    def __init__(self, exit_code):
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code


class FakeSystem:
    """vLLM starts at once and answers unless told otherwise; a benchmark takes 100 s."""

    def __init__(self, crashing=(), never_ready=(), failing_benchmark=(), leaking_probe=(), leaking_answers=()):
        self.crashing = crashing
        self.never_ready = never_ready
        self.failing_benchmark = failing_benchmark
        self.leaking_probe = leaking_probe
        self.leaking_answers = leaking_answers
        self.probes = []
        self.now = 0.0
        self.current = None
        self.started = []
        self.stopped = []
        self.removed = []
        self.benchmarks = []

    def start(self, command, log):
        self.current = command[2]
        self.started.append(self.current)
        log.write_text("loading weights\n", encoding="utf-8")
        return FakeProcess(1 if self.current in self.crashing else None)

    def stop(self, process):
        self.stopped.append(self.current)

    def ready(self):
        return self.current not in self.never_ready

    def run(self, command, log, cwd):
        self.benchmarks.append(command)
        self.now += 100
        model = command[command.index("--model") + 1]
        if model in self.failing_benchmark:
            log.write_text("endpoint kept failing\n", encoding="utf-8")
            return 1
        log.write_text(f"Evaluating {model}\n", encoding="utf-8")
        answer = "<think>\nOkay...\n</think>\n\n목요일" if model in self.leaking_answers else "목요일"
        output = Path(command[command.index("--output") + 1])
        run = {"metrics": {"f05": 0.5}, "results": [{"answer": "목요일"}, {"answer": answer}]}
        output.write_text(json.dumps(run), encoding="utf-8")
        return 0

    def probe(self, model, messages):
        self.probes.append((model, messages))
        return "<think>\nOkay\n</think>\n목요일" if model in self.leaking_probe else "목요일"

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def weights_bytes(self, model):
        return 1234

    def remove_model_cache(self, model):
        self.removed.append(model)

    def vllm_version(self):
        return "0.29.0"


MODELS = ("a/one", "b/two")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_runner(self, system, storage=None, status=None):
        manifest = {
            "batch": "B",
            "image": "vllm/vllm-openai:test",
            "models": [{"model": model, "vllm_args": []} for model in MODELS],
            "benchmark_args": ["--limit", "0"],
        }
        if status is None:
            status = common.new_status("B", MODELS)
        status.update(instance_id=7, gpu_name="H100", num_gpus=1, dph=3.6)
        storage = storage or FakeStorage()
        return runner.Runner(storage, system, status, manifest, self.work, heartbeat_seconds=3600), storage

    def stored_status(self, storage):
        return json.loads(storage.objects["kolla/B/status.json"])

    def states(self, storage):
        return {record["model"]: record["state"] for record in self.stored_status(storage)["models"]}

    def test_runs_every_model_and_uploads_runs_sidecars_and_logs(self):
        system = FakeSystem()
        batch_runner, storage = self.make_runner(system)

        self.assertEqual(batch_runner.run(), "finished")

        self.assertEqual(self.states(storage), {"a/one": "done", "b/two": "done"})
        self.assertEqual(self.stored_status(storage)["models"][0]["f05"], 0.5)
        for name in ("results/a_one.json", "results/a_one.vast.json", "logs/a_one.vllm.log", "logs/a_one.bench.log"):
            self.assertIn(f"kolla/B/{name}", storage.objects)
        sidecar = json.loads(storage.objects["kolla/B/results/b_two.vast.json"])
        self.assertEqual(sidecar["cost_usd"], 0.1)
        self.assertEqual(sidecar["weights_bytes"], 1234)
        self.assertEqual(sidecar["gpu_name"], "H100")
        command = system.benchmarks[0]
        self.assertEqual(command[command.index("--url") + 1], common.ENDPOINT)
        self.assertEqual(command[-2:], ["--limit", "0"])
        self.assertEqual(system.stopped, list(MODELS))
        self.assertEqual(system.removed, list(MODELS))

    def test_a_model_vllm_cannot_serve_fails_and_the_batch_continues(self):
        system = FakeSystem(crashing=("a/one",))
        batch_runner, storage = self.make_runner(system)

        self.assertEqual(batch_runner.run(), "finished")

        record = self.stored_status(storage)["models"][0]
        self.assertEqual(record["state"], "failed")
        self.assertIn("vllm exited with 1", record["error"])
        self.assertIn("loading weights", record["error"])
        self.assertNotIn("kolla/B/results/a_one.json", storage.objects)
        self.assertIn("kolla/B/logs/a_one.vllm.log", storage.objects)
        self.assertEqual(self.states(storage)["b/two"], "done")

    def test_gives_up_on_a_server_that_never_answers(self):
        system = FakeSystem(never_ready=("a/one",))
        batch_runner, storage = self.make_runner(system)

        batch_runner.run()

        record = self.stored_status(storage)["models"][0]
        self.assertIn("not ready after 45 minutes", record["error"])
        self.assertGreaterEqual(system.now, runner.STARTUP_TIMEOUT_SECONDS)
        self.assertEqual(system.stopped[0], "a/one")

    def test_a_failing_benchmark_fails_the_model(self):
        system = FakeSystem(failing_benchmark=("b/two",))
        batch_runner, storage = self.make_runner(system)

        batch_runner.run()

        record = self.stored_status(storage)["models"][1]
        self.assertEqual(record["state"], "failed")
        self.assertIn("benchmark exited with 1", record["error"])
        self.assertIn("endpoint kept failing", record["error"])

    def test_a_model_whose_reasoning_leaks_fails_before_the_benchmark(self):
        system = FakeSystem(leaking_probe=("a/one",))
        batch_runner, storage = self.make_runner(system)

        batch_runner.run()

        record = self.stored_status(storage)["models"][0]
        self.assertEqual(record["state"], "failed")
        self.assertIn("--reasoning-parser", record["error"])
        self.assertEqual([command[command.index("--model") + 1] for command in system.benchmarks], ["b/two"])
        self.assertEqual(
            system.probes[0],
            ("a/one", common.benchmark_messages([], runner.PROBE_SENTENCE)),
        )
        self.assertEqual(system.stopped, list(MODELS))

    def test_a_run_with_reasoning_in_its_answers_is_rejected(self):
        system = FakeSystem(leaking_answers=("a/one",))
        batch_runner, storage = self.make_runner(system)

        batch_runner.run()

        record = self.stored_status(storage)["models"][0]
        self.assertEqual(record["state"], "failed")
        self.assertIn("1 of 2 answers", record["error"])
        self.assertNotIn("kolla/B/results/a_one.json", storage.objects)
        self.assertIn("kolla/B/logs/a_one.rejected.json", storage.objects)
        self.assertEqual(self.states(storage)["b/two"], "done")

    def test_a_failed_upload_fails_the_model(self):
        system = FakeSystem()
        batch_runner, storage = self.make_runner(system, FakeStorage(failing_prefix="kolla/B/results/"))

        self.assertEqual(batch_runner.run(), "failed")

        self.assertIn("ConnectionError: bucket unreachable", self.stored_status(storage)["models"][0]["error"])

    def test_the_batch_fails_when_every_model_fails(self):
        batch_runner, _ = self.make_runner(FakeSystem(crashing=MODELS))

        self.assertEqual(batch_runner.run(), "failed")

    def test_resumes_after_a_restart(self):
        status = common.new_status("B", MODELS)
        status["models"][0].update(state="done", attempts=1, f05=0.4)
        status["models"][1].update(state="benchmarking", attempts=1)
        system = FakeSystem()
        batch_runner, storage = self.make_runner(system, status=status)

        batch_runner.run()

        self.assertEqual(system.started, ["b/two"])
        records = self.stored_status(storage)["models"]
        self.assertEqual(records[0]["f05"], 0.4)
        self.assertEqual((records[1]["state"], records[1]["attempts"]), ("done", 2))

    def test_gives_up_on_a_model_interrupted_twice(self):
        status = common.new_status("B", MODELS)
        status["models"][0].update(state="serving", attempts=2)
        system = FakeSystem()
        batch_runner, storage = self.make_runner(system, status=status)

        batch_runner.run()

        self.assertEqual(system.started, ["b/two"])
        record = self.stored_status(storage)["models"][0]
        self.assertEqual((record["state"], record["error"]), ("failed", "interrupted 2 times"))


class InstallPackagesTests(unittest.TestCase):
    def test_nothing_to_install(self):
        commands = []

        self.assertTrue(runner.install_packages({"pip": []}, lambda command: commands.append(command) or 0))
        self.assertTrue(runner.install_packages({}, lambda command: commands.append(command) or 0))
        self.assertEqual(commands, [])

    def test_installs_the_batch_packages(self):
        commands = []

        installed = runner.install_packages({"pip": ["cohere-melody>=0.11.1"]}, lambda command: commands.append(command) or 0)

        self.assertTrue(installed)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][-1], "cohere-melody>=0.11.1")
        self.assertIn("pip", commands[0])

    def test_falls_back_when_pip_refuses_and_reports_total_failure(self):
        commands = []

        installed = runner.install_packages({"pip": ["a"]}, lambda command: commands.append(command) or 1)

        self.assertFalse(installed)
        self.assertGreaterEqual(len(commands), 2)
        self.assertTrue(any("--break-system-packages" in command for command in commands))
        self.assertTrue(all(command[-1] == "a" for command in commands))


class CurrentStatusTests(unittest.TestCase):
    manifest = {"batch": "B", "models": [{"model": "a/one", "vllm_args": []}]}
    env = {"CONTAINER_ID": "42", "KOLLA_GPU_NAME": "H100", "KOLLA_DPH": "2.5", "GPU_COUNT": "2"}

    def test_starts_a_status_for_this_instance(self):
        status = runner.current_status(FakeStorage(), self.env, self.manifest)

        self.assertEqual((status["instance_id"], status["num_gpus"], status["dph"]), (42, 2, 2.5))
        self.assertEqual(status["models"][0]["state"], "pending")
        self.assertIsNotNone(status["started_at"])

    def test_keeps_progress_and_restarts_the_clock_for_a_new_instance(self):
        storage = FakeStorage()
        stored = common.new_status("B", ["a/one"])
        stored.update(instance_id=41, started_at="2026-09-15T00:00:00Z")
        stored["models"][0]["state"] = "done"
        storage.put_bytes("kolla/B/status.json", json.dumps(stored).encode())

        status = runner.current_status(storage, self.env, self.manifest)

        self.assertEqual(status["models"][0]["state"], "done")
        self.assertNotEqual(status["started_at"], "2026-09-15T00:00:00Z")

        storage.put_bytes("kolla/B/status.json", json.dumps(status).encode())
        again = runner.current_status(storage, self.env, self.manifest)
        self.assertEqual(again["started_at"], status["started_at"])


if __name__ == "__main__":
    unittest.main()
