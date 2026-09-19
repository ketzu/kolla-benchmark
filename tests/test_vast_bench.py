import argparse
import json
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vast"))

import vast_bench  # noqa: E402
from common import RUNNER_FILES  # noqa: E402


class FakeStorage:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.downloads = []

    def put_bytes(self, key, data):
        self.objects[key] = data

    def put_file(self, key, path):
        self.objects[key] = Path(path).read_bytes()

    def get_bytes(self, key):
        return self.objects.get(key)

    def presign_get(self, key, expires):
        return f"https://bucket.example/{key}?expires={expires}"

    def list(self, prefix):
        return [(key, len(data)) for key, data in sorted(self.objects.items()) if key.startswith(prefix)]

    def download(self, key, path):
        self.downloads.append(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.objects[key])


class FakeVast:
    def __init__(self, offers=(), instances=(), unavailable=()):
        self.offers = list(offers)
        self.unavailable = unavailable
        self.listed = list(instances)
        self.queries = []
        self.created = []
        self.destroyed = []

    def search_offers(self, query):
        self.queries.append(query)
        return self.offers

    def create_instance(self, offer_id, body):
        if offer_id in self.unavailable:
            raise vast_bench.VastError(f"vast.ai PUT /asks/{offer_id}/ failed with 400: no_such_ask")
        self.created.append((offer_id, body))
        return 1000 + len(self.created)

    def instances(self):
        return self.listed

    def instance(self, instance_id):
        return next((item for item in self.listed if item["id"] == instance_id), None)

    def destroy_instance(self, instance_id):
        self.destroyed.append(instance_id)


OFFERS = [
    {"id": 1, "gpu_name": "H100 SXM", "num_gpus": 1, "dph_total": 2.0},
    {"id": 2, "gpu_name": "H100 NVL", "num_gpus": 1, "dph_total": 2.5},
]
ENV = {
    "VAST_API_KEY": "account-secret",
    "S3_BUCKET": "bucket",
    "S3_ENDPOINT": "https://s3.example",
    "S3_ACCESS_KEY_ID": "id",
    "S3_SECRET_ACCESS_KEY": "secret",
    "HF_TOKEN": "hf",
}


class OfferQueryTests(unittest.TestCase):
    def test_reads_the_cli_filter_syntax(self):
        query = vast_bench.parse_offer_filter("gpu_ram>=80 num_gpus=1 gpu_name=H100_SXM,H100_NVL verified=false dph_total<2.5")

        self.assertEqual(
            query,
            {
                "gpu_ram": {"gte": 80000},
                "num_gpus": {"eq": 1},
                "gpu_name": {"in": ["H100 SXM", "H100 NVL"]},
                "verified": {"eq": False},
                "dph_total": {"lt": 2.5},
            },
        )

    def test_refuses_what_it_cannot_read(self):
        with self.assertRaises(ValueError):
            vast_bench.parse_offer_filter("gpu_ram~80")

    def test_user_filter_overrides_defaults(self):
        query = vast_bench.offer_query("reliability>=0.9 gpu_name=RTX_4090", disk=100, max_dph=1.5)

        self.assertEqual(query["reliability"], {"gte": 0.9})
        self.assertEqual(query["gpu_name"], {"eq": "RTX 4090"})
        self.assertEqual(query["disk_space"], {"gte": 100})
        self.assertEqual(query["dph_total"], {"lte": 1.5})
        self.assertEqual(query["order"], [["dph_total", "asc"]])
        self.assertEqual(query["rentable"], {"eq": True})


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.models = Path(self.temp.name) / "models.txt"
        self.models.write_text("Qwen/Qwen3-0.6B\nQwen/Qwen3-8B-FP8 --max-model-len 8192\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def options(self, **changes):
        values = dict(
            models=str(self.models),
            gpu="gpu_ram>=24",
            max_dph=None,
            disk=200,
            deadline="12h",
            boot_timeout="30m",
            offers=3,
            image="vllm/vllm-openai:test",
            pip=[],
            prompts=None,
        )
        values.update(changes)
        return argparse.Namespace(**values)

    def launch(self, vast, storage, boots, benchmark_args=("--limit", "0"), **changes):
        answers = iter(boots)
        return vast_bench.launch(
            self.options(**changes),
            list(benchmark_args),
            ENV,
            storage,
            vast,
            bundle=lambda: ("abcdef1234567890", True, b"tarball"),
            now=lambda: datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc),
            wait=lambda *args: next(answers),
        )

    def test_uploads_the_batch_and_moves_to_the_next_offer_when_one_does_not_boot(self):
        vast = FakeVast(OFFERS)
        storage = FakeStorage()

        batch = self.launch(vast, storage, boots=[False, True])

        self.assertEqual(batch, "20260915-120000")
        manifest = json.loads(storage.objects["kolla/20260915-120000/batch.json"])
        self.assertEqual(manifest["models"][1], {"model": "Qwen/Qwen3-8B-FP8", "vllm_args": ["--max-model-len", "8192"]})
        self.assertEqual(manifest["benchmark_args"], ["--limit", "0"])
        self.assertTrue(manifest["dirty"])
        self.assertEqual(storage.objects["kolla/20260915-120000/source.tar.gz"], b"tarball")
        for name in RUNNER_FILES:
            self.assertIn(f"kolla/20260915-120000/{name}", storage.objects)

        self.assertEqual([offer for offer, _ in vast.created], [1, 2])
        self.assertEqual(vast.destroyed, [1001])
        body = vast.created[1][1]
        self.assertEqual(body["label"], "kolla-20260915-120000")
        self.assertEqual(body["image"], "vllm/vllm-openai:test")
        self.assertEqual(body["env"]["KOLLA_DPH"], "2.5")
        self.assertEqual(body["env"]["KOLLA_DEADLINE_AT"], str(int(datetime(2026, 9, 15, 12, tzinfo=timezone.utc).timestamp()) + 43200))
        self.assertEqual(body["env"]["HF_TOKEN"], "hf")
        self.assertNotIn("account-secret", json.dumps(body))
        self.assertIn("bash onstart.sh", body["onstart"])
        self.assertEqual(vast.queries[0]["gpu_ram"], {"gte": 24000})

    def test_uploads_the_instance_scripts_with_unix_line_ends(self):
        scripts = Path(self.temp.name) / "vast"
        scripts.mkdir()
        for name in RUNNER_FILES:
            (scripts / name).write_bytes(b"#!/usr/bin/env bash\r\nset -eu\r\n")
        storage = FakeStorage()

        with mock.patch.object(vast_bench, "HERE", scripts):
            batch = self.launch(FakeVast(OFFERS), storage, boots=[True])

        for name in RUNNER_FILES:
            self.assertEqual(storage.objects[f"kolla/{batch}/{name}"], b"#!/usr/bin/env bash\nset -eu\n")

    def test_records_extra_packages_for_the_instance(self):
        storage = FakeStorage()

        batch = self.launch(FakeVast(OFFERS), storage, boots=[True], pip=["cohere-melody>=0.11.1"])

        manifest = json.loads(storage.objects[f"kolla/{batch}/batch.json"])
        self.assertEqual(manifest["pip"], ["cohere-melody>=0.11.1"])

    def test_records_the_prompt_list_for_the_instance(self):
        prompts = Path(self.temp.name) / "prompts.json"
        prompts.write_text('[{"name": "a", "user": "{sentence}"}, {"system": "S", "user": "{sentence}"}]', encoding="utf-8")
        storage = FakeStorage()

        batch = self.launch(FakeVast(OFFERS), storage, boots=[True], prompts=str(prompts))

        manifest = json.loads(storage.objects[f"kolla/{batch}/batch.json"])
        self.assertEqual(manifest["prompts"], [{"name": "a", "user": "{sentence}"}, {"system": "S", "user": "{sentence}"}])

    def test_refuses_a_broken_prompt_list_or_prompt_flags_before_renting(self):
        prompts = Path(self.temp.name) / "prompts.json"
        prompts.write_text('[{"user": "{sentence}"}]', encoding="utf-8")
        broken = Path(self.temp.name) / "broken.json"
        broken.write_text('[{"user": "no placeholder"}]', encoding="utf-8")
        vast = FakeVast(OFFERS)

        with self.assertRaises(SystemExit):
            self.launch(vast, FakeStorage(), boots=[True], prompts=str(broken))
        with self.assertRaises(SystemExit):
            self.launch(vast, FakeStorage(), boots=[True], benchmark_args=["--system", "S"], prompts=str(prompts))
        self.assertEqual(vast.created, [])

    def test_pip_option_collects_every_package(self):
        options, _ = vast_bench.parse_args(["launch", "--models", "m.txt", "--pip", "a>=1", "--pip", "b"])

        self.assertEqual(options.pip, ["a>=1", "b"])

    def test_refuses_benchmark_flags_the_runner_owns(self):
        vast = FakeVast(OFFERS)

        with self.assertRaises(SystemExit):
            self.launch(vast, FakeStorage(), boots=[True], benchmark_args=["--model", "x"])
        self.assertEqual(vast.created, [])

    def test_skips_an_offer_that_is_listed_but_cannot_be_rented(self):
        vast = FakeVast(OFFERS, unavailable=(1,))

        batch = self.launch(vast, FakeStorage(), boots=[True])

        self.assertEqual(batch, "20260915-120000")
        self.assertEqual([offer for offer, _ in vast.created], [2])

    def test_gives_up_when_no_offer_can_be_rented(self):
        vast = FakeVast(OFFERS, unavailable=(1, 2))

        with self.assertRaises(SystemExit):
            self.launch(vast, FakeStorage(), boots=[])
        self.assertEqual(vast.created, [])

    def test_gives_up_after_the_offers_it_may_try(self):
        vast = FakeVast(OFFERS)

        with self.assertRaises(SystemExit):
            self.launch(vast, FakeStorage(), boots=[False], offers=1)
        self.assertEqual(vast.destroyed, [1001])

    def test_stops_when_no_offer_matches(self):
        storage = FakeStorage()

        with self.assertRaises(SystemExit):
            self.launch(FakeVast([]), storage, boots=[])
        self.assertEqual(storage.objects, {})

    def test_onstart_command_must_fit_vast_limit(self):
        class LongUrls(FakeStorage):
            def presign_get(self, key, expires):
                return "https://bucket.example/" + "x" * 2000

        with self.assertRaises(SystemExit):
            vast_bench.onstart_command(LongUrls(), "B", 3600)


class WaitForBootTests(unittest.TestCase):
    def wait(self, storage, vast):
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds
            storage.objects["kolla/B/status.json"] = json.dumps({"instance_id": 5}).encode()

        return vast_bench.wait_for_boot(storage, vast, "B", 5, 600, sleep=sleep, monotonic=lambda: clock[0])

    def test_boots_once_this_instance_reports(self):
        storage = FakeStorage({"kolla/B/status.json": json.dumps({"instance_id": 4}).encode()})

        self.assertTrue(self.wait(storage, FakeVast(instances=[{"id": 5, "actual_status": "loading"}])))

    def test_an_exited_instance_does_not_boot(self):
        vast = FakeVast(instances=[{"id": 5, "actual_status": "exited", "status_msg": "bad gpu"}])

        self.assertFalse(self.wait(FakeStorage(), vast))


class VastClientTests(unittest.TestCase):
    def test_lists_instances_across_pages(self):
        requests = []
        pages = {
            "/instances/": {"instances": [{"id": 1}], "next_token": "abc"},
            "/instances/?next_token=abc": {"instances": [{"id": 2}], "next_token": None},
        }

        class PagedClient(vast_bench.VastClient):
            def _request(self, method, path, body=None, base=vast_bench.VAST_API):
                requests.append((method, base, path))
                return pages[path]

        self.assertEqual([item["id"] for item in PagedClient("key").instances()], [1, 2])
        self.assertEqual({base for _, base, _ in requests}, {vast_bench.VAST_API_V1})


class CommandTests(unittest.TestCase):
    def test_pull_downloads_new_results_only(self):
        storage = FakeStorage(
            {
                "kolla/B/results/a.json": b"{}",
                "kolla/B/results/a.vast.json": b"{\"cost_usd\": 1}",
                "kolla/B/logs/a.vllm.log": b"log",
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir)
            (destination / "B").mkdir()
            (destination / "B" / "a.json").write_bytes(b"{}")

            fetched = vast_bench.pull(storage, "B", destination, logs=True)

            self.assertEqual(fetched, 2)
            self.assertEqual(storage.downloads, ["kolla/B/results/a.vast.json", "kolla/B/logs/a.vllm.log"])
            self.assertTrue((destination / "B" / "logs" / "a.vllm.log").is_file())

    def test_prompt_batches_are_pulled_next_to_the_other_prompt_runs(self):
        self.assertEqual(vast_bench.default_results_dir({"prompts": []}).name, "results")
        self.assertEqual(vast_bench.default_results_dir(None).name, "results")
        self.assertEqual(vast_bench.default_results_dir({"prompts": [{"user": "{sentence}"}]}).name, "multiprompt-results")

    def test_iterated_batches_are_pulled_next_to_the_other_iterated_runs(self):
        iterated = {"benchmark_args": ["--iterate", "--limit", "0"], "prompts": []}
        self.assertEqual(vast_bench.default_results_dir(iterated).name, "iterate-results")
        iterated["prompts"] = [{"user": "{sentence}"}]
        self.assertEqual(vast_bench.default_results_dir(iterated).name, "iterate-results")
        self.assertEqual(vast_bench.default_results_dir({"benchmark_args": ["--limit", "0"]}).name, "results")

    def test_destroy_only_touches_the_batch_instance(self):
        vast = FakeVast(instances=[{"id": 1, "label": "kolla-B"}, {"id": 2, "label": "something else"}])

        self.assertEqual(vast_bench.destroy(vast, "B"), 1)
        self.assertEqual(vast.destroyed, [1])

    def test_benchmark_arguments_follow_the_double_dash(self):
        options, benchmark_args = vast_bench.parse_args(["launch", "--models", "m.txt", "--", "--limit", "0"])

        self.assertEqual((options.command, options.models), ("launch", "m.txt"))
        self.assertEqual(benchmark_args, ["--limit", "0"])

    def test_dotenv_does_not_override_the_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv = Path(temp_dir) / ".env"
            dotenv.write_text("# keys\nS3_BUCKET='from-file'\nexport HF_TOKEN=hf\n", encoding="utf-8")
            env = {"S3_BUCKET": "from-env"}

            vast_bench.load_dotenv(dotenv, env)

            self.assertEqual(env, {"S3_BUCKET": "from-env", "HF_TOKEN": "hf"})


if __name__ == "__main__":
    unittest.main()
