import tempfile
import unittest
from pathlib import Path

from scripts import model_info


class ModelInfoTests(unittest.TestCase):
    def test_names_the_provider_after_the_endpoint(self):
        self.assertEqual(model_info.provider("https://openrouter.ai/api/v1"), "openrouter")
        self.assertEqual(model_info.provider("http://localhost:1234/v1"), "lmstudio")
        self.assertEqual(model_info.provider("http://gpu-box:8000/v1"), "gpu-box:8000")
        self.assertEqual(model_info.provider(None), "")

    def test_describes_known_models_and_leaves_unknown_ones_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            info_file = Path(temp_dir) / "model-info.csv"
            info_file.write_text(
                "model,company,params,quantization,file_bytes\n"
                "local/model,Maker,3B,Q8_0,1234\n"
                "remote/model,Maker,,,\n",
                encoding="utf-8",
            )
            models = model_info.load(info_file)

            self.assertEqual(
                model_info.describe(
                    {"model": "local/model", "endpoint": "http://localhost:1234/v1"}, models
                ),
                {
                    "provider": "lmstudio",
                    "company": "Maker",
                    "params": "3B",
                    "quantization": "Q8_0",
                    "file_bytes": "1234",
                },
            )
            self.assertEqual(
                model_info.describe({"model": "remote/model"}, models)["quantization"], ""
            )
            self.assertEqual(
                model_info.describe({"model": "missing/model"}, models),
                dict.fromkeys(model_info.COLUMNS, ""),
            )

    def test_a_missing_info_file_describes_nothing(self):
        self.assertEqual(model_info.load(Path("does-not-exist.csv")), {})


if __name__ == "__main__":
    unittest.main()
