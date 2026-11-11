import json
import runpy
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sanitize_metadata = runpy.run_path(str(SCRIPTS / "sanitize_metadata.py"))["sanitize_metadata"]
build_archive = runpy.run_path(str(SCRIPTS / "package_release.py"))["build_archive"]


class ReleaseTests(unittest.TestCase):
    def test_sanitization_preserves_settings_and_nested_model_consistency(self):
        original = {
            "model_path": "/private/models/Qwen2-VL-7B",
            "checkpoint_metadata": {
                "adapter_base_model": "/private/models/Qwen2-VL-7B",
                "item_prompt": "Represent this product.",
                "temperature": 0.1,
                "num_negatives": 10,
                "hub_token": "private-test-token",
                "train_file_sha256": "unchanged-fingerprint",
            },
            "files": ["/private/feature.npy"],
            "source": "https://huggingface.co/Qwen/Qwen2-VL-7B",
        }
        public = sanitize_metadata(original)
        self.assertNotIn("/private/", json.dumps(public))
        self.assertNotIn("private-test-token", json.dumps(public))
        self.assertEqual(public["model_path"], public["checkpoint_metadata"]["adapter_base_model"])
        for field in ("item_prompt", "temperature", "num_negatives", "train_file_sha256"):
            self.assertEqual(public["checkpoint_metadata"][field], original["checkpoint_metadata"][field])
        self.assertEqual(public["source"], original["source"])
        self.assertEqual(original["checkpoint_metadata"]["hub_token"], "private-test-token")

    def test_packaging_excludes_caches_weights_and_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            for name in [
                "README.md", "src/mmemb/__init__.py", "tests/test_example.py",
                "tests/__pycache__/test_example.cpython-311.pyc",
                "models/adapter_model.safetensors", "outputs/private_config.json",
                "examples/private.npy", "examples/processed/Demo/text.tsv",
                ".private.json",
            ]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            archive = Path(directory) / "release.zip"
            build_archive(root, archive)
            with zipfile.ZipFile(archive) as handle:
                self.assertIsNone(handle.testzip())
                self.assertEqual(set(handle.namelist()), {
                    "UniMI/README.md", "UniMI/src/mmemb/__init__.py", "UniMI/tests/test_example.py",
                    "UniMI/examples/processed/Demo/text.tsv",
                })


if __name__ == "__main__":
    unittest.main()
