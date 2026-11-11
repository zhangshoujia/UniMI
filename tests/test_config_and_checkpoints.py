import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mmemb.checkpoint import (
    choose_processor_source,
    load_checkpoint_metadata,
    resolve_model_path,
    sha256,
    validate_resume,
)
from mmemb.config import load_config, project_root


class ConfigTests(unittest.TestCase):
    def test_reference_profiles_and_explicit_overrides(self):
        spa = load_config("spa", ["--dry-run"])
        cca = load_config("cca", ["--dry-run"])
        self.assertEqual(
            (spa["batch_size"], spa["reference_num_processes"], spa["max_steps"]), (32, 2, 1000)
        )
        self.assertEqual(
            (cca["num_negatives"], cca["temperature"], cca["max_steps"]), (10, 0.1, 1250)
        )
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "experiment.json"
            overlay.write_text(json.dumps({"batch_size": 3, "max_steps": 7}))
            config = load_config(
                "spa",
                [
                    "--config",
                    str(overlay),
                    "--batch-size",
                    "2",
                    "--no-bf16",
                    "--train-file",
                    "examples/spa.jsonl",
                ],
            )
            self.assertEqual(
                (config["batch_size"], config["max_steps"], config["bf16"]), (2, 7, False)
            )
            self.assertEqual(Path(config["train_file"]), project_root() / "examples/spa.jsonl")
            overlay.write_text('{"batch_szie": 2}')
            with self.assertRaises(ValueError):
                load_config("spa", ["--config", str(overlay)])

    def test_cli_paths_do_not_depend_on_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "mmemb.train_spa", "--dry-run"],
                cwd=directory,
                text=True,
                capture_output=True,
                check=True,
                env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
            )
        config = json.loads(result.stdout)
        self.assertEqual(Path(config["lora_config"]), project_root() / "configs/lora.json")
        self.assertEqual(Path(config["model_path"]), project_root() / "models/Qwen2-VL-7B")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text('{"model_type":"qwen2_vl"}')
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        adapter = {
            "peft_type": "LORA",
            "base_model_name_or_path": str(self.base),
            "target_modules": ["q_proj"],
        }
        (self.checkpoint / "adapter_config.json").write_text(json.dumps(adapter))
        # These tests inspect metadata and paths; they do not deserialize weights.
        (self.checkpoint / "adapter_model.safetensors").write_bytes(b"metadata-fixture")
        self.legacy = dict(
            stage="semantic_preference_alignment",
            preference_loss="pairwise",
            beta=14.2857,
            direct_pooling_method="last",
            item_text_mode="title",
            item_prompt="Represent item.",
            max_length=512,
            max_char_length=20480,
            image_min_pixels=3136,
            image_max_pixels=262144,
            model_path=str(self.base),
            attn_implementation="sdpa",
            bf16=True,
        )
        self.metadata_file = self.checkpoint / "semantic_preference_config.json"
        self.metadata_file.write_text(json.dumps(self.legacy))

    def tearDown(self):
        self.temporary.cleanup()

    def test_legacy_conversion_and_unsupported_objectives(self):
        metadata = load_checkpoint_metadata(self.checkpoint)
        self.assertEqual(
            (metadata["stage"], metadata["pooling"], metadata["title_normalization"]),
            ("spa", "eos", "strip"),
        )
        self.metadata_file.write_text(json.dumps(self.legacy | {"preference_loss": "listwise"}))
        with self.assertRaises(ValueError):
            load_checkpoint_metadata(self.checkpoint)
        self.metadata_file.unlink()
        self.metadata_file = self.checkpoint / "collaborative_contrastive_config.json"
        cca = self.legacy | dict(
            stage="collaborative_contrastive_alignment",
            runtime_model_path=str(self.base),
            stage1_model_path=str(self.base),
            num_normal_negatives=10,
            num_hard_negatives=0,
            in_batch_negatives=False,
            temperature=0.1,
        )
        self.metadata_file.write_text(json.dumps(cca))
        metadata = load_checkpoint_metadata(self.checkpoint)
        self.assertEqual(metadata["num_negatives"], 10)
        self.assertEqual(metadata["title_normalization"], "collapse_whitespace_then_strip")
        self.metadata_file.write_text(json.dumps(cca | {"in_batch_negatives": True}))
        with self.assertRaises(ValueError):
            load_checkpoint_metadata(self.checkpoint)

    def test_relocation_checks_model_configuration_and_resume_content(self):
        metadata = load_checkpoint_metadata(self.checkpoint)
        metadata.update(
            schema_version=1,
            base_model_config_sha256=sha256(self.base / "config.json"),
            train_file_sha256="same-content",
            world_size=1,
            batch_size=2,
        )
        (self.checkpoint / "embedding_config.json").write_text(json.dumps(metadata))
        relocated = self.root / "relocated"
        relocated.mkdir()
        (relocated / "config.json").write_bytes((self.base / "config.json").read_bytes())
        self.assertEqual(resolve_model_path(metadata, relocated), relocated)
        current = copy.deepcopy(metadata)
        current.update(model_path=str(relocated), train_file="/new/location/data.jsonl")
        validate_resume(self.checkpoint, current)
        current["train_file_sha256"] = "changed-content"
        with self.assertRaises(ValueError):
            validate_resume(self.checkpoint, current)
        (relocated / "config.json").write_text('{"model_type":"different"}')
        with self.assertRaises(ValueError):
            resolve_model_path(metadata, relocated)

    def test_native_processor_is_preserved_and_adapter_lineage_is_checked(self):
        legacy = load_checkpoint_metadata(self.checkpoint)
        self.assertEqual(choose_processor_source(self.checkpoint, legacy, self.base), self.base)
        native = legacy | {"schema_version": 1}
        with self.assertRaises(FileNotFoundError):
            choose_processor_source(self.checkpoint, native, self.base)
        for name in ["preprocessor_config.json", "tokenizer_config.json"]:
            (self.checkpoint / name).write_text("{}")
        self.assertEqual(
            choose_processor_source(self.checkpoint, native, self.base), self.checkpoint
        )
        native["adapter_base_model"] = "/different/base-model"
        (self.checkpoint / "embedding_config.json").write_text(json.dumps(native))
        with self.assertRaises(ValueError):
            load_checkpoint_metadata(self.checkpoint)


if __name__ == "__main__":
    unittest.main()
