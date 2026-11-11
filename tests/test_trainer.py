"""Exercise Trainer accumulation and adapter restoration without model downloads."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, set_seed

from mmemb.checkpoint import CONFIG_NAME, resolve_resume_checkpoint
from mmemb.data import SPACollator
from mmemb.models import SPAModel
from mmemb.trainer import AlignmentTrainer


class _Processor:
    def save_pretrained(self, directory):
        (Path(directory) / "preprocessor_config.json").write_text("{}")


class _TinyEncoder(torch.nn.Module):
    def __init__(self, base, adapter=None):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(base)
        if adapter is None:
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    lora_dropout=0,
                    target_modules=["c_attn"],
                    fan_in_fan_out=True,
                    task_type="CAUSAL_LM",
                ),
            )
        else:
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=True)
        self.processor = _Processor()

    def forward(self, inputs):
        outputs = self.model(**inputs, use_cache=False, output_hidden_states=True)
        return outputs.hidden_states[-1][:, -1]


def _tokenize(items):
    return {
        "input_ids": torch.tensor([[1, int(item["item_id"]) + 2] for item in items]),
        "attention_mask": torch.ones(len(items), 2, dtype=torch.long),
    }


def _item(index):
    return {"dataset": "Demo", "item_id": str(index)}


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="mmemb_trainer_test_")
        cls.root = Path(cls.directory.name)
        cls.base = cls.root / "base"
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        set_seed(123)
        config = GPT2Config(
            vocab_size=24,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
        GPT2LMHeadModel(config).save_pretrained(cls.base)
        cls.records = [
            {
                "anchor": _item(anchor),
                "candidates": [_item(10), _item(11), _item(12)],
                "reward_scores": [0.9, 0.5, 0.1],
            }
            for anchor in range(4)
        ]
        cls.metadata = {
            "schema_version": 1,
            "stage": "spa",
            "beta": 2,
            "model_path": str(cls.base),
            "adapter_base_model": str(cls.base),
            "item_prompt": "Example item instruction.",
            "item_text_mode": "title",
            "pooling": "eos",
            "title_normalization": "strip",
            "max_length": 8,
            "max_char_length": 100,
            "image_min_pixels": 1,
            "image_max_pixels": 16,
            "attn_implementation": "sdpa",
            "bf16": False,
        }

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        cls.directory.cleanup()

    def trainer(self, name, *, steps, batch, accumulation, adapter=None, resume=None):
        set_seed(42)
        model = SPAModel(_TinyEncoder(self.base, adapter), beta=2)
        arguments = TrainingArguments(
            output_dir=str(self.root / name),
            use_cpu=True,
            bf16=False,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=accumulation,
            max_steps=steps,
            learning_rate=0.02,
            lr_scheduler_type="constant",
            optim="sgd",
            weight_decay=0,
            max_grad_norm=0,
            logging_steps=1,
            save_steps=1,
            remove_unused_columns=False,
            report_to=[],
            disable_tqdm=True,
            dataloader_pin_memory=False,
            label_names=[],
            seed=42,
        )
        return AlignmentTrainer(
            model=model,
            args=arguments,
            train_dataset=self.records,
            eval_dataset=self.records[:3],
            data_collator=SPACollator(_tokenize),
            processing_class=model.encoder.processor,
            checkpoint_metadata=self.metadata,
            resume_checkpoint=resume,
        )

    def assert_parameters_match(self, left, right):
        for (name, actual), (other_name, expected) in zip(
            left.model.named_parameters(), right.model.named_parameters()
        ):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)

    def test_accumulation_and_incomplete_eval_batch(self):
        full_batch = self.trainer("full-batch", steps=1, batch=2, accumulation=1)
        accumulated = self.trainer("accumulated", steps=1, batch=1, accumulation=2)
        with contextlib.redirect_stdout(io.StringIO()):
            full_batch.train()
            accumulated.train()
        self.assert_parameters_match(full_batch, accumulated)
        self.assertTrue(any("spa_loss" in log for log in full_batch.state.log_history))
        full_batch.model.eval()
        with torch.no_grad():
            losses = [
                full_batch.model(**full_batch.data_collator([record]))["loss"]
                for record in self.records[:3]
            ]
        expected = torch.stack(losses).mean().item()
        with contextlib.redirect_stdout(io.StringIO()):
            actual = full_batch.evaluate()["eval_loss"]
        self.assertAlmostEqual(actual, expected, places=6)

    def test_adapter_checkpoint_and_native_resume(self):
        uninterrupted = self.trainer("uninterrupted", steps=2, batch=1, accumulation=2)
        with contextlib.redirect_stdout(io.StringIO()):
            uninterrupted.train()
        checkpoint = self.root / "uninterrupted" / "checkpoint-1"
        for name in (
            "adapter_model.safetensors",
            "adapter_config.json",
            CONFIG_NAME,
            "preprocessor_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
        ):
            self.assertTrue((checkpoint / name).is_file(), name)
        self.assertTrue(list(checkpoint.glob("rng_state*.pth")))
        self.assertFalse((checkpoint / "model.safetensors").exists())
        self.assertEqual(resolve_resume_checkpoint(str(checkpoint), self.root), checkpoint)
        resumed = self.trainer(
            "resumed",
            steps=2,
            batch=1,
            accumulation=2,
            adapter=checkpoint,
            resume=checkpoint,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            resumed.train(resume_from_checkpoint=str(checkpoint))
        self.assertEqual(resumed.state.global_step, 2)
        self.assert_parameters_match(uninterrupted, resumed)


if __name__ == "__main__":
    unittest.main()
