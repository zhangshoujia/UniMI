import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from mmemb.io import collect_image_paths, load_item2id, read_item_text_records
from mmemb.tokenization import ItemTokenizer, last_active_positions, validate_terminal_eos


class RecordingProcessor:
    image_token = "<|image_pad|>"

    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token_id=999)
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        rows = [[ord(char) for char in text][: kwargs["max_length"]] for text in kwargs["text"]]
        size = max(map(len, rows))
        return {
            "input_ids": torch.tensor([[0] * (size - len(row)) + row for row in rows]),
            "attention_mask": torch.tensor(
                [[0] * (size - len(row)) + [1] * len(row) for row in rows]
            ),
        }


class TokenizationTests(unittest.TestCase):
    def test_prompt_truncation_padding_and_active_eos(self):
        processor = RecordingProcessor()
        tokenizer = ItemTokenizer(
            processor, "Find similar products.", max_length=12, max_char_length=4, input_mode="text"
        )
        self.assertEqual(
            tokenizer.format_text({"title": "  title  "}),
            "Instruct: Find similar products.\n\nItem title:   ti",
        )
        encoded = tokenizer([{"title": "one"}, {"title": "two"}])
        self.assertEqual(processor.last_kwargs["max_length"], 11)
        self.assertEqual(encoded["input_ids"].shape, (2, 12))
        self.assertEqual(validate_terminal_eos(encoded, 999).tolist(), [11, 11])
        self.assertNotIn("images", processor.last_kwargs)
        mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(last_active_positions(mask).tolist(), [3, 1])
        encoded["input_ids"][0, -1] = 123
        with self.assertRaises(ValueError):
            validate_terminal_eos(encoded, 999)

    def test_visual_and_image_only_prompts(self):
        processor = RecordingProcessor()
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "item.png"
            Image.new("RGB", (4, 4)).save(image)
            tokenizer = ItemTokenizer(processor, "Represent product.")
            item = {"title": "Product", "image_path": image}
            self.assertEqual(
                tokenizer.format_text(item),
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                "Instruct: Represent product.\n\nItem title: Product",
            )
            tokenizer([item])
            self.assertEqual(processor.last_kwargs["images"][0].mode, "RGB")
            tokenizer = ItemTokenizer(processor, "Represent product.", input_mode="image")
            self.assertTrue(tokenizer.format_text(item).endswith("Item image:"))


class ExportTests(unittest.TestCase):
    def test_tsv_duplicates_and_image_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "item2id.json").write_text('{"B": 0, "A": 1}')
            mapping = load_item2id(root / "item2id.json")
            (root / "text.tsv").write_text("parent_asin\ttitle\nA\tFirst\nA\tSecond\nB\tnan\n")
            records, skipped, duplicates = read_item_text_records(root / "text.tsv", mapping)
            self.assertIsNone(records[0])
            self.assertEqual(records[1]["title"], "First")
            self.assertEqual((skipped, duplicates), (1, 1))
            Image.new("RGB", (4, 4)).save(root / "A.jpg")
            Image.new("RGB", (4, 4)).save(root / "A.png")
            paths, _, _ = collect_image_paths(root, mapping)
            self.assertEqual(paths[1].name, "A.jpg")
            (root / "A.jpg").unlink()
            Image.new("RGB", (4, 4)).save(root / "A.webp")
            with self.assertRaises(ValueError):
                collect_image_paths(root, mapping)

    def test_export_preserves_rows_normalizes_and_protects_outputs(self):
        from mmemb.export_embeddings import build_parser, export_dataset

        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.processor = RecordingProcessor()
                self.calls = 0

            def encode(self, inputs):
                self.calls += 1
                return torch.tensor([[3.0, 4.0] if self.calls == 1 else [5.0, 12.0]])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "item2id.json").write_text('{"C": 0, "A": 2, "B": 1}')
            (root / "text.tsv").write_text("parent_asin\ttitle\nA\tAlpha\nB\tBeta\nC\tGamma\n")
            for raw_id in ["A", "B"]:
                Image.new("RGB", (4, 4)).save(root / "images" / f"{raw_id}.jpg")
            args = build_parser().parse_args(
                [
                    "--checkpoint",
                    str(root / "checkpoint"),
                    "--data-dir",
                    str(root),
                    "--batch-size",
                    "1",
                    "--allow-missing-image",
                ]
            )
            args.dataset = "Demo"
            metadata = dict(
                stage="cca",
                pooling="eos",
                item_prompt="Represent product.",
                item_text_mode="title",
                max_length=512,
                max_char_length=20480,
            )
            args.dry_run = True
            summary = export_dataset(args, metadata)
            self.assertEqual(summary["covered_items"], 2)
            self.assertFalse((root / "features").exists())
            args.dry_run = False
            result = export_dataset(args, metadata, FakeEncoder())
            matrix = np.load(result["output"])
            self.assertEqual(matrix.dtype, np.float32)
            np.testing.assert_array_equal(matrix[0], [0.0, 0.0])
            np.testing.assert_allclose(matrix[1], [0.6, 0.8], rtol=1e-6)
            np.testing.assert_allclose(matrix[2], [5 / 13, 12 / 13], rtol=1e-6)
            metadata_path = Path(result["output"]).with_suffix(".meta.json")
            saved = json.loads(metadata_path.read_text())
            self.assertEqual(saved["coverage"]["missing_items"], 1)
            self.assertGreater(saved["export_runtime"]["items_per_second"], 0)
            with self.assertRaises(FileExistsError):
                export_dataset(args, metadata, FakeEncoder())


if __name__ == "__main__":
    unittest.main()
