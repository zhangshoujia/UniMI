import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from mmemb.data import CCACollator, CCADataset, SPACollator, SPADataset
from mmemb.data.cca import fixed_sample_without_replacement
from mmemb.data.common import split_dataset
from mmemb.losses.cca import contrastive_loss
from mmemb.losses.spa import pairwise_preference_loss, preference_metrics
from mmemb.models import CCAModel, SPAModel


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class LossTests(unittest.TestCase):
    def test_spa_pair_sum_and_reward_ties(self):
        scores = torch.tensor([[0.8, 0.3, -0.4], [0.2, 0.4, -0.1]], requires_grad=True)
        rewards = torch.tensor([[0.9, 0.6, 0.1], [0.5, 0.5, 0.2]])
        # Explicitly enumerate the three preference pairs for each anchor.
        expected = (
            -0.3 * F.logsigmoid(scores[0, 0] - scores[0, 1])
            - 0.8 * F.logsigmoid(scores[0, 0] - scores[0, 2])
            - 0.5 * F.logsigmoid(scores[0, 1] - scores[0, 2])
            - 0.3 * F.logsigmoid(scores[1, 0] - scores[1, 2])
            - 0.3 * F.logsigmoid(scores[1, 1] - scores[1, 2])
        ) / 2
        actual = pairwise_preference_loss(scores, rewards)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, scores, retain_graph=True)[0],
            torch.autograd.grad(expected, scores)[0],
        )
        metrics = preference_metrics(scores, rewards)
        self.assertEqual(metrics["preference_pair_count"].item(), 5)
        self.assertEqual(metrics["preference_accuracy"].item(), 1.0)
        with self.assertRaises(ValueError):
            pairwise_preference_loss(scores, rewards.flip(1))

    def test_cca_uniform_logits_and_gradient(self):
        positive = torch.zeros(2, requires_grad=True)
        negatives = torch.zeros(2, 10, requires_grad=True)
        loss = contrastive_loss(positive, negatives, temperature=0.1)
        self.assertAlmostEqual(loss.item(), math.log(11), places=6)
        loss.backward()
        self.assertTrue(torch.all(positive.grad < 0))
        self.assertTrue(torch.all(negatives.grad > 0))
        with self.assertRaises(ValueError):
            contrastive_loss(positive, negatives, temperature=0)

    def test_models_use_cosine_scores_and_propagate_gradients(self):
        class TableEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [3.0, 0.0],
                            [4.0, 3.0],
                            [0.0, 2.0],
                            [-1.0, 0.0],
                        ]
                    )
                )

            def forward(self, inputs):
                return self.table

        encoder = TableEncoder()
        spa = SPAModel(encoder, beta=2)
        outputs = spa(
            None, torch.tensor([0]), torch.tensor([[1, 2, 3]]), torch.tensor([[0.9, 0.5, 0.1]])
        )
        expected = pairwise_preference_loss(
            torch.tensor([[1.6, 0.0, -2.0]]), torch.tensor([[0.9, 0.5, 0.1]])
        )
        torch.testing.assert_close(outputs["loss"], expected)
        outputs["loss"].backward()
        self.assertTrue(torch.isfinite(encoder.table.grad).all())
        cca = CCAModel(encoder, temperature=0.1)
        result = cca(None, torch.tensor([0]), torch.tensor([1]), torch.tensor([[2, 3]]))
        self.assertEqual(result["metrics"]["negative_preference_accuracy"][0].item(), 1.0)
        torch.testing.assert_close(
            result["loss"], contrastive_loss(torch.tensor([0.8]), torch.tensor([[0.0, -1.0]]))
        )


class DataTests(unittest.TestCase):
    def test_complete_examples_and_batch_deduplication(self):
        spa = SPADataset(EXAMPLES / "spa.jsonl", EXAMPLES / "processed")
        cca = CCADataset(EXAMPLES / "cca.jsonl", EXAMPLES / "processed")
        self.assertEqual(len(spa[0]["candidates"]), 3)
        self.assertEqual(len(cca[0]["negatives"]), 10)
        batch = CCACollator(lambda items: items)([cca[0], cca[0]])
        self.assertEqual(len(batch["item_inputs"]), 12)
        self.assertEqual(batch["negative_indices"].shape, (2, 10))
        self.assertEqual(batch["anchor_indices"].tolist(), [0, 0])
        other_dataset = copy.deepcopy(spa[0])
        for item in [other_dataset["anchor"], *other_dataset["candidates"]]:
            item["dataset"] = "Other"
        batch = SPACollator(lambda items: items)([spa[0], other_dataset])
        self.assertEqual(len(batch["item_inputs"]), 8)

    def test_fixed_negative_selection_matches_reference_and_pool_order(self):
        record = json.loads((EXAMPLES / "cca.jsonl").read_text())
        kwargs = dict(global_seed=42, dataset="Demo", anchor_mapped_item_id=0)
        selected, seed = fixed_sample_without_replacement(record["negatives"], 10, **kwargs)
        self.assertEqual(seed, 18365475829966227529)
        self.assertEqual(
            [item["mapped_item_id"] for item in selected], [6, 3, 9, 7, 8, 5, 2, 11, 4, 10]
        )
        reversed_selection, _ = fixed_sample_without_replacement(
            record["negatives"][::-1], 6, **kwargs
        )
        self.assertEqual(reversed_selection, selected[:6])

    def test_legacy_hard_pool_is_excluded_and_titles_keep_stage_rules(self):
        cca_record = json.loads((EXAMPLES / "cca.jsonl").read_text())
        cca_record["anchor"]["title"] = "  First   product  "
        for negative in cca_record["negatives"]:
            negative["negative_type"] = "normal"
        # No image exists for this excluded legacy hard negative.
        cca_record["negatives"].append({"negative_type": "hard", "item_id": "unused"})
        spa_record = json.loads((EXAMPLES / "spa.jsonl").read_text())
        spa_record["anchor"]["title"] = "  First   product  "
        spa_record["candidates"] = list(reversed(spa_record["candidates"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(cca_record) + "\n")
            cca = CCADataset(path, EXAMPLES / "processed")
            self.assertEqual(cca[0]["anchor"]["title"], "First product")
            self.assertNotIn("unused", [item["item_id"] for item in cca[0]["negatives"]])
            path.write_text(json.dumps(spa_record) + "\n")
            spa = SPADataset(path, EXAMPLES / "processed")
            self.assertEqual(spa[0]["anchor"]["title"], "First   product")
            self.assertEqual(
                [item["item_id"] for item in spa[0]["candidates"]], ["D01", "D02", "D03"]
            )

    def test_split_is_disjoint_deterministic_and_nonempty(self):
        dataset = list(range(9))
        train, valid = split_dataset(dataset, 0.2, 42)
        repeat_train, repeat_valid = split_dataset(dataset, 0.2, 42)
        self.assertEqual(train.indices, repeat_train.indices)
        self.assertEqual(valid.indices, repeat_valid.indices)
        self.assertEqual(set(train.indices) | set(valid.indices), set(dataset))
        self.assertFalse(set(train.indices) & set(valid.indices))
        self.assertIsNone(split_dataset(dataset, 0, 42)[1])


if __name__ == "__main__":
    unittest.main()
