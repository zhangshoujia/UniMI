# Supervision construction

The supplied JSONL files are the supervision used for the released adapters.
Mining and teacher-scoring scripts are not included in this code package.
This document records the construction settings; training directly from the
supplied supervision does not require running mining or teacher inference.

## Domains and fixed splits

Upstream supervision uses Arts, CDs, MoviesTV, Office, Pet, and Toys. The
downstream embedding release covers Arts, Office, Musical, and Industrial.
Both groups use the supplied item mappings. CCA relations use only training
interactions within each domain. Validation/test interactions do not enter
co-occurrence mining. Keep the supplied recommendation splits unchanged.

## SPA: semantic preference alignment

1. Encode each domain's item catalog using the pretrained Qwen2-VL-7B Base,
   image plus title, and the Base instruction below. Use terminal EOS pooling
   and cosine retrieval over normalized vectors (dimension 3,584).
2. Sample 10,000 distinct anchor items per domain, for 60,000 records in total.
   The gallery is the full mapped item catalog of the same domain. Retrieve
   exactly three neighbors excluding the anchor. Search is exact cosine;
   equal similarities are resolved by ascending mapped item ID.
3. Score all 180,000 anchor/candidate pairs with the frozen
   Qwen3-VL-Reranker-8B in BF16, using each product's image and title and the
   teacher instruction below. The pointwise score is
   `sigmoid(logit_yes - logit_no)`. There is no cross-candidate normalization.
   Original retrieval order is retained in the JSONL; the SPA loader orders
   candidates by teacher score for pairwise training.

The global construction seed is 42. For a namespace, derive the seed from the
first eight bytes, big-endian, of `SHA256("42\0" + namespace)`. Anchor namespaces
are `anchor:<domain>`; NumPy `default_rng` samples without replacement, and the
selected anchor IDs are sorted before per-domain output. The recorded seeds are:

| Domain | Anchor seed |
| --- | --- |
| Arts | 8768303125530005604 |
| CDs | 9877136720819010360 |
| MoviesTV | 12784512896291494658 |
| Office | 9249048658174947497 |
| Pet | 6247006912658661573 |
| Toys | 4843374276103613297 |

The mixed record order uses Python `random.Random(11233169848793260888)` to
shuffle offsets after concatenating those domains in the order listed above.
Use the released JSONL order for training rather than generating a new order.

## CCA: collaborative contrastive alignment

For each domain, let `d_i` be the number of training users of item `i`, and
`n_ij` the number shared by items `i` and `j`. Define
`c_ij = n_ij / sqrt(d_i * d_j)`. A positive must be distinct from its anchor and
satisfy `d_i, d_j >= 10`, `n_ij >= 5`, and `c_ij >= 0.05`. Select the eligible
positive with the largest `c_ij`, breaking ties by ascending mapped item ID.

The semantic index for CCA construction uses **SPA embeddings**, not Base
embeddings. Search is exact cosine with self matches excluded and equal scores
resolved by ascending mapped item ID. A training negative has degree at least
10 and zero training co-occurrence with the anchor. Ten distinct negatives
are sampled without replacement, excluding the anchor, its positive, and the
anchor's first 100 semantic neighbors.

The archived construction also built an auxiliary semantic negative pool that
was not used in the released CCA loss. It still affected which records and
negatives were available: a record had to supply ten auxiliary candidates,
and those candidates were additionally excluded from the sampled training
negative pool. For faithful reconstruction, that preprocessing used:

- Semantic search prefixes of 100, 200, then 500 neighbors; the first prefix
  with ten valid auxiliary negatives was accepted.
- Auxiliary candidates with degree at least 10, zero training co-occurrence,
  and cosine similarity no greater than 0.6; the positive was excluded.
- Title filtering that removed equal normalized titles, token Jaccard
  similarity at least 0.5, and a contiguous shorter-title token sequence of at
  least three tokens contained in the other title. Exact-title comparison
  lowercases and collapses whitespace. Token-based filters additionally
  replace Unicode punctuation with spaces and keep tokens containing letters
  or digits.

Anchors were visited in a deterministic NumPy permutation, taking up to 5,000
feasible records per domain. Construction seeds use the first eight SHA-256
bytes, big-endian, of NUL-separated fields: `(42, "stage2-anchor-order", domain)`
for anchor order and `(42, "stage2-normal", domain, anchor_mapped_id)` for
negative selection. Negative sampling uses NumPy `default_rng`, rejection
sampling from ascending degree-eligible IDs, and a no-replacement choice from
the remaining valid IDs if needed. Selected negative IDs are sorted for output.

| Domain | CCA records |
| --- | --- |
| Arts | 2,078 |
| CDs | 5,000 |
| MoviesTV | 5,000 |
| Office | 1,788 |
| Pet | 4,555 |
| Toys | 5,000 |
| Total | 23,421 |

The publication JSONL retains exactly the ten negatives used by CCA and omits
the unused auxiliary pool and its category fields. This preserves record
order, anchors, positives, and the ten selected negatives. Loader ordering
uses its original SHA-256 namespace, including the historical `normal` label,
so the publication format produces identical training inputs at seed 42.
CCA uses temperature 0.1, no in-batch negatives, and no co-occurrence-based
loss weighting. Training thresholds here are separate from diagnostic-set
thresholds.

## Exact instructions

Base retrieval instruction:

```text
Represent this Amazon product for retrieving other products commonly purchased or interacted with by the same users in a recommendation task.
```

SPA instruction:

```text
Represent this product using its image and text as a unified embedding for semantic similarity retrieval.
```

CCA instruction:

```text
Represent this product using its image and text as a unified embedding for recommendation.
```

Teacher instruction:

```text
Given an anchor product and a candidate product, determine whether they are semantically aligned based on their textual and visual information. Consider product category or type, function, key attributes, and intended use. Answer only yes or no.
```

These instructions differ across stages. The released comparison should not
be interpreted as an experiment holding the instruction fixed.

## Representation and training

SPA and CCA use image plus title, maximum sequence length 512, title character
limit 20,480, and image pixel bounds 3,136–262,144. Pool the final hidden state
at the appended EOS token. SPA strips outer title whitespace; CCA also
collapses internal whitespace. Export L2-normalizes encoded rows and saves
float32 matrices. Row indices follow each domain's `item2id.json`.

The training seed is 42; each stage holds out 1% of its supplied JSONL using
the deterministic split in the loader. SPA uses 1,000 optimizer steps on two
GPUs (effective anchor batch 128); CCA uses 1,250 on three GPUs (batch 48).
`max_steps` overrides `num_epochs=1`: the original runs ended at approximately
2.151 and 2.583 epochs. CCA continues the same SPA LoRA adapter, so final export
loads one cumulative adapter. The released files for these stages are listed in
`UniMI_data/FILE_INVENTORY.md`.
