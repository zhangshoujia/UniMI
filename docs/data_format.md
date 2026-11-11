# Data Formats

Training input is UTF-8 JSONL, one record per line. Full examples and generated
images are available in [`../examples/`](../examples/).

## SPA

Each record has a dataset, one anchor, and exactly three candidates:

```json
{"dataset":"Demo","anchor":{"item_id":"D00","title":"Anchor"},"candidates":[{"item_id":"D01","title":"First","semantic_alignment_score":0.9},{"item_id":"D02","title":"Second","semantic_alignment_score":0.6},{"item_id":"D03","title":"Third","semantic_alignment_score":0.2}]}
```

The loader sorts candidates by decreasing `semantic_alignment_score`, breaking
ties by raw item ID. Scores must be finite and in `[0, 1]`. Tied rewards have
zero pairwise weight and are excluded from preference-accuracy pair counts.
Each item needs a nonempty ID and title. `description` and candidate
`cosine_similarity` are optional audit fields and do not enter the title-only
encoder or SPA loss. SPA strips outer title whitespace and preserves internal
whitespace, matching the original representation.

## CCA

Each record has a dataset, one anchor, one positive, and a list of at least ten
distinct negatives. Every item requires `item_id`, `mapped_item_id` (a
nonnegative integer within its dataset), and a nonempty `title`. The complete
ten-negative example is [`../examples/cca.jsonl`](../examples/cca.jsonl).

The anchor, positive, and negatives must have distinct raw and mapped IDs
within a record. Repeated items across records must have consistent metadata.
Anchors must be unique within each dataset. CCA collapses title whitespace
before character truncation, matching the original representation.

`interaction_count`, `cooccurrence`, `collaborative_score`,
`semantic_similarity`, and `semantic_rank` are optional audit fields. If
present, counts must be valid integers and scores must be finite. Positive
co-occurrence must be at least one; negative co-occurrence and collaborative
score must be zero. Positive semantic rank must be at least one; negative
semantic rank must be null. These fields do not weight the CCA objective or
replace upstream relation mining.

Release records use one negative pool without `negative_type`. Legacy records
containing `negative_type` are accepted for compatibility.
The supplied publication JSONL has exactly ten negatives per record. Its
construction rules are documented in [data construction](data_construction.md).

## Images and Batching

Training finds images under `IMAGE_ROOT/<dataset>/images/<item_id>.jpg`. A JPG
takes priority; otherwise exactly one matching `.jpeg`, `.png`, `.webp`, or
`.bmp` is required. Missing or ambiguous images fail during dataset loading.
Item IDs and dataset names must be safe path components. The collators encode
each distinct dataset/item identity once per batch and construct index tensors
for the stage objective.

## Embedding Export

Each dataset needs:

```text
processed/<Dataset>/
├── item2id.json
├── text.tsv
└── images/
```

`item2id.json` maps raw IDs to unique contiguous row indices starting at zero.
The TSV uses `parent_asin` and `title` by default; `--item-col` and `--title-col`
change these names. Extra columns are ignored. Duplicate text rows keep the
first usable title by default; `--duplicate keep_last` keeps the last.

```bash
python -m mmemb.export_embeddings --checkpoint outputs/cca \
  --datasets Office --processed-root data/processed
```

The instruction, title normalization, EOS pooling, sequence limits, and image
bounds come from the checkpoint. `--input-mode text` or `--input-mode image`
selects a modality ablation; the default is `text_image`.

Outputs go under the dataset's `features/` directory, with readable default
names `spa_embeddings` or `cca_embeddings`. Modality and partial-export suffixes
are added when applicable. `--feature-name` and single-dataset `--output-dir`
override the destination. Each export writes:

- `.npy`: an item-by-hidden-dimension float32 matrix; row `i` maps to item `i`.
- `.meta.json`: checkpoint/representation settings, coverage, and runtime.
- `.missing_items.tsv`: raw and mapped IDs for uncovered rows.
- `feature_coverage.tsv`: coverage by feature name.

Encoded rows are L2-normalized. Missing titles or images raise errors unless
`--allow-missing-text` / `--allow-missing-image` are explicitly set. Uncovered
rows, including rows outside `--limit`, remain zero and appear in the coverage
report. Existing outputs require `--allow-overwrite` before replacement.
