# Checkpoints

SPA creates a LoRA adapter on Qwen2-VL. CCA loads that adapter as trainable and
continues updating it. Its final adapter therefore includes both stages.
Export loads the base model and one final adapter; the SPA adapter is not
stacked onto a CCA adapter.

The supplied `UniMI_data/checkpoints/unimi_spa` and `unimi_spa_cca` directories
use the legacy stage metadata described below. They contain only final
adapters, processor/tokenizer files, and public configurations. Detailed trainer
records and optimizer/scheduler/RNG states are omitted. These adapters support
export and SPA-to-CCA initialization; they do not provide exact interrupted-run
resume. `UniMI_data/FILE_INVENTORY.md` describes the contents of the released
checkpoint directories.

## Native Layout

Native outputs and intermediate `checkpoint-*` directories contain the PEFT
adapter configuration/weights, saved processor files, and
`embedding_config.json`. The metadata records the stage, resolved training
configuration, representation settings, provenance, and data fingerprints.
Keep a complete native checkpoint directory for continuation or export. CCA,
resumed training, and export use the processor saved with native checkpoints.

Trainer checkpoints also hold the optimizer, scheduler, RNG, and trainer
state. These are needed to resume training, whereas export requires only the
adapter, metadata, processor, and base model. Use an intermediate checkpoint
for an exact training continuation. The separately distributed final adapters
omit these training states and do not support exact resume.

```bash
CUDA_DEVICES=0,1 bash scripts/train_spa.sh \
  --resume-from-checkpoint latest

CUDA_DEVICES=0,1,2 bash scripts/train_cca.sh \
  --resume-from-checkpoint outputs/cca/checkpoint-500
```

Keep the same effective configuration, input contents, world size, and training
environment for a reproducible continuation. Resume checks compare critical
configuration and training-JSONL fingerprints. Moving an identical JSONL does
not itself change its fingerprint. Images are not individually hashed; keep
their contents unchanged when continuing a run.

## Legacy Adapters

The loader accepts the original `semantic_preference_config.json` and
`collaborative_contrastive_config.json` at the input boundary. It translates
stage names and `direct_pooling_method="last"` into the native schema. The
underlying PEFT adapter format stays compatible. Only title-only, terminal-EOS
checkpoints are supported by this release. Legacy loading retains the original
behavior of reading the processor from the matching base-model directory.

```bash
# Initialize CCA using an original SPA checkpoint.
python -m mmemb.train_cca --stage1-checkpoint /path/to/original-spa-checkpoint

# Export an original SPA or CCA checkpoint.
python -m mmemb.export_embeddings --checkpoint /path/to/original-checkpoint \
  --datasets Office --processed-root /path/to/processed
```

Legacy adapters support initialization and export. Their old optimizer states
are not accepted for full training resume across this structural refactor.

## Base-Model Paths

The recorded base model must match the adapter. If the same base weights have
moved, specify `--model-path /path/to/Qwen2-VL-7B` when training CCA or exporting.
The override records the runtime location; it must refer to the same model
weights and architecture as the checkpoint's original base model.

The released adapters use `Qwen/Qwen2-VL-7B` Base, without instruction tuning.
Their public configurations use `/path/to/Qwen2-VL-7B`; pass the actual local
path with `--model-path`. The original download revision is unknown. A base
configuration SHA-256 is supplied, but it is not a hash of the model weights.

Model paths are locations, not proofs of weight identity. The metadata retains
the original model reference and Stage-I adapter provenance; the user supplies
the matching base weights. Adapters for a 7B model cannot be loaded onto a 2B
model by changing a path.
