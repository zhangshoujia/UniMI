# Preparing a publication copy

Keep working training outputs separate from the public copy. Runtime metadata
records resolved paths for loading and resume checks. To sanitize a JSON
configuration or export sidecar without changing its source:

```bash
python scripts/sanitize_metadata.py /path/to/private/embedding_config.json \
  --output /path/to/public/embedding_config.json
```

The same command accepts a legacy stage configuration or a feature `.meta.json`.
It removes credential fields and replaces absolute path values with public
placeholders. It does not process arbitrary text files, model cards, or binary
training arguments; inspect text separately and omit `training_args.bin` from
the publication copy. Pass `--model-path` when loading a sanitized adapter.

Keep adapter weights/configuration, the representation metadata, and required
tokenizer/processor files. Exclude optimizer, scheduler, RNG state, intermediate
checkpoints, `trainer_state.json`, `training_args.bin`, result logs, and duplicate
run configurations. Record an original file fingerprint separately from a
changed publication file fingerprint, and update references to released files.
Prepare SPA first, then update CCA's references to the released SPA configuration.
Sanitizing metadata does not change adapter weights or embedding arrays.

The supplied `UniMI_data` follows this minimal final-adapter layout. Its
`FILE_INVENTORY.md` describes the contents of the checkpoint, dataset,
supervision, and embedding directories. Its CCA JSONL contains exactly the ten
negatives used in training; construction details are explained in
[data construction](data_construction.md).

For a sanitization regression check, run
the standard `unittest discover` command. Its release-tool tests verify that
path cleanup preserves configuration fields and that the packager excludes
caches and runtime files.

To build the code archive from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python scripts/package_release.py --output ../UniMI.zip
```

The packager explicitly selects source, scripts, configs, docs, tests, and
synthetic examples. It excludes caches, compiled Python, model weights, feature
matrices, and runtime outputs independently of `.gitignore`. Review the final
archive contents before uploading. `UniMI_data` is distributed separately.
