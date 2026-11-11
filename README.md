# UniMI

UniMI trains Qwen2-VL item embeddings in two stages: Semantic Preference
Alignment (SPA) ranks semantic candidates, and Collaborative Contrastive
Alignment (CCA) continues the same LoRA adapter using collaborative positives
and ten negatives. A shared encoder and exporter serve both stages.

The project is named `UniMI`; its importable Python package is named `mmemb`.
The base encoder is [Qwen/Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B),
the pretrained **Base** model without instruction tuning. The original download
revision was not recorded.

## Data and Weights

All released data and adapters are in the
[UniMI_data Google Drive folder](https://drive.google.com/drive/folders/1f0qRauq0qWhQHdX_jfhmiubDRIbUJ1mg?usp=drive_link).

| Archive | Contents | Size |
| --- | --- | --- |
| `checkpoints.zip` | SPA adapter (`unimi_spa`) and final UniMI adapter (`unimi_spa_cca`, SPA → CCA) | `<size>` |
| `datasets.zip` | Item and user mappings, MMRec interaction files, fixed train/validation/test splits, titles, and image URLs for eight domains | `<size>` |
| `supervision.zip` | Stage I SPA records (60,000 anchors) and Stage II CCA records (23,421 samples) | `<size>` |
| `embeddings.zip` | Final UniMI item embeddings for Arts, Office, Musical, and Industrial | `<size>` |

The folder also contains the data-package `README.md`, `FILE_INVENTORY.md`,
`DATA_SOURCES.md`, `LICENSE`, and `SHA256SUMS.txt`. Product images are not
included; per-item image URLs are in
`datasets/source_metadata/<domain>/image_urls.tsv`. The base model
[Qwen/Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B) must be downloaded
separately.

### Download and Extract

```bash
pip install gdown
gdown --folder "<DRIVE_LINK>" -O UniMI_data
cd UniMI_data
sha256sum -c SHA256SUMS.txt        # macOS: shasum -a 256 -c SHA256SUMS.txt
for f in checkpoints datasets supervision embeddings; do unzip -q "${f}.zip"; done
```

Place `UniMI_data/` next to this repository so that the relative paths in the
commands below resolve:

```text
release/
├── UniMI/          # this repository
└── UniMI_data/     # extracted data package
```

## Use with MMRec

Row `i` of `embeddings/<domain>/unimi/unimi.npy` corresponds to item `i` in
`datasets/processed/<domain>/item2id.json`. Each row is a 3,584-dimensional,
L2-normalized float32 vector. For MMRec models with separate visual and textual
inputs, supply the same matrix as both feature files; VBPR takes it as a single
content input. Keep the supplied splits and train each recommender on the
training split of its target dataset.

## Install

Create a Conda environment named `unimi` with Python **3.10.20**, PyTorch
**2.8.0+cu128**, and torchvision **0.23.0+cu128** (CUDA 12.8 wheels).
[`requirements.txt`](requirements.txt) records the installed versions of the
packages needed for training, export, and tests. From this directory, run:

```bash
conda create -n unimi python=3.10.20 -y
conda activate unimi
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The requirements include the [official PyTorch CUDA wheel index](https://pytorch.org/get-started/previous-versions/)
and pin the reference CUDA 12.8 builds. `pyproject.toml` declares the same
package versions, with portable PyTorch version constraints that accept these
CUDA builds. Other dependencies are resolved by pip.

SDPA is the default attention backend; Flash Attention requires a separate
compatible `flash-attn` installation. Training uses BF16 by default.

## Train and Export

Place Qwen2-VL-7B under `models/Qwen2-VL-7B`, training JSONL files under
`data/spa_train.jsonl` and `data/cca_train.jsonl`, and images under
`data/processed/<Dataset>/images/`. Alternatively, override these paths on the
command line. All relative paths are resolved from this project's root,
including when a command is invoked from another directory.

```bash
# Stage I: two GPUs, effective anchor batch 32 × 2 × 2 = 128.
CUDA_DEVICES=0,1 bash scripts/train_spa.sh

# Stage II: initializes from outputs/spa; three GPUs, batch 8 × 2 × 3 = 48.
CUDA_DEVICES=0,1,2 bash scripts/train_cca.sh

# Export from either outputs/spa or outputs/cca.
CUDA_DEVICES=0 bash scripts/export_embeddings.sh \
  --checkpoint outputs/cca --datasets Office --processed-root data/processed
```

The shell scripts handle GPU selection and launch. The same training options
are available through Python, with defaults read from `configs/spa.json` and
`configs/cca.json`:

```bash
python -m mmemb.train_spa --model-path /path/to/Qwen2-VL-7B \
  --train-file /path/to/spa.jsonl --image-root /path/to/processed

python -m mmemb.train_cca --stage1-checkpoint /path/to/spa-checkpoint \
  --train-file /path/to/cca.jsonl --image-root /path/to/processed

python -m mmemb.export_embeddings --checkpoint /path/to/cca-checkpoint \
  --datasets Office,Arts --processed-root /path/to/processed
```

For the sibling data package, run the following after obtaining the images
and placing the base model under `models/Qwen2-VL-7B`:
Download each URL in `../UniMI_data/datasets/source_metadata/<domain>/image_urls.tsv` to
`../UniMI_data/datasets/processed/<domain>/images/<mapped_item_id>.jpg`.

```bash
CUDA_DEVICES=0,1 bash scripts/train_spa.sh \
  --train-file ../UniMI_data/supervision/stage1_spa/semantic_candidates_qwen3-vl-reranker-8b.jsonl \
  --image-root ../UniMI_data/datasets/processed

CUDA_DEVICES=0,1,2 bash scripts/train_cca.sh \
  --stage1-checkpoint ../UniMI_data/checkpoints/unimi_spa \
  --model-path models/Qwen2-VL-7B \
  --train-file ../UniMI_data/supervision/stage2_cca/collaborative_contrastive_train.jsonl \
  --image-root ../UniMI_data/datasets/processed

python -m mmemb.export_embeddings \
  --checkpoint ../UniMI_data/checkpoints/unimi_spa_cca \
  --model-path models/Qwen2-VL-7B --datasets Arts,Office,Musical,Industrial \
  --processed-root ../UniMI_data/datasets/processed --feature-name unimi
```

Run the shell training commands for distributed execution. A direct Python
training command uses one process unless launched through Accelerate.
`--config /path/to/experiment.json` overlays the stage defaults; explicit CLI
arguments take precedence. The final resolved configuration is saved with
every checkpoint.

## Reference Settings

| Setting | SPA | CCA |
| --- | --- | --- |
| Objective | Reward-margin-weighted pairwise loss | InfoNCE with 10 explicit negatives |
| Candidates / negatives per anchor | 3 candidates | 10 negatives |
| Beta / temperature | `14.2857` | `0.1` |
| Per-device anchor batch | 32 | 8 |
| Gradient accumulation | 2 | 2 |
| Reference GPU count | 2 | 3 |
| Learning rate | `5e-6` | `5e-6` |
| Optimizer steps | 1000 | 1250 |
| Seed / validation fraction | 42 / 0.01 | 42 / 0.01 |
| Attention / precision | SDPA / BF16 | SDPA / BF16 |
| Text / pooling | Image + title / last token (appended EOS) | Inherited from SPA |

SPA uses maximum sequence length 512 and image pixel bounds 3136–262144.
CCA inherits these representation settings from its SPA checkpoint. Exact
instructions and all optimization settings are stored in the configuration
files. SPA and CCA use different instructions. `max_steps` controls training
length and overrides `num_epochs=1`; the supplied runs finished at approximately
2.151 and 2.583 epochs, respectively. Construction rules and all four instructions
(Base, SPA, CCA, teacher) are in [data construction](docs/data_construction.md).
The release includes the SPA pairwise and CCA single-negative-pool methods;
mining/scoring scripts and downstream recommendation evaluation are separate.

## Small Checks

`examples/` contains generated product images and complete records for both
stages. These illustrate the input format; they are not experimental data.

```bash
# Print the effective configuration without loading weights or data.
python -m mmemb.train_spa --dry-run

# A two-step, single-GPU check using a separately supplied base model.
CUDA_DEVICES=0 bash scripts/train_spa.sh \
  --train-file examples/spa.jsonl --image-root examples/processed \
  --batch-size 1 --max-steps 2 --validation-ratio 0 \
  --dataloader-num-workers 0 --output-dir outputs/spa-smoke

CUDA_DEVICES=0 bash scripts/train_cca.sh \
  --stage1-checkpoint outputs/spa-smoke \
  --train-file examples/cca.jsonl --image-root examples/processed \
  --batch-size 1 --max-steps 2 --validation-ratio 0 \
  --dataloader-num-workers 0 --output-dir outputs/cca-smoke

python -m mmemb.export_embeddings --checkpoint outputs/cca-smoke \
  --datasets Demo --processed-root examples/processed \
  --output-dir outputs/demo-features

PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Changing GPU count or batch size changes the effective training batch.
Training launchers also accept `DRY_RUN=1` to print the Accelerate command.
Export `--dry-run` validates checkpoint metadata and dataset coverage without
loading model weights.
The original 18 unit checks were validated with Python 3.10.20 and the pinned
dependency versions; release-tool checks additionally cover metadata cleanup
and archive exclusions. These checks do not run full 7B GPU training.

See [data formats](docs/data_format.md) for required fields, image lookup,
missing-item handling, and modality export. See [checkpoints](docs/checkpoints.md)
for legacy adapter support, base-model relocation, and resuming training.

## Code Map

`src/mmemb/encoder.py` implements Qwen2-VL with last-token pooling at the appended EOS;
`tokenization.py` constructs the item prompt and appends EOS. `data/` loads and
collates stage-specific records, `losses/` defines the objectives, and
`models.py` combines the encoder and objectives. `trainer.py` handles training
metrics and adapter saves; `checkpoint.py` loads representation metadata and
adapters. `export_embeddings.py` writes L2-normalized float32 matrices, metadata,
and coverage reports using the same encoder and tokenizer as training.

## Citation

UniMI paper (under review): `TODO: add the submitted paper title`.

## License and Packaging

Code is licensed under [Apache-2.0](LICENSE). See [third-party sources](THIRD_PARTY.md)
for the base model, teacher, and dataset attribution. See
[release preparation](docs/release.md) for metadata sanitization and cache-free
packaging. No full 7B fine-tuning run is required for the small unit checks above.
