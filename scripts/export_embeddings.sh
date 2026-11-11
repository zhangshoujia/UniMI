#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
if [[ ! "${CUDA_DEVICES}" =~ ^[0-9]+$ ]]; then
  echo "Export uses one GPU; set CUDA_DEVICES to one physical GPU id." >&2
  exit 2
fi
command=("${PYTHON_BIN}" -m mmemb.export_embeddings "$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then command+=(--dry-run); fi
exec env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}" "${command[@]}"
