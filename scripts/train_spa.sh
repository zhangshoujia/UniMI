#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "${PYTHON_BIN}" -m mmemb.train_spa --help
fi

if [[ -z "${CUDA_DEVICES:-}" ]]; then
  reference_count="$("${PYTHON_BIN}" -m mmemb.config spa "$@")"
  CUDA_DEVICES="0"
  for ((index=1; index<reference_count; index++)); do CUDA_DEVICES+=",${index}"; done
fi
if [[ ! "${CUDA_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "CUDA_DEVICES must be a comma-separated GPU list." >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "${CUDA_DEVICES}"
seen=","
for device in "${devices[@]}"; do
  if [[ "${seen}" == *",${device},"* ]]; then echo "Duplicate GPU id: ${device}" >&2; exit 2; fi
  seen+="${device},"
done
config="${PROJECT_ROOT}/configs/accelerate_single.yaml"
if (( ${#devices[@]} > 1 )); then config="${PROJECT_ROOT}/configs/accelerate_multi.yaml"; fi
mixed_precision="$("${PYTHON_BIN}" -m mmemb.config spa --print-field mixed_precision "$@")"
command=("${PYTHON_BIN}" -m accelerate.commands.launch --config_file "${config}"
  --num_processes "${#devices[@]}" --mixed_precision "${mixed_precision}" -m mmemb.train_spa "$@")
printf 'Command:'
printf ' %q' env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}" "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m mmemb.train_spa --dry-run "$@"
  exit 0
fi
exec env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}" "${command[@]}"
