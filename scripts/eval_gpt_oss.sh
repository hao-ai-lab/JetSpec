#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/jetspec_venv}"
TARGET_MODEL="${TARGET_MODEL:-/mnt/lanxiangh/models/gpt-oss-20b}"
DRAFT_HEAD="${DRAFT_HEAD:-/mnt/lanxiangh/models/jetspec-gpt-oss-20b}"
DATASETS="${DATASETS:-}"
SAMPLES="${SAMPLES:-0}"
MAX_NEW="${MAX_NEW:-1024}"
WIDTH="${WIDTH:-7}"
BUDGET="${BUDGET:-256}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing virtual environment Python: ${VENV_DIR}/bin/python" >&2
  exit 1
fi
if [[ ! -d "${TARGET_MODEL}" ]]; then
  echo "Missing GPT-OSS target checkpoint: ${TARGET_MODEL}" >&2
  exit 1
fi
if [[ ! -d "${DRAFT_HEAD}" ]]; then
  echo "Missing GPT-OSS draft checkpoint: ${DRAFT_HEAD}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${DATASETS}" ]]; then
  DATASETS="$("${VENV_DIR}/bin/python" \
    "${ROOT_DIR}/bench/reference/benchmark.py" --list-datasets)"
fi
if [[ -n "${DATASET:-}" ]]; then
  DATASET_LIST=("${DATASET}")
else
  read -r -a DATASET_LIST <<< "${DATASETS}"
fi

cd "${ROOT_DIR}"
for dataset in "${DATASET_LIST[@]}"; do
  echo "[JetSpec] GPT-OSS: starting dataset=${dataset} samples=${SAMPLES}" >&2
  "${VENV_DIR}/bin/torchrun" \
    --standalone \
    --nproc-per-node "${NPROC_PER_NODE}" \
    bench/reference/benchmark.py \
    --model "${TARGET_MODEL}" \
    --draft-head "${DRAFT_HEAD}" \
    --attn-implementation flash_attention_2 \
    --tree-attn-implementation triton \
    --dataset "${dataset}" \
    --samples "${SAMPLES}" \
    --algos accum_logp \
    --width "${WIDTH}" \
    --budget "${BUDGET}" \
    --max-new "${MAX_NEW}" \
    --warmup-samples-per-rank "${WARMUP_SAMPLES}"
  echo "[JetSpec] GPT-OSS: completed dataset=${dataset}" >&2
done
