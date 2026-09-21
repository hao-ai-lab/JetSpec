#!/usr/bin/env bash
# Self-refinement (J) x chain/tree evaluation sweep for a DFlash draft head.
#
#   DRAFT_HEAD=/path/to/checkpoint bash bench/reference/run_self_refine_sweep.sh
#
# Runs one 8-GPU distributed benchmark.py invocation per (dataset, J) pair,
# sequentially -- only one job can own all 8 GPUs at a time. Each invocation
# covers BOTH chain (--include-dflash-baseline) and tree (--algos) in a single
# pass, since both read from the same J-pass-refined propose_logits() call.
#
# Required:
#   DRAFT_HEAD          path to a DFlashDraftModel checkpoint directory
#
# Optional (defaults cover the full evaluation; override for a cheap check):
#   JETSPEC_VENV_PATH     the eval venv's root (default: /workspace/jetspec_venv, or
#                         $JETSPEC_VENV_PATH if specforge/scripts/a100_env.sh was
#                         sourced). Its OWN torchrun/python are invoked by full path
#                         -- never resolved from PATH -- so this never silently runs
#                         under a different venv (e.g. specforge_venv, which a100_env.sh
#                         puts ahead on PATH) even when the two happen to be pinned
#                         to identical package versions today.
#   MODEL_PATH           target model (default: $DRAFTER_TARGET_MODEL_PATH from
#                         specforge/scripts/a100_env.sh, i.e. the pinned Qwen3-8B)
#   DATASETS              comma-separated (default: gsm8k,math500)
#   J_VALUES               comma-separated self-refine pass counts (default: 1,2,3,4)
#   SAMPLES                 per dataset; unset/empty = FULL test set (1319 / 500)
#   MAX_NEW                   default 1024
#   WARMUP_ROUNDS               default 3
#   ALGO / DEPTH / WIDTH / BUDGET   tree settings; defaults 20/7/256, algo accum_logp
#   ATTN_IMPL                        default sdpa (no flash-attn on this node)
#   OUT_ROOT                          default $DRAFT_HEAD/self-refine-eval
#
# Known node issue this script works around: `torchrun --standalone` hangs during
# rendezvous on this node's container networking (unresolvable hostname; same
# class of issue specforge's scripts/train_full_dflash.sh hit and fixed the same
# way). This script always uses the explicit c10d/127.0.0.1 rendezvous instead.
set -euo pipefail

die() { echo "[self-refine-sweep] ERROR: $*" >&2; exit 1; }
say() { echo "[self-refine-sweep] $*"; }

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

[ -n "${DRAFT_HEAD:-}" ] || die "set DRAFT_HEAD=/path/to/checkpoint"
[ -d "$DRAFT_HEAD" ] || die "not a directory: $DRAFT_HEAD"
[ -f "$DRAFT_HEAD/config.json" ] || die "missing config.json under $DRAFT_HEAD -- not a complete checkpoint"

MODEL_PATH=${MODEL_PATH:-${DRAFTER_TARGET_MODEL_PATH:-}}
[ -n "$MODEL_PATH" ] || die "set MODEL_PATH, or source specforge/scripts/a100_env.sh first"
[ -d "$MODEL_PATH" ] || die "target model not found: $MODEL_PATH"

JETSPEC_VENV_PATH=${JETSPEC_VENV_PATH:-/workspace/jetspec_venv}
TORCHRUN="$JETSPEC_VENV_PATH/bin/torchrun"
[ -x "$TORCHRUN" ] || die "no torchrun at $TORCHRUN -- set JETSPEC_VENV_PATH, or build it: bash specforge/scripts/setup_node.sh"

IFS=',' read -ra DATASET_LIST <<< "${DATASETS:-gsm8k,math500}"
IFS=',' read -ra J_LIST <<< "${J_VALUES:-1,2,3,4}"
for j in "${J_LIST[@]}"; do
  [[ "$j" =~ ^[0-9]+$ ]] && [ "$j" -ge 1 ] || die "J values must be positive integers, got: $j"
done
for d in "${DATASET_LIST[@]}"; do
  case "$d" in gsm8k|math500) ;; *) die "unknown dataset: $d (expected gsm8k or math500)";; esac
done

SAMPLES=${SAMPLES:-}
SAMPLES_ARGS=(); [ -n "$SAMPLES" ] && SAMPLES_ARGS=(--samples "$SAMPLES")
MAX_NEW=${MAX_NEW:-1024}
WARMUP_ROUNDS=${WARMUP_ROUNDS:-3}
ALGO=${ALGO:-accum_logp}
DEPTH=${DEPTH:-20}
WIDTH=${WIDTH:-7}
BUDGET=${BUDGET:-256}
ATTN_IMPL=${ATTN_IMPL:-sdpa}
OUT_ROOT=${OUT_ROOT:-$DRAFT_HEAD/self-refine-eval}
mkdir -p "$OUT_ROOT"

say "checkpoint : $DRAFT_HEAD"
say "target     : $MODEL_PATH"
say "eval venv  : $JETSPEC_VENV_PATH"
say "datasets   : ${DATASET_LIST[*]}   samples=${SAMPLES:-ALL}"
say "J values   : ${J_LIST[*]}"
say "logs       : $OUT_ROOT"
say "$(( ${#DATASET_LIST[@]} * ${#J_LIST[@]} )) sequential 8-GPU runs ahead -- this owns the whole node"

for dataset in "${DATASET_LIST[@]}"; do
  for j in "${J_LIST[@]}"; do
    LOG="$OUT_ROOT/${dataset}-j${j}.log"
    say "=== dataset=$dataset J=$j -> $LOG ==="
    PYTHONPATH="$REPO_ROOT" "$TORCHRUN" \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 --nproc_per_node=8 \
      bench/reference/benchmark.py \
      --model "$MODEL_PATH" \
      --draft-head "$DRAFT_HEAD" \
      --dataset "$dataset" "${SAMPLES_ARGS[@]}" --max-new "$MAX_NEW" --warmup-rounds "$WARMUP_ROUNDS" \
      --algos "$ALGO" --depth "$DEPTH" --width "$WIDTH" --budget "$BUDGET" --self-refine-j "$j" \
      --attn-implementation "$ATTN_IMPL" --include-dflash-baseline \
      2>&1 | tee "$LOG"
    grep -qE "^DFlash blocksize=" "$LOG" || die "no DFlash result line in $LOG -- run did not complete cleanly"
  done
done

echo
say "summary (chain / tree accept_len, tokens per target forward):"
printf "%-10s %5s %14s %14s\n" "dataset" "J" "chain" "tree($ALGO)"
for dataset in "${DATASET_LIST[@]}"; do
  for j in "${J_LIST[@]}"; do
    LOG="$OUT_ROOT/${dataset}-j${j}.log"
    chain=$(grep -oE "accept_len=[0-9.]+" "$LOG" | head -1 | cut -d= -f2)
    tree=$(grep -E "^$ALGO " "$LOG" | awk '{print $2}')
    printf "%-10s %5s %14s %14s\n" "$dataset" "$j" "${chain:-?}" "${tree:-?}"
  done
done
say "done. Per-run logs and raw generations under $OUT_ROOT"
