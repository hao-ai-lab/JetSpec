#!/usr/bin/env bash
# Answer one prompt with JetSpec++ on qwen3-8b. Serving needs two GPUs with peer access.
# Usage: ./examples/jetspec_plus_plus/launch_8b.sh ["prompt"]
# Set DEVICES or MAX_TOKENS to change the GPU pair or the answer length.
set -euo pipefail

PROMPT="${1:-How many positive whole-number divisors does 196 have?}"

python -m jetspec.engines.jetspec_plus_plus generate \
    --model qwen3-8b \
    --devices "${DEVICES:-0,1}" \
    --max-tokens "${MAX_TOKENS:-2048}" \
    "$PROMPT"
