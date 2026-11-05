#!/usr/bin/env bash
# Expects two peer-access A100, B200 or GB200 GPUs for Qwen3-8B J1-R1.
# Run from the repository root; see the serving engine README's schedule table.
set -euo pipefail

python -m jetspec.engines.jetspec_plus_plus generate \
    --model qwen3-8b --schedule j1-r1 \
    "How many positive whole-number divisors does 196 have?"
