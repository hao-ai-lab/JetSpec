#!/usr/bin/env bash
# Expects two peer-access A100, B200 or GB200 GPUs for Qwen3-8B speedup over AR.
# Run the Evaluation command from the root README from the repository root.
set -euo pipefail

python bench/jetspec_plus_plus/speedup.py --model qwen3-8b --dataset math500
