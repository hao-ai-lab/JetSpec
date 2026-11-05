#!/usr/bin/env bash
# Expects a GB200 GPU pair for the Qwen3-8B Table 3 reproduction.
# Run from the repository root; set HOST_VALUES and OUTPUT_DIR for this host.
# See bench/jetspec_plus_plus/reproduce_paper_results.md for the host file.
set -euo pipefail

python bench/jetspec_plus_plus/reproduce_paper_results.py --model qwen3-8b --assets release \
    --host-values "${HOST_VALUES:-/path/to/host-8b.json}" --devices 0,1 \
    --output-dir "${OUTPUT_DIR:-/path/to/table3-8b}"
