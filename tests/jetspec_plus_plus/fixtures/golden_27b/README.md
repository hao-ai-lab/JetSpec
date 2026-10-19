# Hybrid tail references

The maintainers generated these frozen values from the JetSpec++ reference
implementation. Regeneration is not part of this repository.

`tail-{0,1,2}.safetensors` preserve fp32 CPU weights, committed GDN state,
full-attention K/V, predicted windows, and bitwise logits. `provenance.json`
records fixture file hashes, seeds, tensor shapes, and the tiny `qwen3_5_text`
configuration. Normal tests exercise the public checkpoint loader before
comparing these outputs.

The public implementation uses the reference implementation's torch GDN
recurrence and SDPA window math on both CPU and CUDA. Its target verification
records each causal step so rejected speculative recurrent states are discarded.
