# Reader reference snapshots

These snapshots preserve the historical reader implementations compared by the parity tests. Loading local files keeps the comparisons usable in shallow public clones without historical Git objects. The initial snapshot is byte-for-byte unchanged. The fp32 snapshot only renames reference terminology in comments and a CLI help string; computations and signatures are unchanged.

The frozen public core is used only by this branch, which changes its generation hooks. Reader snapshots supply executable references for the same generated inputs and capture options; their comparisons require the historical implementation, not just a source hash.

| File | Source revision and path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `reader_initial.py` | `b3d3a15:jetspec/jetspec_plus_plus/reader.py` | 12025 | `7fdc4a6ae8857a09d52b9adecebf0c65ae966f4b2be5aa97a3f86b0063a7250b` |
| `reader_fp32.py` | `8dce203:jetspec/jetspec_plus_plus/reader.py` | 17293 | `60285e4786e664217fbffc6c7f952de397723b1b5a809422fbc466ca7c32433d` |
| `llm.py` | `2c7b3fa:jetspec/core/llm.py` | 30619 | `3a543603afa1f2ada32bb4af77242a5661c6497ede5bba5ffb7d9c5f02421305` |
