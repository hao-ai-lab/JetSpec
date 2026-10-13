"""profile_guided — tree shape driven by external acceptance profiles.

Implemented:
- `depth_rank_histogram` (B2) — per-(depth, rank) acceptance cap; the budget-aware
  generalization of the low-budget winners. Profiler: ``bench/profiling/collect_depth_rank_stats.py``.

Planned (placeholders, NOT registered — build() raises; see ``jetspec/tree/ROADMAP.md``):
- `online_warmup` (B5), `template_bandit` (B7) — need serving-loop feedback (vLLM phase).

Importing this package registers the implemented algorithms only.
"""
from jetspec.tree.profile_guided import depth_rank_histogram  # noqa: F401  (registers B2)
