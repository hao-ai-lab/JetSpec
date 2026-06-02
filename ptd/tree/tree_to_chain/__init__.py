"""tree_to_chain — single-pass, uncertainty-aware tree shaping.

Every algorithm here builds the verification tree from ONE drafter pass (no
mid-tree drafter re-run, no external profile, no prompt routing), so all of
them run on the offline engine and ship today. They share one idea: spend the
fixed node budget where the drafter is confident and chain-extend where it
isn't, recovering crossproduct (the full-fanout baseline) at their identity
knob settings.

Grouped by the mechanism they use to bias the tree toward the spine:

- `fanout_cap/`   — cap children per depth/node from a confidence signal
                    (top-2 gap, marginal/top-K entropy, top-K mass).
- `score_adjust/` — leave fanout open, reorder the expansion heap key
                    (entropy penalty, budget-modulated depth reward).
- `path_state/`   — cap per node from accumulated path state
                    (drift off the rank-1 chain, sibling-rank decay).
- `composition/`  — combinations of the above (e.g. budget-gated top-2 gap).

Importing this package registers every algorithm via the @register_tree_algo
decorator (side-effect imports below).
"""
from ptd.tree.tree_to_chain import fanout_cap    # noqa: F401
from ptd.tree.tree_to_chain import score_adjust  # noqa: F401
from ptd.tree.tree_to_chain import path_state     # noqa: F401
from ptd.tree.tree_to_chain import composition    # noqa: F401
