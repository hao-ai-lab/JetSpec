"""tree_to_chain — single-pass, uncertainty-aware tree shaping.

Builds the verification tree from ONE drafter pass (no mid-tree drafter re-run,
no external profile, no prompt routing), so it runs on the offline engine and
recovers crossproduct at its identity knob. `fanout_cap/` caps children per
depth from a confidence signal (the top-2 logprob gap).

Importing this package registers its algorithms via @register_tree_algo.
"""
from jetflow.tree.tree_to_chain import fanout_cap    # noqa: F401
