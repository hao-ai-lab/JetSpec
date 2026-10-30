"""depth_rank_histogram (B2) unit gate — the profile -> per-depth-cap mapping.

Pure-tensor, CPU, no model: B2's value is how it turns an offline per-(depth,rank)
acceptance table into a fanout cap, so these tests pin that mapping directly
(identity recovery, chain profile, mixed profile, tau threshold). Losslessness of
B2 through the engine is gated separately on b200 (test_tree_algos_lossless.py,
which now includes depth_rank_histogram in ACTIVE_KWARGS).
"""
import torch

from jetspec.tree import get_algorithm

DEV = torch.device("cpu")
BLOCK, WIDTH, BUDGET = 4, 4, 63   # D = block_size - 1 = 3 depths, top-4 per depth


def _logits(seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, BLOCK - 1, 50)


def test_no_profile_recovers_accum_logp():
    """profile_table=None -> b_per_depth = K everywhere -> byte-identical accum_logp."""
    dl = _logits()
    xp = get_algorithm("accum_logp").build(7, dl, BLOCK, WIDTH, BUDGET, DEV)
    b2 = get_algorithm("depth_rank_histogram").build(7, dl, BLOCK, WIDTH, BUDGET, DEV, profile_table=None)
    assert b2.num_nodes == xp.num_nodes
    assert torch.equal(b2.token_ids, xp.token_ids)
    assert torch.equal(b2.parent_indices, xp.parent_indices)


def test_chain_profile_yields_chain():
    """A profile where only rank-0 clears tau at every depth -> a pure chain
    (root + one node per depth)."""
    dl = _logits()
    prof = {"depth_rank_accept": [[0.9, 0.0, 0.0, 0.0]] * (BLOCK - 1)}
    b2 = get_algorithm("depth_rank_histogram", tau=0.5).build(7, dl, BLOCK, WIDTH, BUDGET, DEV, profile_table=prof)
    assert b2.num_nodes == BLOCK            # root + (block_size-1) depths
    assert b2.depth.tolist() == list(range(BLOCK))


def test_mixed_profile_shapes_per_depth():
    """Keep ranks {0,1} at depth 0, only rank 0 deeper -> 2 children at depth 1,
    then a single chain below each."""
    dl = _logits()
    prof = {"depth_rank_accept": [
        [0.9, 0.30, 0.01, 0.0],   # depth 0: ranks 0,1 clear tau=0.1
        [0.9, 0.01, 0.0, 0.0],    # depth 1: only rank 0
        [0.9, 0.01, 0.0, 0.0],    # depth 2: only rank 0
    ]}
    b2 = get_algorithm("depth_rank_histogram", tau=0.1).build(7, dl, BLOCK, WIDTH, BUDGET, DEV, profile_table=prof)
    # 1 root + 2 (depth1) + 2 (depth2) + 2 (depth3) = 7
    assert b2.num_nodes == 7
    assert b2.depth.tolist() == [0, 1, 1, 2, 2, 3, 3]


def test_tau_controls_pruning_monotonically():
    """Higher tau keeps fewer ranks -> never a larger tree."""
    dl = _logits()
    prof = {"depth_rank_accept": [[0.9, 0.4, 0.2, 0.05]] * (BLOCK - 1)}
    sizes = [
        get_algorithm("depth_rank_histogram", tau=t).build(7, dl, BLOCK, WIDTH, BUDGET, DEV, profile_table=prof).num_nodes
        for t in (0.01, 0.1, 0.3, 0.6)
    ]
    assert sizes == sorted(sizes, reverse=True), f"tree size should shrink with tau: {sizes}"
    assert sizes[-1] == BLOCK          # tau=0.6 keeps only rank-0 -> chain
