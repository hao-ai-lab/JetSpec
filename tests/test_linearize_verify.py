import torch

from ptd.draft import RandomTreeDrafter, TargetEchoTreeDrafter
from ptd.engine.llm import SamplingParams
from ptd.tree import DraftTree
from ptd.tree._core.base import TreeAlgorithm
from ptd.tree._core.registry import register_tree_algo
from tests.test_jetflow_tree import PROMPT, _tiny_jetflow, _tiny_model


SP = SamplingParams(0.0, 18)


def _set_linearize_flag(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PTD_LINEARIZE_VERIFY", raising=False)
    else:
        monkeypatch.setenv("PTD_LINEARIZE_VERIFY", value)


def _run_tree(monkeypatch, model, drafter, *, flag, seed=1, sp=SP, **kwargs):
    _set_linearize_flag(monkeypatch, flag)
    torch.manual_seed(seed)
    return _tiny_jetflow(model, block_size=4).generate_tree(
        PROMPT,
        drafter,
        block_size=4,
        tree_width=2,
        budget=kwargs.pop("budget", 15),
        sampling_params=sp,
        return_stats=True,
        **kwargs,
    )


def _assert_same_decode(actual, expected):
    assert actual["token_ids"] == expected["token_ids"]
    assert actual["accept_lengths"] == expected["accept_lengths"]
    assert actual["tree_sizes"] == expected["tree_sizes"]


def test_linearize_flag_off_env_value_keeps_dense_path(monkeypatch):
    import ptd.tree.linearize as linearize

    calls = []
    real_expand = linearize.expand_tree_to_paths

    def wrapped_expand(tree):
        calls.append(int(tree.num_nodes))
        return real_expand(tree)

    monkeypatch.setattr(linearize, "expand_tree_to_paths", wrapped_expand)

    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    dense_absent = _run_tree(monkeypatch, model, drafter, flag=None, seed=1)
    calls.clear()
    dense_zero = _run_tree(monkeypatch, model, drafter, flag="0", seed=1)

    assert calls == []
    _assert_same_decode(dense_zero, dense_absent)


def test_linearized_verify_matches_dense_for_random_and_echo(monkeypatch):
    import ptd.tree.linearize as linearize

    calls = []
    real_expand = linearize.expand_tree_to_paths

    def wrapped_expand(tree):
        calls.append(int(tree.num_nodes))
        return real_expand(tree)

    monkeypatch.setattr(linearize, "expand_tree_to_paths", wrapped_expand)

    for seed, drafter_factory in (
        (7, lambda model: RandomTreeDrafter(model.config.vocab_size)),
        (1, lambda model: TargetEchoTreeDrafter(model)),
    ):
        model = _tiny_model(seed)
        dense = _run_tree(monkeypatch, model, drafter_factory(model), flag=None, seed=11)
        calls.clear()
        linearized = _run_tree(monkeypatch, model, drafter_factory(model), flag="1", seed=11)

        assert calls, "PTD_LINEARIZE_VERIFY=1 did not consume PathPlan"
        _assert_same_decode(linearized, dense)


class _RootGreedyDuplicateDrafter:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def propose_logits(self, context_ids, depth, target_hidden=None, **kwargs):
        from transformers import DynamicCache

        cache = DynamicCache()
        pos = torch.arange(context_ids.shape[1], device=context_ids.device).unsqueeze(0)
        logits = self.model(
            input_ids=context_ids,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        ).logits
        out = torch.zeros(
            1,
            depth,
            self.model.config.vocab_size,
            dtype=logits.dtype,
            device=context_ids.device,
        )
        out[:, 0, :] = logits[:, -1, :]
        return out


@register_tree_algo("_test_linearize_duplicate_root")
class _DuplicateRootAlgorithm(TreeAlgorithm):
    def build(self, root_token, draft_logits, block_size, tree_width, budget, device, **kwargs):
        vocab_size = int(draft_logits.shape[-1])
        greedy_child = int(draft_logits[0, 0].argmax().item())
        other_child = (greedy_child + 1) % vocab_size
        return DraftTree(
            token_ids=torch.tensor(
                [int(root_token), greedy_child, other_child, greedy_child],
                dtype=torch.long,
                device=device,
            ),
            parent_indices=torch.tensor([-1, 0, 0, 0], dtype=torch.long, device=device),
            depth=torch.tensor([0, 1, 1, 1], dtype=torch.long, device=device),
            num_nodes=4,
        )


def test_linearized_verify_duplicate_sibling_row_gather_matches_dense(monkeypatch):
    model = _tiny_model(3)
    drafter = _RootGreedyDuplicateDrafter(model)
    kwargs = {
        "algo": "_test_linearize_duplicate_root",
        "budget": 4,
        "sp": SamplingParams(0.0, 12),
    }

    dense = _run_tree(monkeypatch, model, drafter, flag=None, seed=1, **kwargs)
    linearized = _run_tree(monkeypatch, model, drafter, flag="1", seed=1, **kwargs)

    _assert_same_decode(linearized, dense)
    assert dense["accept_lengths"]
    assert all(length == 2 for length in dense["accept_lengths"])
