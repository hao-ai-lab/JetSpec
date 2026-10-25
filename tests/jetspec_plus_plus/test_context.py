"""Staged tails are used for exactly one round and checked against verification."""
from golden_values import golden
import pytest
import torch

from jetspec.methods.jetspec_plus_plus.context import (
    OneRoundAdaptedContext, build_hypothesis_contexts, build_protocol_hypothesis_contexts,
)
from jetspec.methods.jetspec_plus_plus.models.drafter import StockAdaptedMark


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_context_equal(seed):
    torch.manual_seed(seed)
    exact = torch.randn(1, 9, 16)
    ids = torch.randint(0, 41, (1, 10))
    predicted = torch.randn(1, 4, 16)
    actual = OneRoundAdaptedContext()
    for context in (actual,):
        context.stage(start_position=5, token_ids=ids[:, 5:9], predicted_hidden=predicted)
    for left, right in zip(actual.consume(exact, ids), golden('value_1')):
        assert torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
    embedding = StockAdaptedMark(16, dtype=torch.float32, device=torch.device('cpu'))
    args = dict(predicted_hidden=predicted, k_hypotheses=[0, 1, 3],
                adapted_position_embedding=embedding)
    for left, right in zip(build_hypothesis_contexts(exact, **args),
                           golden('value_2')):
        assert torch.equal(left, right)
    for left, right in zip(build_protocol_hypothesis_contexts(exact, block_size=16, **args),
                           golden('value_3')):
        assert torch.equal(left, right)
    with pytest.raises(RuntimeError, match='token mismatch'):
        actual.consume(exact, ids + 1)


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('mode', ['kth', 'token', 'noise'])
@pytest.mark.parametrize('device_gather', [False, True])
def test_random_context_equal(seed, mode, device_gather):
    from jetspec.methods.jetspec_plus_plus.context import build_random_context_block_token_ids

    ids = torch.tensor([[2, 3, 5, 7, 11, 13]])
    kwargs = dict(block_size=5, seed=seed, prompt_index=17, mode=mode,
                  device_gather=device_gather)
    actual = build_random_context_block_token_ids(ids, **kwargs)
    assert torch.equal(actual, golden('value_4'))
    if mode == 'kth':
        assert actual[0, 0] == ids[0, -2]
    torch.manual_seed(999)
    assert torch.equal(actual, build_random_context_block_token_ids(ids, **kwargs))


def test_random_context_rejects_missing_predecessor():
    from jetspec.methods.jetspec_plus_plus.context import build_random_context_block_token_ids

    with pytest.raises(ValueError, match='before the correction'):
        build_random_context_block_token_ids(torch.tensor([[3]]), block_size=5,
                                             seed=0, prompt_index=0, mode='kth')


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dense', [False, True])
def test_jacobi_pass_equal(seed, dense):
    from jetspec.methods.jetspec_plus_plus.method import passes

    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, 5, 13, generator=generator)
    inputs = torch.randint(13, (3, 5), generator=generator)
    first, second = inputs[:, 0], inputs[:, 1]
    pins = ((torch.rand(3, 5, generator=generator) < .4, inputs) if dense else
            (torch.tensor([-1, 0, 3]), torch.tensor([2, 4, 6])))
    assert torch.equal(passes.apply_pins(inputs, pins), golden('value_5'))
    tokens = passes.jacobi_tokens(logits, first, second, pins)
    assert torch.equal(tokens, golden('value_6'))
    assert torch.equal(passes.jacobi_converged(tokens, inputs),
                       golden('value_7'))
    for input_tokens in (inputs, tokens):
        actual = passes.reference_pass(logits, input_tokens, first, second, pins)
        for name in ('tokens', 'fixed', 'p_top1'):
            assert torch.equal(getattr(actual, name), golden('value_8'))
        assert torch.equal(actual.fixed, tokens.eq(input_tokens).all(-1))


@pytest.fixture
def tiny_draft_head():

    from transformers import Qwen3Config

    from jetspec.methods.jetspec.draft_head import DFlashDraftModel
    config = Qwen3Config(vocab_size=13, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8)
    config.num_target_layers = 4
    config.block_size = 4
    config.dflash_config = dict(target_layer_ids=[1], causal_head=True, mask_token_id=13)
    config._attn_implementation = 'sdpa'
    return DFlashDraftModel(config).eval()


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('start', [0, 1])
def test_uncached_draft_forward_equal(tiny_draft_head, seed, start):
    """Check frozen CPU head outputs; bucketed capture requires CUDA, its step does not."""
    from types import SimpleNamespace
    from jetspec.methods.jetspec_plus_plus.method.passes import DraftForwardPass, draft_forward

    torch.manual_seed(seed)
    head = tiny_draft_head
    embedding, lm_head = torch.nn.Embedding(13, 16), torch.nn.Linear(16, 13, bias=False)
    ids = torch.randint(13, (1, 4))
    context = torch.randn(1, 5, 16)
    calls = []
    handle = head.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs),
                                            with_kwargs=True)
    actual = draft_forward(head, embedding, lm_head, ids, context, start=start)
    assert isinstance(actual, DraftForwardPass)
    assert torch.equal(actual.input_tokens, ids[:, start:])
    assert calls[0]['past_key_values'] is None
    assert calls[0]['use_cache'] is False
    assert calls[0]['is_causal'] is True
    assert calls[0]['attention_backend'] == 'sdpa'
    assert torch.equal(calls[0]['position_ids'], torch.arange(9)[None])
    if start:
        assert torch.equal(actual.logits, golden('value_9'))
        assert torch.equal(actual.input_tokens, golden('value_10'))
    else:
        assert torch.equal(actual.logits, golden('value_11'))
    changed = draft_forward(head, embedding, lm_head, ids, context + 1, start=start)
    assert not torch.equal(actual.logits, changed.logits)
    handle.remove()


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_refresh_cutoff_equal(seed):
    from jetspec.methods.jetspec_plus_plus.method.refresh import refresh_cutoff, refresh_khat_anchor0

    probabilities = torch.rand(3, 15, generator=torch.Generator().manual_seed(seed))
    for start, final in ((1, None), (1, 15), (2, 19)):
        assert torch.equal(refresh_cutoff(probabilities, .1, start=start, final=final),
                           golden('value_12'))
    edge = torch.ones(3, 15)
    edge[0, :2] = torch.tensor([.5, .5])
    edge[1, -1] = 0
    assert torch.equal(refresh_cutoff(edge, .5, final=15), torch.tensor([2, 15, 15]))
    logits = torch.randn(3, 16, 13, generator=torch.Generator().manual_seed(seed))
    p_top1 = logits.float().softmax(-1).amax(-1)
    p_drafts = torch.softmax(logits[:, 1:], -1, dtype=torch.float32).amax(-1)
    assert torch.equal(p_drafts, p_top1[:, 1:])
    assert torch.equal(refresh_khat_anchor0(p_drafts, .1),
                       golden('value_13'))
    assert torch.equal(refresh_khat_anchor0(p_drafts, .1),
                       refresh_cutoff(p_top1[:, 1:], .1, final=15))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_refresh_inject_equal(monkeypatch, seed):
    """Serving cache commit needs a bucket; test its CPU projection and visibility math.

    Compare the packed N=1 oracle and CPU dense bucket mask, without a GPU cache.
    """
    from types import SimpleNamespace
    from jetspec.methods.jetspec_plus_plus.method.refresh import inject

    generator = torch.Generator().manual_seed(seed)
    base, rows, block = 8, 3, 16
    context = torch.randn(rows, base, 16, generator=generator)
    hidden = torch.randn(rows, block, 16, generator=generator)
    cutoff = torch.tensor([0, seed + 2, 15])
    combined, positions, mask = inject(context, hidden, cutoff)
    assert torch.equal(combined[:, :base], context)
    assert torch.equal(positions, torch.cat([torch.arange(base + block),
                                           torch.arange(base, base + block)])[None].repeat(rows, 1))
    assert torch.equal(mask[..., :base], torch.zeros(rows, 1, block, base))
    assert torch.equal(mask[..., base + block:], torch.zeros(rows, 1, block, block))
    monkeypatch.setenv('JETSPEC_EVAL_MODE', 'pipelined')
    head = SimpleNamespace(fc=torch.nn.Identity(), hidden_norm=torch.nn.Identity(),
                           _refresh_inject_kv=(torch.eye(16), None, None, None))
    bucket = SimpleNamespace(head=head, R=rows, B=block, dtype=torch.float32,
                             _refresh_mark=False)
    projected = golden('value_14')
    assert torch.equal(combined[:, base:], projected)

    packed = torch.zeros(rows * block * ((base + rows * block) // 8), dtype=torch.uint8)
    bits = golden('packed_bits')
    allowed = bits.reshape(rows, block, base + rows * block)
    assert torch.equal(allowed.reshape(rows * block, -1), golden('value_15'))
    for row in range(rows):
        own = allowed[row, :, base + row * block:base + (row + 1) * block]
        assert torch.equal(mask[row, 0, :, base:base + block].eq(0), own)
        assert not own[:, -1].any()
        for other in range(rows):
            if other != row:
                assert not allowed[row, :, base + other * block:base + (other + 1) * block].any()
    # A later refresh rebuilds visibility, hiding all entries beyond the new cutoff.
    smaller = torch.zeros_like(cutoff)
    _, _, next_mask = inject(context, hidden, smaller)
    bits = golden('packed_bits')
    for row in range(rows):
        own = bits.reshape(rows, block, -1)[row, :, base + row * block:base + (row + 1) * block]
        assert torch.equal(next_mask[row, 0, :, base:base + block].eq(0), own)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_recipe_inclusive_count_conversion(monkeypatch, seed):
    """Run the actual eager recipe's count-to-cutoff hook on CPU tensor state."""
    from types import SimpleNamespace
    from jetspec.methods.jetspec_plus_plus.method.refresh import inject

    monkeypatch.setenv('JETSPEC_REFRESH_INJECT_INCLUSIVE', '1')
    used = torch.tensor([seed, seed + 1, 15])
    hidden = torch.randn(3, 16, 4, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(golden('value_16'), used)
    _, _, mask = inject(torch.zeros(3, 5, 4), hidden, golden('value_16'))
    slots = torch.arange(16)
    expected = ((slots[None, :, None] > slots[None, None, :]) &
                (slots[None, None, :] <= used[:, None, None]))
    assert torch.equal(mask[:, 0, :, 5:21].eq(0), expected)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_refreshed_forward_equal(tiny_draft_head, seed):
    """Run the stale reader's pure-torch refresh head call on an equal fixed layout."""
    from types import SimpleNamespace
    from jetspec.methods.jetspec_plus_plus.method.passes import draft_forward
    from jetspec.methods.jetspec_plus_plus.method.refresh import inject

    torch.manual_seed(seed)
    embedding, lm_head = torch.nn.Embedding(13, 16), torch.nn.Linear(16, 13, bias=False)
    ids = torch.randint(13, (1, 4))
    context, refreshed = torch.randn(1, 5, 16), torch.randn(1, 4, 16)
    refreshed[:, -1] = 0
    combined, positions, mask = inject(context, refreshed, torch.tensor([3]), block_size=4)
    actual = draft_forward(tiny_draft_head, embedding, lm_head, ids, combined,
                           position_ids=positions, attention_mask=mask, start=1)
    assert torch.equal(actual.logits, golden('value_9'))
    assert torch.equal(actual.input_tokens, golden('value_10'))


def test_public_modules_are_standalone():
    """Import the leaf API in a fresh process without the test golden path setup."""
    import subprocess
    import sys

    check = '''
import sys
from jetspec.methods.jetspec_plus_plus import context
from jetspec.methods.jetspec_plus_plus.method import passes, refresh
for module in (context, passes, refresh):
    assert module.__all__
    assert all(hasattr(module, name) for name in module.__all__)
from pathlib import Path
from importlib.metadata import packages_distributions
import sysconfig
known_roots = set(sys.stdlib_module_names) | set(packages_distributions()) | {'__main__', '__mp_main__', 'jetspec', 'tests'}
allowed = [Path(sysconfig.get_path(key)).resolve() for key in ('stdlib', 'platstdlib', 'purelib', 'platlib')]
allowed += [Path.cwd() / 'jetspec', Path.cwd() / 'tests']
templates = sys.modules.get('torch.distributed.nn.jit.instantiator')
if templates:
    allowed.append(Path(templates.INSTANTIATED_TEMPLATE_DIR_PATH).resolve())
for name, module in list(sys.modules.items()):
    origin = getattr(module, '__file__', None)
    if origin and (Path(origin).is_absolute() or Path(origin).exists()):
        assert any(Path(origin).resolve().is_relative_to(root) for root in allowed), origin
    else:
        assert name.split('.')[0] in known_roots or name.startswith(('_cython_', 'cython_runtime')), name
assert not any(name.split('.')[0] in {'triton', 'flashinfer'} for name in sys.modules)
'''
    subprocess.run([sys.executable, '-c', check], check=True)
