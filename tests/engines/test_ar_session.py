"""AR session replay preserves greedy tokens across ragged prompt boundaries.

The CPU executor runs the actual graph body and compiled-stack math eagerly.
CUDA capture and Triton execution are separately gated by the eight-prompt read.
"""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from jetspec.core.llm import SamplingParams
from jetspec.core.model_runner import ModelRunner
from jetspec.engines.jetspec.engine import JetSpecEngine


class _Tokenizer:
    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))


def engine(seed):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=16)
    config._attn_implementation = 'sdpa'
    eng = object.__new__(JetSpecEngine)
    eng.model = Qwen3ForCausalLM(config).eval()
    eng.tokenizer = _Tokenizer()
    eng.runner = ModelRunner(eng.model)
    eng.device = 'cpu'
    eng.dtype = torch.float32
    eng.block_size = 4
    eng.eos_token_ids = set()
    return eng


def cpu_graphs(monkeypatch, eng):
    from jetspec.engines.jetspec.serving.compiled_verify_stack import CompiledVerifyStack
    from jetspec.engines.jetspec.verify.graph_capture import GraphedVerify
    from jetspec.engines.jetspec.kernels.paged_tree_attn_op import paged_tree_attn

    def attention(q, kp, vp, tables, cu, lengths, bias, scale, groups, block_size,
                  *logical):
        length = int(lengths[0])
        positions = torch.arange(length)
        blocks = tables[0, positions // block_size].long()
        keys = kp[blocks, positions % block_size].transpose(0, 1).unsqueeze(0)
        values = vp[blocks, positions % block_size].transpose(0, 1).unsqueeze(0)
        query = q.transpose(0, 1).unsqueeze(0)
        result = torch.nn.functional.scaled_dot_product_attention(
            query, keys, values, scale=scale, enable_gqa=True)
        return result.squeeze(0).transpose(0, 1).contiguous()

    paged_tree_attn.register_kernel('cpu', attention)
    eng.compiled_ar = CompiledVerifyStack(eng.model, block_size=eng.block_size)
    eng.compiled_ar._compiled = eng.compiled_ar._stack
    eng.attn_backend = 'triton_paged_tree_cudagraph'
    eng._use_cudagraph = True
    counts = SimpleNamespace(captures=0, replays=0, reserves=0, decode=False)

    # Prefill uses the real SDPA runner; only the GPU-only handoff is disabled.
    forward = eng.runner.forward
    def prefill(ids, cache, pos):
        counts.decode = False
        cache._paged_handoff = False
        return forward(ids, cache, pos)
    monkeypatch.setattr(eng.runner, 'forward', prefill)

    def capture(self, B):
        counts.captures += 1
        self.outputs[B] = self._call_stack(B).clone()
        def replay():
            counts.decode = True
            counts.replays += 1
            self.outputs[B].copy_(self._call_stack(B))
        self.graphs[B] = SimpleNamespace(replay=replay)
    monkeypatch.setattr(GraphedVerify, '_capture_bucket', capture)

    from jetspec.engines.jetspec.serving.paged_kv_cache import PagedKVCache
    reserve = PagedKVCache.reserve_tree_slots
    def reserve_once(self, *args, **kwargs):
        assert not counts.decode, 'slot reservation returned to the token loop'
        counts.reserves += 1
        return reserve(self, *args, **kwargs)
    monkeypatch.setattr(PagedKVCache, 'reserve_tree_slots', reserve_once)
    return counts


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_session_reuses_cache_graph_and_keeps_tokens(monkeypatch, seed):
    torch.set_num_threads(1)
    eng = engine(seed)
    prompts = [torch.tensor([[1, 2, 3]]), torch.tensor([[7, 6, 5, 4, 3, 2, 1]]),
               torch.tensor([[4, 9, 8, 2]])]
    sp = SamplingParams(0.0, 24)
    expected = [eng.generate(prompt, sp)['token_ids'] for prompt in prompts]
    counts = cpu_graphs(monkeypatch, eng)
    actual = [eng.generate(prompt, sp, session=True, session_prompt_capacity=8)['token_ids']
              for prompt in prompts]
    assert actual == expected
    assert counts.captures == 1
    assert counts.replays == len(prompts) * (sp.max_new_tokens - 1)
    assert counts.reserves == len(prompts)


def test_session_resets_after_eos_and_rejects_overflow(monkeypatch):
    torch.set_num_threads(1)
    eng = engine(0)
    prompts = [torch.tensor([[1, 2, 3]]), torch.tensor([[7, 6, 5, 4, 3, 2, 1]])]
    sp = SamplingParams(0.0, 24)
    eng.eos_token_ids = {eng.generate(prompts[0], sp)['token_ids'][0]}
    expected = [eng.generate(prompt, sp)['token_ids'] for prompt in prompts]
    assert len(expected[0]) == 1
    counts = cpu_graphs(monkeypatch, eng)
    actual = []
    identities = []
    for prompt in prompts:
        actual.append(eng.generate(prompt, sp, session=True, session_prompt_capacity=8)['token_ids'])
        state = eng._ar_session
        identities.append((id(state['cache']), state['cache'].pool(0)[0].data_ptr(),
                           id(state['graph'])))
    assert actual == expected
    assert identities[0] == identities[1]
    assert counts.replays == sum(len(ids) - 1 for ids in expected)
    with pytest.raises(ValueError, match='prompt_len'):
        eng.generate(torch.ones(1, 9, dtype=torch.long), sp, session=True)
    with pytest.raises(ValueError, match='max_new_tokens'):
        eng.generate(prompts[0], SamplingParams(0.0, 25), session=True)


def test_temperature_sampling_keeps_seeded_tokens(monkeypatch):
    torch.set_num_threads(1)
    eng = engine(1)
    prompt = torch.tensor([[2, 3, 4, 5]])
    sp = SamplingParams(0.7, 24)
    torch.manual_seed(7)
    expected = eng.generate(prompt, sp)['token_ids']
    cpu_graphs(monkeypatch, eng)
    torch.manual_seed(7)
    actual = eng.generate(prompt, sp, session=True)['token_ids']
    assert actual == expected
