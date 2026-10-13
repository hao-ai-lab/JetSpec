"""Both benchmark arms receive the same prompt warmup and session capacity."""
from types import SimpleNamespace
import sys

import torch

from bench.jetspec.engine import tps_walltime as bench


def test_warm_all_and_session_apply_to_both_arms(monkeypatch):
    calls = []

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            return SimpleNamespace(input_ids=torch.ones((1, len(prompt)), dtype=torch.long))

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]['content']

    class Engine:
        resolved_attn_implementation = 'sdpa'
        fused_moe_blocks = 0
        _use_cudagraph = True
        tokenizer = Tokenizer()

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, sp, **kwargs):
            calls.append(('ar', prompt, kwargs))
            return {'token_ids': [1, 2]}

        def generate_tree(self, prompt, drafter, **kwargs):
            calls.append(('tree', prompt, kwargs))
            return {'token_ids': [1, 2], 'rounds': 1, 'accept_lengths': [1]}

    monkeypatch.setattr(bench, 'JetSpecEngine', Engine)
    monkeypatch.setattr(bench, 'load_draft_head', lambda *a, **k: SimpleNamespace(
        target_layer_ids=[0], block_size=4))
    monkeypatch.setattr(bench, '_build_drafter', lambda *a, **k: object())
    monkeypatch.setattr(bench, '_load_prompt_bank', lambda *a: ['short', 'longer prompt'])
    monkeypatch.setattr(bench.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(bench.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('JETSPEC_DRAFT_HEAD', 'local')
    monkeypatch.setenv('JETSPEC_BACKEND', 'triton_paged_tree_cudagraph')
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(sys, 'argv', ['tps_walltime', '--samples', '2', '--max-tokens', '4',
                                    '--warmup-samples-per-rank', '0', '--warm-all', '--session'])
    bench.main()
    assert [(arm, prompt) for arm, prompt, _ in calls] == [
        ('ar', 'short'), ('tree', 'short'),
        ('ar', 'longer prompt'), ('tree', 'longer prompt'),
        ('ar', 'short'), ('ar', 'longer prompt'),
        ('tree', 'short'), ('tree', 'longer prompt')]
    for _, _, options in calls:
        assert options['session'] is True
        assert options['session_prompt_capacity'] == 256
