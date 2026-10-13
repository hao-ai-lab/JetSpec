"""Seeded CPU fixtures and a session-wide public-module isolation gate."""
from pathlib import Path
import sys
import sysconfig

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

collect_ignore = ['golden']

@pytest.fixture(autouse=True, scope='session')
def public_module_isolation(tmp_path_factory):
    def check():
        repo = Path(__file__).resolve().parents[2]
        test_outputs = tmp_path_factory.getbasetemp().resolve()
        allowed = [Path(sysconfig.get_path(key)).resolve()
                   for key in ('stdlib', 'platstdlib', 'purelib', 'platlib')]
        allowed += [repo / 'jetspec', repo / 'tests', repo / 'bench']
        templates = sys.modules.get('torch.distributed.nn.jit.instantiator')
        if templates:
            allowed.append(Path(templates.INSTANTIATED_TEMPLATE_DIR_PATH).resolve())
        for name, module in list(sys.modules.items()):
            origin = getattr(module, '__file__', None)
            if origin and (Path(origin).is_absolute() or Path(origin).exists()):
                path = Path(origin).resolve()
                if name == 'original_llm':
                    assert path.read_bytes() == (repo / 'tests/jetspec_plus_plus/fixtures/reference/llm.py').read_bytes()
                elif name.startswith('jetspec_drafter_'):
                    assert path.is_relative_to(test_outputs), f'draft test module outside test outputs: {path}'
                else:
                    assert any(path.is_relative_to(root) for root in allowed), f'external first-party module loaded: {name}: {path}'
    check()
    yield
    check()


@pytest.fixture(autouse=True)
def seeded_case(request):
    from golden_values import begin_case
    torch.manual_seed(0)
    begin_case(request.node.nodeid)


@pytest.fixture(autouse=True, scope='session')
def single_thread():
    torch.set_num_threads(1)


def target_config():
    config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=4, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8)
    config._attn_implementation = 'sdpa'
    return config


def tiny_target(seed):
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(target_config()).eval()
