"""The public example loads tiny local artifacts and checks greedy equality."""
import re

import pytest

from test_bench_cli import cli, local_cli_inputs


@pytest.mark.parametrize('environment_paths', [False, True])
def test_example_generates_with_local_artifacts(tmp_path, environment_paths):
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    inputs = local_cli_inputs(tmp_path)
    # The bench fixture uses token IDs; the example exercises text tokenization.
    target_path = inputs[inputs.index('--model') + 1]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(target_path)
    tokenizer.backend_tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save_pretrained(target_path)
    args, env = [], {}
    for flag, name in [('--model', 'JETSPEC_MODEL'), ('--drafter', 'JETSPEC_DRAFTER'),
                       ('--adapter', 'JETSPEC_ADAPTER'),
                       ('--heads-bundle', 'JETSPEC_HEADS_BUNDLE')]:
        value = inputs[inputs.index(flag) + 1]
        if environment_paths:
            env[name] = value
        else:
            args.extend([flag, value])
    result = cli('examples/jetspec_plus_plus/simple_generate.py', *args,
                 '--prompt', '1 2 3 4 5 6', '--dataset', 'tiny',
                 '--device', 'cpu', '--max-tokens', '16', env=env)
    assert result.returncode == 0, result.stderr
    text, mean, equality = result.stdout.strip().splitlines()
    assert text
    assert float(re.fullmatch(r'accepted-length mean: (\S+)', mean).group(1)) > 0
    assert equality == 'AR equality: True'
