import json
from . import protocol
from ..dataloader import pinned_prompts


def sealed_source(model, dataset, tokenizer=None):
    name = ('27b-sys' if model == '27b' else model) + f'/prompts_{dataset}128.json'
    path = pinned_prompts('27b-sys' if model == '27b' else model, dataset)
    seals = json.loads((protocol.INPUTS / 'seals.json').read_text())
    assert protocol.sha256(path) == seals[name], 'SEALED_SOURCE_SHA'
    source = json.loads(path.read_text())
    assert [r['index'] for r in source['prompts']] == list(range(128)), 'SEALED_SOURCE_INDICES'
    if tokenizer is not None:
        for row in source['prompts']:
            assert tokenizer.encode(row['text'], add_special_tokens=False) == row['token_ids'], 'TOKEN_IDENTITY'
    return source
