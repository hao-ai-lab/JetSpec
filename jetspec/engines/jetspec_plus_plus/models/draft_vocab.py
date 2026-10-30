"""Load a fixed frequent-token draft support; expose only target vocabulary IDs."""
import json
from pathlib import Path

from jetspec.engines.jetspec_plus_plus.models.config import setting


def frequent_ids(path, k, vocab):
    """Frequency ties prefer the lowest target ID; projection order preserves that order."""
    table = json.loads(Path(path).read_text())
    counts = table.get('counts')
    if (table.get('schema') != 'jetspec.draft_vocab/1' or not isinstance(counts, list)
            or len(counts) != vocab or any(type(n) is not int or n < 0 for n in counts)
            or not any(counts)):
        raise ValueError('draft vocabulary table must contain one nonnegative count per target token')
    return sorted(sorted(range(vocab), key=lambda token: (-counts[token], token))[:k])


def configure_draft_vocab(head, run_config):
    """Gather after strict checkpoint load and precision rounding, before packs/capture.

    Keep checkpoint storage for the full-vocabulary loader contract. Fused
    top-1 consumes the gathered rows directly; general logits callers receive
    negative infinity outside the support so rejection/tree/record IDs retain
    their target-vocabulary meaning.
    """
    if not hasattr(run_config, 'engine'):
        return
    k = setting(run_config, 'engine', 'engine_settings.draft_vocab_k', '0')
    if type(k) is str and k.isdigit():
        k = int(k)
    vocab = head.weight.shape[0]
    if type(k) is not int or not 0 <= k <= vocab:
        raise ValueError('draft_vocab_k must be between zero and the target vocabulary size')
    if k == 0 or k == vocab:
        return
    path = setting(run_config, 'engine', 'engine_settings.draft_vocab_path', '')
    if not path:
        raise ValueError('draft_vocab_path is required for a reduced draft vocabulary')
    import torch
    ids = torch.tensor(frequent_ids(path, k, vocab), device=head.weight.device, dtype=torch.long)
    weight = head.weight.detach().index_select(0, ids).contiguous()
    head.weight._draft_vocab = (ids, weight)
    from jetspec.engines.jetspec_plus_plus.models.target_linear import bind_weight_role
    bind_weight_role(weight, 'lm_head')
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    if weight.is_cuda and current_engine().precision_lm_head == 'mxfp4_w4a8':
        from jetspec.engines.jetspec_plus_plus.models.target_linear import _w4a8_registry
        _w4a8_registry.pack(weight)
    import jetspec.engines.jetspec_plus_plus.runtime.run_manifest as run_manifest
    run_manifest.record_file('draft_vocabulary', path, run_config=run_config)
    print(f'DRAFT_VOCAB k={k} full={vocab} table={path}', flush=True)


def expand_logits(logits, ids, vocab):
    """Restore the target ID axis for consumers that also query rejected-token logits."""
    full = logits.new_full((*logits.shape[:-1], vocab), float('-inf'))
    return full.index_copy_(-1, ids, logits)
