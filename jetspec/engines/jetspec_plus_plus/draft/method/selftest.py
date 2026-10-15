"""Run a complete tiny CPU round using only the method and torch."""
import torch
from torch import nn
from types import SimpleNamespace

from .accept import greedy_accept_device
from .adapter import CtxKVC1Head
from .decoder import AnchorDecoder
from .round import MethodRound, MethodState
from .schedule import Schedule


class TinyTarget(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def tap(self, prefix):
        return self.embedding(prefix)

    def verify(self, tokens):
        return self.head(self.embedding(tokens))


class TinyAdapter(nn.Module):
    def __init__(self, embedding, hidden):
        super().__init__()
        self.embedding = embedding
        config = SimpleNamespace(hidden_size=hidden, intermediate_size=16,
                                 hidden_act='silu', num_attention_heads=2,
                                 num_key_value_heads=1, attention_bias=False,
                                 rms_norm_eps=1e-6, rope_theta=10000.,
                                 dflash_config={'target_layer_ids': [0]})
        self.head = CtxKVC1Head(config, resolved_config={})

    def tap(self, hidden):
        return hidden

    def block(self, ids, context):
        from .round import adapter_block
        return adapter_block(self.head, ids, context, embedding=self.embedding)


class TinyDrafter(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, ids, adapted):
        return self.head(self.embedding(ids) + adapted)


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.Linear(hidden, hidden)

    def forward(self, hidden, **kwargs):
        return hidden + torch.tanh(self.projection(hidden))


def _tiny_round(*, schedule=None, width=5, tau=0.):
    torch.manual_seed(0)
    vocab, hidden = 41, 8
    target = TinyTarget(vocab, hidden)
    drafter = TinyDrafter(vocab, hidden)
    adapter = TinyAdapter(drafter.embedding, hidden)
    decoder = AnchorDecoder(layers=nn.ModuleList([TinyDecoderLayer(hidden), TinyDecoderLayer(hidden)]),
                            norm=nn.LayerNorm(hidden), lm_head=target.head, hidden_size=hidden)
    state = MethodState(prefix=torch.tensor([1, 3, 2, 7, 5, 4]), anchor=torch.tensor(9), width=width)
    method = MethodRound(adapter=adapter, target=target, drafter=drafter, decoder=decoder,
                         schedule=schedule or Schedule(2, 2), cap=4, offer_width=32, tau=tau)
    return method, state


@torch.inference_mode()
def run(*, schedule=None, outcome='hit'):
    method, state = _tiny_round(schedule=schedule)
    initial = method.run(state)
    true_k, correction = map(int, initial.keys[0])
    if outcome == 'miss':
        correction = next(token for token in range(41)
                          if not ((initial.keys[:, 0] == true_k) &
                                  (initial.keys[:, 1] == token)).any())
    elif outcome != 'hit':
        raise ValueError('outcome must be hit or miss')
    # The caller supplies the committed verifier prefix, just as the engine
    # uploads its outcome before the next admission/segment A.
    state.prefix = torch.cat((state.prefix, initial.tokens[0, :true_k + 1]))
    candidate = method.run(state, true_k=true_k, correction=correction)
    accepted = greedy_accept_device(method.target.verify(candidate.tokens), candidate.tokens)
    assert candidate.tokens.shape == (4, 5) and candidate.keys.shape == (4, 2)
    assert torch.isfinite(state.values.p_top1).all()
    assert state.missed == (outcome == 'miss')
    assert state.err.item() == 0
    return state, candidate, accepted


def main():
    for outcome in ('hit', 'miss'):
        state, candidate, accepted = run(outcome=outcome)
        print('METHOD_SELFTEST=PASS device=cpu outcome=' + outcome
              + ' steps=' + ','.join(state.steps)
              + f' rows={candidate.keys.shape[0]} accepted={accepted[0][0].item()}')


if __name__ == '__main__':
    main()
