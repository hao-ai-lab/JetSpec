"""Compare the fused Markov correction with the stock per-position loop on the real DSpark W1/W2.

usage: python bench/jetspec_plus_plus/dspark_opt/check_markov.py /path/to/dspark-qwen3-8b-800k [trials]
"""
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jetspec.methods.baselines.blockhead_markov import FusedMarkov  # noqa: E402


class Markov(torch.nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.markov_w1 = torch.nn.Embedding.from_pretrained(w1)
        self.markov_w2 = torch.nn.Linear(w2.shape[1], w2.shape[0], bias=False)
        self.markov_w2.weight.data.copy_(w2)


def stock(markov, logits, anchor):
    prev, corrected, tokens = anchor, [], []
    for position in range(logits.shape[1]):
        step = logits[:, position, :] + markov.markov_w2(markov.markov_w1(prev.long()))
        corrected.append(step.unsqueeze(1))
        prev = step.argmax(dim=-1)
        tokens.append(prev.unsqueeze(1))
    return torch.cat(corrected, 1), torch.cat(tokens, 1)


def main():
    root, trials = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200
    with safe_open(root / 'model.safetensors', 'pt', device='cuda') as f:
        w1 = f.get_tensor('markov_head.markov_w1.weight')
        w2 = f.get_tensor('markov_head.markov_w2.weight')
        lm = f.get_tensor('lm_head.weight')
    markov = Markov(w1, w2).cuda().to(torch.bfloat16)
    vocab, block = w2.shape[0], 15
    fused = FusedMarkov(markov, vocab, torch.device('cuda'))
    gen = torch.Generator(device='cuda').manual_seed(0)
    token_diff = rows_diff = logit_diff = 0
    for t in range(trials):
        # Realistic logits: the LM head applied to random unit hidden states.
        hidden = torch.randn(1, block, lm.shape[1], device='cuda', generator=gen, dtype=torch.bfloat16)
        logits = (hidden @ lm.T).to(torch.bfloat16)
        anchor = torch.randint(0, vocab, (1,), device='cuda', generator=gen)
        ref_c, ref_t = stock(markov, logits, anchor)
        out_c = torch.empty_like(logits)
        out_t = torch.empty(1, block, device='cuda', dtype=torch.long)
        fused(logits, anchor, out_c, out_t)
        token_diff += int((ref_t != out_t).sum())
        rows_diff += int((ref_t != out_t).any())
        logit_diff += int((ref_c != out_c).sum())
    print(f'trials {trials}: differing tokens {token_diff}/{trials * block}, blocks with a difference '
          f'{rows_diff}, differing corrected logits {logit_diff}/{trials * block * vocab}')

    # Graph-replay timing of both, as the static block head runs them.
    logits = torch.randn(1, block, vocab, device='cuda', dtype=torch.bfloat16)
    anchor = torch.zeros(1, device='cuda', dtype=torch.long)
    out_c = torch.empty_like(logits)
    out_t = torch.empty(1, block, device='cuda', dtype=torch.long)
    for name, fn in (('stock', lambda: stock(markov, logits, anchor)),
                     ('fused', lambda: fused(logits, anchor, out_c, out_t))):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            g.replay()
        torch.cuda.synchronize()
        print(f'{name}: {(time.perf_counter() - t0) / 200 * 1e3:.3f} ms per block (graph replay)')


if __name__ == '__main__':
    main()
