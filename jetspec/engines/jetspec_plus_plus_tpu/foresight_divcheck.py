"""At each divergence between two-chip and one-chip outputs, are the two tokens the target's top 2?"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main():
    from . import loader, model
    from .bench import load_rows
    from .engine import TPUJetSpecPP
    from .foresight import TwoChipJetSpecPP
    cpu = jax.devices('cpu')[0]
    d0, d1, d2 = jax.devices()[:3]
    orig = model.load_target

    def host_target(r, d):
        with jax.default_device(cpu):
            return orig(r, d)
    loader.load_target = host_target
    weights, dims, _, ckpt = loader.load_weights(draft_dtype='int8', verbose=False)
    tw = weights['target']
    with jax.default_device(cpu):
        q = jax.jit(model.quantize)
        tw = dict(tw, layers={k: (q(v) if k in ('qkv', 'o', 'gate_up', 'down') else v) for k, v in tw['layers'].items()})
    th = json.loads((Path(ckpt) / 'calibration' / 'thresholds.json').read_text())
    coef = {k: float(v['value']) for k, v in th['e04_kscore_blend']['pooled'].items()}
    temp = float(th['e04_rowrule_joint']['pooled']['temperature']['value'])
    w0 = jax.device_put(dict(target=tw), d0)
    w1 = jax.device_put(dict(drafter=weights['drafter'], adapter=weights['adapter'], decoder=weights['decoder'],
                             embed=tw['embed'], norm=tw['norm']), d1)
    w2 = jax.device_put(dict(target=tw, drafter=weights['drafter'], adapter=weights['adapter'],
                             decoder=weights['decoder']), d2)
    two = TwoChipJetSpecPP(w0, w1, coef, temp, dims=dims, schedule='J1-R0', rows=8)
    one = TPUJetSpecPP(w2, dims=dims, schedule='J1-R0', capacity=2048 + 512)
    rows = dict(load_rows('gsm8k', 8))
    tgt2 = w2['target']
    for index in [int(x) for x in sys.argv[1:]]:
        ids = rows[index]
        out2, _ = two.generate(ids, 256, prompt_index=index)
        with jax.default_device(d2):
            out1 = one.generate(ids, 256, prompt_index=index)
            pos = next(i for i in range(min(len(out1), len(out2))) if out1[i] != out2[i])
            seq = list(ids) + out1[:pos]
            padded = -(-len(seq) // 64) * 64
            cache = model.new_cache(dims, 1024, adapter=False, draft=False)
            logits, _, _ = model.target_forward(tgt2, cache, jnp.asarray(np.pad(np.asarray(seq, np.int32),
                                                (0, padded - len(seq)))), 0, dims, 1024, logit_row=len(seq) - 1)
            lg = np.asarray(logits[0])
        top = np.argsort(-lg)[:3]
        print(json.dumps(dict(index=index, pos=pos, one=out1[pos], two=out2[pos], top3=top.tolist(),
                              gap_top2=float(lg[top[0]] - lg[top[1]]),
                              both_in_top2=bool({out1[pos], out2[pos]} <= set(top[:2].tolist())))), flush=True)


if __name__ == '__main__':
    main()
