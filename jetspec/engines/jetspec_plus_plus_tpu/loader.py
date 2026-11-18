"""Load the pinned Qwen3-8B target and the JetSpec++ checkpoint onto one TPU chip."""
import os
import time

import jax

from .model import Dims, load_adapter, load_drafter, load_target, quantize_draft

CACHE_DIR = os.path.expanduser(os.environ.get('JETSPEC_TPU_XLA_CACHE', '~/.cache/jetspec-plus-plus-tpu/xla'))


def enable_compile_cache():
    jax.config.update('jax_compilation_cache_dir', CACHE_DIR)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

TARGET = ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218')
CHECKPOINT = ('JetSpec/jetspec-plus-plus-qwen3-8b', '9bd8b157')
DFLASH = ('z-lab/Qwen3-8B-DFlash-b16', '9b41424b7109f9c5413454f481b09a82b85333f4')


def resolve(target=None, checkpoint=None):
    from huggingface_hub import snapshot_download
    target = target or snapshot_download(TARGET[0], revision=TARGET[1])
    checkpoint = checkpoint or snapshot_download(CHECKPOINT[0], revision=CHECKPOINT[1])
    return target, checkpoint


def load_weights(target=None, checkpoint=None, *, draft_dtype='int8', verbose=True):
    enable_compile_cache()
    target, checkpoint = resolve(target, checkpoint)
    dims = Dims()
    start = time.perf_counter()
    weights = dict(target=load_target(target, dims), drafter=load_drafter(checkpoint, dims),
                   adapter=load_adapter(checkpoint, dims))
    if draft_dtype == 'int8':
        weights = quantize_draft(weights, dims)   # the bf16 drafter and adapter are dropped here
    jax.block_until_ready(weights)
    if verbose:
        stats = jax.devices()[0].memory_stats() or {}
        print(f'weights loaded in {time.perf_counter() - start:.1f}s; '
              f'HBM in use {stats.get("bytes_in_use", 0) / 2**30:.2f} GiB')
    return weights, dims, target, checkpoint


def load_method_weights(method, *, draft_dtype='int8', target=None, verbose=True):
    """Weights for one serving method: 'ar' (target only), 'jetspec' or 'dflash'."""
    from huggingface_hub import snapshot_download
    enable_compile_cache()
    if method == 'jetspec':
        return load_weights(target, draft_dtype=draft_dtype, verbose=verbose)
    target = target or snapshot_download(TARGET[0], revision=TARGET[1])
    dims = Dims()
    start = time.perf_counter()
    weights = dict(target=load_target(target, dims))
    if method == 'dflash':
        drafter = load_drafter(snapshot_download(DFLASH[0], revision=DFLASH[1]), dims)
        if draft_dtype == 'int8':
            drafter = quantize_draft(dict(weights, drafter=drafter, adapter={}), dims)['drafter']
        weights['drafter'] = drafter
    jax.block_until_ready(weights)
    if verbose:
        stats = jax.devices()[0].memory_stats() or {}
        print(f'{method} weights loaded in {time.perf_counter() - start:.1f}s; '
              f'HBM in use {stats.get("bytes_in_use", 0) / 2**30:.2f} GiB')
    return weights, dims, target, None
