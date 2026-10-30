"""EAGLE-3 draft heads for the targets the fork's own table refuses.

``bench_helpers._get_draft_model_path`` (upstream, not edited here) resolves an EAGLE draft by
target family and size and raises for every Qwen3 size but 32B.  This module holds the public
Qwen3 heads it lacks, keyed the same way, plus the one override every ``--eagle`` run honours:
``JETSPEC_EAGLE_DRAFT_PATH``, a draft directory (a snapshot holding ``config.json``, or a
HuggingFace cache base ``models--<org>--<name>``) that replaces the table.  ``bench.py``
consults ``eagle_draft_base`` before ``get_model_paths`` and hands the fork the directory as if
``--draft`` had named it; the fork's own snapshot resolution then runs unchanged.

Stdlib only, like ``env.py``.
"""
from __future__ import annotations

import os

from jetspec.engines.jetspec_plus_plus.launch_env import environment_name
from jetspec.engines.jetspec_plus_plus.run_config import RunConfig

DRAFT_PATH_ENV = environment_name('model', 'model_settings.eagle_draft_path')

# Qwen3 target size -> HuggingFace repo of a public EAGLE-3 head trained for it.  The 32B row
# stays with the fork (bench_helpers.EAGLE3_QWEN_32B / SSD_EAGLE3_QWEN_32B).
QWEN3_EAGLE3_HEADS = {
    # SpecForge-shaped: config.json + pytorch_model.bin, draft vocab 32000, no embed_tokens
    # (the loader takes them from the target), hooks the outputs of layers [1, 17, 32].
    "8": "AngelSlim/Qwen3-8B_eagle3",
}


def hf_cache_base(cache_dir: str, repo_id: str) -> str:
    """The ``models--<org>--<name>`` directory ``huggingface-cli download`` makes under the hub."""
    return os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))


def eagle_draft_base(args, cache_dir: str, run_config=None) -> str | None:
    """The draft directory ``bench.py`` hands the fork for ``--eagle``; None leaves the fork's table alone.

    ``JETSPEC_EAGLE_DRAFT_PATH`` wins over everything.  Otherwise an explicit ``--draft`` and
    every Llama target are the fork's, and a Qwen3 size is looked up here.  A directory that
    does not exist raises, naming what to fetch or set, instead of falling through to the
    fork's "not available" error for a head that is available.
    """
    config = run_config if run_config is not None else RunConfig.from_environment({})
    override = config.model.model_settings.eagle_draft_path
    if override:
        if not os.path.isdir(override):
            raise FileNotFoundError(f"{DRAFT_PATH_ENV}={override!r} is not a directory")
        return override
    if getattr(args, "draft", None) or getattr(args, "llama", False):
        return None
    repo = QWEN3_EAGLE3_HEADS.get(str(args.size))
    if repo is None:
        return None
    base = hf_cache_base(cache_dir, repo)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"EAGLE-3 draft for Qwen3-{args.size}B: {repo} is not under SSD_HF_CACHE ({base}); "
            f"`huggingface-cli download {repo}` there, or point {DRAFT_PATH_ENV} at the head")
    return base
