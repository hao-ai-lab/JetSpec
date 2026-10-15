import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
from random import randint
from typing import List, Optional, Tuple
from transformers import AutoTokenizer
from jetspec.engines.jetspec_plus_plus.serving.paths import resolve_paths






def get_model_paths(args, cache_dir: str = None, *, run_config=None) -> Tuple[str, str, Optional[str]]:
    """Resolve model and draft paths (pointing to snapshot dirs with config.json)."""
    if cache_dir is None:
        cache_dir = resolve_paths(run_config).HF_CACHE_DIR
    if args.llama:
        size_to_model = {
            "1": "Llama-3.2-1B-Instruct",
            "3": "Llama-3.2-3B-Instruct",
            "8": "Llama-3.1-8B-Instruct",
            "70": "Llama-3.3-70B-Instruct" if getattr(args, "eagle", False) else "Llama-3.1-70B-Instruct",
        }
        if args.size not in size_to_model:
            raise ValueError(
                f"Size {args.size} not available for Llama models. Available sizes: {list(size_to_model.keys())}"
            )
        model_name = size_to_model[args.size]
        model_base = os.path.join(
            cache_dir, f"models--meta-llama--{model_name}")
        default_draft_base = os.path.join(
            cache_dir, "models--meta-llama--Llama-3.2-1B-Instruct")
    else:
        size_to_model = {
            "0.6": "Qwen3-0.6B",
            "1.7": "Qwen3-1.7B",
            "4": "Qwen3-4B",
            "8": "Qwen3-8B",
            "14": "Qwen3-14B",
            "32": "Qwen3-32B",
        }
        if args.size not in size_to_model:
            raise ValueError(
                f"Size {args.size} not available for Qwen models. Available sizes: {list(size_to_model.keys())}"
            )
        model_name = size_to_model[args.size]
        model_base = os.path.join(cache_dir, f"models--Qwen--{model_name}")
        default_draft_base = os.path.join(
            cache_dir, "models--Qwen--Qwen3-0.6B")

    model_path = _get_snapshot_path(model_base)

    # Always resolve a draft path so callers can pass it through unchanged
    if getattr(args, "eagle", False) or getattr(args, "draft", None):
         draft_base = _get_draft_model_path(args, cache_dir, run_config=run_config)
    else:
         draft_base = default_draft_base
         
    draft_path = _get_snapshot_path(draft_base)

    return model_name, model_path, draft_path






