"""Fail-closed guards around the upstream bench helpers, which this tree does not edit.

``bench_helpers.load_dataset_token_ids`` returns None on any dataset error (unknown name,
missing file, a tokenizer or json failure) after printing "falling back to random tokens",
and ``generate_benchmark_inputs`` then substitutes random token ids.  A run that asked for a
dataset would be measured on noise and still print a throughput.
"""

# Locate the component from either its category or its stable forwarding path.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_TOOL_ROOT = next((p for p in _ToolPath(__file__).resolve().parents
                   if (p / "artifacts/manifest.json").is_file() or (p / "serve/README.md").is_file()), _ToolPath.cwd())
_tool_sys.path[:0] = [str(_TOOL_ROOT), str(_TOOL_ROOT / 'serve'), str(_ToolPath(__file__).resolve().parent)]
if __name__ == "__main__" and any(a in ("-h", "--help") for a in _tool_sys.argv[1:]):
    from jetspec.engines.jetspec_plus_plus.serving.tools.lib.entrypoints import cpu_help as _cpu_help
    _cpu_help(__file__)



import json
from pathlib import Path

from transformers import AutoTokenizer


class DatasetFallbackError(RuntimeError):
    """Dataset prompts were requested and the loader fell back to random tokens."""


def generate_inputs_fail_closed(args, model_path, helpers):
    """``helpers.generate_benchmark_inputs(args, model_path)``, refusing the silent random fallback.

    ``helpers`` is the ``bench_helpers`` module.  Its loader is observed, not replaced: every
    value it returns is handed on untouched, so a load that succeeds yields the same prompts
    as before.  ``--random`` is the explicit request for random tokens and never reaches the
    loader.
    """
    failed = []
    loader = helpers.load_dataset_token_ids

    def observed(dataset_name, *positional, **keywords):
        prompts = loader(dataset_name, *positional, **keywords)
        if prompts is None:
            failed.append(dataset_name)
        return prompts

    helpers.load_dataset_token_ids = observed
    try:
        inputs = helpers.generate_benchmark_inputs(args, model_path)
    finally:
        helpers.load_dataset_token_ids = loader
    if failed and not getattr(args, "random", False):
        raise DatasetFallbackError(
            f"dataset prompts failed to load ({', '.join(failed)}; the loader's warning above says why) "
            "and bench_helpers substituted random tokens. Fix the dataset (SSD_DATASET_DIR), pass "
            "--prompts-json, or pass --random to ask for random tokens explicitly.")
    return inputs


def exported_prompt_ids(rows, model_path):
    """Use the target vocabulary for exported text when serving a hybrid target."""
    config = json.loads((Path(model_path) / "config.json").read_text())
    if config.get("model_type") in ("qwen3_5", "qwen3_5_text"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return [tokenizer.encode(row["text"], add_special_tokens=False) for row in rows]
    return [row["token_ids"] for row in rows]
