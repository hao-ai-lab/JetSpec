from importlib import import_module


_EXPORTS = {
    "LLM": ("jetspec.engines.jetspec_plus_plus.serving.llm", "LLM"),
    "SamplingParams": ("jetspec.engines.jetspec_plus_plus.serving.sampling_params", "SamplingParams"),
    "Sequence": ("jetspec.engines.jetspec_plus_plus.serving.sequence", "Sequence"),
    "SequenceStatus": ("jetspec.engines.jetspec_plus_plus.serving.sequence", "SequenceStatus"),
    "ModelRunner": ("jetspec.engines.jetspec_plus_plus.serving.model_runner", "ModelRunner"),
    "Config": ("jetspec.engines.jetspec_plus_plus.serving.config", "Config"),
    "infer_model_family": ("jetspec.engines.jetspec_plus_plus.serving.utils.misc", "infer_model_family"),
    "Attention": ("jetspec.engines.jetspec_plus_plus.models.layers.attention", "Attention"),
    "set_context": ("jetspec.engines.jetspec_plus_plus.serving.utils.context", "set_context"),
    "get_context": ("jetspec.engines.jetspec_plus_plus.serving.utils.context", "get_context"),
    "reset_context": ("jetspec.engines.jetspec_plus_plus.serving.utils.context", "reset_context"),
    "verify": ("jetspec.engines.jetspec_plus_plus.serving.utils.verify", "verify"),
    "make_glue_decode_input_ids": ("jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.async_spec_helpers", "make_glue_decode_input_ids"),
    "get_forked_recovery_tokens_from_logits": (
        "jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.async_spec_helpers",
        "get_forked_recovery_tokens_from_logits",
    ),
    "apply_sampler_x_rescaling": ("jetspec.engines.jetspec_plus_plus.serving.utils.async_helpers.async_spec_helpers", "apply_sampler_x_rescaling"),
    "get_custom_mask": ("jetspec.engines.jetspec_plus_plus.verify.helpers.mask_helpers", "get_custom_mask"),
    "run_verify_cudagraph": ("jetspec.engines.jetspec_plus_plus.serving.helpers.cudagraph_helpers", "run_verify_cudagraph"),
    "run_decode_cudagraph": ("jetspec.engines.jetspec_plus_plus.serving.helpers.cudagraph_helpers", "run_decode_cudagraph"),
    "capture_cudagraph": ("jetspec.engines.jetspec_plus_plus.serving.helpers.cudagraph_helpers", "capture_cudagraph"),
    "capture_verify_cudagraph": ("jetspec.engines.jetspec_plus_plus.serving.helpers.cudagraph_helpers", "capture_verify_cudagraph"),
    "run_fi_tree_decode_cudagraph": ("jetspec.engines.jetspec_plus_plus.serving.helpers.cudagraph_helpers", "run_fi_tree_decode_cudagraph"),
    "prepare_decode_tensors_from_seqs": (
        "jetspec.engines.jetspec_plus_plus.serving.helpers.runner_helpers",
        "prepare_decode_tensors_from_seqs",
    ),
    "prepare_block_tables_from_seqs": (
        "jetspec.engines.jetspec_plus_plus.serving.helpers.runner_helpers",
        "prepare_block_tables_from_seqs",
    ),
    "prepare_prefill_tensors_from_seqs": (
        "jetspec.engines.jetspec_plus_plus.serving.helpers.runner_helpers",
        "prepare_prefill_tensors_from_seqs",
    ),
    "prepare_prefill_payload": ("jetspec.engines.jetspec_plus_plus.serving.helpers.runner_helpers", "prepare_prefill_payload"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
