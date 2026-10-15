import os
import json
from dataclasses import dataclass
from types import SimpleNamespace
from transformers import AutoConfig
import torch
from jetspec.engines.jetspec_plus_plus.serving.paths import DEFAULT_TARGET
from jetspec.engines.jetspec_plus_plus.serving.paths import DEFAULT_DRAFT


@dataclass
class Config:
    @staticmethod
    def _target_config(model: str):
        with open(os.path.join(model, "config.json")) as config_file:
            raw = json.load(config_file)
        if raw.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
            return AutoConfig.from_pretrained(model)
        text = raw.get("text_config", raw)
        if text.get("model_type") != "qwen3_5_text":
            raise ValueError("hybrid target needs a qwen3_5_text text_config")
        layers = text.get("layer_types")
        if (not isinstance(layers, list)
                or len(layers) != text.get("num_hidden_layers")
                or any(layer not in {"linear_attention", "full_attention"} for layer in layers)):
            raise ValueError("hybrid target layer_types must cover every text layer")
        if text.get("num_attention_heads", 0) % text.get("num_key_value_heads", 1):
            raise ValueError("hybrid target attention heads must divide KV heads")
        text = dict(text)
        text["rope_theta"] = text.get("rope_theta", text.get("rope_parameters", {}).get("rope_theta"))
        dtype = text.get("dtype", "bfloat16")
        if dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"unsupported hybrid target dtype: {dtype}")
        text["torch_dtype"] = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        return SimpleNamespace(**text)

    model: str = DEFAULT_TARGET
    run_config: object = None
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    speculate_k: int = 1
    draft_async: bool = False
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # JetSpec target hidden-state payload
    jetspec_taps: bool = False
    jetspec_tap_hidden_size: int | None = None
    jetspec_taps_dump: str | None = None
    jetspec_target_layer_ids: tuple[int, ...] | None = None
    jetspec_config_is_draft: bool = False  # set only on DraftRunner.create_draft_config's copy

    # JetSpec protocol (separate draft process; kth block has 1 guess + 14 drafts)
    jetspec: bool = False
    jetspec_sync: bool = False
    jetspec_hyp: int = 16
    jetspec_passes: int = 3
    jetspec_miss: str = "ar"
    jetspec_verify_backend: str = "flashinfer_paged"
    jetspec_draft_graphs: bool = True
    jetspec_wait_ms: float = float("inf")
    jetspec_checkpoint_dir: str | None = None
    jetspec_adapter_config: str | None = None
    jetspec_adapter_head: str | None = None
    jetspec_target_release_weights: str | None = None
    target_dtype: str | None = None

    # Debugging
    verbose: bool = False 
    debug_mode: bool = False 
    max_steps: int | None = None
    run_config: object = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    def __post_init__(self):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        if self.jetspec:
            from jetspec.engines.jetspec_plus_plus.jetspec.live_rows import require_segmented_rows
            require_segmented_rows(self.run_config)
        if self.model is None or self.draft is None:
            from jetspec.engines.jetspec_plus_plus.serving.paths import resolve_paths
            paths = resolve_paths(self.run_config)
            if self.model is None:
                self.model = paths.DEFAULT_TARGET
            if self.draft is None:
                self.draft = paths.DEFAULT_DRAFT
        if setting(self, 'method.method_settings.tree_b', '0') != "0":
            raise ValueError('retired tree verify requires a sealed historical read')
        model = self.model 
        assert os.path.isdir(model)

        if self.jetspec_sync:
            assert self.jetspec, "JetSpec sync proof mode requires JetSpec"
        if self.jetspec:
            from jetspec.engines.jetspec_plus_plus.wire.protocol import normalize_wait_ms

            self.jetspec_wait_ms = normalize_wait_ms(self.jetspec_wait_ms)
            assert self.speculate and self.draft_async, (
                "JetSpec requires speculative decoding with the separate draft process"
            )
            from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
            assert self.speculate_k == draft_width(run_config=getattr(self, 'run_config', None)), "JetSpec verifies exactly JETSPEC_DRAFTS (14) drafts"
            assert self.jetspec_taps, "JetSpec requires target tap transport"
            assert 1 <= self.jetspec_hyp <= 16
            assert self.jetspec_passes >= 1
            assert self.jetspec_miss in {"ar", "verify"}
            if self.jetspec_verify_backend not in {"flashinfer_paged", "sdpa"}:
                raise ValueError(
                    "jetspec_verify_backend must be flashinfer_paged or sdpa")
            for name in (
                "jetspec_checkpoint_dir",
                "jetspec_adapter_config",
                "jetspec_adapter_head",
                "jetspec_target_release_weights",
            ):
                path = getattr(self, name)
                assert path and os.path.exists(path), f"JetSpec path missing: {name}={path}"

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.hf_config = self._target_config(model)
        # The draft runner's copy (DraftRunner.create_draft_config) swaps `model` for the draft model and
        # carries the target's already-validated taps; the target-only checks below must not re-run on it.
        if self.jetspec_taps and self.jetspec_checkpoint_dir and not self.jetspec_config_is_draft:
            source = os.path.join(self.jetspec_checkpoint_dir, "config.json")
            with open(source) as config_file:
                declared = json.load(config_file).get("dflash_config", {}).get("target_layer_ids")
            text_config = getattr(self.hf_config, "text_config", self.hf_config)
            layers = text_config.num_hidden_layers
            if (not isinstance(declared, list) or not declared
                    or any(type(layer) is not int or layer < 0 or layer >= layers for layer in declared)
                    or any(left >= right for left, right in zip(declared, declared[1:]))):
                raise ValueError(f"{source}: target_layer_ids must be strictly increasing integers "
                                 f"below target text_config.num_hidden_layers={layers}")
            self.jetspec_target_layer_ids = tuple(declared)
        if self.jetspec and self.jetspec_adapter_config and not self.jetspec_config_is_draft:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
            from jetspec.engines.jetspec_plus_plus.jetspec.drafter import load_qwen3_config

            draft_rotary = Qwen3RotaryEmbedding(load_qwen3_config(self.jetspec_checkpoint_dir))
            adapter_rotary = Qwen3RotaryEmbedding(load_qwen3_config(self.jetspec_adapter_config))
            with open(os.path.join(model, "config.json")) as target_file:
                target = json.load(target_file)
            text_config = target.get("text_config", target)
            rope_parameters = text_config.get("rope_parameters") or {}
            theta = text_config.get("rope_theta", rope_parameters.get("rope_theta"))
            if not isinstance(theta, (int, float)) or isinstance(theta, bool) or theta <= 0:
                raise ValueError(f"{model}/config.json: target text_config.rope_theta is absent or invalid")
            width = draft_rotary.inv_freq.numel() * 2
            expected = 1.0 / (float(theta) ** (torch.arange(0, width, 2).float() / width))
            if not (torch.equal(draft_rotary.inv_freq, expected)
                    and torch.equal(adapter_rotary.inv_freq, expected)):
                raise ValueError("JetSpec drafter/adapter inv_freq disagrees with target text config")
        if self.target_dtype is not None:
            dtype_by_name = {
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            if self.target_dtype not in dtype_by_name:
                raise ValueError("target_dtype must be bf16 or fp32")
            self.hf_config.torch_dtype = dtype_by_name[self.target_dtype]
            print(
                f"W217_TARGET_DTYPE={self.target_dtype} "
                f"torch_dtype={self.hf_config.torch_dtype}",
                flush=True,
            )
        if self.jetspec_taps and self.jetspec_tap_hidden_size is None:
            self.jetspec_tap_hidden_size = self.hf_config.hidden_size
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings) 
        if self.speculate: 
            draft = self.draft
            self.draft_hf_config = (self._target_config(draft)
                                    if self.hf_config.model_type == "qwen3_5_text"
                                    else AutoConfig.from_pretrained(draft))
            self.max_model_len = min(
                self.max_model_len, self.draft_hf_config.max_position_embeddings)
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
                
        if self.use_eagle:
            if self.eagle_layers is None:
                L = self.hf_config.num_hidden_layers
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos
        
        assert self.max_num_batched_tokens >= self.max_model_len
