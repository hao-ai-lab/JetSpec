"""NanoEngine — single-stream AR decode over a paged KV cache (nano_vllm N0).

The owned-substrate analogue of `ptd.engine.llm.LLM.generate()`: plain
greedy/temperature decode (prefill the prompt once, then single-token steps
reusing the cache), with the HF `DynamicCache` swapped for `PagedKVCache`. The
paging is invisible to the model forward (`ModelRunner` is reused as-is), so the
loop is the same — this is the N0 lossless gate: token-identical to `LLM.generate`
on the same model/device/dtype (fp32 exact on CPU; bf16 on b200, modulo the
documented SDPA-reduction-order borderline flips).

N0 is vanilla AR (no tree). N1 layers tree spec on the same paged cache, using
`PagedKVCache.gather` for the accepted-path KV (the paged `_select_kv_cache`).
"""
import torch

from ptd.models.qwen3 import load_target
from ptd.engine.llm import SamplingParams
from ptd.engine.model_runner import ModelRunner
from ptd.engine.sampler import sample
from ptd.nano_vllm.paged_kv_cache import PagedKVCache


class NanoEngine:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        block_size: int = 16,
        attn_implementation: str = "sdpa",
    ):
        self.model, self.tokenizer = load_target(
            model_name_or_path, device, dtype, attn_implementation
        )
        self.runner = ModelRunner(self.model)
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.eos_token_ids = self._resolve_eos()

    def _resolve_eos(self) -> set:
        """EOS ids from tokenizer + generation_config (mirrors `LLM._resolve_eos`)."""
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        for src in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(gc, "eos_token_id", None) if gc is not None else None,
        ):
            if src is None:
                continue
            if isinstance(src, (list, tuple, set)):
                ids.update(int(x) for x in src)
            else:
                ids.add(int(src))
        return ids

    @torch.inference_mode()
    def generate(self, prompt, sampling_params: SamplingParams = None) -> dict:
        """Greedy/temperature decode over the paged cache. `prompt` is a str
        (tokenized raw) or an already-tokenized `input_ids` tensor (1, T). Returns
        {token_ids, text}. Token-identical to `LLM.generate` (the N0 gate)."""
        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = prompt.to(self.device)
        prompt_len = input_ids.shape[1]
        cache = PagedKVCache(
            block_size=self.block_size, device=torch.device(self.device), dtype=self.dtype
        )

        # --- prefill: process the whole prompt once, sample the first token ---
        pos = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        logits, cache, _ = self.runner.forward(input_ids, cache, pos)
        next_tok = sample(logits[:, -1:, :], sp.temperature)  # (1, 1)
        out_ids = [int(next_tok.item())]
        cur = prompt_len

        # --- decode: single-token steps reusing the paged cache (no reprocess) ---
        for _ in range(sp.max_new_tokens - 1):
            if out_ids[-1] in self.eos_token_ids:
                break
            pos = torch.tensor([[cur]], device=self.device)
            logits, cache, _ = self.runner.forward(next_tok, cache, pos)
            next_tok = sample(logits[:, -1:, :], sp.temperature)
            out_ids.append(int(next_tok.item()))
            cur += 1

        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return {"token_ids": out_ids, "text": text}
