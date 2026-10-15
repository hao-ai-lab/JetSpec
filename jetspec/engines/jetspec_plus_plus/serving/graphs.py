import os
from jetspec.engines.jetspec_plus_plus._optional import version
import torch
import numpy as np
from typing import List
from jetspec.engines.jetspec_plus_plus.models.context import set_context
from jetspec.engines.jetspec_plus_plus.models.context import get_context
from jetspec.engines.jetspec_plus_plus.models.context import reset_context
from jetspec.engines.jetspec_plus_plus.serving.mask_helpers import get_custom_mask
from jetspec.engines.jetspec_plus_plus.serving.mask_helpers import get_verify_chain_mask
from jetspec.engines.jetspec_plus_plus.serving.mask_helpers import get_verify_chain_mask_cached
from jetspec.engines.jetspec_plus_plus.models.attention import ATTENTION_BACKEND
from time import perf_counter


# The upstream 0.5.2 FA2 planner lacks the two newer trailing options.
_FA2_PLAN_OPTIONS = (-1, False) if version("flashinfer-python") == "0.5.2" else (-1, False, 0, 0)


## RUN CUDAGRAPHS
@torch.inference_mode()
def run_verify_cudagraph(model_runner, input_ids, positions, last_only, graph_vars):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    from jetspec.engines.jetspec_plus_plus.verify.verify_plan_helpers import verify_plan_signature
    context = get_context()
    k_plus_1 = model_runner.config.speculate_k + 1
    orig_bs = input_ids.size(0) // k_plus_1  # orig_bs = N here

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["verify"] if x >= orig_bs)
    graph = model_runner.graphs["verify"][wrapper_bs]

    if (orig_bs > 1 and model_runner.jetspec_paged_flashinfer
            and model_runner._staged_verify is not None):
        graph_vars["input_ids"][:orig_bs * k_plus_1].copy_(input_ids)
        timing = getattr(model_runner, '_anatomy_lite', None)
        if timing is not None:
            timing.gpu_begin()
        graph.replay()
        outputs = graph_vars["outputs"][:orig_bs * k_plus_1]
        logits = model_runner.model.compute_logits(outputs, last_only)
        if timing is not None:
            timing.gpu_end()
        check = getattr(model_runner, "_verify_masked_graphs", {}).get(wrapper_bs)
        if check is not None and wrapper_bs not in model_runner._verify_masked_checked:
            masked_graph, masked_outputs, masked_wrapper = check
            key = (wrapper_bs, wrapper_bs * k_plus_1)
            causal_wrapper = model_runner.flashinfer_graph_wrappers[key]
            model_runner.flashinfer_graph_wrappers[key] = masked_wrapper
            try:
                mask = get_verify_chain_mask(
                    graph_vars["context_lens"][:wrapper_bs], k_plus_1)
                model_runner.plan_flashinfer_paged_attention(
                    graph_vars["context_lens"][:wrapper_bs],
                    graph_vars["block_tables"][:wrapper_bs],
                    qo_indptr=graph_vars["cu_seqlens_q"][:wrapper_bs + 1],
                    graph_batch_size=wrapper_bs, custom_mask=mask,
                    static_q_len=wrapper_bs * k_plus_1)
            finally:
                model_runner.flashinfer_graph_wrappers[key] = causal_wrapper
            masked_graph.replay()
            masked_logits = model_runner.model.compute_logits(
                masked_outputs[:orig_bs * k_plus_1], last_only)
            a, b = logits.detach(), masked_logits.detach()
            bits = 16 if a.dtype in (torch.bfloat16, torch.float16) else 32
            integer = torch.int16 if bits == 16 else torch.int32
            mask = (1 << bits) - 1
            sign = 1 << (bits - 1)
            left = a.contiguous().view(integer).to(torch.int64) & mask
            right = b.contiguous().view(integer).to(torch.int64) & mask
            left = torch.where((left & sign) != 0, mask - left, left + sign)
            right = torch.where((right & sign) != 0, mask - right, right + sign)
            delta = (a.float() - b.float()).abs()
            print("W630_ATTN_CHECK " + __import__("json").dumps(dict(
                batch=orig_bs, bucket=wrapper_bs, rows=orig_bs * k_plus_1,
                bitwise=bool(torch.equal(a, b)),
                max_abs=float(delta.max().item()),
                max_ulp=int((left - right).abs().max().item()),
                argmax_agree=int((a.argmax(-1) == b.argmax(-1)).sum().item()),
                argmax_total=orig_bs * k_plus_1,
            )), flush=True)
            model_runner._verify_masked_checked.add(wrapper_bs)
        if "jetspec_taps" in graph_vars:
            return logits, graph_vars["jetspec_taps"][:orig_bs * k_plus_1]
        return logits

    for k, v in graph_vars.items():
        if k != "outputs":
            v.zero_()

    # Pad to graph bucket size if needed (fixes B>=6 crash from non-monotonic cu_seqlens_q)
    if wrapper_bs > orig_bs:
        pad_bs = wrapper_bs - orig_bs
        pad_flat = pad_bs * k_plus_1
        dev = input_ids.device

        input_ids = torch.cat([input_ids, torch.zeros(pad_flat, dtype=input_ids.dtype, device=dev)])
        positions = torch.cat([positions, torch.zeros(pad_flat, dtype=positions.dtype, device=dev)])
        slot_mapping = torch.cat([
            context.slot_mapping,
            torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=dev)])
        # Repeat last real row for ghost sequences (valid page table / context len)
        bt = context.block_tables
        cl = context.context_lens
        block_tables = torch.cat([bt, bt[orig_bs-1:orig_bs].expand(pad_bs, -1).contiguous()])
        context_lens = torch.cat([cl, cl[orig_bs-1:orig_bs].expand(pad_bs).contiguous()])
        bs = wrapper_bs
    else:
        slot_mapping = context.slot_mapping
        block_tables = context.block_tables
        context_lens = context.context_lens
        bs = orig_bs

    graph_vars["input_ids"][:bs * k_plus_1] = input_ids
    graph_vars["positions"][:bs * k_plus_1] = positions
    graph_vars["slot_mapping"][:bs * k_plus_1] = slot_mapping
    graph_vars["context_lens"][:bs] = context_lens
    # Construct cu_seqlens_q for FULL padded batch (monotonically increasing)
    seqlen_q = torch.full(
        (bs,), k_plus_1, dtype=torch.int32, device=graph_vars["cu_seqlens_q"].device)
    cu = graph_vars["cu_seqlens_q"][:bs + 1]
    cu.zero_()
    cu[1:].copy_(torch.cumsum(seqlen_q, 0))

    if block_tables is not None:
        graph_vars["block_tables"][:bs, :block_tables.size(1)] = block_tables

    timing = getattr(model_runner, '_anatomy_lite', None)
    if timing is not None:
        timing.stamp('plan_start')
    if ATTENTION_BACKEND == "flashinfer" or model_runner.jetspec_paged_flashinfer:
        signature = None
        if model_runner.verify_plan_cache:
            signature = verify_plan_signature(
                getattr(model_runner, "_verify_context_lens_cpu", None),
                getattr(model_runner, "_verify_block_tables_cpu", None),
                wrapper_bs, k_plus_1)
        reuse_plan = signature is not None and model_runner._verify_plan_cache.get(wrapper_bs) == signature
        if reuse_plan:
            model_runner._verify_plan_cache_hits += 1
        else:
            verify_mask = None
            causal_batch = model_runner.jetspec_paged_flashinfer and (
                orig_bs > 1 or current_engine().kind == 'reference')
            if model_runner.jetspec_paged_flashinfer and not causal_batch:
                if model_runner.verify_mask_cache:
                    verify_mask = get_verify_chain_mask_cached(
                        model_runner, graph_vars["context_lens"][:bs], k_plus_1)
                else:
                    verify_mask = get_verify_chain_mask(
                        graph_vars["context_lens"][:bs], k_plus_1)
            model_runner.plan_flashinfer_paged_attention(
                graph_vars["context_lens"][:bs],
                graph_vars["block_tables"][:bs],
                qo_indptr=cu,
                graph_batch_size=wrapper_bs,
                custom_mask=verify_mask,
                causal_only=causal_batch,
            )
            if signature is not None:
                model_runner._verify_plan_cache[wrapper_bs] = signature
                model_runner._verify_plan_cache_misses += 1

    _pt = setting(model_runner, 'engine.engine_settings.profile_target', '0') == "1"
    if _pt:
        torch.cuda.synchronize()
        _t0 = perf_counter()

    timing = getattr(model_runner, '_anatomy_lite', None)
    if timing is not None:
        timing.gpu_begin()
    graph.replay()

    if _pt:
        torch.cuda.synchronize()
        _t1 = perf_counter()

    # Extract outputs for the ORIGINAL batch size only
    outputs = graph_vars["outputs"][:orig_bs * k_plus_1]
    logits = model_runner.model.compute_logits(outputs, last_only)
    if timing is not None:
        timing.gpu_end()

    if _pt:
        torch.cuda.synchronize()
        _t2 = perf_counter()
        has_eagle = "eagle_acts" in graph_vars
        print(f"[PROFILE verify_cg] replay={(_t1-_t0)*1000:.2f}ms logits={(_t2-_t1)*1000:.2f}ms eagle={has_eagle} bs={orig_bs} rank={model_runner.rank}", flush=True)

    # For eagle target, also return eagle_acts
    if "eagle_acts" in graph_vars:
        eagle_acts = graph_vars["eagle_acts"][:orig_bs * k_plus_1]
        return logits, eagle_acts
    if "jetspec_taps" in graph_vars:
        jetspec_taps = graph_vars["jetspec_taps"][:orig_bs * k_plus_1]
        return logits, jetspec_taps
    return logits


@torch.inference_mode()
def run_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, hidden_states=None):
    context = get_context()

    flat_batch_size = input_ids.size(0)
    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["decode"] if x >= flat_batch_size)
    graph = model_runner.graphs["decode"][wrapper_bs]

    for k, v in graph_vars.items():
            if k != "outputs":
                v.zero_()

    graph_vars["input_ids"][:flat_batch_size] = input_ids
    graph_vars["positions"][:flat_batch_size] = positions
    graph_vars["slot_mapping"][:flat_batch_size] = context.slot_mapping
    graph_vars["context_lens"][:flat_batch_size] = context.context_lens
    if wrapper_bs > flat_batch_size:
        graph_vars["slot_mapping"][flat_batch_size:wrapper_bs] = -1
        graph_vars["context_lens"][flat_batch_size:wrapper_bs] = (
            context.context_lens[flat_batch_size - 1])

    if hidden_states is not None and "hidden_states" in graph_vars:
        graph_vars["hidden_states"][:flat_batch_size] = hidden_states

    if context.block_tables is not None:
        graph_vars["block_tables"][:flat_batch_size,
                                :context.block_tables.size(1)] = context.block_tables
        if wrapper_bs > flat_batch_size:
            graph_vars["block_tables"][flat_batch_size:wrapper_bs,
                                      :context.block_tables.size(1)] = (
                context.block_tables[flat_batch_size - 1])

    if ATTENTION_BACKEND == "flashinfer" or model_runner.jetspec_paged_flashinfer:
        qo_indptr = torch.arange(
            wrapper_bs + 1, dtype=torch.int32, device=input_ids.device)
        model_runner.plan_flashinfer_paged_attention(
            graph_vars["context_lens"][:wrapper_bs],
            graph_vars["block_tables"][:wrapper_bs],
            qo_indptr=qo_indptr,
            graph_batch_size=wrapper_bs,
        )

    timing = getattr(model_runner, '_anatomy_lite', None)
    if timing is not None:
        timing.gpu_begin()
    graph.replay()

    outputs = graph_vars["outputs"][:flat_batch_size]
    logits = model_runner.model.compute_logits(outputs, last_only)
    if timing is not None:
        timing.gpu_end()
    # EAGLE draft: outputs is prenorm, return both
    if "hidden_states" in graph_vars:
        return logits, outputs
    if "jetspec_taps" in graph_vars:
        return logits, graph_vars["jetspec_taps"][:flat_batch_size]
    return logits


cache = {}

_plan_event = None  # Lazy-init CUDA event for plan() sync
def profile_enabled(run_config=None):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    return setting(run_config, 'engine.engine_settings.profile', '0') == '1'


def draft_profile_enabled(run_config=None):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    return setting(run_config, 'engine.engine_settings.profile_draft', '0') == '1'


_draft_events = []  # [(step, label, start_event, end_event), ...]

def flush_draft_profile():
    """Sync once, read all CUDA events, print per-step breakdown, clear list."""
    if not _draft_events:
        return
    torch.cuda.synchronize()
    by_step = {}
    for step, label, ev0, ev1 in _draft_events:
        by_step.setdefault(step, []).append((label, ev0.elapsed_time(ev1)))
    parts = []
    total = 0.0
    for step in sorted(by_step):
        step_total = sum(t for _, t in by_step[step])
        detail = " ".join(f"{l}={t:.2f}" for l, t in by_step[step])
        parts.append(f"s{step}={step_total:.2f}({detail})")
        total += step_total
    print(f"[PROFILE draft_detail] K={len(by_step)} total={total:.2f}ms avg_step={total/len(by_step):.2f}ms | {' '.join(parts)}", flush=True)
    _draft_events.clear()



## CAPTURE CUDAGRAPHS
@torch.inference_mode()
def capture_cudagraph(model_runner):
    config = model_runner.config
    hf_config = config.hf_config
    max_seqs = min(model_runner.config.max_num_seqs, 512)
    if getattr(model_runner, "hybrid_target", False):
        max_bs = 1
    elif model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft:
        max_bs = max_seqs  # JIT recovery is single-token B; tree decode has its own graphs.
    else:
        max_bs = max_seqs + 1
    max_num_blocks = (config.max_model_len +
                      model_runner.block_size - 1) // model_runner.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    graph_bs_list = [bs for bs in [1, 2, 4, 8] if bs <= max_bs]
    if max_bs not in graph_bs_list:
        graph_bs_list.append(max_bs)

    graphs = {}
    graph_pool = None

    is_jit = (model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft)

    # Eagle models need special handling during CUDA graph capture
    is_eagle_draft = config.use_eagle and model_runner.is_draft
    is_eagle_target = config.use_eagle and not model_runner.is_draft
    is_jetspec_target = config.jetspec_taps and not model_runner.is_draft
    hidden_states = None
    jetspec_taps = None
    if is_eagle_draft:
        # Use hidden_size (d_model_draft) so CG captures the pass-through branch in Eagle3DraftForCausalLM.forward()
        # All callers project target acts via fc() BEFORE passing to CG
        hidden_states = torch.zeros(max_bs, hf_config.hidden_size,
                                    dtype=hf_config.torch_dtype, device=input_ids.device)
    if is_jetspec_target:
        jetspec_taps = torch.zeros(
            max_bs, 5, hf_config.hidden_size, dtype=hf_config.torch_dtype,
            device=input_ids.device,
        )

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        if ATTENTION_BACKEND == "flashinfer" or model_runner.jetspec_paged_flashinfer:
            context_lens[:bs].fill_(1)
        set_context(
            False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], is_jit=is_jit)
        if ATTENTION_BACKEND == "flashinfer" or model_runner.jetspec_paged_flashinfer:
            qo_indptr = torch.arange(
                bs + 1, dtype=torch.int32, device=input_ids.device)
            model_runner.plan_flashinfer_paged_attention(
                context_lens[:bs],
                block_tables[:bs],
                qo_indptr=qo_indptr,
                graph_batch_size=bs,
            )
        if is_eagle_draft:
            outputs[:bs] = model_runner.model(
                input_ids[:bs], positions[:bs], hidden_states[:bs])    # warmup
        elif is_eagle_target:
            out, _ = model_runner.model(
                input_ids[:bs], positions[:bs])    # warmup
            outputs[:bs] = out
        elif is_jetspec_target:
            out, taps = model_runner.model(
                input_ids[:bs], positions[:bs])
            outputs[:bs] = out
            jetspec_taps[:bs] = taps
        else:
            outputs[:bs] = model_runner.model(
                input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, graph_pool):
            if is_eagle_draft:
                outputs[:bs] = model_runner.model(
                    input_ids[:bs], positions[:bs], hidden_states[:bs])    # capture
            elif is_eagle_target:
                out, _ = model_runner.model(
                    input_ids[:bs], positions[:bs])    # capture
                outputs[:bs] = out
            elif is_jetspec_target:
                out, taps = model_runner.model(
                    input_ids[:bs], positions[:bs])
                outputs[:bs] = out
                jetspec_taps[:bs] = taps
            else:
                outputs[:bs] = model_runner.model(
                    input_ids[:bs], positions[:bs])    # capture
        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
    if hidden_states is not None:
        graph_vars["hidden_states"] = hidden_states
    if jetspec_taps is not None:
        graph_vars["jetspec_taps"] = jetspec_taps

    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def capture_verify_cudagraph(model_runner):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    config = model_runner.config
    # assert not model_runner.is_draft, "ERROR in capture_verify_cudagraph: verify path only supported for target model"
    hf_config = config.hf_config
    max_bs = min(model_runner.config.max_num_seqs, 512)
    k_plus_1 = model_runner.config.speculate_k + 1

    is_eagle_target = config.use_eagle and not model_runner.is_draft
    is_jetspec_target = config.jetspec_taps and not model_runner.is_draft

    # For verify, we need to handle k+1 tokens per sequence, and use cu_seqlens_q and max_seqlen_q
    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(
        max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs * k_plus_1, hf_config.hidden_size)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)

    # Eagle target: also capture eagle_acts from model forward
    eagle_acts = None
    jetspec_taps = None
    if is_eagle_target:
        # eagle_acts has shape [num_tokens, 3 * hidden_size] for 3 layers
        eagle_acts = torch.zeros(max_bs * k_plus_1, 3 * hf_config.hidden_size,
                                  dtype=hf_config.torch_dtype)
    if is_jetspec_target:
        jetspec_taps = torch.zeros(
            max_bs * k_plus_1, 5, hf_config.hidden_size,
            dtype=hf_config.torch_dtype,
        )

    base = [1, 2, 4, 8]
    dynamic = list(range(16, max_bs+1, 16))
    all_b = base + dynamic
    if max_bs not in all_b:
        all_b.append(max_bs)
    all_b.sort()
    all_N = [b for b in all_b if b <= max_bs]

    graphs = {}
    graph_pool = None

    for bs in reversed(all_N):
        graph = torch.cuda.CUDAGraph()
        # For verify, each sequence is length K+1, so seqlen_q is [K+1]*bs
        seqlen_q = torch.full((bs,), k_plus_1, dtype=torch.int32)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))
        context_lens[:bs] = seqlen_q

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:bs * k_plus_1],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            cu_seqlens_q=cu,
            max_seqlen_q=k_plus_1,
        )

        causal_batch = False
        if ATTENTION_BACKEND == "flashinfer" or model_runner.jetspec_paged_flashinfer:
            verify_mask = None
            causal_batch = model_runner.jetspec_paged_flashinfer and (
                bs > 1 or current_engine().kind == 'reference')
            if model_runner.jetspec_paged_flashinfer and not causal_batch:
                verify_mask = get_verify_chain_mask(
                    context_lens[:bs], k_plus_1)
            if causal_batch:
                causal = model_runner.get_flashinfer_graph_wrapper(
                    bs, bs * k_plus_1, causal_only=True)
                model_runner.flashinfer_graph_wrappers[bs, bs * k_plus_1] = causal
            model_runner.plan_flashinfer_paged_attention(
                context_lens[:bs],
                block_tables[:bs],
                qo_indptr=cu,
                graph_batch_size=bs,
                custom_mask=verify_mask,
                causal_only=causal_batch,
            )

        # warmup
        model_out = model_runner.model(
            input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
        if isinstance(model_out, tuple):
            outputs[:bs * k_plus_1] = model_out[0]
            if eagle_acts is not None:
                eagle_acts[:bs * k_plus_1] = model_out[1]
            elif jetspec_taps is not None:
                jetspec_taps[:bs * k_plus_1] = model_out[1]
        else:
            outputs[:bs * k_plus_1] = model_out
        with torch.cuda.graph(graph, graph_pool):
            # capture
            model_out = model_runner.model(
                input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
            if isinstance(model_out, tuple):
                outputs[:bs * k_plus_1] = model_out[0]
                if eagle_acts is not None:
                    eagle_acts[:bs * k_plus_1] = model_out[1]
                elif jetspec_taps is not None:
                    jetspec_taps[:bs * k_plus_1] = model_out[1]
            else:
                outputs[:bs * k_plus_1] = model_out

        if (causal_batch and bs in (4, 16)
                and setting(model_runner, 'engine.engine_settings.verify_causal_check', '0') == "1"):
            key = (bs, bs * k_plus_1)
            causal = model_runner.flashinfer_graph_wrappers.pop(key)
            try:
                masked_wrapper = model_runner.get_flashinfer_graph_wrapper(bs, bs * k_plus_1)
                mask = get_verify_chain_mask(context_lens[:bs], k_plus_1)
                model_runner.plan_flashinfer_paged_attention(
                    context_lens[:bs], block_tables[:bs], qo_indptr=cu,
                    graph_batch_size=bs, custom_mask=mask)
                masked_outputs = torch.empty_like(outputs[:bs * k_plus_1])
                model_runner.model(input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
                masked_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(masked_graph, graph.pool()):
                    masked_result = model_runner.model(
                        input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
                    masked_outputs.copy_(masked_result[0] if isinstance(masked_result, tuple)
                                         else masked_result)
                if not hasattr(model_runner, "_verify_masked_graphs"):
                    model_runner._verify_masked_graphs = {}
                    model_runner._verify_masked_checked = set()
                model_runner._verify_masked_graphs[bs] = (
                    masked_graph, masked_outputs, masked_wrapper)
            finally:
                model_runner.flashinfer_graph_wrappers[key] = causal

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        outputs=outputs,
    )
    if eagle_acts is not None:
        graph_vars["eagle_acts"] = eagle_acts
    if jetspec_taps is not None:
        graph_vars["jetspec_taps"] = jetspec_taps

    return graph_vars, graph_pool, graphs, all_N
