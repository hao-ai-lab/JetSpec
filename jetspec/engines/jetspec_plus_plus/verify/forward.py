
from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
import os
import warnings
import time
import numpy as np
from jetspec.engines.jetspec_plus_plus.models.context import set_context, get_context, reset_context
from jetspec.engines.jetspec_plus_plus.models import context as context_state
from jetspec.engines.jetspec_plus_plus.models.attention import ATTENTION_BACKEND
from jetspec.engines.jetspec_plus_plus.verify.verify_plan_helpers import staged_verify_rows
from jetspec.engines.jetspec_plus_plus.verify.verify_plan_helpers import stage_mixed_verify_rows
from jetspec.engines.jetspec_plus_plus.verify.verify_plan_helpers import assemble_full_staged_verify_rows
from jetspec.engines.jetspec_plus_plus.verify.device_verify import stage_single_verify_inputs
from jetspec.engines.jetspec_plus_plus.verify.device_verify import stage_single_verify_inputs_host
from jetspec.engines.jetspec_plus_plus.verify.device_verify import stage_single_verify_pages
from jetspec.engines.jetspec_plus_plus.verify.device_verify import stage_single_verify_inputs_blob
from jetspec.engines.jetspec_plus_plus.verify.device_verify import stage_single_verify_inputs_blob_fast
from jetspec.engines.jetspec_plus_plus.verify.device_verify import verify_stage_blob_views
import torch
from time import perf_counter
from transformers import AutoTokenizer

from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.model_runner import ModelRunner
from jetspec.engines.jetspec_plus_plus.verify.verify import verify
from jetspec.engines.jetspec_plus_plus.engine.helpers.greedy_accept import greedy_accept
from jetspec.engines.jetspec_plus_plus.engine.helpers.greedy_accept import greedy_accept_device
from jetspec.engines.jetspec_plus_plus.engine.helpers.greedy_accept import point_mass_accept_device
from jetspec.engines.jetspec_plus_plus.verify.device_verify import stage_committed_ids
from jetspec.engines.jetspec_plus_plus.verify.device_verify import finish_committed_ids
from jetspec.engines.jetspec_plus_plus.verify.device_verify import stage_recovery_id
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import hash16_committed
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import make_hash16_weights
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import SpeculateResult
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import VerifyResult
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import VerifierBase
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import append_committed_taps
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import stage_committed_taps
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import finish_staged_taps
from jetspec.engines.jetspec_plus_plus.trace.attribution import FirstDivergenceDumper
from jetspec.engines.jetspec_plus_plus.trace.attribution import prompt_index
from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.jetspec.env import require_verify_device_batch


def tree_budget(*, run_config=None):
    """Keep the sealed native verifier guards while refusing the retired tree route."""
    from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
    if int(config_values(run_config).get('method.method_settings.tree_b', '0')) != 0:
        raise ValueError('retired tree verify requires a sealed historical read')
    return 0


def _required_env(var_name: str, note: str, *, run_config=None) -> str:
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name
    value = setting(run_config, var_name)
    if value:
        return value
    category, field = var_name.split('.', 1)
    raise RuntimeError(f"Missing required env var {environment_name(category, field)}. {note}")


def _point_mass_logits_q(logits_q, speculations, temperatures_target, temperatures_draft):
    """Give greedy draft rows the logits of their actual proposed tokens."""
    rows = (temperatures_target > 0) & (temperatures_draft == 0)
    if not rows.any():
        return logits_q
    aligned = logits_q.clone()
    one_hot = torch.zeros_like(aligned[rows])
    one_hot.scatter_(2, speculations[rows, 1:].unsqueeze(-1), 1.0)
    aligned[rows] = one_hot
    return aligned


class TargetRunner(ModelRunner):

    @property
    def hybrid_target(self):
        """Keep the frozen reference verifier's capability read unchanged."""
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        # N=1 commits through the device or reference protocol; C>1 is batch_hybrid_target.
        return has_state(self) and getattr(self.config, 'max_num_seqs', 1) == 1

    @property
    def batch_hybrid_target(self):
        """A stateful target served at C>1: the host verify commits each full row's count."""
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        return has_state(self) and getattr(self.config, 'max_num_seqs', 1) > 1

    def commit_hybrid_state(self, accepted):
        """Route the frozen reference verifier's commit through the protocol."""
        self.commit_state(accepted)

    def _capture_target_graphs(self):
        self._capture_jetspec_mixed_graphs()

    @torch.inference_mode()
    def stage_jetspec_verify(self, seqs: list[Sequence]):
        """Prepare the C>1 paged verify graph while candidate rows are in flight."""
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        self._staged_verify = None
        self._staged_mixed_verify = None
        if setting(self, 'engine.engine_settings.verify_prestage_mixed', '0') == "1":
            if setting(self, 'engine.engine_settings.legacy_kpred_forward', '0') == "1":
                raise ValueError("mixed verify prestage does not support legacy k predictor forward")
            if not hasattr(self, "_verify_prestage_path_counts"):
                self._verify_prestage_path_counts = dict(
                    staged_mixed=0, staged_full=0, unstaged=0)
        if (setting(self, 'engine.engine_settings.verify_prestage_mixed', '0') == "1"
                and setting(self, 'method.method_settings.miss_redraft', '0') == "0"):
            if (len(seqs) > 1 and self.jetspec_paged_flashinfer
                    and not self.enforce_eager
                    and getattr(self, "_jetspec_mixed_graphs", None)
                    and "verify" in self.graph_vars):
                self._staged_mixed_verify = stage_mixed_verify_rows(
                    seqs, self.block_size, self.config.speculate_k + 1)
        if (len(seqs) <= 1 or not self.jetspec_paged_flashinfer
                or self.enforce_eager or "verify" not in self.graph_vars):
            return
        width = self.config.speculate_k + 1
        bucket = next(b for b in self.graph_bs_list["verify"] if b >= len(seqs))
        if self._staged_mixed_verify is not None:
            positions, slots, lengths, tables = assemble_full_staged_verify_rows(
                self._staged_mixed_verify, bucket, width)
        else:
            positions, slots, lengths, tables = staged_verify_rows(
                seqs, self.block_size, width, bucket)
        gv = self.graph_vars["verify"]
        for key, value in gv.items():
            if key != "outputs":
                value.zero_()
        def copy_host(key, values, dtype, count):
            host = torch.tensor(values.tolist() if hasattr(values, "tolist") else values,
                                dtype=dtype, pin_memory=True)
            target = gv[key][:count]
            if host.ndim == 2:
                target = target[:, :host.shape[1]]
            target.copy_(host, non_blocking=True)
        copy_host("positions", positions, torch.int64, bucket * width)
        copy_host("slot_mapping", slots, torch.int32, bucket * width)
        copy_host("context_lens", lengths, torch.int32, bucket)
        copy_host("block_tables", tables, torch.int32, bucket)
        gv["cu_seqlens_q"][:bucket + 1].copy_(
            torch.arange(bucket + 1, dtype=torch.int32, device=self.device) * width)
        self.plan_flashinfer_paged_attention(
            gv["context_lens"][:bucket], gv["block_tables"][:bucket],
            qo_indptr=gv["cu_seqlens_q"][:bucket + 1],
            graph_batch_size=bucket, static_q_len=bucket * width,
            causal_only=True)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        self._staged_verify = (tuple((id(seq), len(seq) + width - 1) for seq in seqs),
                               bucket, temperatures)

    @torch.inference_mode()
    def prepare_staged_verify(self, seqs: list[Sequence]):
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        staged = self._staged_verify
        if staged is None or staged[0] != tuple((id(seq), len(seq)) for seq in seqs):
            self._staged_verify = None
            return None
        if has_state(self):
            self.prepare_state(seqs)
        width = self.config.speculate_k + 1
        bucket = staged[1]
        gv = self.graph_vars["verify"]
        ids = [token for seq in seqs for token in seq.token_ids[-width:]]
        input_ids = torch.tensor(ids, dtype=torch.int64, pin_memory=True).to(
            self.device, non_blocking=True)
        set_context(is_prefill=False,
                    cu_seqlens_q=gv["cu_seqlens_q"][:bucket + 1],
                    cu_seqlens_k=None, max_seqlen_q=width, max_seqlen_k=0,
                    slot_mapping=gv["slot_mapping"][:bucket * width],
                    context_lens=gv["context_lens"][:bucket],
                    block_tables=gv["block_tables"][:bucket])
        counts = getattr(self, "_verify_prestage_path_counts", None)
        if counts is not None:
            counts["staged_full"] += 1
        return input_ids, gv["positions"][:bucket * width], staged[2]

    def jetspec_n1_input_view(self):
        """Return the proposal's N=1 graph input storage to the speculator."""
        width = self.config.speculate_k + 1
        if (not self._verify_input_direct_enabled or self.config.max_num_seqs != 1
                or self.num_tp_gpus != 1):
            raise RuntimeError("direct N=1 verify input is disabled")
        return self._jetspec_mixed_vars["input_ids"][:width].view(1, width)

    @torch.inference_mode()
    def _capture_jetspec_mixed_graphs(self):
        """Warm exact mixed shapes before serving, sharing one replay buffer/pool."""
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        from jetspec.engines.jetspec_plus_plus.serving.graphs import graph_capture_buckets
        stateful = has_state(self)
        model_forward = self.state_model_forward if stateful else self.model
        max_bs = self.config.max_num_seqs
        batches = (graph_capture_buckets(self, verify=True) if stateful
                   else list(range(1, max_bs + 1)))
        max_bs = max(batches)
        width = self.config.speculate_k + 1
        rows = max_bs * width
        if self._verify_stage_blob_enabled:
            if max_bs != 1 or not self._verify_stage_host_enabled or not self._jetspec_verify_device:
                raise RuntimeError("JETSPEC_VERIFY_STAGE_BLOB requires N=1 device and host staging")
            blob_bytes = width * 12 + (20 if self._verify_plan_lean_enabled else 16)
            device_blob = torch.zeros(blob_bytes, dtype=torch.uint8, device=self.device)
            host_blob = torch.zeros(blob_bytes, dtype=torch.uint8, device="cpu",
                                    pin_memory=self.device.type == "cuda")
            device_views = verify_stage_blob_views(device_blob, width)
            self._jetspec_verify_blob_device = device_views
            self._jetspec_verify_blob_host = dict(device=device_blob, base=host_blob,
                                                   views=verify_stage_blob_views(host_blob, width))
            if self._verify_stage_blob_fast_enabled:
                blob = self._jetspec_verify_blob_host
                blob["np"] = {key: value.numpy() for key, value in blob["views"].items()}
                blob["offsets"] = np.arange(width, dtype=np.int64)
                if self._verify_stage_blob_fast_check:
                    check_base = torch.zeros_like(host_blob, device="cpu")
                    blob["check"] = dict(device=torch.zeros_like(check_base),
                                         base=check_base,
                                         views=verify_stage_blob_views(check_base, width))
        if self._verify_input_direct_enabled:
            # The speculator writes this graph input outside inference mode.
            with torch.inference_mode(False):
                input_ids = torch.zeros(rows, dtype=torch.int64, device=self.device)
        else:
            input_ids = torch.zeros(rows, dtype=torch.int64, device=self.device)
        gv = dict(
            input_ids=input_ids,
            position_offsets=torch.arange(rows, dtype=torch.int64, device=self.device),
            positions=torch.zeros(rows, dtype=torch.int64, device=self.device),
            slot_mapping=torch.full((rows,), -1, dtype=torch.int32, device=self.device),
            context_lens=torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            block_tables=torch.zeros(max_bs, self.max_num_blocks, dtype=torch.int32, device=self.device),
            cu_seqlens_q=torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device),
            outputs=torch.empty(rows, self.hf_config.hidden_size,
                                dtype=self.hf_config.torch_dtype, device=self.device),
        )
        if stateful:
            gv["state_slots"] = self.state.graph_slots(max_bs)
        if self._verify_stage_blob_enabled:
            for key in ("positions", "slot_mapping", "context_lens", "cu_seqlens_q"):
                gv[key] = device_views[key]
        if self.config.jetspec_taps:
            gv["taps"] = torch.empty(rows, 5, self.hf_config.hidden_size,
                                     dtype=self.hf_config.torch_dtype, device=self.device)
        self._jetspec_mixed_vars = gv
        if self._verify_input_direct_enabled:
            if max_bs != 1 or self.num_tp_gpus != 1 or not self._jetspec_verify_device:
                raise RuntimeError("JETSPEC_VERIFY_INPUT_DIRECT requires N=1 TP1 device verify")
            self._jetspec_n1_context = {
                query_width: context_state.Context(
                    is_prefill=False, slot_mapping=gv["slot_mapping"][:query_width],
                    context_lens=gv["context_lens"][:1],
                    block_tables=gv["block_tables"][:1],
                    cu_seqlens_q=gv["cu_seqlens_q"][:2], max_seqlen_q=query_width)
                for query_width in (1, width)
            }
        self._jetspec_mixed_graphs = {}
        pool = None
        shapes = ([(bs, full) for bs in reversed(batches) if bs > 1
                   for full in range(bs - 1, 0, -1)]
                  if ATTENTION_BACKEND == "flashinfer" else [])
        if (setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
                or setting(self, 'engine.engine_settings.ssd_verify_device', '0') == "1"):
            shapes.extend(((1, 1), (1, 0)))
        from jetspec.engines.jetspec_plus_plus.models.target_w4a8 import report_target_w4a8, warm_target_w4a8
        warm_target_w4a8(self.model, [full * width + bs - full for bs, full in shapes])
        for bs, full in shapes:
                count = full * width + bs - full
                lengths = torch.tensor([width] * full + [1] * (bs - full),
                                       dtype=torch.int32, device=self.device)
                gv["context_lens"][:bs].copy_(lengths)
                cu = gv["cu_seqlens_q"][:bs + 1]
                cu[0] = 0
                cu[1:].copy_(lengths.cumsum(0))
                set_context(is_prefill=False, slot_mapping=gv["slot_mapping"][:count],
                            context_lens=gv["context_lens"][:bs],
                            block_tables=gv["block_tables"][:bs],
                            cu_seqlens_q=cu, max_seqlen_q=width)
                # A preallocated mask selects CUSTOM in FlashInfer even when
                # plan(causal=True) receives no mask. Mixed rows use causal only.
                # The attention layer looks wrappers up by (bs, rows); a masked
                # verify wrapper may already hold that key, so the causal one
                # stands in for the capture and the masked one comes back after.
                masked = (self.flashinfer_graph_wrappers.get((bs, count))
                          if ATTENTION_BACKEND == "flashinfer" else None)
                if ATTENTION_BACKEND == "flashinfer":
                    causal = self.get_flashinfer_graph_wrapper(bs, count, causal_only=True)
                    self.flashinfer_graph_wrappers[bs, count] = causal
                    self.plan_flashinfer_paged_attention(gv["context_lens"][:bs],
                        gv["block_tables"][:bs], qo_indptr=cu, graph_batch_size=bs,
                        static_q_len=count if bs == 1 else None,
                        static_kv_pages=1 if bs == 1 else None, causal_only=True)
                try:
                    if stateful:
                        self._state_slots = gv['state_slots'][:bs]
                    state_options = (dict(verify=True, query_lengths=(width,) * full + (1,) * (bs - full))
                                     if stateful else {})
                    # The model stacks its taps straight into the replay buffer.
                    tap_out = (gv["taps"][:count] if self.config.jetspec_taps and not stateful
                               and setting(self, 'engine.engine_settings.verify_cut_glue8b', '0') == "1"
                               else None)
                    if tap_out is not None:
                        state_options = dict(state_options, tap_out=tap_out)
                    model_forward(gv["input_ids"][:count], gv["positions"][:count], **state_options)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        result = model_forward(gv["input_ids"][:count], gv["positions"][:count], **state_options)
                        if self.config.jetspec_taps:
                            gv["outputs"][:count].copy_(result[0])
                            if result[1] is not tap_out:
                                gv["taps"][:count].copy_(result[1])
                        else:
                            gv["outputs"][:count].copy_(result)
                finally:
                    # The stand-in lives only for the capture: a later (bs, rows) lookup must
                    # never reach, and replan, the wrapper this graph captured.
                    if masked is not None:
                        self.flashinfer_graph_wrappers[bs, count] = masked
                    elif ATTENTION_BACKEND == "flashinfer":
                        self.flashinfer_graph_wrappers.pop((bs, count), None)
                if pool is None:
                    pool = graph.pool()
                self._jetspec_mixed_graphs[bs, full] = graph
                torch.cuda.synchronize()
                reset_context()
        report_target_w4a8()
        self._jetspec_mixed_pool = pool
        if self._verify_plan_reuse_enabled:
            if max_bs != 1 or not self._jetspec_verify_device:
                raise RuntimeError("JETSPEC_VERIFY_PLAN_REUSE requires N=1 device verify")
            for query_width in (1, width):
                self._guard_verify_plan_reuse(
                    self.get_flashinfer_graph_wrapper(1, query_width, causal_only=True))
        if (setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
                or setting(self, 'engine.engine_settings.ssd_verify_device', '0') == "1"):
            self._jetspec_n1_plan_host = {}
            for query_width in (1, width):
                self._jetspec_n1_plan_host[query_width] = dict(
                    qo_indptr=torch.tensor([0, query_width], dtype=torch.int32,
                                           device="cpu", pin_memory=True),
                    context_lens=torch.empty(1, dtype=torch.int32,
                                             device="cpu", pin_memory=True),
                    kv_indptr=torch.tensor([0, 0], dtype=torch.int32,
                                           device="cpu", pin_memory=True),
                    kv_last_page_len=torch.empty(1, dtype=torch.int32,
                                                 device="cpu", pin_memory=True),
                )
            if self._verify_plan_lean_enabled:
                self._jetspec_n1_plan_wrappers = {}
                self._jetspec_n1_plan_kwargs = {}
                for query_width, plan_host in self._jetspec_n1_plan_host.items():
                    wrapper = self.get_flashinfer_graph_wrapper(1, query_width,
                                                                causal_only=True)
                    self._jetspec_n1_plan_wrappers[query_width] = wrapper
                    wrapper._kv_lens_buffer = self._jetspec_verify_blob_device["kv_lens"]
                    plan_host["np"] = {key: plan_host[key].numpy()
                                       for key in ("context_lens", "kv_indptr",
                                                   "kv_last_page_len")}
                    self._jetspec_n1_plan_kwargs[query_width] = dict(
                        qo_indptr=plan_host["qo_indptr"], graph_batch_size=1,
                        static_q_len=query_width, static_kv_pages=0,
                        static_kv_indptr=plan_host["kv_indptr"],
                        static_kv_last_page_len=plan_host["kv_last_page_len"],
                        causal_only=True)
        if ((setting(self, 'engine.engine_settings.verify_stage_host', '0') == "1" or
             setting(self, 'engine.engine_settings.verify_stage_early', '0') == "1")
                and setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"):
            self._jetspec_verify_stage_host = {
                query_width: torch.empty(query_width * 2 + 2, dtype=torch.int64,
                                         device="cpu", pin_memory=self.device.type == "cuda")
                for query_width in (1, width)
            }
            self._jetspec_staged_page_ids = None
            self._jetspec_prepared = None

    @torch.inference_mode()
    def prepare_jetspec_mixed_device(self, seq: Sequence, full: bool):
        """Stage host-known graph inputs and plan while the candidate is in flight."""
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        width = self.config.speculate_k + 1 if full else 1
        prefix_len = getattr(seq, "_jetspec_last_packet_l", seq.num_cached_tokens)
        if prefix_len != seq.num_cached_tokens:
            raise RuntimeError("verify packet length differs from sequence metadata")
        gv = self._jetspec_mixed_vars
        kv_len = prefix_len + width
        pages = (kv_len + self.block_size - 1) // self.block_size
        page_ids = tuple(seq.block_table[:pages])
        if page_ids != self._jetspec_staged_page_ids:
            stage_single_verify_pages(seq, gv["block_tables"][0], pages)
            self._jetspec_staged_page_ids = page_ids
        _census_mark("stage_pages", run_config=getattr(self, 'run_config', None))
        if getattr(self, "_verify_stage_blob_enabled", False):
            if self._verify_stage_blob_fast_enabled:
                mismatches = stage_single_verify_inputs_blob_fast(
                    gv, None, prefix_len, seq.block_table, self.block_size,
                    self._jetspec_verify_blob_host, query_width=width,
                    check=self._verify_stage_blob_fast_check)
                self._verify_stage_blob_fast_rounds += 1
                self._verify_stage_blob_fast_mismatches += mismatches
                if self._verify_stage_blob_fast_check:
                    print("STAGE_BLOB_FAST_CHECK "
                          f"rounds={self._verify_stage_blob_fast_rounds} "
                          f"mismatches={self._verify_stage_blob_fast_mismatches}", flush=True)
                if mismatches:
                    raise RuntimeError("STAGE_BLOB_FAST_CHECK mismatch")
            else:
                stage_single_verify_inputs_blob(
                    gv, None, prefix_len, seq.block_table, self.block_size,
                    self._jetspec_verify_blob_host, query_width=width)
        else:
            stage_single_verify_inputs_host(
                gv, None, prefix_len, seq.block_table, self.block_size,
                self._jetspec_verify_stage_host[width], query_width=width)
        _census_mark("stage_inputs", run_config=getattr(self, 'run_config', None))
        plan_host = self._jetspec_n1_plan_host[width]
        if getattr(self, "_verify_plan_lean_enabled", False):
            plan_host["np"]["context_lens"][0] = kv_len
            plan_host["np"]["kv_indptr"][1] = pages
            plan_host["np"]["kv_last_page_len"][0] = (kv_len - 1) % self.block_size + 1
        else:
            plan_host["context_lens"][0] = kv_len
            plan_host["kv_indptr"][1] = pages
            plan_host["kv_last_page_len"][0] = (kv_len - 1) % self.block_size + 1
        if (getattr(self, "_verify_input_direct_enabled", False) and
                hasattr(self, "_jetspec_n1_context")):
            context_state._CONTEXT = self._jetspec_n1_context[width]
        else:
            set_context(is_prefill=False, slot_mapping=gv["slot_mapping"][:width],
                        context_lens=gv["context_lens"][:1],
                        block_tables=gv["block_tables"][:1],
                        cu_seqlens_q=gv["cu_seqlens_q"][:2], max_seqlen_q=width)
        try:
            if ATTENTION_BACKEND == "flashinfer":
                if getattr(self, "_verify_plan_reuse_enabled", False):
                    self._plan_jetspec_n1(width, pages, page_ids, plan_host, gv)
                else:
                    self.plan_flashinfer_paged_attention(plan_host["context_lens"],
                        gv["block_tables"][:1], qo_indptr=plan_host["qo_indptr"],
                        graph_batch_size=1, static_q_len=width, static_kv_pages=pages,
                        static_kv_indptr=plan_host["kv_indptr"],
                        static_kv_last_page_len=plan_host["kv_last_page_len"], causal_only=True)
            _census_mark("plan", run_config=getattr(self, 'run_config', None))
        finally:
            reset_context()
        self._jetspec_prepared = (seq.seq_id, prefix_len, width, page_ids)

    @torch.inference_mode()
    def _run_hybrid_mixed_eager(self, seq, proposals, width, prefix_len):
        """Stage an N=1 hybrid verify block through the eager target forward."""
        started = time.perf_counter()
        self.prepare_state([seq])
        positions = torch.arange(prefix_len, prefix_len + width,
                                 dtype=torch.int64, device=self.device)
        kv_len = prefix_len + width
        pages = (kv_len + self.block_size - 1) // self.block_size
        tables = torch.tensor([seq.block_table[:pages]], dtype=torch.int32,
                              device=self.device)
        slots = (tables[0].index_select(0, positions // self.block_size)
                 * self.block_size + positions % self.block_size).to(torch.int32)
        set_context(is_prefill=False, slot_mapping=slots,
                    context_lens=torch.tensor([kv_len], dtype=torch.int32, device=self.device),
                    block_tables=tables,
                    cu_seqlens_q=torch.tensor([0, width], dtype=torch.int32, device=self.device),
                    max_seqlen_q=width)
        try:
            forward_started = time.perf_counter()
            # run_model selects causal eager attention and GDN verify snapshots;
            # the existing verifier commits only the accepted snapshot afterward.
            logits, taps = self.run_model(proposals[:width], positions, False, last_only=False)
            ended = time.perf_counter()
            return logits.reshape(width, -1), taps, {
                "input": (forward_started - started) * 1000,
                "eager_forward": (ended - forward_started) * 1000,
            }
        finally:
            reset_context()

    @torch.inference_mode()
    def run_jetspec_mixed_device(self, seq: Sequence, proposals: torch.Tensor, full: bool):
        """Replay the N=1 mixed graph from device proposal IDs and static widths."""
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import eager_pair
        if eager_pair(self):
            # The same incremental pair protocol, with an uncaptured target
            # forward. All widths use the existing scratch-state commit below.
            if self.is_draft or not self.config.jetspec:
                raise ValueError('eager hybrid verify requires the JetSpec target')
            input_ids, positions = self.prepare_decode([seq], verify=full)
            try:
                logits, taps = self.run_model(input_ids, positions, False, False)
                if full:
                    logits = logits.reshape(self.config.speculate_k + 1, -1)
                return logits, taps, {}
            finally:
                reset_context()
        # Extracted-method contract tests supply a lightweight runner without __init__.
        verify_device = (self._jetspec_verify_device if hasattr(self, "_jetspec_verify_device")
                         else setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1")
        ssd_verify_device = (self._ssd_verify_device if hasattr(self, "_ssd_verify_device")
                             else setting(self, 'engine.engine_settings.ssd_verify_device', '0') == "1")
        stage_early = (self._verify_stage_early_enabled if hasattr(self, "_verify_stage_early_enabled")
                       else setting(self, 'engine.engine_settings.verify_stage_early', '0') == "1")
        stage_host = (self._verify_stage_host_enabled if hasattr(self, "_verify_stage_host_enabled")
                      else setting(self, 'engine.engine_settings.verify_stage_host', '0') == "1")
        if (self.is_draft or not (self.config.jetspec and
                                  verify_device)
                and not (self.config.jetspec_taps and ssd_verify_device)):
            raise ValueError("device mixed verify requires the JetSpec target and its knob")
        width = self.config.speculate_k + 1 if full else 1
        if len(seq) - seq.num_cached_tokens != width or proposals.shape[0] < width:
            raise ValueError("device mixed verify received the wrong query width")
        # The previous outcome's D2H mirror is the exact committed prefix.
        # Scheduler postprocess must agree before this prefix enters the plan.
        prefix_len = getattr(seq, "_jetspec_last_packet_l", seq.num_cached_tokens)
        if prefix_len != seq.num_cached_tokens:
            raise RuntimeError("verify packet length differs from sequence metadata")
        if (has_state(self)
                and getattr(self, "enforce_eager", False) and ssd_verify_device):
            return self._run_hybrid_mixed_eager(seq, proposals, width, prefix_len)
        graph = getattr(self, "_jetspec_mixed_graphs", {}).get((1, int(full)))
        if graph is None:
            raise RuntimeError("N=1 device mixed verify graph was not captured")
        stage_started = time.perf_counter()
        gv = self._jetspec_mixed_vars
        if has_state(self):
            from jetspec.engines.jetspec_plus_plus.models.state import stage_state_slots
            self.prepare_state([seq])
            stage_state_slots(self, gv, 1)
            self.state.prepare_verify(self._state_slots, width)
        plan_host = self._jetspec_n1_plan_host[width]
        kv_len = prefix_len + width
        pages = (kv_len + self.block_size - 1) // self.block_size
        early = (stage_early and
                 getattr(self, "_jetspec_prepared", None) ==
                 (seq.seq_id, prefix_len, width, tuple(seq.block_table[:pages])))
        if early:
            self._jetspec_prepared = None
        direct = (getattr(self, "_verify_input_direct_enabled", False) and
                  proposals.data_ptr() == gv["input_ids"].data_ptr())
        if not (early and direct):
            if getattr(self, "_verify_plan_lean_enabled", False):
                plan_host["np"]["context_lens"][0] = kv_len
                plan_host["np"]["kv_indptr"][1] = pages
                plan_host["np"]["kv_last_page_len"][0] = (kv_len - 1) % self.block_size + 1
            else:
                plan_host["context_lens"][0] = kv_len
                plan_host["kv_indptr"][1] = pages
                plan_host["kv_last_page_len"][0] = (kv_len - 1) % self.block_size + 1
        host_stage = stage_host and verify_device
        if early:
            source = (getattr(seq, "_jetspec_verify_proposal_host", None)
                      if host_stage else proposals)
            if source is None:
                raise RuntimeError("early verify stage lacks the pinned proposal row")
            if not direct:
                gv["input_ids"][:width].copy_(source[:width], non_blocking=gv["input_ids"].is_cuda)
        else:
            _census_mark("stage_head", run_config=getattr(self, 'run_config', None))
            page_ids = tuple(seq.block_table[:pages]) if host_stage else None
            if not host_stage or page_ids != self._jetspec_staged_page_ids:
                stage_single_verify_pages(seq, gv["block_tables"][0], pages)
                if host_stage:
                    self._jetspec_staged_page_ids = page_ids
            _census_mark("stage_pages", run_config=getattr(self, 'run_config', None))
            if host_stage:
                proposal_host = getattr(seq, "_jetspec_verify_proposal_host", None)
                if proposal_host is None or proposal_host.shape[0] < width:
                    raise RuntimeError("host verify staging lacks the pinned proposal row")
                if getattr(self, "_verify_stage_blob_enabled", False):
                    if self._verify_stage_blob_fast_enabled:
                        mismatches = stage_single_verify_inputs_blob_fast(
                            gv, proposal_host[:width], prefix_len, seq.block_table,
                            self.block_size, self._jetspec_verify_blob_host,
                            input_staged=direct, check=self._verify_stage_blob_fast_check)
                        self._verify_stage_blob_fast_rounds += 1
                        self._verify_stage_blob_fast_mismatches += mismatches
                        if self._verify_stage_blob_fast_check:
                            print("STAGE_BLOB_FAST_CHECK "
                                  f"rounds={self._verify_stage_blob_fast_rounds} "
                                  f"mismatches={self._verify_stage_blob_fast_mismatches}", flush=True)
                        if mismatches:
                            raise RuntimeError("STAGE_BLOB_FAST_CHECK mismatch")
                    else:
                        stage_single_verify_inputs_blob(
                            gv, proposal_host[:width], prefix_len, seq.block_table,
                            self.block_size, self._jetspec_verify_blob_host,
                            input_staged=direct)
                else:
                    stage_single_verify_inputs_host(
                        gv, proposal_host[:width], prefix_len, seq.block_table,
                        self.block_size, self._jetspec_verify_stage_host[width],
                        input_staged=direct)
            else:
                if direct:
                    stage_single_verify_inputs(
                        gv, proposals[:width], prefix_len,
                        gv["block_tables"][0], self.block_size,
                        input_staged=True)
                else:
                    stage_single_verify_inputs(
                        gv, proposals[:width], prefix_len,
                        gv["block_tables"][0], self.block_size)
            _census_mark("stage_inputs", run_config=getattr(self, 'run_config', None))
        if (getattr(self, "_verify_input_direct_enabled", False) and
                hasattr(self, "_jetspec_n1_context")):
            context_state._CONTEXT = self._jetspec_n1_context[width]
        else:
            set_context(is_prefill=False, slot_mapping=gv["slot_mapping"][:width],
                        context_lens=gv["context_lens"][:1],
                        block_tables=gv["block_tables"][:1],
                        cu_seqlens_q=gv["cu_seqlens_q"][:2], max_seqlen_q=width)
        _census_mark("stage", run_config=getattr(self, 'run_config', None))
        try:
            plan_started = time.perf_counter()
            if ATTENTION_BACKEND == "flashinfer" and not early:
                if getattr(self, "_verify_plan_reuse_enabled", False):
                    self._plan_jetspec_n1(
                        width, pages, tuple(seq.block_table[:pages]), plan_host, gv)
                else:
                    self.plan_flashinfer_paged_attention(plan_host["context_lens"],
                        gv["block_tables"][:1], qo_indptr=plan_host["qo_indptr"],
                        graph_batch_size=1, static_q_len=width, static_kv_pages=pages,
                        static_kv_indptr=plan_host["kv_indptr"],
                        static_kv_last_page_len=plan_host["kv_last_page_len"], causal_only=True)
            replay_started = time.perf_counter()
            if not early:
                _census_mark("plan", run_config=getattr(self, 'run_config', None))
            kernel_profile = getattr(self, "_verify_kernel_profile", None)
            if kernel_profile is not None and kernel_profile.active is not None:
                kernel_profile.step("replay")
            timing = getattr(self, '_anatomy_lite', None)
            if timing is not None:
                timing.gpu_begin()
            graph.replay()
            logits_started = time.perf_counter()
            _census_mark("replay", run_config=getattr(self, 'run_config', None))
            if kernel_profile is not None and kernel_profile.active is not None:
                kernel_profile.step("logits")
            # lm_head keeps a leading batch dim; greedy accept wants [width, vocab].
            logits = self.model.compute_logits(gv["outputs"][:width], last_only=False).reshape(width, -1)
            if timing is not None:
                timing.gpu_end()
            ended = time.perf_counter()
            site_ms = {
                "input": (plan_started - stage_started) * 1000,
                "plan": (replay_started - plan_started) * 1000,
                "replay": (logits_started - replay_started) * 1000,
                "logits": (ended - logits_started) * 1000,
            }
            return logits, (gv["taps"][:width] if "taps" in gv else None), site_ms
        finally:
            reset_context()

    @torch.inference_mode()
    def run_jetspec_mixed(self, seqs: list[Sequence]):
        """One paged-prefill forward over ready blocks and one-token AR rows.

        The existing prefill backend already accepts ragged query offsets and
        cached prefixes. Return every query's logits/taps; the verifier commits
        only each sequence's accepted prefix, exactly as in the split path.
        """
        if self.is_draft or not self.config.jetspec:
            raise ValueError("mixed verification requires a JetSpec target")
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        if has_state(self):
            raise ValueError('hybrid target mixed verification requires JETSPEC_PERSEQ=0')
        widths = [len(seq) - seq.num_cached_tokens for seq in seqs]
        if any(width not in (1, self.config.speculate_k + 1) for width in widths):
            raise ValueError("mixed queries must be AR or a complete draft block")
        staged = getattr(self, "_staged_mixed_verify", None)
        self._staged_mixed_verify = None
        if (staged is not None and getattr(self, "_jetspec_mixed_graphs", None)):
            try:
                result = self._run_jetspec_mixed_staged(seqs, widths, staged)
                if result is not None:
                    self._verify_prestage_path_counts["staged_mixed"] += 1
                    return result
            finally:
                reset_context()
        if getattr(self, "_verify_prestage_path_counts", None) is not None:
            self._verify_prestage_path_counts["unstaged"] += 1
        input_ids, positions = self.prepare_prefill(seqs)
        try:
            if getattr(self, "_jetspec_mixed_graphs", None) and get_context().block_tables is not None:
                return self._run_jetspec_mixed_graph(input_ids, positions, get_context(), widths)
            if ATTENTION_BACKEND == "flashinfer":
                self.plan_flashinfer_prefill_attention()
            result = self.model(input_ids, positions)
            if self.config.jetspec_taps:
                hidden, taps = result
                return self.model.compute_logits(hidden, last_only=False), taps
            return self.model.compute_logits(result, last_only=False)
        finally:
            reset_context()


class Verifier(VerifierBase):
    _round_env = {
        "_jetspec_verify_device": 'engine.engine_settings.jetspec_verify_device',
        "_jetspec_verify_sync_audit": 'engine.engine_settings.jetspec_verify_sync_audit',
        "_jetspec_verify_post_lean": 'verify_post_lean',
        "_jetspec_publish_first": 'engine.engine_settings.verify_publish_first',
        "_jetspec_verify_outcome_fused": 'verify_outcome_fused',
        "_jetspec_verify_outcome_fused_check": 'engine.engine_settings.verify_outcome_fused_check',
        "_verify_cut_glue8b": 'engine.engine_settings.verify_cut_glue8b',
    }

    def __getattr__(self, name):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        if name == '_jetspec_verify_reference':
            from jetspec.engines.jetspec_plus_plus.engine import current_engine
            return current_engine().kind == 'reference'
        env_name = type(self)._round_env.get(name)
        if env_name is None:
            raise AttributeError(name)
        if "." in env_name:
            from jetspec.engines.jetspec_plus_plus.serving.settings import setting
            value = setting(self, env_name, "0") == "1"
        else:
            value = kernel_enabled(env_name)
            if not value:
                from jetspec.engines.jetspec_plus_plus.profiles import verify_cut_lean27b
                value = verify_cut_lean27b(
                    env_name, self, getattr(self, 'target_model_runner', None))
        setattr(self, name, value)
        return value

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        sampler_x: float | None = None,
        async_fan_out: int | None = None,
        jit_speculate: bool = False,
        tokenizer: AutoTokenizer = None,
        metrics: dict = None,
        run_config=None,
    ):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.run_config = run_config if run_config is not None else getattr(target_model_runner, 'run_config', None)
        self.sampler_x = sampler_x
        self.async_fan_out = async_fan_out
        self.jit_speculate = jit_speculate
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.lossless_dumper = FirstDivergenceDumper.from_env(run_config=run_config)
        self.hybrid_verify_audit = setting(self, 'engine.engine_settings.hybrid_verify_audit', '0') == "1"
        # The EAGLE target prefill returns its conditioning in place of logits, so the fork skips
        # the dumper there; JETSPEC_LOGIT_DUMP_EAGLE=1 asks the target for both (registered in env.py).
        self.dump_eagle_prefill = setting(self, 'engine.engine_settings.logit_dump_eagle', '0') == "1"
        self._jetspec_offsets: dict[int, int] = {}
        self.oracle_anchor = getattr(getattr(self.target_model_runner, "config", None), "jetspec", False) and anchor_mode(run_config=run_config) == "oracle"
        self.perseq = setting(self, 'engine.engine_settings.perseq', '0') == "1"
        self._hash16_weights = None
        self._jetspec_deferred_metrics = []
        if setting(self, 'engine.engine_settings.verify_kernel_profile'):
            from jetspec.engines.jetspec_plus_plus.trace.verify_kprofile import VerifyKernelProfile
            self._verify_kernel_profile = VerifyKernelProfile.from_env(run_config=run_config)
            self.target_model_runner._verify_kernel_profile = self._verify_kernel_profile

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        if self.hybrid_verify_audit and any(seq.temperature > 0 for seq in seqs):
            raise ValueError("JETSPEC_HYBRID_VERIFY_AUDIT only supports T=0 argmax auditing")
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import eager_pair
        if (self._jetspec_verify_device or eager_pair(self.target_model_runner)) and self._hash16_weights is None:
            self._hash16_weights = make_hash16_weights(self.device)
        if (self.lossless_dumper is not None or self.hybrid_verify_audit) and (not eagle or self.dump_eagle_prefill):
            result = self.target_model_runner.call("run", seqs, True, True, True)
            if eagle:
                token_ids, eagle_acts, prefill_logits = result
            else:
                token_ids, prefill_logits = result
        else:
            result = self.target_model_runner.call("run", seqs, True)
            prefill_logits = None
            if eagle:
                token_ids, eagle_acts = result
            else:
                token_ids = result

        offset = 0
        for index, (seq, token_id) in enumerate(zip(seqs, token_ids)):
            seq.recovery_token_id = token_id
            if self.hybrid_verify_audit:
                top = prefill_logits[index].float().topk(2)
                greedy = int(prefill_logits[index].argmax().item())
                self.metrics.setdefault("jetspec_hybrid_prefill_verify", []).append({
                    "prompt_index": prompt_index(seq), "emitted": int(token_id),
                    "verify_argmax": greedy,
                    "verify_top2_ids": top.indices.tolist(),
                    "verify_top2_logits": top.values.tolist(),
                })
                if int(token_id) != greedy:
                    raise AssertionError("hybrid prefill token differs from target verify argmax")
            if self.lossless_dumper is not None and prefill_logits is not None:
                self.lossless_dumper.consider(
                    prompt_index=prompt_index(seq),
                    seq_id=seq.seq_id,
                    position=0,
                    round_id="prefill",
                    decision="prefill",
                    actual=token_id,
                    logits=prefill_logits[index],
                    source="prefill",
                )
            if eagle:
                seq_len = seq.num_prompt_tokens
                # this doesn't move acts onto cpu does it? 
                seq.last_target_hidden_state = eagle_acts[offset + seq_len - 1].clone()
                offset += seq_len

        return VerifyResult(
            [], # no accepted tokens for prefill, just recovery tokens (first sampled token).
            [seq.recovery_token_id for seq in seqs],
            eagle_acts if eagle else None,
        )

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        """Verify speculative tokens using the target model."""
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        if speculate_result.jetspec_verify_mask is not None:
            return self._verify_jetspec(seqs, speculate_result)
        if (setting(self, 'engine.engine_settings.ssd_verify_device', '0') == "1"
                and speculate_result.blockhead_anatomy is not None):
            require_verify_device_batch(len(seqs), run_config=getattr(self, 'run_config', None))
            if seqs[0].draft_temperature not in (None, 0):
                raise ValueError("SSD_VERIFY_DEVICE requires a greedy draft (draft_temperature None or 0)")
            if not seqs[0].temperature >= 0:
                raise ValueError("SSD_VERIFY_DEVICE requires target temperature >= 0")
            if setting(self, 'engine.engine_settings.ssd_verify_sync_audit', '0') == "1":
                torch.cuda.set_sync_debug_mode("warn")
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    result = self._verify_jetspec_device(seqs[0], speculate_result)
                self.metrics.setdefault("ssd_device_sync_warnings", []).append(
                    sum("synchroniz" in str(warning.message).lower()
                        for warning in seen))
                return result
            return self._verify_jetspec_device(seqs[0], speculate_result)
        if has_state(self.target_model_runner):
            raise RuntimeError("hybrid speculative verify requires SSD_VERIFY_DEVICE=1")
        _prof = setting(self, 'engine.engine_settings.profile', '0') == "1"
        batch_size = len(seqs)

        if _prof:
            torch.cuda.synchronize()
            _vt0 = perf_counter()

        _pt = setting(self, 'engine.engine_settings.profile_target', '0') == "1"
        _tv0 = perf_counter()
        result = self.target_model_runner.call("run", seqs, False, False, True)

        if _prof:
            torch.cuda.synchronize()
            _vt1 = perf_counter()

        if _pt:
            torch.cuda.synchronize()
            _vt_call = perf_counter()
            print(f"[PROFILE verifier] target_call={(_vt_call-_tv0)*1000:.2f}ms eagle={eagle} bs={batch_size}", flush=True)

        if eagle:
            logits_p_flat, eagle_acts_flat = result
        elif self.target_model_runner.config.jetspec_taps:
            logits_p_flat, jetspec_taps_flat = result
        else:
            logits_p_flat = result

        for s in seqs:
            s.num_cached_tokens += self.lookahead + 1

        logits_p = logits_p_flat.view(
            batch_size, self.lookahead + 1, -1)  # [b, k+1, v]

        # Build per-seq temps for target verify and draft q respectively.
        temps_target = [seq.temperature for seq in seqs]
        temps_draft = [
            seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            for seq in seqs
        ]
        temperatures_target = torch.tensor(temps_target, dtype=torch.float32, device=self.device)
        temperatures_draft = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)

        new_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=_point_mass_logits_q(
                speculate_result.logits_q, speculate_result.speculations,
                temperatures_target, temperatures_draft),
            speculations=speculate_result.speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
            cache_hits=speculate_result.cache_hits,
            sampler_x=self.sampler_x,
            async_fan_out=self.async_fan_out,
            jit_speculate=self.jit_speculate,
        )

        if self.target_model_runner.config.jetspec_taps:
            jetspec_taps = jetspec_taps_flat.view(
                batch_size, self.lookahead + 1, 5, -1
            )
            for seq, candidates, new_suffix in zip(
                seqs, jetspec_taps, new_suffixes
            ):
                append_committed_taps(seq, candidates, len(new_suffix))

        self.metrics["target_verify_times"].append(perf_counter() - _tv0)

        if _prof:
            torch.cuda.synchronize()
            _vt2 = perf_counter()
            print(f"[PROFILE verify] target_fwd={(_vt1-_vt0)*1000:.2f}ms verify_compute={(_vt2-_vt1)*1000:.2f}ms", flush=True)


        # # Debug: print recovery tokens detokenized
        if getattr(self.target_model_runner.config, "verbose", False) and recovery_tokens is not None and len(recovery_tokens) > 0:
            recovery_texts = []
            for token in recovery_tokens:
                try:
                    text = self.tokenizer.decode([token], skip_special_tokens=False)
                    recovery_texts.append(text)
                except Exception:
                    recovery_texts.append(f"<token_id:{token}>")
            print(f"[verify] recovery tokens: {recovery_texts}", flush=True)

        self.metrics["accepted_suffix_lens_with_recovery"].extend(
            [len(s) for s in new_suffixes])

        # For async mode, also track accepted suffix lengths only for cache hits
        if speculate_result.cache_hits is not None:
            _ch_cpu = speculate_result.cache_hits.cpu()
            self.metrics["cache_hits"].append(_ch_cpu.float().mean().item())
            hit_count = int(_ch_cpu.sum().item())
            self.metrics["cache_hit_count"] += hit_count
            self.metrics["cache_miss_count"] += _ch_cpu.numel() - hit_count
            for i, suffix_len in enumerate([len(s) for s in new_suffixes]):
                if _ch_cpu[i] == 1:
                    self.metrics["accepted_suffix_lens_on_hit"].append(suffix_len)
                else:
                    self.metrics["accepted_suffix_lens_on_miss"].append(suffix_len)

        # Print mean length of new suffixes for monitoring
        if getattr(self.target_model_runner.config, "verbose", False) and new_suffixes:
            mean_suffix_len = sum([len(suffix) for suffix in new_suffixes]) / len(new_suffixes)
            print(f"[verify] mean new suffix length: {mean_suffix_len:.2f}", flush=True)

        eagle_acts = None
        if eagle:
            eagle_acts = eagle_acts_flat.view(batch_size, self.lookahead + 1, -1)
        
        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=eagle_acts,
        )

    def _verify_jetspec(
        self,
        seqs: list[Sequence],
        speculate_result: SpeculateResult,
    ) -> VerifyResult:
        """Run full block verification for hits and a true one-token AR for misses."""
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        require_verify_device_batch(len(seqs), run_config=getattr(self, 'run_config', None))
        if any(getattr(seq, "draft_temperature", None) not in (None, 0) for seq in seqs):
            raise ValueError("JetSpec verification requires a greedy draft")
        reference = getattr(self, "_jetspec_verify_reference", False)
        if reference:
            if (len(seqs) != 1 or seqs[0].temperature != 0 or self.lookahead != 15
                    or self._jetspec_verify_device or tree_budget(run_config=self.target_model_runner.run_config) or self.oracle_anchor
                    or getattr(self.target_model_runner.config, "enforce_eager", False)
                    or not self.target_model_runner.config.jetspec_taps):
                raise ValueError("reference verify requires graphed N=1 T0 b16 with taps")
            reference_start = seqs[0].num_cached_tokens
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import eager_hybrid_pair
        if eager_hybrid_pair(getattr(self, 'target_model_runner', None)):
            if len(seqs) != 1 or not seqs[0].temperature >= 0 or tree_budget(run_config=self.target_model_runner.run_config) or self.oracle_anchor:
                raise ValueError('eager hybrid pair requires N=1 target temperature >= 0, no tree budget or oracle anchor')
            # Reuse the pair protocol's accept/state/outcome work exactly once.
            # The runner executes its target forward directly under enforce_eager.
            return self._verify_jetspec_device(seqs[0], speculate_result)
        if (self._jetspec_verify_device and len(seqs) == 1
                and seqs[0].temperature >= 0 and not tree_budget(run_config=self.target_model_runner.run_config)
                and not self.oracle_anchor and self.lossless_dumper is None):
            if self._jetspec_verify_sync_audit:
                torch.cuda.set_sync_debug_mode("warn")
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    result = self._verify_jetspec_device(seqs[0], speculate_result)
                self.metrics.setdefault("jetspec_device_sync_warnings", []).append(
                    sum("synchroniz" in str(warning.message).lower()
                        for warning in seen))
                return result
            return self._verify_jetspec_device(seqs[0], speculate_result)
        if not reference and getattr(getattr(self, "target_model_runner", None), "hybrid_target", False):
            raise ValueError("hybrid target requires device verification to commit its state")
        started = perf_counter()
        timing = getattr(self.target_model_runner, '_anatomy_lite', None)
        if timing is not None:
            timing.stamp('native_verify_start')
        mask = speculate_result.jetspec_verify_mask
        verify_mask = (torch.as_tensor(mask, dtype=torch.bool) if isinstance(mask, list)
                       else mask.to(torch.bool).cpu())
        full_indices = [i for i, value in enumerate(verify_mask.tolist()) if value]
        ar_indices = [i for i, value in enumerate(verify_mask.tolist()) if not value]
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        if ar_indices and current_engine().budget.uncut:
            raise RuntimeError("uncut verify cannot execute an AR fallback")
        if reference and ar_indices:
            raise RuntimeError("reference verify refuses an AR step without a full tap block")
        new_suffixes = [None] * len(seqs)
        recovery_tokens = [None] * len(seqs)
        records = self.metrics["jetspec_round_records"][-len(seqs):]
        by_seq = {int(record["seq_id"]): record for record in records}
        independent_chain = not tree_budget(run_config=self.target_model_runner.run_config) and setting(self, 'engine.engine_settings.w264_independent_records', '0') == '1'
        if independent_chain:
            for index, seq in enumerate(seqs):
                record = by_seq[seq.seq_id]
                width = self.lookahead + 1 if index in full_indices else 1
                record['chain_prefix_ids'] = list(seq.token_ids[:seq.num_cached_tokens])
                record['chain_query_ids'] = list(seq.token_ids[seq.num_cached_tokens:seq.num_cached_tokens+width])
                if len(record['chain_query_ids']) != width or record['chain_query_ids'][0] != seq.recovery_token_id:
                    raise RuntimeError('CHAIN_INPUT_LINKAGE')

        mixed_logits = mixed_taps = None
        full_rows = len(full_indices) * (self.lookahead + 1)
        if self.perseq and full_indices and ar_indices:
            # Keep each sequence contiguous; scatter accepted results back below.
            ordered = [seqs[i] for i in full_indices + ar_indices]
            result = self.target_model_runner.call("run_jetspec_mixed", ordered)
            if self.target_model_runner.config.jetspec_taps:
                mixed_logits, mixed_taps = result
            else:
                mixed_logits = result

        if full_indices and tree_budget(run_config=self.target_model_runner.run_config):
            full_seqs = [seqs[i] for i in full_indices]
            full_suffixes, full_recoveries = self._verify_jetspec_tree_hits(full_seqs, by_seq)
            for index, suffix, recovery in zip(full_indices, full_suffixes, full_recoveries):
                new_suffixes[index] = suffix
                recovery_tokens[index] = recovery
        elif full_indices:
            full_seqs = [seqs[i] for i in full_indices]
            if mixed_logits is None:
                result = self.target_model_runner.call(
                    "run", full_seqs, False, False, True
                )
            elif mixed_taps is None:
                result = mixed_logits[:full_rows]
            else:
                result = mixed_logits[:full_rows], mixed_taps[:full_rows]
            if self.target_model_runner.config.jetspec_taps:
                logits_flat, taps_flat = result
            else:
                logits_flat = result
                taps_flat = None
            for seq in full_seqs:
                seq.num_cached_tokens += self.lookahead + 1
            logits_p = logits_flat.view(
                len(full_indices), self.lookahead + 1, -1
            )
            full_predictions = None
            if not reference and all(
                seq.temperature == 0 and (seq.draft_temperature is None or seq.draft_temperature == 0)
                for seq in full_seqs
            ):
                proposal_rows = [seq.token_ids[-(self.lookahead + 1):] for seq in full_seqs]
                decisions = greedy_accept(logits_p, None, proposal_rows=proposal_rows,
                                          return_predictions=self.oracle_anchor)
                full_suffixes, full_recoveries = decisions[:2]
                if self.oracle_anchor:
                    full_predictions = decisions[2]
            else:
                temperatures_target = torch.tensor(
                    [seq.temperature for seq in full_seqs],
                    dtype=torch.float32,
                    device=self.device,
                )
                temperatures_draft = torch.tensor(
                    [
                        seq.draft_temperature
                        if seq.draft_temperature is not None
                        else 0
                        for seq in full_seqs
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                if reference and full_indices == [0]:
                    full_suffixes, full_recoveries = verify(
                        logits_p=logits_p,
                        logits_q=_point_mass_logits_q(
                            speculate_result.logits_q, speculate_result.speculations,
                            temperatures_target, temperatures_draft),
                        speculations=speculate_result.speculations,
                        temperatures_target=temperatures_target,
                        temperatures_draft=temperatures_draft,
                        cache_hits=speculate_result.cache_hits,
                        sampler_x=self.sampler_x,
                        async_fan_out=self.async_fan_out,
                        jit_speculate=self.jit_speculate,
                    )
                else:
                    full_suffixes, full_recoveries = verify(
                        logits_p=logits_p,
                        logits_q=_point_mass_logits_q(
                            speculate_result.logits_q[full_indices],
                            speculate_result.speculations[full_indices],
                            temperatures_target, temperatures_draft),
                        speculations=speculate_result.speculations[full_indices],
                        temperatures_target=temperatures_target,
                        temperatures_draft=temperatures_draft,
                        cache_hits=(speculate_result.cache_hits[full_indices] if reference
                                    else torch.ones(len(full_indices), dtype=torch.int64,
                                                    device=self.device)),
                        sampler_x=self.sampler_x if reference else None,
                        async_fan_out=self.async_fan_out if reference else None,
                        jit_speculate=self.jit_speculate if reference else True,
                    )
            if self.oracle_anchor and full_predictions is None:
                full_predictions = logits_p.argmax(-1).tolist()
            if reference and getattr(self.target_model_runner, "hybrid_target", False):
                # The reference accept has synchronized its host count. Commit
                # that GDN snapshot before the existing hybrid envelope is sent.
                self.target_model_runner.commit_hybrid_state(torch.tensor(
                    len(full_suffixes[0]), dtype=torch.int64, device=self.device))
            elif getattr(self.target_model_runner, 'batch_hybrid_target', False):
                # C>1: commit while the runner still holds the full rows' slot order;
                # the AR forward below prepares a different set of slots.
                self.target_model_runner.commit_state(torch.tensor(
                    [len(suffix) for suffix in full_suffixes],
                    dtype=torch.int64, device=self.device))
            if taps_flat is not None:
                taps = taps_flat.view(
                    len(full_indices), self.lookahead + 1, 5, -1
                )
                for seq, candidates, suffix in zip(
                    full_seqs, taps, full_suffixes
                ):
                    append_committed_taps(seq, candidates, len(suffix))
            for local_index, (index, seq, suffix, recovery) in enumerate(
                zip(full_indices, full_seqs, full_suffixes, full_recoveries)
            ):
                new_suffixes[index] = suffix
                recovery_tokens[index] = recovery
                if reference and getattr(self, "hybrid_verify_audit", False):
                    record = by_seq[seq.seq_id]
                    predictions = logits_p[local_index].argmax(-1).tolist()
                    emitted = [int(token) for token in suffix[1:]] + [int(recovery)]
                    record["verify_argmax_ids"] = predictions
                    record["verify_emitted_ids"] = emitted
                    if emitted != predictions[:len(emitted)]:
                        raise AssertionError("reference verify emitted a non-argmax token")
                    self.metrics["jetspec_hybrid_verify_rows"] = (
                        self.metrics.get("jetspec_hybrid_verify_rows", 0) + 1)
                    self.metrics["jetspec_hybrid_verify_tokens"] = (
                        self.metrics.get("jetspec_hybrid_verify_tokens", 0) + len(emitted))
                if independent_chain:
                    self._save_chain_logits(by_seq[seq.seq_id], logits_p[local_index, :len(suffix)])
                if self.oracle_anchor:
                    by_seq[seq.seq_id]['greedy_predictions'] = full_predictions[local_index]
                if self.lossless_dumper is not None:
                    record = by_seq[seq.seq_id]
                    generated_offset = self._jetspec_offsets.get(seq.seq_id, 0)
                    for suffix_index in range(1, len(suffix)):
                        self.lossless_dumper.consider(
                            prompt_index=prompt_index(seq),
                            seq_id=seq.seq_id,
                            position=generated_offset + suffix_index,
                            round_id=record["round"],
                            decision=record["decision"],
                            actual=suffix[suffix_index],
                            logits=logits_p[local_index, suffix_index - 1],
                            source="block_verify_draft",
                        )
                    self.lossless_dumper.consider(
                        prompt_index=prompt_index(seq),
                        seq_id=seq.seq_id,
                        position=generated_offset + len(suffix),
                        round_id=record["round"],
                        decision=record["decision"],
                        actual=recovery,
                        logits=logits_p[local_index, len(suffix) - 1],
                        source="block_verify_recovery",
                    )

        if ar_indices:
            ar_seqs = [seqs[i] for i in ar_indices]
            if tree_budget(run_config=self.target_model_runner.run_config):
                ar_started = perf_counter()
                ar_start_event = torch.cuda.Event(enable_timing=True)
                ar_end_event = torch.cuda.Event(enable_timing=True)
                ar_start_event.record()
            if mixed_logits is not None:
                logits = mixed_logits[full_rows:]
                temperatures = self.target_model_runner.prepare_sample(ar_seqs)
                ar_recoveries = self.target_model_runner.sampler(
                    logits, temperatures
                ).tolist()
                if mixed_taps is not None:
                    for index, seq in enumerate(ar_seqs):
                        append_committed_taps(
                            seq, mixed_taps[full_rows + index:full_rows + index + 1], 1
                        )
                ar_logits = logits if self.lossless_dumper is not None or independent_chain else None
            elif self.lossless_dumper is not None or independent_chain:
                ar_recoveries, ar_logits = self.target_model_runner.call(
                    "run", ar_seqs, False, True, True
                )
            else:
                ar_recoveries = self.target_model_runner.call(
                    "run", ar_seqs, False
                )
                ar_logits = None
            if tree_budget(run_config=self.target_model_runner.run_config):
                ar_end_event.record(); ar_end_event.synchronize()
                for seq in ar_seqs:
                    by_seq[seq.seq_id].update(tree_hit=False, tree_budget=tree_budget(run_config=self.target_model_runner.run_config),
                        tree_nodes=0, tree_topk_present=False, tree_graph_replay=False,
                        tree_build_ms=0., realized_queries=1,
                        verify_forward_ms=ar_start_event.elapsed_time(ar_end_event),
                        verify_forward_wall_ms=(perf_counter()-ar_started)*1000)
            for local_index, (index, seq, recovery) in enumerate(
                zip(ar_indices, ar_seqs, ar_recoveries)
            ):
                new_suffixes[index] = [int(seq.recovery_token_id)]
                recovery_tokens[index] = int(recovery)
                if independent_chain:
                    self._save_chain_logits(by_seq[seq.seq_id], ar_logits[local_index:local_index+1])
                if self.oracle_anchor:
                    # An AR miss has only its one real target prediction.
                    by_seq[seq.seq_id]['greedy_predictions'] = [int(recovery)] + [-1] * 14
                if ar_logits is not None and self.lossless_dumper is not None:
                    record = by_seq[seq.seq_id]
                    generated_offset = self._jetspec_offsets.get(seq.seq_id, 0)
                    self.lossless_dumper.consider(
                        prompt_index=prompt_index(seq),
                        seq_id=seq.seq_id,
                        position=generated_offset + 1,
                        round_id=record["round"],
                        decision=record["decision"],
                        actual=recovery,
                        logits=ar_logits[local_index],
                        source="ar_recovery",
                    )

        assert all(suffix is not None for suffix in new_suffixes)
        assert all(token is not None for token in recovery_tokens)
        for seq, suffix in zip(seqs, new_suffixes):
            self._jetspec_offsets[seq.seq_id] = (
                self._jetspec_offsets.get(seq.seq_id, 0) + len(suffix)
            )
        if timing is not None:
            timing.stamp('native_verify_end')
        self.metrics["target_verify_times"].append(perf_counter() - started)
        self.metrics["accepted_suffix_lens_with_recovery"].extend(
            [len(suffix) for suffix in new_suffixes]
        )
        self.metrics["jetspec_verify_round_ms"].append(
            (perf_counter() - started) * 1000
        )
        self.metrics["jetspec_ar_steps"] += len(ar_indices)
        self.metrics["jetspec_full_verify_steps"] += len(full_indices)
        for seq, suffix, recovery in zip(
            seqs, new_suffixes, recovery_tokens
        ):
            record = by_seq[seq.seq_id]
            record["accepted_len"] = len(suffix)
            record["accepted_ids"] = [int(token) for token in suffix[1:]]
            record["next_correction"] = int(recovery)
            if self.oracle_anchor:
                record['verify_complete_ns'] = int(perf_counter() * 1e9)
        if reference:
            from jetspec.engines.jetspec_plus_plus.verify.baseline import build_reference_outcome
            if timing is not None:
                timing.stamp('envelope_start')
            outcome = build_reference_outcome(
                seqs[0], new_suffixes[0], recovery_tokens[0], taps[0],
                start=reference_start,
                max_model_len=self.target_model_runner.config.max_model_len,
                eos_id=getattr(self.tokenizer, "eos_token_id", None),
                record=records[-1], run_config=getattr(self, 'run_config', None))
            if timing is not None:
                timing.stamp('envelope_end')
            return VerifyResult(new_suffixes, recovery_tokens, device_outcome=outcome)
        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=None,
        )

    def _verify_jetspec_device(self, seq: Sequence, speculate_result: SpeculateResult) -> VerifyResult:
        """N=1 verify with one small host outcome read after device commit."""
        from jetspec.engines.jetspec_plus_plus.models.state import has_state
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        # Recovery is staged for the next suffix, not accepted in this one.
        # The scheduler finishes only after EOS enters the committed suffix.
        recovery_eos_terminal = False
        temperature = getattr(seq, "temperature", 0)
        if temperature > 0 and getattr(self, "hybrid_verify_audit", False):
            raise ValueError("JETSPEC_HYBRID_VERIFY_AUDIT only supports T=0 argmax auditing")
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        _census_mark("verify_entry", run_config=getattr(self, 'run_config', None))
        started = perf_counter()
        blockhead = getattr(speculate_result, "blockhead_anatomy", None) is not None
        if blockhead:
            full = True
        else:
            mask = speculate_result.jetspec_verify_mask
            if isinstance(mask, torch.Tensor) and mask.is_cuda:
                raise RuntimeError("device verify needs the host-known N=1 query width")
            full = bool(mask[0])
        width = self.lookahead + 1 if full else 1
        proposal = speculate_result.speculations[0, :width]
        kernel_profile = getattr(self, "_verify_kernel_profile", None)
        profiled = not blockhead and kernel_profile is not None and kernel_profile.begin()
        if profiled:
            kernel_profile.step("runner_stage")
        logits, taps, site_ms = self.target_model_runner.call(
            "run_jetspec_mixed_device", seq, proposal, full)
        verify_top2 = logits.float().topk(2, dim=-1) if getattr(self, "hybrid_verify_audit", False) else None
        verify_argmax = logits.argmax(dim=-1) if getattr(self, "hybrid_verify_audit", False) else None
        launched = perf_counter()
        if profiled:
            kernel_profile.step("late_record")
        late_record = getattr(self, "_jetspec_late_record_hook", None)
        if late_record is not None:
            late_record(seq)
        for site, duration in site_ms.items():
            self.metrics.setdefault(f"jetspec_device_{site}_host_ms", []).append(duration)
        _census_mark("logits", run_config=getattr(self, 'run_config', None))
        if profiled:
            kernel_profile.step("accept")
        if temperature == 0:
            counts, recoveries = greedy_accept_device(logits[None], proposal[None])
        else:
            temperatures = torch.tensor([temperature], dtype=torch.float32, device=logits.device)
            counts, recoveries = point_mass_accept_device(
                logits[None], proposal[None], temperatures,
                sampler_x=self.sampler_x, async_fan_out=self.async_fan_out)
        accepted = perf_counter()
        _census_mark("accept", run_config=getattr(self, 'run_config', None))
        count, recovery = counts[0], recoveries[0]
        suffix_len = count + 1
        if has_state(self.target_model_runner):
            self.target_model_runner.commit_state(suffix_len)
        outcome_fused = getattr(self, "_jetspec_verify_outcome_fused", False)
        if outcome_fused:
            if blockhead or self.device.type != "cuda" or not self._jetspec_verify_device:
                raise RuntimeError("JETSPEC_VERIFY_OUTCOME_FUSED requires N=1 CUDA JetSpec device verify, not blockhead")
        else:
            if profiled:
                kernel_profile.step("stage_ids")
            stage_committed_ids(seq, proposal,
                capacity=self.target_model_runner.config.max_model_len + self.lookahead + 2)
            # The second stage below writes the same slot after the packet D2H is queued.
            if taps is not None and not getattr(self, "_verify_cut_glue8b", False):
                if profiled:
                    kernel_profile.step("stage_taps_first")
                stage_committed_taps(seq, taps,
                    capacity=self.target_model_runner.config.max_model_len + self.lookahead + 1)

        if blockhead:
            # Same replay, greedy decision and device commit as V. The local
            # block head consumes accepted target taps directly; it has no wire.
            if taps is None:
                raise RuntimeError("block-head device verify requires target hidden states")
            stage_committed_taps(seq, taps,
                capacity=self.target_model_runner.config.max_model_len + self.lookahead + 1)
            host_pair = torch.empty(2, dtype=torch.int64, device="cpu",
                                    pin_memory=self.device.type == "cuda")
            host_pair.copy_(torch.stack((count, recovery)),
                            non_blocking=self.device.type == "cuda")
            if self.device.type == "cuda":
                mirror_event = torch.cuda.Event(enable_timing=False)
                mirror_event.record(torch.cuda.current_stream(self.device))
                mirror_event.synchronize()
            committed, recovery_id = int(host_pair[0]) + 1, int(host_pair[1])
            suffix = seq.token_ids[seq.num_cached_tokens:seq.num_cached_tokens + committed]
            if len(suffix) != committed:
                raise RuntimeError("device verify host proposal mirror is incomplete")
            finish_staged_taps(seq, committed)
            finish_committed_ids(seq, committed)
            seq.num_cached_tokens += width
            self.metrics["target_verify_times"].append(perf_counter() - started)
            self.metrics["accepted_suffix_lens_with_recovery"].append(committed)
            self.metrics.setdefault("ssd_verify_round_ms", []).append(
                (perf_counter() - started) * 1000)
            self.metrics.setdefault("ssd_device_launch_host_ms", []).append(
                (launched - started) * 1000)
            self.metrics.setdefault("ssd_device_accept_host_ms", []).append(
                (accepted - launched) * 1000)
            self.metrics.setdefault("ssd_device_commit_host_ms", []).append(
                (perf_counter() - accepted) * 1000)
            return VerifyResult([suffix], [recovery_id])

        fused_check = outcome_fused and getattr(self, "_jetspec_verify_outcome_fused_check", False)
        if outcome_fused:
            from jetspec.engines.jetspec_plus_plus.verify.outcome import launch_outcome_fused
            from jetspec.engines.jetspec_plus_plus.verify.outcome import prepare_outcome_storage
            row_k, row_a = getattr(seq, "_jetspec_verified_row_key", (-1, -1))
            storage, tap_storage, tap_base = prepare_outcome_storage(
                seq, proposal, taps,
                token_capacity=self.target_model_runner.config.max_model_len + self.lookahead + 2,
                tap_capacity=self.target_model_runner.config.max_model_len + self.lookahead + 1)
            host_words = torch.empty(OUTCOME_WORDS, dtype=torch.int64,
                                     device="cpu", pin_memory=True)
            outcome = torch.empty(OUTCOME_WORDS, dtype=torch.int64, device=self.device)
            if profiled:
                kernel_profile.step("outcome_fused")
            launch_outcome_fused(
                proposal, count, recovery, storage, taps, tap_storage,
                self._hash16_weights, outcome, start=seq.num_cached_tokens,
                tap_base=tap_base, seq_id=seq.seq_id,
                round_id=seq.jetspec_round + 1, row_k=row_k, row_a=row_a,
                prompt_tokens=seq.num_prompt_tokens,
                max_new_tokens=seq.max_new_tokens,
                max_model_len=self.target_model_runner.config.max_model_len,
                eos_id=getattr(self.tokenizer, "eos_token_id", None),
                ignore_eos=seq.ignore_eos,
                recovery_eos_terminal=recovery_eos_terminal)
            seq._jetspec_device_outcome = outcome
            _census_mark("outcome_head", run_config=getattr(self, 'run_config', None))
            _census_mark("recovery_id", run_config=getattr(self, 'run_config', None))
            if fused_check:
                from types import SimpleNamespace
                scratch = SimpleNamespace(
                    num_cached_tokens=seq.num_cached_tokens, token_ids=seq.token_ids,
                    _jetspec_token_storage=torch.empty_like(storage),
                    _jetspec_device_valid_length=seq.num_cached_tokens,
                    jetspec_taps=seq.jetspec_taps)
                scratch._jetspec_token_storage[:seq.num_cached_tokens].copy_(
                    storage[:seq.num_cached_tokens])
                stage_committed_ids(scratch, proposal, capacity=storage.numel())
                scratch_length = seq.num_cached_tokens + suffix_len
                stage_recovery_id(scratch, scratch_length, recovery)
                scratch_words = torch.zeros(OUTCOME_WORDS, dtype=torch.int64,
                                            device="cpu", pin_memory=True)
                scratch_words[OutcomeOffset.CMD] = 1
                scratch_words[OutcomeOffset.SEQ_ID] = seq.seq_id
                scratch_words[OutcomeOffset.ROUND] = seq.jetspec_round + 1
                scratch_words[OutcomeOffset.K] = row_k
                scratch_words[OutcomeOffset.A] = row_a
                scratch_outcome = scratch_words.to(self.device, non_blocking=True)
                scratch_outcome[OutcomeOffset.TRUE_K].copy_(count)
                scratch_outcome[OutcomeOffset.RECOVERY].copy_(recovery)
                scratch_outcome[OutcomeOffset.TAP_COUNT].copy_(suffix_len)
                scratch_outcome[OutcomeOffset.L].copy_(scratch_length)
                scratch_outcome[OutcomeOffset.HASH16].copy_(
                    hash16_committed(scratch._jetspec_token_storage,
                                     scratch_length + 1, self._hash16_weights))
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                eos = torch.zeros((), dtype=torch.bool, device=self.device)
                if eos_id is not None and not seq.ignore_eos:
                    within = torch.arange(width, device=self.device) < suffix_len
                    eos = (proposal.eq(eos_id) & within).any()
                    if recovery_eos_terminal:
                        eos = eos | recovery.eq(eos_id)
                prior = seq.num_cached_tokens - seq.num_prompt_tokens
                limit = ((prior + suffix_len >= seq.max_new_tokens)
                         | (seq.num_cached_tokens + suffix_len >= self.target_model_runner.config.max_model_len))
                scratch_outcome[OutcomeOffset.EOS_FLAG].copy_((eos | limit).to(torch.int64))
                if taps is not None:
                    scratch._jetspec_tap_storage = torch.empty_like(tap_storage)
                    stage_committed_taps(scratch, taps, capacity=tap_storage.shape[0])
        else:
            if profiled:
                kernel_profile.step("packet_head")
            host_words = torch.zeros(OUTCOME_WORDS, dtype=torch.int64,
                                     device="cpu", pin_memory=self.device.type == "cuda")
            host_words[OutcomeOffset.CMD] = 1
            host_words[OutcomeOffset.SEQ_ID] = seq.seq_id
            host_words[OutcomeOffset.ROUND] = seq.jetspec_round + 1
            row_k, row_a = getattr(seq, "_jetspec_verified_row_key", (-1, -1))
            host_words[OutcomeOffset.K] = row_k
            host_words[OutcomeOffset.A] = row_a
            outcome = host_words.to(self.device, non_blocking=True)
            if profiled:
                kernel_profile.step("outcome_writes")
            outcome[OutcomeOffset.TRUE_K].copy_(count)
            outcome[OutcomeOffset.RECOVERY].copy_(recovery)
            outcome[OutcomeOffset.TAP_COUNT].copy_(suffix_len)
            committed_length = seq.num_cached_tokens + suffix_len
            _census_mark("outcome_head", run_config=getattr(self, 'run_config', None))
            if profiled:
                kernel_profile.step("recovery_id")
            stage_recovery_id(seq, committed_length, recovery)
            _census_mark("recovery_id", run_config=getattr(self, 'run_config', None))
            if profiled:
                kernel_profile.step("hash")
            outcome[OutcomeOffset.L].copy_(committed_length)
            outcome[OutcomeOffset.HASH16].copy_(
                hash16_committed(seq._jetspec_token_storage, committed_length + 1,
                                 self._hash16_weights))
            if profiled:
                kernel_profile.step("eos")
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            eos = torch.zeros((), dtype=torch.bool, device=self.device)
            if eos_id is not None and not seq.ignore_eos:
                within = torch.arange(width, device=self.device) < suffix_len
                eos = (proposal.eq(eos_id) & within).any()
                if recovery_eos_terminal:
                    eos = eos | recovery.eq(eos_id)
            prior = seq.num_cached_tokens - seq.num_prompt_tokens
            limit = ((prior + suffix_len >= seq.max_new_tokens)
                     | (seq.num_cached_tokens + suffix_len >= self.target_model_runner.config.max_model_len))
            outcome[OutcomeOffset.EOS_FLAG].copy_((eos | limit).to(torch.int64))
            seq._jetspec_device_outcome = outcome

        # Post the packet D2H as soon as it is complete, then enqueue tap
        # staging before the one wait needed by host sequence metadata.
        if profiled:
            kernel_profile.step("packet_mirror")
        host_words.copy_(outcome, non_blocking=outcome.is_cuda)
        if outcome.is_cuda:
            mirror_event = getattr(seq, "_jetspec_packet_mirror_event", None)
            if mirror_event is None:
                mirror_event = torch.cuda.Event(enable_timing=False)
                seq._jetspec_packet_mirror_event = mirror_event
            mirror_event.record(torch.cuda.current_stream(self.device))
        if taps is not None and not outcome_fused:
            if profiled:
                kernel_profile.step("stage_taps_second")
            stage_committed_taps(seq, taps,
                capacity=self.target_model_runner.config.max_model_len + self.lookahead + 1)
        presync_pack = getattr(self, "_jetspec_presync_pack_hook", None)
        if presync_pack is not None:
            if profiled:
                kernel_profile.step("presync_pack")
            presync_pack(seq)
        _census_mark("pre_sync", run_config=getattr(self, 'run_config', None))
        if profiled:
            kernel_profile.step("sync")
        if outcome.is_cuda:
            mirror_event.synchronize()
        if fused_check:
            mismatch = not torch.equal(outcome, scratch_outcome)
            mismatch |= not torch.equal(
                storage[:seq.num_cached_tokens + width],
                scratch._jetspec_token_storage[:seq.num_cached_tokens + width])
            mismatch |= not torch.equal(
                storage.index_select(0, scratch_length.reshape(1)),
                scratch._jetspec_token_storage.index_select(0, scratch_length.reshape(1)))
            if taps is not None:
                mismatch |= not torch.equal(
                    tap_storage[tap_base:tap_base + width],
                    scratch._jetspec_tap_storage[tap_base:tap_base + width])
            self._outcome_fused_check_rounds = getattr(self, "_outcome_fused_check_rounds", 0) + 1
            self._outcome_fused_check_mismatches = getattr(self, "_outcome_fused_check_mismatches", 0) + int(mismatch)
            print(f"OUTCOME_FUSED_CHECK rounds={self._outcome_fused_check_rounds} "
                  f"mismatches={self._outcome_fused_check_mismatches}", flush=True)
        if profiled:
            kernel_profile.step("verify_post")
            kernel_profile.round_meta = dict(seq_id=seq.seq_id,
                                             round=seq.jetspec_round + 1,
                                             width=width, full=full)
        _census_mark("sync", run_config=getattr(self, 'run_config', None))
        words = host_words.tolist() if getattr(self, "_jetspec_verify_post_lean", False) else host_words
        committed = int(words[OutcomeOffset.TAP_COUNT])
        recovery_id = int(words[OutcomeOffset.RECOVERY])
        packet_l = int(words[OutcomeOffset.L])
        if packet_l != seq.num_cached_tokens + committed:
            raise RuntimeError("verify packet length differs from accepted prefix")
        seq._jetspec_last_packet_l = packet_l
        seq._jetspec_last_packet_eos = int(words[OutcomeOffset.EOS_FLAG])
        suffix = seq.token_ids[seq.num_cached_tokens:seq.num_cached_tokens + committed]
        if len(suffix) != committed:
            raise RuntimeError("device verify host proposal mirror is incomplete")
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        if setting(self, 'engine.engine_settings.chain_tail_trace', '0') == '1':
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            branch = ('eos' if eos_id in suffix and not seq.ignore_eos else 'cap'
                      if seq._jetspec_last_packet_eos else 'ar_step' if not full else 'verify')
            print(f"MAILBOX_ID side=verify event=outcome seq={seq.seq_id} round={seq.jetspec_round + 1} branch={branch} recovery={recovery_id} eos={seq._jetspec_last_packet_eos} suffix={suffix}", flush=True)
        if verify_top2 is not None:
            top_ids = verify_top2.indices.tolist()
            top_logits = verify_top2.values.tolist()
            argmax_ids = verify_argmax.tolist()
            record = self.metrics["jetspec_round_records"][-1]
            record["verify_top2_ids"] = top_ids
            record["verify_top2_logits"] = top_logits
            record["verify_argmax_ids"] = argmax_ids
            record["verify_emitted_ids"] = [int(token) for token in suffix[1:]] + [recovery_id]
            for position, token in enumerate(record["verify_emitted_ids"]):
                if token != argmax_ids[position]:
                    raise AssertionError(
                        f"hybrid verify argmax mismatch seq={seq.seq_id} "
                        f"round={seq.jetspec_round + 1} row={position} "
                        f"emitted={token} verify={argmax_ids[position]}")
            self.metrics["jetspec_hybrid_verify_rows"] = (
                self.metrics.get("jetspec_hybrid_verify_rows", 0) + 1)
            self.metrics["jetspec_hybrid_verify_tokens"] = (
                self.metrics.get("jetspec_hybrid_verify_tokens", 0) + len(record["verify_emitted_ids"]))
        if getattr(self, "_jetspec_publish_first", False):
            seq._jetspec_device_tap_count = committed
            seq._jetspec_last_recovery_id = recovery_id
            self._jetspec_pending_postpublish = (
                seq, suffix, committed, recovery_id, width, full, taps is not None,
                started, launched, accepted, perf_counter(),
                self.metrics["jetspec_round_records"][-1])
            _census_mark("verify_post", run_config=getattr(self, 'run_config', None))
            return VerifyResult([suffix], [recovery_id], device_outcome=outcome)
        if taps is not None:
            finish_staged_taps(seq, committed)
        finish_committed_ids(seq, committed)
        seq._jetspec_device_tap_count = committed
        seq.num_cached_tokens += width
        self._jetspec_offsets[seq.seq_id] = self._jetspec_offsets.get(seq.seq_id, 0) + committed
        self.metrics["target_verify_times"].append(perf_counter() - started)
        self.metrics["accepted_suffix_lens_with_recovery"].append(committed)
        self.metrics["jetspec_verify_round_ms"].append((perf_counter() - started) * 1000)
        self.metrics["jetspec_ar_steps"] += int(not full)
        self.metrics["jetspec_full_verify_steps"] += int(full)
        record = self.metrics["jetspec_round_records"][-1]
        record["accepted_len"] = committed
        record["accepted_ids"] = [int(token) for token in suffix[1:]]
        record["next_correction"] = recovery_id
        self.metrics.setdefault("jetspec_device_launch_host_ms", []).append(
            (launched - started) * 1000)
        self.metrics.setdefault("jetspec_device_accept_host_ms", []).append(
            (accepted - launched) * 1000)
        self.metrics.setdefault("jetspec_device_commit_host_ms", []).append(
            (perf_counter() - accepted) * 1000)
        _census_mark("verify_post", run_config=getattr(self, 'run_config', None))
        return VerifyResult([suffix], [recovery_id], device_outcome=outcome)

    def finish_jetspec_postpublish(self, seq):
        """Finish sequence state after the unchanged outcome send site."""
        pending = getattr(self, "_jetspec_pending_postpublish", None)
        if pending is None or pending[0] is not seq:
            raise RuntimeError("post-publish verify row does not match")
        self._jetspec_pending_postpublish = None
        (_, suffix, committed, recovery_id, width, full, has_taps,
         started, launched, accepted, word_stamp, record) = pending
        if has_taps:
            finish_staged_taps(seq, committed)
        finish_committed_ids(seq, committed)
        seq.num_cached_tokens += width
        self._jetspec_offsets[seq.seq_id] = self._jetspec_offsets.get(seq.seq_id, 0) + committed
        self._jetspec_deferred_metrics.append((
            record, suffix, committed, recovery_id, full,
            started, launched, accepted, word_stamp))
        if seq._jetspec_last_packet_eos:
            self.flush_jetspec_deferred_metrics()

    def flush_jetspec_deferred_metrics(self):
        """Apply metrics in verify order, using each round's captured record."""
        for (record, suffix, committed, recovery_id, full,
             started, launched, accepted, word_stamp) in self._jetspec_deferred_metrics:
            self.metrics["target_verify_times"].append(word_stamp - started)
            self.metrics["accepted_suffix_lens_with_recovery"].append(committed)
            self.metrics["jetspec_verify_round_ms"].append((word_stamp - started) * 1000)
            self.metrics["jetspec_ar_steps"] += int(not full)
            self.metrics["jetspec_full_verify_steps"] += int(full)
            record["accepted_len"] = committed
            record["accepted_ids"] = [int(token) for token in suffix[1:]]
            record["next_correction"] = recovery_id
            self.metrics.setdefault("jetspec_device_launch_host_ms", []).append(
                (launched - started) * 1000)
            self.metrics.setdefault("jetspec_device_accept_host_ms", []).append(
                (accepted - launched) * 1000)
            self.metrics.setdefault("jetspec_device_commit_host_ms", []).append(
                (word_stamp - accepted) * 1000)
        self._jetspec_deferred_metrics.clear()


    def _save_chain_logits(self, record, logits):
        from jetspec.engines.jetspec_plus_plus.serving.settings import required_setting
        from pathlib import Path
        import hashlib
        cell_name = required_setting(self, 'engine.engine_settings.w264_cell_name')
        dump_root = _required_env(
            'engine.engine_settings.debug_dump_dir',
            'Set it to the root directory for debug dumps.',
            run_config=self.run_config,
        )
        directory = Path(dump_root) / 'chain-inputs' / cell_name
        directory.mkdir(parents=True, exist_ok=True)
        evidence = directory / f'p{record["prompt_index"]}-r{record["round"]}.pt'
        if evidence.exists():
            raise RuntimeError('CHAIN_INPUT_EVIDENCE_EXISTS')
        torch.save(logits.cpu(), evidence)
        record['chain_served_logits_path'] = str(evidence)
        record['chain_served_logits_sha256'] = hashlib.sha256(evidence.read_bytes()).hexdigest()
