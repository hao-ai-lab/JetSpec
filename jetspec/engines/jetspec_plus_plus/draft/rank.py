# Adaptation: package relocation and deferred optional dependencies.
from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
from jetspec.engines.jetspec_plus_plus.draft.block_geometry import staged_context
from jetspec.engines.jetspec_plus_plus.draft.config import legacy_name, named_setting, legacy_values, with_settings
import os
import time
import json
import sys
import torch
import torch.distributed as dist
import dataclasses

from jetspec.engines.jetspec_plus_plus.serving.model_runner import ModelRunner
from jetspec.engines.jetspec_plus_plus.serving.config import Config
from jetspec.engines.jetspec_plus_plus.models.context import set_context
from jetspec.engines.jetspec_plus_plus.models.context import reset_context
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import append_jetspec_tap_mirrors
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import recv_int64
from jetspec.engines.jetspec_plus_plus.draft.drafter import DraftSuperseded
from jetspec.engines.jetspec_plus_plus.draft.drafter import JetSpecDrafter
from jetspec.engines.jetspec_plus_plus.draft.drafter import JetSpecSequence
from jetspec.engines.jetspec_plus_plus.draft.drafter import commit_sequences_batch
from jetspec.engines.jetspec_plus_plus.draft.drafter import anchor_known_correction
from jetspec.engines.jetspec_plus_plus.draft.drafter import following_candidate_from_token_block
from jetspec.engines.jetspec_plus_plus.draft.live_rows import admission_hypotheses
from jetspec.engines.jetspec_plus_plus.draft.live_rows import select_live_block
from jetspec.engines.jetspec_plus_plus.draft.live_rows import select_live_blocks_batch
from jetspec.engines.jetspec_plus_plus.draft.live_rows import cache_packed_metadata_batch
from jetspec.engines.jetspec_plus_plus.draft.live_rows import e04_lean
from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft
from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft_force
from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft_passes
from jetspec.engines.jetspec_plus_plus.draft.live_rows import draft_deadline
from jetspec.engines.jetspec_plus_plus.draft.live_rows import DeadlineClock
from jetspec.engines.jetspec_plus_plus.draft.live_rows import DraftDeadlineMiss
from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft_anchor
from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateSet
from jetspec.engines.jetspec_plus_plus.wire.protocol import unpack_request_wire
from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
from jetspec.engines.jetspec_plus_plus.wire.protocol import oracle_candidate_blocks
from jetspec.engines.jetspec_plus_plus.wire.protocol import branch_count
from jetspec.engines.jetspec_plus_plus.wire.protocol import candidate_blocks_tensor
from jetspec.engines.jetspec_plus_plus.wire.protocol import pack_candidate_sets
from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
from jetspec.engines.jetspec_plus_plus.wire.protocol import select_candidate_drafts
from jetspec.engines.jetspec_plus_plus.wire.protocol import ProtocolCounters
from jetspec.engines.jetspec_plus_plus.wire.protocol import validate_candidate_row_keys_batch
from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_verifier_kind_from_host_pins

# serving JETSPEC_TAIL_LEAN=1 (default off): the round's W270_HEAD / W258_KSET log lines (and the
# posterior-hit bookkeeping that feeds them) run after the candidate is sent, not between the
# commit and the adapter or between the last pass and the send; same lines, later in the log
_DEFERRED = []


def _defer_logs():
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    return kernel_enabled('tail_lean', '1')


def _later(call):
    if _defer_logs():
        _DEFERRED.append(call)
    else:
        call()


def _flush_deferred():
    _run_idle()
    while _DEFERRED:
        _DEFERRED.pop(0)()


# serving JETSPEC_TAIL_PUBLISH: host work with no reader before the next round (the round's
# deferred log lines; measurement tools may queue their own writes here with idle_later) runs
# while the next round's host waits on its own passes, just before the early publish's wait,
# not between the publish and the next request.  Order is kept; the exit command drains it.
_IDLE = []


def idle_later(call):
    _IDLE.append(call)


def _run_idle(event=None, keep=0):
    """Drain the queue; with ``event``, only while the device is still busy (event not done),
    then down to ``keep`` pending items (oldest first)."""
    while _IDLE and (event is None or not event.query()):
        _IDLE.pop(0)()
    if event is not None:
        while len(_IDLE) > keep:
            _IDLE.pop(0)()


def _tail_publish(run_config=None):
    """serving JETSPEC_TAIL_PUBLISH=1 (default off): the round-ahead candidate is published from
    inside the refinement, straight from its device rows (one host copy of the rows and keys
    into the pinned packet, the control fields written on the host), before the metadata trip
    home, the published blocks' host bookkeeping, the candidate checks' sync and the trace."""
    from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
    return setting(run_config, 'engine.engine_settings.tail_publish', '0') == '1'


def _device_publish(run_config=None):
    from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
    return setting(run_config, 'engine.engine_settings.device_publish', '0') == '1'


def _pack_jetspec_blocks(blocks):
    return blocks.wire if hasattr(blocks, 'wire') else candidate_blocks_tensor(blocks)


def _candidate_set_from_rows(seq_id, round_id, blocks, host_keys=None, keys_validated=False,
                             joined_metrics=None, *, run_config=None):
    # serving JETSPEC_TAIL_PUBLISH: host_keys = the published keys, already copied home (the same
    # values as blocks.row_keys), so the CandidateSet checks read no device value
    candidate = CandidateSet(
        seq_id=int(seq_id), round_id=int(round_id), blocks=_pack_jetspec_blocks(blocks),
        row_keys=getattr(blocks, 'row_keys', None) if host_keys is None else host_keys,
        _keys_validated=keys_validated, run_config=run_config)
    metrics = getattr(blocks, 'head_metrics', None)
    if metrics is not None:
        record = dict(seq_id=int(seq_id), round_id=int(round_id), **metrics)
        if joined_metrics is None:
            _later(lambda: print('W270_HEAD ' + json.dumps(record), flush=True))
        else:
            joined_metrics.append('W270_HEAD ' + json.dumps(record) + '\n')
    return candidate


def _draft_verifier_kind_from_pins(candidate, *, seq_id, round_id, true_k, correction):
    """Mirror the verifier's first live keyed pin without copying every draft token home."""
    if candidate.seq_id != seq_id or candidate.round_id != round_id:
        return 'late'
    key = [true_k, correction]
    for index, row_key in enumerate(candidate.row_keys.tolist()):
        if row_key == key:
            pin = int(candidate.blocks[index, 0].item())
            if pin >= 0:
                return 'hit' if pin == correction else 'miss'
    return 'miss'


class DraftRunner(ModelRunner):
    
    @classmethod
    def create_draft_config(cls, cfg: Config, run_config=None) -> Config:
        """Create a draft config from the main config without instantiating DraftRunner."""
        run_config = (run_config if run_config is not None else
                      getattr(cfg, 'run_config', None))
        if run_config is None:
            raise ValueError('DraftRunner missing RunConfig launch_overlay')
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import resolve_admission_count
        if cfg.jetspec and run_config.engine.engine_settings.auto == '1':
            raise ValueError('retired variant fuse: the automatic route requests all fusion')
        draft_cfg = dataclasses.replace(
            cfg,
            model=cfg.draft,
            jetspec_config_is_draft=True,
            jetspec_passes=cfg.jetspec_passes,
            jetspec_hyp=(resolve_admission_count(cfg.jetspec_hyp, run_config=run_config)
                         if cfg.jetspec and run_config.engine.engine_settings.rows == "1" else cfg.jetspec_hyp),
            # The rows module has its own KV and logits buffers after this
            # ordinary draft cache is allocated (prefill can exceed decode N).
            gpu_memory_utilization=((0.35 if run_config.engine.engine_settings.rows_fast == "1" else 0.5)
                                    if cfg.jetspec and run_config.engine.engine_settings.rows == "1"
                                    else (0.75 if not cfg.draft_async else 0.8)),
            tokenizer_path=cfg.model if cfg.use_eagle else None,
            d_model_target=cfg.hf_config.hidden_size if cfg.use_eagle and cfg.hf_config else None,
            jetspec_tap_hidden_size=(
                cfg.hf_config.hidden_size if cfg.jetspec_taps else None
            ),
            enforce_eager=True if cfg.jetspec else cfg.enforce_eager,
            target_dtype=None,
        )
        # Config.__post_init__ skipped its target-only checks for this copy; the taps came validated.
        assert draft_cfg.jetspec_config_is_draft
        assert draft_cfg.jetspec_target_layer_ids == cfg.jetspec_target_layer_ids
        draft_cfg.run_config = run_config
        return draft_cfg

    def __init__(self, cfg: Config, rank: int = 0, init_q = None, eagle3_model_class=None, engine=None, run_config=None):
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
        from jetspec.engines.jetspec_plus_plus.draft.config import with_method_settings
        self.run_config = (run_config if run_config is not None else
                           getattr(cfg, 'run_config', None))
        if self.run_config is None:
            raise ValueError('DraftRunner missing RunConfig launch_overlay')
        self.run_config = with_method_settings(self.run_config, getattr(cfg, 'method_settings', None))
        self.profile_draft = self.run_config.engine.engine_settings.profile_draft == "1"
        from jetspec.engines.jetspec_plus_plus.engine import current_engine, install_engine
        self.engine = install_engine(engine if engine is not None else getattr(cfg, 'engine', None) or current_engine())
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import install_method_settings
        install_method_settings(getattr(cfg, 'method_settings', None))
        self._hybrid_pair_layout = None
        if self.run_config.engine.engine_settings.hybrid_pair == "1":
            from jetspec.engines.jetspec_plus_plus.wire.hybrid import HybridPairLayout
            if cfg.speculate_k not in (15, 31):
                raise ValueError("hybrid pair transfer requires b16/b32")
            self._hybrid_pair_layout = HybridPairLayout.from_config(
                cfg.hf_config, lanes=cfg.speculate_k + 1)
        self._chain_mode = cfg.jetspec and self.run_config.engine.engine_settings.chain_graph == "seg"
        if self._chain_mode:
            from jetspec.engines.jetspec_plus_plus.serving.cache_capacity import apply_chain_capacity
            capacities = legacy_values(self.run_config)
            apply_chain_capacity(capacities, cfg.max_model_len, lanes=cfg.speculate_k + 1)
            self.run_config = with_settings(self.run_config, {
                path: capacities[legacy_name(path)] for path in (
                    'engine.engine_settings.w259_c1_cache_capacity',
                    'engine.engine_settings.decoder_cache_capacity')})
            if cfg.max_num_seqs != 1:
                raise ValueError("JETSPEC_CHAIN_GRAPH=seg requires N=1")
            if (setting(self.run_config, 'engine.engine_settings.jetspec_verify_device', "0") != "1"
                    and current_engine().kind != 'reference'):
                raise ValueError("JETSPEC_CHAIN_GRAPH=seg requires device or reference verification")
        self.draft_cfg = self.create_draft_config(cfg, run_config=self.run_config)
        self.run_config = self.draft_cfg.run_config
        self.draft_cfg.method_settings = getattr(cfg, 'method_settings', None)
        super().__init__(self.draft_cfg, rank=rank, event=None, is_draft=True, num_tp_gpus=1, init_q=init_q, eagle3_model_class=eagle3_model_class, engine=self.engine, run_config=self.run_config)
        
        if self.config.use_eagle:
            assert self.config.jit_speculate, \
                "EAGLE requires jit_speculate=True (cache misses need draft activations)"

        self.jetspec_tap_mirrors = {}
        self._jetspec_tap_recv_ms = []
        self._jetspec_tap_payload_bytes = []
        self.jetspec_drafter = None
        self.jetspec_sequences = {}
        self.jetspec_candidate_sets = {}
        self.jetspec_block_objects = {}
        self.jetspec_candidate_ms = {}
        self._jetspec_candidate_send_id = 0
        self._jetspec_next_request_receive_id = 0
        self._jetspec_control_receive_id = 0
        self._jetspec_pending_rounds = {}
        self._jetspec_last_drafted_round = {}
        self._jetspec_sent_candidates = set()
        self._jetspec_wire_capacity = None
        self._chain_runtime = None
        self._pair_transport = None
        self.oracle_anchor = self.config.jetspec and anchor_mode(run_config=self.run_config) == 'oracle'
        if self.oracle_anchor and (setting(self.run_config, 'engine.engine_settings.rows') != '1'
                or self.config.jetspec_sync or self.config.jetspec_wait_ms != float('inf')):
            raise ValueError('oracle anchor requires asynchronous rows with infinite wait')
        if self.config.jetspec:
            self.jetspec_drafter = JetSpecDrafter.load(
                self.config.jetspec_checkpoint_dir,
                self.config.jetspec_adapter_config,
                self.config.jetspec_adapter_head,
                self.config.jetspec_target_release_weights,
                self.device,
            run_config=run_config)
            if setting(self.run_config, 'model.model_settings.draft_dtype') == "fp32":
                for module in (self.jetspec_drafter.head, self.jetspec_drafter.embed_tokens,
                               self.jetspec_drafter.lm_head, self.jetspec_drafter.adapter.head):
                    module.float()
                self.jetspec_drafter.dtype = torch.float32
            self.jetspec_rows = None
            if setting(self.run_config, 'engine.engine_settings.rows', "0") == "1":
                if self.config.jetspec_sync or (self.config.jetspec_wait_ms != float("inf")
                        and setting(self.run_config, 'engine.engine_settings.rows_fast', "0") != "1"
                        and setting(self.run_config, 'engine.engine_settings.w241_async_rows', "0") != "1"):
                    raise ValueError("JETSPEC_ROWS requires the waiting kth protocol")
                from jetspec.engines.jetspec_plus_plus.draft.live_rows import KthRowsDrafter
                backend = "sdpa" if self.jetspec_drafter.dtype == torch.float32 else "flashinfer"
                self.jetspec_rows = KthRowsDrafter(self.jetspec_drafter, backend=backend,
                    optimized=setting(self.run_config, 'engine.engine_settings.rows_fast', "0") == "1",
                    graphs=setting(self.run_config, 'engine.engine_settings.rows_graphs', "0") == "1", branches=branch_count(run_config=self.run_config),
                    f_graphs=setting(self.run_config, 'engine.engine_settings.f_graphs', "0") == "1", run_config=self.run_config, target_model_path=cfg.model)
                print(f"W234_ROWS=enabled F={branch_count(run_config=run_config)} live-kth adapter=shared-prefix", flush=True)
                # serving: the miss fallback row and the round-ahead draft deadline (both default off)
                self._w320_redraft = miss_redraft(run_config=run_config)
                self._w320_force = self._w320_redraft and miss_redraft_force(run_config=run_config)
                self._w320_clock = DeadlineClock(run_config=self.run_config) if draft_deadline() else None
                self.jetspec_rows.deadline = self._w320_clock
                if (self._w320_redraft or self._w320_clock is not None) and (
                        ((self.config.jetspec_wait_ms != float("inf")) or (self._w320_clock is not None and self.config.max_num_seqs != 1) or (anchor_mode(run_config=run_config) == 'oracle'))):
                    raise ValueError("these flags require infinite wait, no tree budget, no oracle; "
                                     "the deadline also requires N=1")
            if self.config.jetspec_draft_graphs and self.jetspec_rows is None:
                raise ValueError('retired graphed draft head requires a sealed historical read')
            print("W220_DRAFT_GRAPHS=disabled adapter=eager", flush=True)

        self.jetspec_catchup_drafter = self.jetspec_drafter
        if (self.config.jetspec and self.jetspec_rows is not None and self.jetspec_rows.optimized
                and self.config.jetspec_wait_ms != float('inf') and self.config.jetspec_draft_graphs):
            raise ValueError('retired graphed catchup head requires a sealed historical read')

        if self.is_draft and self.draft_async:
            self._prepare_async_draft()
            self._draft_step_times = []
            print(f'DraftRunner set up, starting draft_loop', flush=True)
            self.draft_loop(run_config=self.run_config)

    def _prepare_async_draft(self):
        """Prepare method-specific buffers before entering the async loop."""
        pass

    def draft_async_prefill(self, run_config=None):
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
        assert self.draft_async and self.is_draft
        request_pg = self.jetspec_request_pg if self.config.jetspec else self.async_pg

        # 1) Receive metadata then individual tensors
        # First recv metadata to learn sizes
        metadata = torch.zeros(5, dtype=torch.int64, device=self.device)
        dist.recv(metadata, src=0, group=request_pg)
        total_new_tokens, batch_size, max_blocks, use_eagle, eagle_act_dim = metadata.tolist()
        if use_eagle:
            assert eagle_act_dim == 3 * self.config.d_model_target, (
                f"EAGLE activation dimension {eagle_act_dim} does not match expected dimension 3 * {self.config.d_model_target}"
            )

        # 2) receive fused int64 payload (input_ids + num_tokens + draft_block_table)
        fused_total = total_new_tokens + batch_size + batch_size * max_blocks
        fused = recv_int64(request_pg, src=0, total_length=fused_total, device=self.device)
        off = 0
        input_ids = fused[off:off + total_new_tokens]; off += total_new_tokens
        num_tokens = fused[off:off + batch_size]; off += batch_size
        draft_block_table = fused[off:off + batch_size * max_blocks].view(batch_size, max_blocks).to(torch.int32); off += batch_size * max_blocks
        assert off == fused_total

        eagle_acts = None
        if use_eagle:
            eagle_acts = torch.zeros(
                total_new_tokens, eagle_act_dim, dtype=self.hf_config.torch_dtype, device=self.device,
            )
            dist.recv(eagle_acts, src=0, group=request_pg)

        if self.config.jetspec:
            fused_jetspec = recv_int64(
                request_pg,
                src=0,
                total_length=3 * batch_size,
                device=self.device,
            )
            seq_ids = fused_jetspec[:batch_size]
            prompt_indices = fused_jetspec[batch_size:2 * batch_size]
            recovery_tokens = fused_jetspec[2 * batch_size:]
            tap_lengths = torch.empty(
                batch_size, dtype=torch.int64, device=self.device
            )
            dist.recv(tap_lengths, src=0, group=request_pg)
            total_taps = int(tap_lengths.sum().item())
            tap_payload = torch.empty(
                total_taps,
                5,
                self.config.jetspec_tap_hidden_size,
                dtype=self.hf_config.torch_dtype,
                device=self.device,
            )
            if total_taps:
                dist.recv(tap_payload, src=0, group=request_pg)
            hybrid_admission = None
            if self._hybrid_pair_layout is not None:
                layout = self._hybrid_pair_layout
                hybrid_admission = []
                for length in tap_lengths.tolist():
                    tensors = (
                        torch.empty(layout.recurrent_shape, dtype=torch.float32,
                                    device=self.device),
                        torch.empty(layout.conv_shape, dtype=layout.dtype,
                                    device=self.device),
                        torch.empty((int(length), layout.kv_heads, layout.attention_dim),
                                    dtype=layout.dtype, device=self.device),
                        torch.empty((int(length), layout.kv_heads, layout.attention_dim),
                                    dtype=layout.dtype, device=self.device),
                    )
                    for payload in tensors:
                        dist.recv(payload, src=0, group=request_pg)
                    hybrid_admission.append(tensors)
            tap_chunks = []
            offset = 0
            for length in tap_lengths.tolist():
                next_offset = offset + int(length)
                tap_chunks.append(tap_payload[offset:next_offset])
                offset = next_offset

            from jetspec.engines.jetspec_plus_plus.engine import current_engine
            if current_engine().release_finished_requests:
                # Free every finished request's taps and state before admission.
                self.state.free_requests(
                    (self.jetspec_sequences, self.jetspec_tap_mirrors,
                     self.jetspec_candidate_sets, self.jetspec_block_objects),
                    seq_ids.tolist())

            id_offset = 0
            assert self.jetspec_drafter is not None
            for row, (seq_id, prompt_index, length, correction, taps) in enumerate(
                zip(
                    seq_ids.tolist(),
                    prompt_indices.tolist(),
                    num_tokens.tolist(),
                    recovery_tokens.tolist(),
                    tap_chunks,
                )
            ):
                prompt_ids = input_ids[id_offset:id_offset + int(length)]
                id_offset += int(length)
                committed_ids = torch.cat(
                    [
                        prompt_ids,
                        torch.tensor(
                            [correction], dtype=torch.int64, device=self.device
                        ),
                    ]
                ).reshape(1, -1)
                state = JetSpecSequence(
                    taps=taps,
                    committed_ids=committed_ids,
                    prompt_index=int(prompt_index),
                    staged=staged_context(getattr(self, 'run_config', None)))
                if current_engine().tag_sequence_ids:
                    state._jetspec_seq_id = int(seq_id)
                if hybrid_admission is not None:
                    recurrent, conv, key, value = hybrid_admission[row]
                    # The decoder computes in the draft dtype; only the recurrence stays fp32.
                    dtype = self.hf_config.torch_dtype
                    (state.hybrid_recurrent, state.hybrid_conv,
                     state.hybrid_prefix_key, state.hybrid_prefix_value) = (
                        recurrent, conv.to(dtype), key.to(dtype), value.to(dtype))
                    if setting(run_config, 'engine.engine_settings.chain_graph', '0') == '0':
                        from jetspec.engines.jetspec_plus_plus.wire.hybrid import seed_eager_hybrid_state
                        seed_eager_hybrid_state(state, capacity=self.config.max_model_len + 32)
                if self.jetspec_rows is not None:
                    self.jetspec_sequences[int(seq_id)] = state
                    self.jetspec_tap_mirrors[int(seq_id)] = taps.clone()
                    continue
                torch.cuda.synchronize()
                draft_started = time.perf_counter()
                if self.config.jetspec_sync:
                    blocks = self.jetspec_drafter.propose_sync(
                        state,
                        mode="kth",
                        hypotheses=list(range(self.config.jetspec_hyp)),
                        passes=self.config.jetspec_passes,
                    )
                else:
                    blocks = self.jetspec_drafter.propose(
                        state,
                        mode="kth",
                        k_hypotheses=list(range(self.config.jetspec_hyp)),
                        passes=self.config.jetspec_passes,
                    )
                torch.cuda.synchronize()
                candidate_set = CandidateSet(
                    seq_id=int(seq_id),
                    round_id=0,
                    blocks=_pack_jetspec_blocks(blocks),
                 run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                self.jetspec_sequences[int(seq_id)] = state
                self.jetspec_tap_mirrors[int(seq_id)] = taps.clone()
                self.jetspec_candidate_sets[int(seq_id)] = candidate_set
                self.jetspec_block_objects[int(seq_id)] = (0, blocks)
                self.jetspec_candidate_ms[(int(seq_id), 0)] = (
                    time.perf_counter() - draft_started
                ) * 1000
                self._jetspec_last_drafted_round[int(seq_id)] = 0
            assert id_offset == input_ids.numel()
            if self.jetspec_rows is not None:
                states = [self.jetspec_sequences[int(s)] for s in seq_ids.tolist()]
                torch.cuda.synchronize()
                started = time.perf_counter()
                oracle = {}
                if self.oracle_anchor:
                    pins = torch.full((len(states), 15), -1, dtype=torch.int64, device=self.device)
                    pins[:, 0] = recovery_tokens
                    oracle['oracle_predictions'] = pins
                outputs = self.jetspec_rows.propose(states,
                    k_hypotheses=[0] if self.oracle_anchor else admission_hypotheses(self.config.jetspec_hyp, run_config=run_config if run_config is not None else getattr(self, 'run_config', None)),
                    passes=self.config.jetspec_passes, initial=True, **oracle, run_config=run_config)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                for seq_id, blocks in zip(seq_ids.tolist(), outputs):
                    seq_id = int(seq_id)
                    self.jetspec_candidate_sets[seq_id] = (
                        (CandidateSet(
                            seq_id=seq_id, round_id=0,
                            blocks=oracle_candidate_blocks(blocks, hypotheses=self.config.jetspec_hyp), run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                            if self.oracle_anchor else _candidate_set_from_rows(seq_id, 0, blocks, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))))
                    self.jetspec_block_objects[seq_id] = (0, blocks)
                    self.jetspec_candidate_ms[(seq_id, 0)] = elapsed
                    self._jetspec_last_drafted_round[seq_id] = 0
            if not self.config.jetspec_sync:
                if self._jetspec_wire_capacity is None:
                    self._jetspec_wire_capacity = max(self.config.max_num_seqs, int(batch_size))
                if int(batch_size) > self._jetspec_wire_capacity:
                    raise RuntimeError("JetSpec prefill batch exceeds wire capacity")
                if self._chain_mode:
                    self._init_chain_runtime(int(seq_ids[0]), run_config=run_config)
                self._send_jetspec_candidates(seq_ids, run_config=run_config)
            return

        prefill_ctxt = self.prepare_prefill_ctxt(num_tokens, draft_block_table)

        # 5) set up context exactly like prepare_prefill() does:
        set_context(
            is_prefill=True,
            cu_seqlens_q=prefill_ctxt["cu_seqlens_q"],
            cu_seqlens_k=prefill_ctxt["cu_seqlens_k"],
            max_seqlen_q=prefill_ctxt["max_seqlen_q"],
            max_seqlen_k=prefill_ctxt["max_seqlen_k"],
            slot_mapping=prefill_ctxt["slot_map"],
            context_lens=None,
        ) # , block_tables=block_tables, commenting this out essentially removes prefix caching

        # 6) run the draft model in prefill mode
        positions = prefill_ctxt["positions"]
        if self.config.use_eagle:
            self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)
        else:
            self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)

        # 7) clean up
        reset_context()





    def _advance_batch_rows(self, payload, run_config=None):
        """C>1 rows: the returned round's commit, the chosen block's stage, and round+1's batched
        draft for every sequence of the request (port of the retired ``_advance_jetspec_rows``
        without the miss redraft, the deadline, the tree budget and device publish)."""
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting
        from jetspec.engines.jetspec_plus_plus.draft.batch_round import prefetch_batch, propose_batch
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import host_mark
        if getattr(self, '_w320_redraft', False) or getattr(self, '_w320_clock', None) is not None:
            raise ValueError('C>1 rows serve miss redraft off and the deadline off')
        if self.oracle_anchor or _device_publish(run_config):
            raise ValueError('C>1 rows serve neither the oracle anchor nor device publish')
        if not (e04_lean() and setting(run_config, 'method.method_settings.rowrule') == 'joint'):
            raise ValueError('C>1 rows require JETSPEC_E04_LEAN=1 and E04_ROWRULE=joint')
        seq_ids = [int(seq_id) for seq_id in payload["seq_ids"].tolist()]
        round_ids = [int(round_id) for round_id in payload["rounds"].tolist()]
        true_ks = [int(true_k) for true_k in payload["cache_keys"][:, 1].tolist()]
        corrections = [int(correction) for correction in payload["cache_keys"][:, 2].tolist()]
        previous = [self.jetspec_block_objects.get(seq_id) for seq_id in seq_ids]
        stale = [(seq_id, round_id, None if item is None else item[0])
                 for seq_id, round_id, item in zip(seq_ids, round_ids, previous)
                 if item is None or item[0] != round_id]
        if stale:
            raise RuntimeError(f"JetSpec candidate round mismatch (seq, requested, drafted): {stale}")
        if len(seq_ids) > 1:
            cache_packed_metadata_batch([item[1] for item in previous])
        pending = [(self.jetspec_sequences[seq_id], true_k, correction)
                   for seq_id, true_k, correction in zip(seq_ids, true_ks, corrections)
                   if self.jetspec_sequences[seq_id].pending_block is not None]
        if pending:
            commit_sequences_batch([item[0] for item in pending], [item[1] for item in pending],
                                   [item[2] for item in pending])
        selected = select_live_blocks_batch([item[1] for item in previous], true_ks, corrections)
        states, chosen, keys = [], [], []
        for row, (seq_id, round_id) in enumerate(zip(seq_ids, round_ids)):
            state = self.jetspec_sequences[seq_id]
            if state.pending_block is not None:
                mirror = self.jetspec_tap_mirrors[seq_id]
                if mirror.shape[0] != state.committed_ids.shape[1] - 1:
                    raise RuntimeError("JetSpec mirror/commit length mismatch")
                state.replace_taps(mirror)
            block, selection_miss = selected[row]
            state._jetspec_previous_selection_miss = selection_miss
            state._jetspec_seq_id = seq_id
            state._jetspec_candidate_round = round_id + 1
            states.append(state)
            chosen.append(block)
            keys.append((seq_id, round_id + 1))
        host_mark('pre_stage', run_config=run_config)
        started = time.perf_counter()
        self.jetspec_rows.stage(states, chosen, run_config=run_config)
        prefetch_batch(self.jetspec_rows, states, run_config=run_config)
        outputs = propose_batch(self.jetspec_rows, states,
                                k_hypotheses=list(range(draft_width(run_config=run_config) + 1)),
                                passes=self.config.jetspec_passes, run_config=run_config)
        host_mark('propose_ret', run_config=run_config)
        elapsed = (time.perf_counter() - started) * 1000
        keys_validated = False
        if len(outputs) > 1 and all(getattr(blocks, 'row_keys', None) is not None for blocks in outputs):
            validate_candidate_row_keys_batch(torch.stack([blocks.row_keys for blocks in outputs]),
                                              run_config=run_config)
            keys_validated = True
        for (seq_id, next_round), blocks in zip(keys, outputs):
            self.jetspec_candidate_sets[seq_id] = _candidate_set_from_rows(
                seq_id, next_round, blocks, keys_validated=keys_validated, run_config=run_config)
            self.jetspec_block_objects[seq_id] = (next_round, blocks)
            self.jetspec_candidate_ms[(seq_id, next_round)] = elapsed
            self._jetspec_last_drafted_round[seq_id] = next_round
        # The read joins every verified round to its proposal row (launch/runner.py join_populations)
        root = setting(run_config, 'engine.engine_settings.pred_serve_read_records')
        if root:
            lines = []
            for (seq_id, next_round), blocks in zip(keys, outputs):
                metrics = blocks.head_metrics
                lines.append(json.dumps(dict(
                    seq_id=seq_id, candidate_round=next_round, pass_label=str(metrics['kpred_pass']),
                    rows_requested=int(metrics['rows_requested']),
                    rows_published=int(metrics['rows_published']),
                    rows_truncated=int(metrics['rows_truncated']), refresh_calls=0),
                    separators=(',', ':')))
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, f'{os.getpid()}.jsonl'), 'a') as handle:
                handle.write('\n'.join(lines) + '\n')

    def _send_jetspec_candidates(
        self,
        seq_ids: torch.Tensor | list[int],
        *,
        candidate_sets: list[CandidateSet | None] | None = None,
        coalesced: int = 0,
        gaps: dict[tuple[int, int], int] | None = None,
     run_config=None) -> None:
        """Push one fixed-capacity response into the target's posted mailbox."""
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
        seq_ids = [int(seq_id) for seq_id in seq_ids]
        if candidate_sets is None:
            candidate_sets = [
                self.jetspec_candidate_sets.get(seq_id) for seq_id in seq_ids
            ]
        if len(candidate_sets) != len(seq_ids):
            raise ValueError("candidate response rows must match sequence ids")
        capacity = (
            len(seq_ids)
            if self.config.jetspec_sync
            else self._jetspec_wire_capacity
        )
        if capacity is None:
            raise RuntimeError("JetSpec candidate capacity is unknown before prefill")
        if len(candidate_sets) > capacity:
            raise ValueError("candidate response exceeds configured capacity")
        candidate_sets = list(candidate_sets) + [None] * (
            capacity - len(candidate_sets)
        )
        if self.jetspec_rows is not None and self.jetspec_rows.optimized and not self.config.jetspec_sync:
            from jetspec.engines.jetspec_plus_plus.draft.live_rows import CandidateWire
            if not hasattr(self, '_jetspec_packed_mailbox'):
                self._jetspec_packed_mailbox=CandidateWire(capacity,self.config.jetspec_hyp,device=self.device,branches=branch_count(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)), run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            mailbox = self._jetspec_packed_mailbox
            direct = (_device_publish(run_config) and getattr(mailbox, '_device_publish_ready', False)
                      and all(candidate is None or candidate.blocks.data_ptr() ==
                              mailbox.buffers[2][index].data_ptr()
                              for index, candidate in enumerate(candidate_sets)))
            if _device_publish(run_config) and getattr(mailbox, '_device_publish_ready', False) and not direct:
                raise RuntimeError('JETSPEC_DEVICE_PUBLISH candidate order differs from packet')
            packet = (mailbox.pack_published if direct else mailbox.pack)(
                candidate_sets, times=self.jetspec_candidate_ms,
                coalesced=coalesced, gaps={} if gaps is None else gaps)
            mailbox._device_publish_ready = False
            tensors=(packet,)
        else:
            available, candidate_keys, candidate_blocks = pack_candidate_sets(
                candidate_sets,
                hypotheses=self.config.jetspec_hyp,
                branches=branch_count(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)),
                device=self.device,
             run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            candidate_ms = torch.zeros(
                capacity, dtype=torch.float32, device=self.device
            )
            telemetry = torch.zeros(
                capacity, 2, dtype=torch.int64, device=self.device
            )
            if capacity:
                telemetry[0, 0] = int(coalesced)
            gaps = {} if gaps is None else gaps
            for row, candidate_set in enumerate(candidate_sets):
                if candidate_set is None:
                    continue
                key = (candidate_set.seq_id, candidate_set.round_id)
                candidate_ms[row] = self.jetspec_candidate_ms.get(key, 0.0)
                telemetry[row, 1] = int(gaps.get(key, 0))
            tensors = (
                available,
                candidate_keys,
                candidate_blocks,
                candidate_ms,
            )
            if not self.config.jetspec_sync:
                tensors += (telemetry,)
        if (len(tensors) == 1 and True
                and self.config.jetspec_wait_ms == float("inf")):
            packet = tensors[0]
            if (len(seq_ids) > 1 and self.jetspec_rows is not None
                    and self.jetspec_rows.optimized and torch.device(self.device).type == 'cuda'):
                # The whole candidate wire travels home once. Keep its keyed pins for the
                # following round's miss check, instead of copying each request's blocks.
                from jetspec.engines.jetspec_plus_plus.draft.live_rows import candidate_wire_views
                mailbox = self._jetspec_packed_mailbox
                source = mailbox.wire_all if direct else packet
                host_name = 'host_wire_checked' if direct else 'host_wire'
                if not hasattr(mailbox, host_name):
                    setattr(mailbox, host_name, torch.empty(source.shape,
                            dtype=source.dtype, pin_memory=True))
                host = getattr(mailbox, host_name)
                host.copy_(source, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                event.synchronize()
                packet = mailbox.published_host(host, run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) if direct else host
                if direct:
                    for index, candidate_set in enumerate(candidate_sets):
                        if candidate_set is None:
                            continue
                        entry = self.jetspec_block_objects.get(candidate_set.seq_id)
                        if entry is None or entry[0] != candidate_set.round_id:
                            raise RuntimeError('JETSPEC_DEVICE_PUBLISH metadata owner differs from packet')
                        blocks = entry[1]
                        blocks._host_metadata = mailbox.host_metadata[index, :len(blocks)].clone()
                        fallback = getattr(blocks, '_fallback_blocks', None)
                        if fallback is not None:
                            fallback._host_metadata = mailbox.host_metadata[index, :1].clone()
                if mailbox.keyed:
                    views = candidate_wire_views(packet, capacity, self.config.jetspec_hyp,
                                                 branch_count(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)), run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                    if direct:
                        checked = getattr(self, '_jetspec_publish_checked', 0)
                        if setting(run_config, 'engine.engine_settings.device_publish_check', '0') == '1' and checked < 200:
                            if not hasattr(self, '_jetspec_publish_reference'):
                                self._jetspec_publish_reference = CandidateWire(
                                    capacity, self.config.jetspec_hyp, device=self.device,
                                    branches=branch_count(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)), run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                            reference = self._jetspec_publish_reference.pack(
                                candidate_sets, times=self.jetspec_candidate_ms,
                                coalesced=coalesced, gaps={} if gaps is None else gaps).cpu()
                            same = torch.equal(packet.view(torch.uint8), reference.view(torch.uint8))
                            self._jetspec_publish_checked = checked + 1
                            print('W641_PUBLISH_CHECK ' + json.dumps(dict(
                                rounds=self._jetspec_publish_checked, mismatches=int(not same))), flush=True)
                            if not same:
                                raise AssertionError('JETSPEC_DEVICE_PUBLISH packet differs from legacy')
                    pins = torch.cat((views[5], views[2][:, :, :1]), dim=2).tolist()
                    pin_home = self.__dict__.setdefault('_jetspec_pin_home', {})
                    for index, candidate_set in enumerate(candidate_sets):
                        if candidate_set is not None:
                            pin_home[candidate_set.seq_id] = (candidate_set.round_id, pins[index])
            if setting(run_config, 'engine.engine_settings.chain_tail_trace', '0') == '1':
                print(f"MAILBOX_ID side=draft event=candidate_send id={self._jetspec_candidate_send_id} sets={[(c.seq_id, c.round_id) if c is not None else None for c in candidate_sets]} branch=admission", flush=True)
            self.jetspec_mailbox_store.set(
                f"candidate_payload_{self._jetspec_candidate_send_id}",
                packet.cpu().numpy().tobytes())
        else:
            for tensor in tensors:
                dist.send(tensor, dst=0, group=self.jetspec_candidate_pg)
        if not self.config.jetspec_sync and self.config.jetspec_wait_ms != float("inf"):
            self.jetspec_mailbox_store.set(
                f"candidate_{self._jetspec_candidate_send_id}", "1"
            )
        for candidate_set in candidate_sets:
            if candidate_set is not None:
                self._jetspec_sent_candidates.add(
                    (candidate_set.seq_id, candidate_set.round_id)
                )
        if not self.config.jetspec_sync:
            self._jetspec_candidate_send_id += 1
        _flush_deferred()


    def _receive_jetspec_request(self, run_config=None):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
        key = f"request_{self._jetspec_control_receive_id}"
        control = self.jetspec_mailbox_store.get(key)
        self._jetspec_control_receive_id += 1
        if (control == b'1' and getattr(self, '_hybrid_pair_layout', None) is not None
                and getattr(self.config, 'max_num_seqs', 1) == 1
                and setting(run_config, 'engine.engine_settings.chain_graph', '0') == '0'):
            from jetspec.engines.jetspec_plus_plus.wire.hybrid import receive_eager_hybrid_outcome
            return receive_eager_hybrid_outcome(self, receive=lambda tensor:
                dist.recv(tensor, src=0, group=self.jetspec_request_pg))
        wire = torch.frombuffer(bytearray(control), dtype=torch.int64)
        payload = unpack_request_wire(wire, capacity=self._jetspec_wire_capacity,
            tree=bool(False), oracle=self.oracle_anchor)
        lengths = payload["tap_lengths"]
        total = int(lengths.sum())
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        taps = torch.empty(total, 5, self.config.jetspec_tap_hidden_size,
            dtype=self.hf_config.torch_dtype, device=self.device)
        if total:
            dist.recv(taps, src=0, group=self.jetspec_request_pg)
        if self.config.jetspec_wait_ms == float("inf"):
            append_jetspec_tap_mirrors(
                self.jetspec_tap_mirrors, payload["seq_ids"], taps, lengths)
        if (getattr(self, '_hybrid_pair_layout', None) is not None
                and getattr(self.config, 'max_num_seqs', 1) > 1):
            from jetspec.engines.jetspec_plus_plus.wire.hybrid import receive_batch_hybrid_outcomes
            receive_batch_hybrid_outcomes(self, payload, receive=lambda tensor:
                dist.recv(tensor, src=0, group=self.jetspec_request_pg))
        end.record()
        if kernel_enabled('request_lean', '1'):
            # serving JETSPEC_REQUEST_LEAN=1: no host wait on the tap receive (the device orders the
            # consumers after it); its time is read one round later, once the events are done
            previous = getattr(self, '_jetspec_tap_events', None)
            if previous is not None:
                self._jetspec_tap_recv_ms.append(previous[0].elapsed_time(previous[1]))
            self._jetspec_tap_events = (start, end)
        else:
            end.synchronize()
            self._jetspec_tap_recv_ms.append(start.elapsed_time(end))
        self._jetspec_tap_payload_bytes.append(
            lengths.numel() * lengths.element_size() + taps.numel() * taps.element_size())
        if setting(run_config, 'engine.engine_settings.w328_defer_taps'):
            from jetspec.engines.jetspec_plus_plus.serving.compile_pin import digest_defer
            digest_defer("taps", 'engine.engine_settings.w328_defer_taps', run_config=run_config, taps=taps)
        if setting(run_config, 'engine.engine_settings.w328_digest_taps'):
            # serving diagnostic: the received taps' bytes (host sync), first n requests
            from jetspec.engines.jetspec_plus_plus.serving.compile_pin import digest_log
            digest_log("taps", path='engine.engine_settings.w328_digest_taps', run_config=run_config, taps=taps, lengths=lengths)
        payload["tap_payload"] = taps
        return payload

    def _service_spec_request(self, run_config=None):
        """Receives a speculation request, serves it from cache, and sends results back in a single response."""
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        if self.config.jetspec and not self.config.jetspec_sync and not self.config.use_eagle:
            return None, self._receive_jetspec_request(run_config=run_config)
        request_pg = self.jetspec_request_pg if self.config.jetspec else self.async_pg
        meta = self.recv_tensor((3,), torch.int64, group=request_pg)
        B, K, F = meta.tolist()

        # Receive all request payload in one fused int64 burst (includes temperatures encoded as int64)
        max_blocks = self.config.max_blocks
        fused_total = (3 * B) + B + (B * max_blocks) + B  # +B for temps_as_int64
        if self.config.jetspec:
            fused_total += B
            if self.oracle_anchor:
                fused_total += B * 16
        fused_req = recv_int64(request_pg, src=0,
                               total_length=fused_total, device=self.device)
        off = 0
        cache_keys = fused_req[off:off + (3 * B)].view(B, 3)
        off += 3 * B
        seq_ids = cache_keys[:, 0]
        num_tokens = fused_req[off:off + B].to(torch.int64)
        off += B
        draft_block_tables = fused_req[off:off + B *
                                       max_blocks].view(B, max_blocks).to(torch.int32)
        off += B * max_blocks
        temps_as_int64 = fused_req[off:off + B]
        off += B
        jetspec_rounds = None
        if self.config.jetspec:
            jetspec_rounds = fused_req[off:off + B]
            off += B
        accepted_tree_ids = None
        oracle_verify = None
        if self.oracle_anchor:
            oracle_verify = fused_req[off:off + B * 16].view(B, 16)
            off += B * 16
        assert off == fused_total
        temperatures = temps_as_int64.to(torch.int32).view(torch.float32)

        target_recovery_activations = torch.zeros(
            B, 3 * self.config.d_model_target, dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle else None

        extend_counts = None
        extend_eagle_acts = None
        extend_token_ids = None

        if self.config.use_eagle:
            dist.recv(target_recovery_activations, src=0, group=request_pg)

            # Receive extend data for fused glue decode
            act_dim = 3 * self.config.d_model_target
            extend_counts = torch.zeros(B, dtype=torch.int64, device=self.device)
            extend_eagle_acts = torch.zeros(B, K, act_dim, dtype=self.hf_config.torch_dtype, device=self.device)
            extend_token_ids = torch.zeros(B, K, dtype=torch.int64, device=self.device)
            dist.recv(extend_counts, src=0, group=request_pg)
            dist.recv(extend_eagle_acts, src=0, group=request_pg)
            dist.recv(extend_token_ids, src=0, group=request_pg)

            if self.config.verbose:
                recovery_tokens_target = cache_keys[:, 2].clone()
                print(f"\n{'='*80}", flush=True)
                print(f"[CACHE REQUEST] Batch size: {B}, Spec depth: {K}", flush=True)
                for i in range(B):
                    seq_id = cache_keys[i, 0].item()
                    keep_idx = cache_keys[i, 1].item()
                    rec_token_target = recovery_tokens_target[i].item()
                    rec_token_text = self.tokenizer.decode([rec_token_target])
                    n_ext = extend_counts[i].item()
                    print(f"  Seq {seq_id}: keep_idx={keep_idx}, recovery_token={rec_token_target} ('{rec_token_text}'), n_ext={n_ext}", flush=True)
                print(f"{'='*80}\n", flush=True)

        if self.config.jetspec_taps:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            tap_lengths = torch.empty(B, dtype=torch.int64, device=self.device)
            dist.recv(tap_lengths, src=0, group=request_pg)
            total_taps = int(tap_lengths.sum().item())
            tap_payload = torch.empty(
                total_taps,
                5,
                self.config.jetspec_tap_hidden_size,
                dtype=self.hf_config.torch_dtype,
                device=self.device,
            )
            if total_taps > 0:
                dist.recv(tap_payload, src=0, group=request_pg)
            if (
                self.config.jetspec_sync
                or self.config.jetspec_wait_ms == float("inf")
            ):
                append_jetspec_tap_mirrors(
                    self.jetspec_tap_mirrors, seq_ids, tap_payload, tap_lengths
                )
            end_event.record()
            end_event.synchronize()
            self._jetspec_tap_recv_ms.append(
                start_event.elapsed_time(end_event)
            )
            self._jetspec_tap_payload_bytes.append(
                tap_lengths.numel() * tap_lengths.element_size()
                + tap_payload.numel() * tap_payload.element_size()
            )

        if self.config.jetspec:
            assert jetspec_rounds is not None
            jetspec_payload = {
                "cache_keys": cache_keys,
                "seq_ids": seq_ids,
                "rounds": jetspec_rounds,
                "tap_lengths": tap_lengths,
                "tap_payload": tap_payload,
                "accepted_tree_ids": accepted_tree_ids,
            }
            if self.oracle_anchor:
                jetspec_payload['oracle_verify'] = oracle_verify
            return None, jetspec_payload

    def prepare_prefill_ctxt(
        self,
        num_tokens: torch.Tensor,  # [B]
        draft_block_table: torch.Tensor,  # [B, max_blocks]
    ) -> dict:
        """
        Prepare context for prefill forward pass.
        """
        B = num_tokens.shape[0]
        total = num_tokens.sum().item()
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(num_tokens, dim=0)
        batch_indices = torch.arange(B, device=self.device, dtype=torch.int64).repeat_interleave(num_tokens)
        positions = torch.arange(total, device=self.device, dtype=torch.int64) - cu_seqlens_q[:-1].to(torch.int64).repeat_interleave(num_tokens)
        max_seqlen_q = num_tokens.max().item()

        # Calculate block indices and offsets for ALL positions
        block_indices = (positions // self.block_size).to(torch.int64)
        offsets = (positions % self.block_size).to(torch.int32)

        # Get block IDs for each position from dbt
        block_ids = draft_block_table[batch_indices, block_indices]

        # Calculate slot_map for each position
        slot_map = (block_ids * self.block_size + offsets).to(torch.int32)

        return {
            "positions": positions,
            "slot_map": slot_map,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_q.clone(),
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_q,
        }

    
























    @torch.inference_mode()
    def _warm_jetspec_session(self, run_config=None):
        """Exercise the real T module before any mailbox request is admitted."""
        from jetspec.engines.jetspec_plus_plus.profiles import transport_enabled
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
        if setting(run_config, 'benchmark.benchmark_settings.w240_warmup', '0') != '1' or not self.config.jetspec:
            raise RuntimeError('session warm-up requires the explicit measurement flag')
        if transport_enabled('pair_outcome_push', run_config=run_config):
            from jetspec.engines.jetspec_plus_plus.wire.pair_transport import PairOutcomeTransport

            self._pair_transport = PairOutcomeTransport(
                store=self.jetspec_mailbox_store, group=self.jetspec_request_pg,
                device=self.device, lanes=self.config.speculate_k + 1,
                hidden=self.config.jetspec_tap_hidden_size,
                tap_dtype=self.jetspec_drafter.dtype,
                envelope_bytes=(None if getattr(self, "_hybrid_pair_layout", None) is None else
                                self._hybrid_pair_layout.outcome_bytes))
            self._chain_inbox.transport = self._pair_transport
        d=self.jetspec_drafter
        state=JetSpecSequence(
            taps=torch.zeros(32,5,self.config.jetspec_tap_hidden_size,device=d.device,dtype=d.dtype),
            committed_ids=torch.zeros(1,33,device=d.device,dtype=torch.long),prompt_index=240, staged=staged_context(getattr(self, 'run_config', None)))
        # Admission tail seeding is guarded by the live sequence id. Exercise
        # that path in cold warmup, before any measured request is admitted.
        state._jetspec_seq_id = -1
        ks=admission_hypotheses(self.config.jetspec_hyp, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        if self.oracle_anchor:
            ks = [k for k in ks if k < 15]
        if self.jetspec_rows is not None:
            blocks=self.jetspec_rows.propose([state],k_hypotheses=ks,passes=self.config.jetspec_passes,initial=True, run_config=run_config)
            self.jetspec_rows.stage([state],[select_live_block(blocks[0],true_k=ks[0],correction=-1)], run_config=run_config)
            next_ks=admission_hypotheses(self.config.jetspec_hyp,
                block=state.pending_block, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            oracle = {}
            if self.oracle_anchor:
                next_ks = [k for k in next_ks if k < 15] or [0]
                oracle['oracle_predictions'] = torch.zeros(1, 15, dtype=torch.int64, device=d.device)
            self.jetspec_rows.propose([state],k_hypotheses=next_ks,
                passes=self.config.jetspec_passes, initial=True, **oracle, run_config=run_config)
            lengths = json.loads(setting(run_config, 'benchmark.benchmark_settings.warmup_prefix_lengths', '[]'))
            # One prewarm pass per distinct length; a 128-prompt file has 65-77 of them.
            if (not isinstance(lengths, list)
                    or any(type(length) is not int or length < 2
                           or length + 16 > self.config.max_model_len for length in lengths)
                    or len(set(lengths)) > 256):
                raise ValueError('JETSPEC_WARMUP_PREFIX_LENGTHS must be at most 256 distinct valid prompt lengths')
            for length in sorted(set(lengths)):
                if length == 32:
                    continue
                seeded = JetSpecSequence(
                    taps=torch.zeros(length, 5, self.config.jetspec_tap_hidden_size,
                                     device=d.device, dtype=d.dtype),
                    committed_ids=torch.zeros(1, length + 1, device=d.device, dtype=torch.long),
                    prompt_index=240, staged=staged_context(getattr(self, 'run_config', None)))
                seeded._jetspec_seq_id = -1
                seeded_blocks = self.jetspec_rows.propose(
                    [seeded], k_hypotheses=ks, passes=self.config.jetspec_passes, initial=True, run_config=run_config)
                self.jetspec_rows.stage([seeded], [select_live_block(
                    seeded_blocks[0], true_k=ks[0], correction=-1)], run_config=run_config)
                seeded_ks = admission_hypotheses(self.config.jetspec_hyp,
                                             block=seeded.pending_block, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                if self.oracle_anchor:
                    seeded_ks = [k for k in seeded_ks if k < 15] or [0]
                self.jetspec_rows.propose([seeded], k_hypotheses=seeded_ks,
                    passes=self.config.jetspec_passes, initial=True, **oracle, run_config=run_config)
            if lengths:
                print(f'W409_PREFIX_SHAPE_PREWARM lengths={sorted(set(lengths))}', flush=True)
            # Row counts are Triton constexprs: compile every size a live round can bring
            # (an anchor-prefix append of up to a full block, a rebuilt adapter cache)
            # here, not inside a measured round.
            decoder = getattr(self.jetspec_rows, 'decoder', None)
            if decoder is not None and hasattr(decoder, 'prewarm_prefix_appends'):
                def throwaway(length):
                    sequence = JetSpecSequence(
                        taps=torch.zeros(length, 5, self.config.jetspec_tap_hidden_size,
                                         device=d.device, dtype=d.dtype),
                        committed_ids=torch.zeros(1, length + 1, device=d.device,
                                                  dtype=torch.long),
                        prompt_index=240, staged=staged_context(getattr(self, 'run_config', None)))
                    sequence._jetspec_seq_id = -1
                    return sequence
                counts = decoder.prewarm_prefix_appends(throwaway, range(1, draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 3))
                print(f'A03_DECODER_APPEND_PREWARM counts=1..{max(counts, default=0)}', flush=True)
            head = getattr(getattr(d, 'adapter', None), 'head', None)
            if head is not None and hasattr(head, 'prewarm_rebuild_rows'):
                buckets = head.prewarm_rebuild_rows(state.taps)
                print(f'A03_ADAPTER_REBUILD_PREWARM buckets={buckets}', flush=True)
            from jetspec.engines.jetspec_plus_plus.draft.batch_round import batch_round_enabled, prewarm_batch_graphs
            if self.config.max_num_seqs > 1 and batch_round_enabled(self.jetspec_rows, run_config=run_config):
                def batch_sequence(length):
                    sequence = JetSpecSequence(
                        taps=torch.zeros(length, 5, self.config.jetspec_tap_hidden_size,
                                         device=d.device, dtype=d.dtype),
                        committed_ids=torch.zeros(1, length + 1, device=d.device, dtype=torch.long),
                        prompt_index=240, staged=staged_context(getattr(self, 'run_config', None)))
                    sequence._jetspec_seq_id = -2  # a warm-up id until the prewarm assigns its own
                    return sequence
                prewarm_batch_graphs(
                    self.jetspec_rows, batch_sequence, max_batch=self.config.max_num_seqs,
                    max_model_len=self.config.max_model_len, initial_ks=ks,
                    round_ks=list(range(draft_width(run_config=run_config) + 1)),
                    passes=self.config.jetspec_passes, run_config=run_config)
            if setting(run_config, 'engine.engine_settings.c1_graph', '0') == '1':
                head = getattr(d.adapter, 'head', None)
                if head is not None and hasattr(head, 'prewarm_live_query_window'):
                    head.prewarm_live_query_window(15, run_config=run_config)
                    if setting(run_config, 'engine.engine_settings.keylen_bucket', '0') == '1':
                        head.prewarm_keylen_graphs(run_config=run_config)
        else:
            blocks=d.propose(state,mode='kth',k_hypotheses=ks,passes=self.config.jetspec_passes)
            d.stage(state,blocks[0])
            d.propose_protocol(state,mode='kth',k_hypotheses=ks,passes=self.config.jetspec_passes)
        if setting(run_config, 'engine.engine_settings.c1_loop', '0') == '1':
            # The synthetic state can miss the live D1 window-mask path. Its first
            # call builds this CUDA extension (~42 s), so build it before decode.
            from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _loop_extension
            _loop_extension()
        if self._chain_mode:
            # Capture every segment before the measured request. The synthetic
            # admission only supplies pointers and shapes; real prefill re-seeds
            # the same buffers without recapturing.
            warm_state = JetSpecSequence(
                taps=torch.zeros(32, 5, self.config.jetspec_tap_hidden_size,
                                 device=d.device, dtype=d.dtype),
                committed_ids=torch.zeros(1, 33, dtype=torch.int64, device=d.device),
                prompt_index=240, staged=staged_context(getattr(self, 'run_config', None)))
            # Eval admission stages the decoder swap; the hybrid decoder warms only seq -1.
            warm_state._jetspec_seq_id = -1
            warm_rows = self.jetspec_rows.propose(
                [warm_state], k_hypotheses=ks,
                passes=self.config.jetspec_passes, initial=True, run_config=run_config)[0]
            warm_id = -1
            self.jetspec_sequences[warm_id] = warm_state
            self.jetspec_candidate_sets[warm_id] = _candidate_set_from_rows(
                warm_id, 0, warm_rows, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            self.jetspec_block_objects[warm_id] = (0, warm_rows)
            try:
                chain_warm_started = time.perf_counter()
                self._init_chain_runtime(warm_id, run_config=run_config)
                chain_warm_seconds = time.perf_counter() - chain_warm_started
            finally:
                self.jetspec_sequences.pop(warm_id, None)
                self.jetspec_candidate_sets.pop(warm_id, None)
                self.jetspec_block_objects.pop(warm_id, None)
                self._chain_runtime = None
            from jetspec.engines.jetspec_plus_plus.engine import current_engine
            # A read can skip per-length admission prewarm when eager admission is byte-equal.
            if (setting(run_config, 'engine.engine_settings.chain_admit_graph', '0') == '1'
                    and setting(run_config, 'engine.engine_settings.chain_admit_batch', '0') == '1'
                    and not current_engine().skip_admission_prewarm):
                read = json.loads(setting(run_config, 'engine.engine_settings.chain_warm_read') or '{}')
                prompt_lengths = read.get('prompt_lengths') or json.loads(
                    setting(run_config, 'benchmark.benchmark_settings.warmup_prefix_lengths', '[]'))
                admit_rows = min(len(dict.fromkeys(int(k) for k in ks)),
                                 int(setting(run_config, 'engine.engine_settings.rows_cap', '128')))
                for length in sorted(set(int(value) for value in prompt_lengths)):
                    self.jetspec_rows.prewarm_admission_graph(length, admit_rows, run_config=run_config)
            decoder = self.jetspec_rows.decoder
            window_states = getattr(decoder, '_window_graph_states', {})
            if window_states:
                window_captures = sum(len(state['graphs']) for state in window_states.values())
            else:
                state = getattr(decoder, '_window_graph_state', None)
                window_captures = 0 if state is None else len(state['graphs'])
            segment_captures = sum(len(entry.graphs) for entry in self._chain_graph.entries.values())
            allocated_stamp = (f' allocated_bytes={torch.cuda.memory_allocated(self.device)}'
                               if setting(run_config, 'engine.engine_settings.chain_miss_redraft', '0') == '1'
                               else '')
            warm_stamp = (f' warm_seconds={chain_warm_seconds:.3f}'
                          if setting(run_config, 'engine.engine_settings.chain_warm_reachable', '0') == '1' else '')
            print('CHAIN_GRAPH_PREWARM '
                  f'keys={len(self._chain_graph.entries)} '
                  f'segment_captures={segment_captures} '
                  f'decoder_window_captures={window_captures} '
                  f'reserved_bytes={torch.cuda.memory_reserved(self.device)}'
                  f'{allocated_stamp}'
                  f'{warm_stamp}', flush=True)
        torch.cuda.synchronize()
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        if 'mxfp4_w4a8' in (current_engine().precision_draft, current_engine().precision_lm_head):
            # W4A8 coverage: which GEMM each draft-lane site captured, and the rounded rows.
            from jetspec.engines.jetspec_plus_plus.models.target_linear import report_nvfp4_census, report_w4a8_sites
            report_w4a8_sites()
            report_nvfp4_census(dict(drafter=d, rows=self.jetspec_rows), self.device)
        print('W240_T_MODULE_WARMUP=complete synthetic_state_unpublished',flush=True)

    # new one, with true asynchrony
    def _check_draft_compile_pin_served(self, engine=None):
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        from jetspec.engines.jetspec_plus_plus.profiles import get_profile
        engine = engine or getattr(self, 'engine', None) or current_engine()
        if not get_profile(engine.profile).compile_pin:
            return False
        from jetspec.engines.jetspec_plus_plus.serving.compile_pin import process_check
        import jetspec.engines.jetspec_plus_plus.runtime.run_manifest as run_manifest
        process_check("served")
        run_manifest.record_value("compile_pin_served_draft", True, run_config=getattr(self, 'run_config', None))
        return True

    @torch.inference_mode()
    def _init_chain_runtime(self, seq_id, run_config=None):
        """Bind the admission sequence and capture every reachable chain key."""
        from jetspec.engines.jetspec_plus_plus.profiles import transport_enabled
        from jetspec.engines.jetspec_plus_plus.trace.anatomy_lite import enabled as anatomy_enabled
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
        from jetspec.engines.jetspec_plus_plus.draft.graph import hostpath_setting
        cached_setting = hostpath_setting(run_config)
        if cached_setting is not None:
            setting = cached_setting
        from jetspec.engines.jetspec_plus_plus.wire.contract import ChainKey
        from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
        from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
        from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomePacket
        from jetspec.engines.jetspec_plus_plus.wire.contract import allocate_token_buffers
        from jetspec.engines.jetspec_plus_plus.draft.graph import ChainGraph
        from jetspec.engines.jetspec_plus_plus.draft.graph import chain_fixed_choice
        from jetspec.engines.jetspec_plus_plus.draft.graph import reachable_chain_capture
        from jetspec.engines.jetspec_plus_plus.draft.graph import reachable_chain_lengths
        from jetspec.engines.jetspec_plus_plus.wire.contract import HASH_WEIGHTS
        from jetspec.engines.jetspec_plus_plus.draft.state import ChainLengthTracker
        from jetspec.engines.jetspec_plus_plus.draft.state import ChainRngTable
        from jetspec.engines.jetspec_plus_plus.draft.state import ChainTapBuffers
        from jetspec.engines.jetspec_plus_plus.wire.transport import OutcomeReceiveRing
        from jetspec.engines.jetspec_plus_plus.wire.transport import PushedOutcomeRing
        from jetspec.engines.jetspec_plus_plus.draft.state import PublishedRowRing
        from jetspec.engines.jetspec_plus_plus.wire.contract import hash_last16
        from jetspec.engines.jetspec_plus_plus.draft.driver import ChainDraftRecorder
        from jetspec.engines.jetspec_plus_plus.draft.driver import ChainRoundDriver
        from jetspec.engines.jetspec_plus_plus.draft.driver import configure_host_lean_entry
        from jetspec.engines.jetspec_plus_plus.draft.segments import build_served_entry
        from jetspec.engines.jetspec_plus_plus.draft.segments import copy_front_state
        from jetspec.engines.jetspec_plus_plus.draft.segments import retire_chain_runtime
        from jetspec.engines.jetspec_plus_plus.draft.segments import reseed_served_graph
        from jetspec.engines.jetspec_plus_plus.draft.segments import seed_front_from_admission
        from jetspec.engines.jetspec_plus_plus.draft.segments import seed_row_ring_from_admission
        from jetspec.engines.jetspec_plus_plus.draft.segments import shadow_state_tensors
        from jetspec.engines.jetspec_plus_plus.draft.segments import stage_admission_rows
        from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
        from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds

        retire_chain_runtime(self, run_config=run_config)

        if self.jetspec_rows is None or seq_id not in self.jetspec_sequences:
            raise RuntimeError("chain admission requires one decoder row sequence")
        state = self.jetspec_sequences[seq_id]
        candidate = self.jetspec_candidate_sets[seq_id]
        _, blocks = self.jetspec_block_objects[seq_id]
        lanes = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_capacity
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_count
        R = rows_min_capacity(int(required_setting(run_config, 'method.method_settings.rows_rule_cap')), run_config=run_config)
        if R not in (4, 8, 16):
            raise ValueError("chain served row cap must be 4, 8 or 16")
        capacity = self.config.max_model_len + lanes + 1
        initial_ids = state.committed_ids.reshape(-1).detach().clone()
        initial_taps = state.taps.detach().clone()
        L_admit = initial_taps.shape[0]
        if initial_ids.numel() != L_admit + 1:
            raise ValueError("chain admission must have one untapped query token")
        state._jetspec_seq_id = int(seq_id)  # real admission, not the synthetic warm-up state
        host_lean = (kernel_enabled('chain_admit_host_lean', '1')
                     and setting(run_config, 'engine.engine_settings.chain_admit_batch', "0") == "1")
        host_check = host_lean and setting(run_config, 'engine.engine_settings.chain_admit_host_lean_check', "0") == "1"
        # Copy the committed IDs before the row stage queues admission work.
        initial_ids_host = initial_ids.cpu() if host_lean else None
        wire_row_cap = int(required_setting(run_config, 'engine.engine_settings.rows_cap'))
        admission_hidden, admission_tau = stage_admission_rows(
            self.jetspec_rows, state, blocks, candidate, row_cap=wire_row_cap, run_config=run_config)
        if host_lean:
            heads_dir = required_setting(run_config, 'model.model_settings.heads_dir')
            cached = getattr(self, "_chain_admit_thresholds", None)
            if cached is None or cached[0] != heads_dir:
                cached = self._chain_admit_thresholds = (heads_dir, read_thresholds(run_config))
            thresholds = cached[1]
        else:
            thresholds = read_thresholds(run_config)
        dataset = required_setting(run_config, 'benchmark.benchmark_settings.dataset')
        rule = dataset_entry(thresholds["e04_rowrule_joint"], dataset)
        from jetspec.engines.jetspec_plus_plus.runtime.pass_labels import chain_tau_values
        tau_values = chain_tau_values(rule['tau'], bundle=required_setting(run_config, 'model.model_settings.heads_dir'), run_config=self.run_config)
        if host_lean:
            from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor
            tau_table = device_tensor(tau_values, self.device, dtype=torch.float64)
        else:
            tau_table = torch.tensor(tau_values, dtype=torch.float64, device=self.device)
        temperature = float(rule["temperature"]["value"])
        head = self.jetspec_drafter.adapter.head
        decoder = self.jetspec_rows.decoder
        max_decoder = max(decoder._window_buckets()) - lanes
        c1_capacity = getattr(head, "_c1_capacity", int(required_setting(run_config, 'engine.engine_settings.w259_c1_cache_capacity')))
        from jetspec.engines.jetspec_plus_plus.serving.cache_capacity import chain_key_limit
        key_limit = chain_key_limit(self.config.max_model_len, lanes)
        if c1_capacity < key_limit or max_decoder < key_limit:
            raise ValueError(f"chain startup requires C1 capacity >= {key_limit} and "
                             f"decoder bucket >= {key_limit + lanes}; "
                             f"got C1={c1_capacity}, decoder={max_decoder + lanes}")
        reachable = setting(run_config, 'engine.engine_settings.chain_warm_reachable', "0") == "1"
        if reachable:
            read = json.loads(required_setting(run_config, 'engine.engine_settings.chain_warm_read'))
            lengths = reachable_chain_lengths(read["prompt_lengths"], read["tokens"],
                                              lanes, key_limit)
            lengths = tuple(sorted(set((*lengths, key_limit))))
            if setting(run_config, 'engine.engine_settings.chain_warm_single_key', "0") == "1":
                # Every shorter exact-L packet is covered by this ceiling key.
                # Keep segment pools private; avoid duplicate hybrid decoder
                # state and graph workspaces for unused smaller prefix keys.
                lengths = (key_limit,)
            row_caps, capture_names = reachable_chain_capture(
                setting(run_config, 'method.method_settings.chain_fixed_config'), R,
                bool(getattr(self._w320_clock, "chain_tau", None)),
                bool(getattr(self._w320_clock, "chain_tau_cap", False)), run_config=run_config)
        else:
            first = max(128, ((L_admit + 2 * lanes + 127) // 128) * 128)
            last = key_limit
            if first > last:
                raise ValueError("chain admission has no safe captured length key")
            lengths = range(first, last + 1, 128)
            row_caps, capture_names = ((R,) if R == 16 or rows_min_count(run_config=run_config) > 4 else (4, 8)), None
        from jetspec.engines.jetspec_plus_plus.draft.method.schedule import Schedule
        fixed_schedule = setting(run_config, 'method.method_settings.chain_fixed_config')
        if (setting(run_config, 'engine.engine_settings.chain_single_key', "0") == "1"
                and fixed_schedule and Schedule.parse(fixed_schedule).r > 1
                and not setting(run_config, 'engine.engine_settings.chain_j_switch_l')):
            # Every key retains its buffers and private pools for every pass.
            # Multi-refresh schedules otherwise multiply that storage across
            # all lengths and both historical row caps until the card fills.
            # One ceiling key covers every exact-L packet at the fixed row cap.
            # Length-switched schedules retain their buckets and switch point.
            lengths, row_caps = (key_limit,), (R,)
        last = lengths[-1]
        decoder_key_bucket = setting(run_config, 'engine.engine_settings.chain_decoder_key_bucket', "0") == "1"

        def chain_key(length, row_cap):
            window_bucket = (next(bucket for bucket in decoder._window_buckets()
                                  if bucket >= length + lanes)
                             if decoder_key_bucket else 0)
            return ChainKey(length, row_cap, 3, "after_j0_j1", lanes,
                            decoder_key_bucket=window_bucket)

        if host_lean:
            key_spec = (tuple(lengths), tuple(row_caps), decoder_key_bucket,
                        tuple(decoder._window_buckets()))
            cached = getattr(self, "_chain_admit_keys", None)
            if cached is None or cached[0] != key_spec:
                cached = self._chain_admit_keys = (
                    key_spec, tuple(chain_key(length, row_cap)
                                    for length in lengths for row_cap in row_caps))
            keys = cached[1]
        else:
            keys = tuple(chain_key(length, row_cap)
                         for length in lengths for row_cap in row_caps)
        graph = getattr(self, "_chain_graph", None)
        if graph is None:
            tokens = allocate_token_buffers(device=self.device, capacity=capacity, lanes=lanes)
            # Row passes pad their prefix to 256, including the new ceiling key.
            tap_capacity = max(capacity, ((key_limit + 255) // 256) * 256)
            tap_store = ChainTapBuffers(device=self.device, capacity=tap_capacity, lanes=lanes,
                                        hidden=initial_taps.shape[-1], dtype=initial_taps.dtype)
            row_ring = PublishedRowRing(device=self.device, rows=wire_row_cap, lanes=lanes, run_config=self.run_config)
        else:
            if (not graph._prewarmed or any(key not in graph.entries for key in keys)
                    or self._chain_static_capacity != capacity
                    or self._chain_static_wire_cap != wire_row_cap
                    or self._chain_static_hidden != initial_taps.shape[-1]):
                raise RuntimeError("chain warm-up graph does not cover this sequence shape")
            shared = next(iter(graph.entries.values())).buffers
            tokens, tap_store, row_ring = (shared[name] for name in
                                           ("tokens", "tap_store", "row_ring"))
        if graph is None:
            tokens.committed_ids.zero_()
            tokens.seed_committed(initial_ids)
            tokens.pending_draft_ids.zero_()
            tap_store.storage.zero_()
            tap_store.seed(initial_taps)
            seed_row_ring_from_admission(row_ring, candidate, blocks, admission_tau, run_config=run_config)
        if host_lean:
            fake_ids = torch.zeros(capacity, dtype=torch.int64)
            fake_ids[:initial_ids_host.numel()].copy_(initial_ids_host)
            host_packet = OutcomePacket(
                1, seq_id, 0, int(initial_ids_host[-1]), 0, 0, 0,
                L=L_admit,
                hash16=int(hash_last16(
                    fake_ids, torch.tensor(L_admit + 1),
                    torch.tensor(HASH_WEIGHTS, dtype=torch.int64))),
                k=-1, a=-1).pack()
            if host_check:
                device_ids = torch.zeros(capacity, dtype=torch.int64, device=self.device)
                device_ids[:initial_ids.numel()].copy_(initial_ids)
                device_packet = OutcomePacket(
                    1, seq_id, 0, int(initial_ids[-1]), 0, 0, 0,
                    L=L_admit,
                    hash16=int(hash_last16(
                        device_ids, torch.tensor(L_admit + 1, device=self.device),
                        torch.tensor(HASH_WEIGHTS, dtype=torch.int64, device=self.device))),
                    k=-1, a=-1).pack()
                mismatches = int((host_packet != device_packet.cpu()).sum())
                print(f"CHAIN_ADMIT_HOST_LEAN_CHECK mismatch={mismatches}")
                if mismatches:
                    raise RuntimeError("host round-0 packet differs from device packet")
            fake_packet = (host_packet.pin_memory().to(self.device, non_blocking=True)
                           if torch.device(self.device).type == "cuda"
                           else host_packet.to(self.device))
        else:
            fake_ids = torch.zeros(capacity, dtype=torch.int64, device=self.device)
            fake_ids[:initial_ids.numel()].copy_(initial_ids)
            fake_packet = OutcomePacket(
                1, seq_id, 0, int(initial_ids[-1]), 0, 0, 0,
                L=L_admit,
                hash16=int(hash_last16(
                    fake_ids, torch.tensor(L_admit + 1, device=self.device),
                    torch.tensor(HASH_WEIGHTS, dtype=torch.int64, device=self.device))),
                k=-1, a=-1).pack().to(self.device)

        def build(key):
            from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
            buffers, bodies = build_served_entry(
                key, rows=self.jetspec_rows, tokens=tokens, tap_store=tap_store,
                row_ring=row_ring, thresholds=thresholds, dataset=dataset,
                tau_table=tau_table, temperature=temperature,
                hypotheses=self.config.jetspec_hyp,
                wire_row_cap=wire_row_cap, admission=state,
                admission_hidden=admission_hidden, admission_tau=admission_tau, run_config=run_config)
            buffers["packet"].copy_(fake_packet)
            buffers["taps"].zero_()
            buffers["rp"].zero_()
            if kernel_enabled('chain_host_lean', '1'):
                bodies = configure_host_lean_entry(buffers, bodies, run_config=run_config)
            return buffers, bodies

        def restore(key, buffers):
            from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
            tokens.committed_ids.zero_()
            tokens.seed_committed(initial_ids)
            tokens.pending_draft_ids.zero_()
            tap_store.storage.zero_()
            tap_store.seed(initial_taps)
            for old_round in range(3):
                row_ring.empty(old_round)
            seed_row_ring_from_admission(row_ring, candidate, blocks, admission_tau, run_config=run_config)
            seed_front_from_admission(buffers["front"], state, decoder,
                                      thresholds, dataset, run_config=run_config)

        if graph is None:
            kwargs = {} if capture_names is None else dict(capture_names=capture_names)
            graph = ChainGraph(
                build, capacity=len(keys), **kwargs,
                shadow_state=lambda key, buffers: shadow_state_tensors(
                    buffers, self.jetspec_rows), run_config=run_config)
            graph.prewarm(keys, after_capture=restore, run_config=run_config)
            self._chain_static_capacity = capacity
            self._chain_static_wire_cap = wire_row_cap
            self._chain_static_hidden = initial_taps.shape[-1]
        else:
            fresh_rng = ChainRngTable(
                capacity=capacity, lanes=lanes, seed=self.jetspec_drafter.seed,
                prompt_index=state.prompt_index, device=self.device, run_config=run_config)
            reseed_served_graph(
                graph, initial_ids=initial_ids, initial_taps=initial_taps,
                candidate=candidate, blocks=blocks,
                admission_hidden=admission_hidden, admission_tau=admission_tau,
                rng_rows=fresh_rng.rows, fake_packet=fake_packet, run_config=run_config)
        self._chain_graph = graph
        self._chain_last_front = None
        self._chain_last_key = None
        self._chain_preselected_choice = (2, False)
        fixed_config = setting(run_config, 'method.method_settings.chain_fixed_config')
        switch_value = setting(run_config, 'engine.engine_settings.chain_j_switch_l')
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        switch_L = current_engine().budget.switch_length(switch_value)
        handoff = setting(run_config, 'engine.engine_settings.chain_handoff', "0") == "1"
        prepost = setting(run_config, 'engine.engine_settings.chain_prepost', "0") == "1"
        post_lean = kernel_enabled('chain_post_lean', '1')
        round_host_lean = kernel_enabled('chain_host_lean', '1') if cached_setting is not None else None
        anatomy_lite = anatomy_enabled(run_config)
        tracker = ChainLengthTracker(capacity=last, lanes=lanes,
                                     L_known=L_admit, mirrored_round=-1)

        def covering_length(length):
            return next((covered for covered in lengths if covered >= length
                         and all(chain_key(covered, row_cap) in graph.entries
                                 for row_cap in row_caps)), None)

        def key_for(tracker, round_id):
            length = covering_length(tracker.bucket(round_id, step=128))
            if length is None:
                return None
            candidates = tuple(chain_key(length, row_cap)
                               for row_cap in row_caps)
            clock = self._w320_clock
            if fixed_config is not None:
                key = candidates[row_caps.index(R)] if R in row_caps else candidates[0]
                choice = chain_fixed_choice(fixed_config, length, switch_L)
                if not (handoff or prepost or clock is None):
                    clock.arrive(time.perf_counter(), clean=True)
                    launch_config = (f"J{choice[0]}-R{int(choice[1])}"
                                     if switch_L is not None else fixed_config)
                    choice = clock.launch_chain_fixed(launch_config, key)
            elif clock is None:
                key, choice = candidates[1 if R == 8 else 0], (2, False)
            else:
                now = time.perf_counter()
                clock.arrive(now, clean=True)
                try:
                    if len(candidates) == 1 and (reachable or not clock.chain_tau_cap):
                        key = candidates[0]
                        choice = clock.launch_chain(now, key)
                    else:
                        key, choice = clock.launch_chain_capacity(now, candidates)
                except DraftDeadlineMiss:
                    key, choice = candidates[1 if R == 8 else 0], current_engine().budget.fallback_choice()
            self._chain_preselected_choice = choice
            return key

        def fallback(packet, taps, round_id):
            del taps
            # Last resort: the received packet supplies the exact live prefix.
            # Re-enter the driver's ordinary captured draft and publication path.
            length = covering_length(int(packet[OutcomeOffset.L].item()))
            if length is None:
                raise RuntimeError("chain capacity has no captured graph covering the live prefix")
            graph.fallback_counts["capacity"] += 1
            print("CHAIN_FALLBACK " + json.dumps(dict(
                seq_id=seq_id, round=round_id + 1, reason="capacity",
                last_resort=True, key=length,
                count=graph.fallback_counts["capacity"])), flush=True)
            self._chain_preselected_choice = (chain_fixed_choice(fixed_config, length, switch_L)
                                             if fixed_config is not None else (1, False))
            return chain_key(length, R if R in row_caps else row_caps[0])

        def select(key):
            from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
            entry = graph.entries[key]
            if handoff and fixed_config is not None and not prepost:
                clock = self._w320_clock
                if clock is not None:
                    clock.arrive(time.perf_counter(), clean=True)
                    choice = self._chain_preselected_choice
                    launch_config = (f"J{choice[0]}-R{int(choice[1])}"
                                     if switch_L is not None else fixed_config)
                    clock.launch_chain_fixed(launch_config, key)
            if self._chain_last_key != key:
                if self._chain_last_front is None:
                    seed_front_from_admission(entry.buffers["front"], state, decoder,
                                              thresholds, dataset, run_config=run_config)
                else:
                    copy_front_state(self._chain_last_front, entry.buffers["front"],
                                     prefix_length=max(0, tracker.L_known))
            self._chain_last_front = entry.buffers["front"]
            self._chain_last_key = key
            choice = self._chain_preselected_choice
            self._chain_last_choice = choice
            if not (round_host_lean if cached_setting is not None else kernel_enabled('chain_host_lean', '1')):
                entry.buffers["rp"][Offset.J_ROUND].fill_(choice[0])
            return choice

        def publish_redraft(buffers, round_id):
            if setting(run_config, 'engine.engine_settings.chain_tail_trace', '0') == '1':
                print(f"MAILBOX_ID side=draft event=candidate_send id={self._jetspec_candidate_send_id} seq={seq_id} round={round_id} branch=redraft", flush=True)
            self.jetspec_mailbox_store.set(
                f"candidate_payload_{self._jetspec_candidate_send_id}",
                buffers["redraft_home"].numpy().tobytes())
            self._jetspec_sent_candidates.add((seq_id, round_id))
            self._jetspec_candidate_send_id += 1

        def publish(buffers, round_id):
            timing = setting(run_config, 'engine.engine_settings.chain_handoff_timing', "0") == "1"
            publish_start = time.perf_counter_ns() if timing else None
            records = []
            if setting(run_config, 'engine.engine_settings.chain_tail_trace', '0') == '1':
                print(f"MAILBOX_ID side=draft event=candidate_send id={self._jetspec_candidate_send_id} seq={seq_id} round={round_id + 1} branch=next_round", flush=True)
            self.jetspec_mailbox_store.set(
                f"candidate_payload_{self._jetspec_candidate_send_id}",
                buffers["wire_home"].numpy().tobytes())
            lite_publish_ns = time.monotonic_ns() if anatomy_lite else None
            if timing:
                publish_end = time.perf_counter_ns()
                records.append(("CHAIN_HANDOFF_PUBLISH ", dict(
                    seq_id=seq_id, round=round_id + 1,
                    publish_ns=publish_end,
                    store_us=round((publish_end - publish_start) / 1000, 1))))
            self._jetspec_sent_candidates.add((seq_id, round_id + 1))
            self._jetspec_candidate_send_id += 1
            if self._w320_clock is not None and not prepost:
                self._w320_clock.learn_chain_times(
                    self._chain_last_key, graph.last_ms.get(self._chain_last_key, {}), 0.0)
                deadline = self._w320_clock.published(time.perf_counter())
                records.append(('W320_DEADLINE ', dict(
                    seq_id=seq_id, round_id=round_id + 1, **deadline)))
            records.append(("CHAIN_ROUND ", dict(
                seq_id=seq_id, round=round_id, published_round=round_id + 1,
                L=int(receiver.mirrors[receiver.slot_for_round(round_id)][OutcomeOffset.L]),
                key=self._chain_last_key.L_bucket,
                config=f"J{self._chain_last_choice[0]}-R{int(self._chain_last_choice[1])}",
                row_cap=self._chain_last_key.R, sync_total=graph.round_syncs,
                **({'lite': [buffers['_lite_command_ns'], buffers['_lite_chain_end_ns'],
                    lite_publish_ns, round(graph.last_span_ms[self._chain_last_key], 6),
                    {name: round(value, 6) for name, value in
                     graph.last_ms[self._chain_last_key].items()}]} if anatomy_lite else {}),
                **({"redraft": bool(buffers.get("host_redrafted", False))}
                   if setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1"
                   else {}),
                mix=dict(graph.mix), cap_mix=dict(graph.cap_mix),
                fallback=dict(graph.fallback_counts))))
            if prepost or post_lean:
                def emit():
                    for prefix, payload in records:
                        print(prefix + json.dumps(payload), flush=True)
                return emit
            for prefix, payload in records:
                print(prefix + json.dumps(payload), flush=True)

        pair_push = transport_enabled('pair_outcome_push', run_config=run_config)
        pair_check = setting(run_config, 'engine.engine_settings.pair_outcome_push_check', "0") == "1"
        shadow = (OutcomeReceiveRing(device=self.device, lanes=lanes,
                                     hidden=initial_taps.shape[-1],
                                     tap_dtype=initial_taps.dtype,
                                     hybrid_layout=getattr(self, "_hybrid_pair_layout", None), run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                  if not pair_push or pair_check else None)
        receiver = (PushedOutcomeRing(
            transport=self._pair_transport, inbox=self._chain_inbox,
            lanes=lanes, hidden=initial_taps.shape[-1],
            tap_dtype=initial_taps.dtype, shadow=shadow,
            hybrid_layout=getattr(self, "_hybrid_pair_layout", None))
            if pair_push else shadow)
        def pre_stage(tracker, round_id):
            # The tracker is intentionally one packet behind here. Its length
            # bound covers the next round without waiting for this mirror.
            length = covering_length(tracker.bucket(round_id, step=128))
            if length is None:
                return None  # let the fresh mirror decide near capacity
            key = chain_key(length, R if fixed_config is not None else row_caps[0])
            self._chain_preselected_choice = chain_fixed_choice(fixed_config, length, switch_L)
            return round_id, key, graph.prepare(key, run_config=run_config)

        recorder = None
        record_root = setting(run_config, 'engine.engine_settings.chain_record')
        if record_root and seq_id != -1:
            seen = getattr(self, "_chain_record_requests_seen", 0)
            self._chain_record_requests_seen = seen + 1
            if seen < int(setting(run_config, 'engine.engine_settings.chain_record_skip_requests', "0")):
                record_root = None
        if record_root and seq_id != -1:  # synthetic graph prewarm is not a request
            recorder = getattr(self, "_chain_recorder", None)
            if recorder is None or not recorder.closed:
                from jetspec.engines.jetspec_plus_plus.runtime.env import legacy_registry as _legacy_registry
                REGISTRY = _legacy_registry()

                prefix = decoder._prefixes[id(state)]
                admission = dict(
                    environment={name: value for name, value in legacy_values(run_config).items()
                                 if (name in REGISTRY or name == 'ssd_attention_backend')
                                 and name not in (legacy_name('engine.engine_settings.chain_record'), legacy_name('engine.engine_settings.pins'))},
                    pins=json.loads(open(required_setting(run_config, 'engine.engine_settings.pins')).read()),
                    reference_head=required_setting(run_config, 'engine.engine_settings.chain_reference_head'),
                    capacity=capacity, wire_row_cap=wire_row_cap,
                    hypotheses=self.config.jetspec_hyp,
                    prompt_index=state.prompt_index,
                    initial_ids=initial_ids, initial_taps=initial_taps,
                    staged_hidden=state.staged.predicted_hidden,
                    prefix_length=prefix.length, prefix_kv=prefix.kv,
                    admission_hidden=admission_hidden, admission_tau=admission_tau,
                    fake_packet=fake_packet,
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    captured_segments=sum(len(entry.graphs) for entry in graph.entries.values()),
                    ring=dict(rows=row_ring.rows, keys=row_ring.keys,
                              p_top1=row_ring.p_top1,
                              passes_run=row_ring.passes_run, rounds=row_ring.rounds))
            if recorder is None:
                recorder = ChainDraftRecorder(
                    record_root,
                    limit=int(setting(run_config, 'engine.engine_settings.chain_record_rounds', "300")),
                    admission=admission)
                self._chain_recorder = recorder
            elif not recorder.closed:
                recorder.mark_request(admission)

        self._chain_runtime = ChainRoundDriver(
            store=self.jetspec_mailbox_store, request_group=self.jetspec_request_pg,
            graph=graph, receiver=receiver, tracker=tracker, seq_id=seq_id,
            key_for=key_for, select=select, publish=publish, inbox=self._chain_inbox,
            publish_redraft=publish_redraft, fallback=fallback, start_round=0,
            pre_stage=(pre_stage if handoff else None), recorder=recorder,
            rows=self.jetspec_rows, run_config=run_config)

    def draft_loop(self, run_config=None):
        """
        Runs the asynchronous draft model loop.
        Handles prefill (1), speculation (0), exit (2), warmup (3), and pin check (4).
        """
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting
        assert self.draft_async, "draft_loop only runs in async-draft mode"
        chain_mode = getattr(self, "_chain_mode", False)

        if chain_mode:
            from jetspec.engines.jetspec_plus_plus.wire.transport import ChainCommandInbox
            self._chain_inbox = ChainCommandInbox(self.jetspec_mailbox_store, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))

        while True:
            # 1) Wait for the next command (may be PREFILL, SPEC_REQUEST, or EXIT)
            command_pg = (
                self.jetspec_request_pg if self.config.jetspec else self.async_pg
            )
            if chain_mode:
                if self._chain_runtime is None:
                    _, command = self._chain_inbox.next()
                else:
                    command = self._chain_runtime.next(run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                cmd = {"outcome": 0, "prefill": 1, "exit": 2,
                       "warmup": 3, "pin_served": 4}[command]
            else:
                cmd = self.recv_cmd(group=command_pg)

            # PREFILL: run the draft prefill and then loop back
            if cmd == 1:
                self.draft_async_prefill(run_config=run_config)
                continue

            elif cmd == 3:
                self._warm_jetspec_session(run_config=run_config)
                dist.send(torch.ones(1,dtype=torch.int64,device=self.device),dst=0,group=command_pg)
                continue

            # Check the draft rank before the target seals the run manifest.
            elif cmd == 4:
                try:
                    self._check_draft_compile_pin_served()
                except Exception:
                    dist.send(torch.zeros(1, dtype=torch.int64, device=self.device),
                              dst=0, group=command_pg)
                    raise
                dist.send(torch.ones(1, dtype=torch.int64, device=self.device),
                          dst=0, group=command_pg)
                continue

            # SPECULATE request: serve out-of-cache or random speculations
            elif cmd == 0:
                if chain_mode:
                    continue
                _ds0 = time.perf_counter()
                _prof = setting(run_config, 'engine.engine_settings.profile', "0") == "1"
                if _prof or self.profile_draft:
                    torch.cuda.synchronize()
                    _d0 = time.perf_counter()

                clock = getattr(self, '_w320_clock', None)
                if clock is not None:
                    clock.arrive(_ds0, clean=False)

                glue_decode_input_ids, partial_tree_decode_args = self._service_spec_request(run_config=run_config)

                if self.config.jetspec:
                    from jetspec.engines.jetspec_plus_plus.draft.batch_round import batch_round_enabled
                    if (self.jetspec_rows is None or self.config.jetspec_sync
                            or self.config.jetspec_wait_ms != float("inf")
                            or not batch_round_enabled(self.jetspec_rows, run_config=run_config)):
                        raise ValueError('retired eager draft loop; use the segmented chain or sealed records')
                    # C>1 rows: commit the returned round, stage, and draft round+1 for every
                    # sequence in one batched round, then push round+1 before the next request
                    self._advance_batch_rows(partial_tree_decode_args, run_config=run_config)
                    self._send_jetspec_candidates(partial_tree_decode_args["seq_ids"], run_config=run_config)
                    self._draft_step_times.append(time.perf_counter() - _ds0)
                    continue

                self._finish_spec_request(
                    glue_decode_input_ids, partial_tree_decode_args, _ds0, _prof,
                    _d0 if _prof or self.profile_draft else None)

                continue

            # EXIT: clean up and break out of the loop
            elif cmd == 2:
                _run_idle()
                if self._draft_step_times:
                    avg_ms = sum(self._draft_step_times) * 1000 / len(self._draft_step_times)
                    print(f"[metrics] Avg draft step time (ms): {avg_ms:.2f}", flush=True)
                if self.config.jetspec_taps:
                    recv_sorted = sorted(self._jetspec_tap_recv_ms)
                    bytes_sorted = sorted(self._jetspec_tap_payload_bytes)
                    p90_index = max(0, (9 * len(recv_sorted) + 9) // 10 - 1)
                    mirror_bytes = sum(
                        taps.numel() * taps.element_size()
                        for taps in self.jetspec_tap_mirrors.values()
                    )
                    summary = {
                        "rounds": len(recv_sorted),
                        "recv_ms": self._jetspec_tap_recv_ms,
                        "recv_ms_mean": (
                            sum(recv_sorted) / len(recv_sorted)
                            if recv_sorted else 0.0
                        ),
                        "recv_ms_p90": recv_sorted[p90_index] if recv_sorted else 0.0,
                        "payload_bytes_mean": (
                            sum(bytes_sorted) / len(bytes_sorted)
                            if bytes_sorted else 0.0
                        ),
                        "payload_bytes_p90": (
                            bytes_sorted[p90_index] if bytes_sorted else 0
                        ),
                        "mirror_bytes": mirror_bytes,
                        "mirror_lengths": {
                            str(seq_id): taps.shape[0]
                            for seq_id, taps in self.jetspec_tap_mirrors.items()
                        },
                    }
                    print(
                        f"[jetspec_taps] draft_metrics={json.dumps(summary, sort_keys=True)}",
                        flush=True,
                    )
                self.exit()
                break

            else:
                raise RuntimeError(f"draft_loop: unknown command {cmd}")
