"""Pure bookkeeping for the JetSpec candidate-set protocol."""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import math
import os
from dataclasses import InitVar, dataclass, field
from concurrent.futures import Future
from queue import SimpleQueue
from threading import Thread
from typing import Literal

import torch

from jetspec.engines.jetspec_plus_plus.trace import host_census


DecisionKind = Literal["hit", "miss", "late"]


def anchor_mode(run_config=None):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    value = setting(run_config, 'method.method_settings.anchor', 'guessed')
    if value not in ('guessed', 'oracle', 'decoder'):
        raise ValueError('JETSPEC_ANCHOR must be guessed, oracle, or decoder')
    return value


def draft_width(run_config=None):
    """Drafts verified per round. Fourteen is the explicit legacy kth width."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    from jetspec.engines.jetspec_plus_plus.draft.method.geometry import Geometry
    raw = setting(run_config, 'method.method_settings.drafts', '14' if run_config is None else None)
    if raw is None:
        raise ValueError('JETSPEC_DRAFTS must be set for JetSpec++')
    value = int(raw)
    geometry = Geometry.from_drafts(value)
    if not geometry.served:
        raise ValueError('JETSPEC_DRAFTS must be 3, 7, 14, 15, or 31; minimum served block is 4')
    if value != 14 and setting(run_config, 'engine.engine_settings.row_layout', 'kth') != 'anchor0':
        raise ValueError('JETSPEC_DRAFTS other than 14 requires JETSPEC_ROW_LAYOUT=anchor0')
    return geometry.drafts


def oracle_request_predictions(records, *, seq_id, round_id):
    """Only a completed preceding verify can source an oracle request."""
    if round_id == 0:
        return [-1] * 16  # Prefill's known correction is already in cache_keys.
    record = next((r for r in reversed(records)
                   if int(r['seq_id']) == seq_id and int(r['round']) == round_id - 1), None)
    if record is None or 'verify_complete_ns' not in record:
        raise RuntimeError('oracle request precedes its verifier result')
    predictions = record.get('greedy_predictions')
    if predictions is None or len(predictions) != 15:
        raise RuntimeError('oracle verifier export must contain 15 positions')
    return [int(token) for token in predictions] + [int(record['verify_complete_ns'])]


def request_wire_numel(capacity, *, tree=False, oracle=False):
    return 1 + capacity * (5 + 14 * tree + 16 * oracle)


def pack_request_wire(seqs, *, capacity, tap_lengths, tree=False, oracle_records=None):
    """Pack only state consumed by the asynchronous JetSpec draft rank."""
    if not 0 < len(seqs) <= capacity or len(tap_lengths) != len(seqs):
        raise ValueError("request batch exceeds capacity or tap lengths disagree")
    oracle = oracle_records is not None
    width = 5 + 14 * tree + 16 * oracle
    values = [len(seqs)]
    for seq, length in zip(seqs, tap_lengths):
        true_k = max(0, seq.last_spec_step_accepted_len - 1)
        values.extend((seq.seq_id, true_k, seq.recovery_token_id, seq.jetspec_round, length))
        if tree:
            if true_k > 14:
                raise ValueError("tree request exceeds its 14 accepted-token slots")
            values.extend(seq.token_ids[-true_k-1:-1] if true_k else [])
            values.extend([-1] * (14 - true_k))
        if oracle:
            values.extend(oracle_request_predictions(oracle_records,
                seq_id=int(seq.seq_id), round_id=int(seq.jetspec_round)))
    values.extend([-1] * ((capacity - len(seqs)) * width))
    return torch.tensor(values, dtype=torch.int64, device="cpu")


def unpack_request_wire(wire, *, capacity, tree=False, oracle=False):
    """Materialize the small control packet once; taps travel separately."""
    if wire.numel() != request_wire_numel(capacity, tree=tree, oracle=oracle):
        raise ValueError("request packet does not match its fixed capacity")
    host = wire.cpu()
    batch = int(host[0])
    if not 0 < batch <= capacity:
        raise ValueError("request packet batch exceeds capacity")
    rows = host[1:].view(capacity, -1)[:batch]
    payload = dict(cache_keys=rows[:, :3], seq_ids=rows[:, 0],
                   rounds=rows[:, 3], tap_lengths=rows[:, 4],
                   accepted_tree_ids=rows[:, 5:19] if tree else None)
    if oracle:
        start = 19 if tree else 5
        payload["oracle_verify"] = rows[:, start:start+16]
    return payload


def branch_count(run_config=None):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    if anchor_mode(run_config=run_config) == 'oracle':
        return 1
    value = int(setting(run_config, 'method.method_settings.f', '1'))
    if value not in (1, 2, 3, 5):
        raise ValueError('JETSPEC_F must be 1, 2, 3, or 5')
    if anchor_mode(run_config=run_config) == 'decoder' and value != 1:
        raise ValueError('decoder anchor requires JETSPEC_F=1')
    return value


def candidate_wire_shape(batch, hypotheses, *, run_config=None):
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    if anchor_mode(run_config=run_config) == 'decoder':
        return (batch, int(setting(run_config, 'engine.engine_settings.rows_cap', '128')), draft_width(run_config=run_config) + 1)
    F = branch_count(run_config=run_config)
    return (batch,hypotheses,15) if F == 1 else (batch,hypotheses,F,15)


def normalize_wait_ms(value: float) -> float:
    """Validate the verifier's bounded-wait budget in milliseconds."""
    wait_ms = float(value)
    if math.isnan(wait_ms) or wait_ms < 0:
        raise ValueError("jetspec_wait_ms must be non-negative or infinity")
    return wait_ms


def summarize_candidate_kinds(records: list[dict]) -> dict:
    """Summarize ready candidates without pooling catch-up and speculative work."""
    total_rounds = len(records)

    def summarize(groups: dict[tuple, list[dict]], key_names: tuple[str, ...]):
        rows = []
        for key in sorted(groups):
            members = groups[key]
            count = len(members)
            row = dict(zip(key_names, key))
            row.update(
                {
                    "count": count,
                    "share_all_rounds": (
                        count / total_rounds if total_rounds else 0.0
                    ),
                    "hit_rate": sum(
                        member["decision"] == "hit" for member in members
                    ) / count,
                    "tau": sum(
                        int(member.get("accepted_len", 0)) for member in members
                    ) / count,
                    "gap_histogram": sorted(
                        int(member.get("gap", 0))
                        for member in members
                        if int(member.get("gap", 0)) > 0
                    ),
                }
            )
            rows.append(row)
        return rows

    ready = [record for record in records if record["decision"] != "late"]
    by_kind = {}
    by_detail = {}
    for record in ready:
        kind = record.get("candidate_kind")
        if kind not in {"speculative", "catchup"}:
            raise ValueError("ready rounds require a candidate kind")
        by_kind.setdefault((kind,), []).append(record)
        by_detail.setdefault(
            (
                kind,
                bool(record.get("previous_round_ar", False)),
                int(record["k"]),
            ),
            [],
        ).append(record)
    return {
        "total_rounds": total_rounds,
        "served_tau": (
            sum(int(record.get("accepted_len", 0)) for record in records)
            / total_rounds
            if total_rounds
            else 0.0
        ),
        "by_kind": summarize(by_kind, ("candidate_kind",)),
        "by_previous_ar_and_k": summarize(
            by_detail,
            ("candidate_kind", "previous_round_ar", "k"),
        ),
    }


class CandidateMessage:
    """Completion of one pre-posted host candidate packet."""

    def __init__(self):
        self.ready = Future()

    def cpu(self):
        return self.ready.result()


class CandidateReceiver:
    """One CPU reader fetches immutable packets while verification proceeds."""

    def __init__(self, store, decode, *, run_config=None):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        # A blocking get must not hold the request publisher's TCP connection.
        store = store.clone()
        self.pending = SimpleQueue()
        census = host_census.enabled(run_config)
        id_trace = setting(run_config, 'engine.engine_settings.chain_tail_trace', '0') == '1'

        def receive():
            while True:
                key, message = self.pending.get()
                try:
                    if id_trace:
                        print(f"MAILBOX_ID side=verify event=candidate_await key={key}", flush=True)
                    wire = self._read(store, key)
                    if census:
                        got = host_census.perf_counter()
                        delivery = decode(wire)
                        message.census_stamps = (got, host_census.perf_counter())
                        message.ready.set_result(delivery)
                        continue
                    message.ready.set_result(decode(wire))
                except Exception as error:
                    message.ready.set_exception(error)

        # Reads can outlive the final round; they must not hold engine exit open.
        Thread(target=receive, name="candidate-receive", daemon=True).start()

    def post(self, key):
        message = CandidateMessage()
        message.key = key
        self.pending.put((key, message))
        return message

    def _read(self, store, key):
        return torch.frombuffer(bytearray(store.get(key)), dtype=torch.int64)


class PublicationWork:
    """A store write tracked alongside asynchronous tensor sends."""

    def __init__(self):
        self.ready = Future()

    def wait(self):
        return self.ready.result()

    def is_completed(self):
        if not self.ready.done():
            return False
        self.ready.result()
        return True


class RequestPublisher:
    """Publish ordered request controls without blocking the target thread."""

    def __init__(self, store):
        store = store.clone()
        self.pending = SimpleQueue()

        def publish():
            while True:
                key, value, work = self.pending.get()
                try:
                    self._write(store, key, value)
                    work.ready.set_result(None)
                except Exception as error:
                    work.ready.set_exception(error)

        Thread(target=publish, name="request-publish", daemon=True).start()

    def post(self, key, value):
        work = PublicationWork()
        self.pending.put((key, value, work))
        return work

    def _write(self, store, key, value):
        store.set(key, value)


class PackedCandidateBuffers(tuple):
    """Keep the packed receive so host bookkeeping needs one device transfer."""

    def __new__(cls, wire, views):
        result = super().__new__(cls, views)
        result.wire = wire
        return result

    def cpu(self):
        host = self.wire.cpu()
        return tuple(host.view(view.dtype).as_strided(
            view.shape, view.stride(), view.storage_offset()) for view in self)


def validate_candidate_row_keys_batch(keys: torch.Tensor, *, run_config=None) -> None:
    """Validate all keyed rows with one scalar read for a batched publication."""
    if keys.ndim != 3 or keys.shape[-1] != 2:
        raise ValueError("batched candidate keys must have shape [N, R, 2]")
    width = draft_width(run_config=run_config) + 1
    live = keys[:, :, 0] >= 0
    bad_live = (live & ((keys[:, :, 0] > width - 1) | (keys[:, :, 1] < 0))).any()
    bad_pad = ((~live)[:, :, None] & (keys != -1)).any()
    invalid_live, invalid_pad = torch.stack((bad_live, bad_pad)).tolist()
    if invalid_live:
        raise ValueError("flat candidate keys must contain live (k=0..drafts, anchor>=0) pairs")
    if invalid_pad:
        raise ValueError("flat candidate padding keys must be (-1, -1)")


def draft_verifier_kind_from_host_pins(rows, true_k: int, correction: int) -> str:
    """Read the first keyed pin from the previous round's batched host copy."""
    for k, guess, pin in rows:
        if k == true_k and guess == correction and pin >= 0:
            return "hit" if pin == correction else "miss"
    return "miss"


@dataclass(frozen=True)
class CandidateSet:
    seq_id: int
    round_id: int
    blocks: torch.Tensor
    topk_ids: torch.Tensor | None = None
    topk_logps: torch.Tensor | None = None
    row_keys: torch.Tensor | None = None
    row_keys_host: InitVar[list[list[int]] | None] = None
    _keys_validated: bool = field(default=False, repr=False, compare=False)
    run_config: InitVar[object] = None

    def __post_init__(self, row_keys_host, run_config=None) -> None:
        width = draft_width(run_config=run_config) + 1
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        if self.blocks.ndim not in (2, 3) or self.blocks.shape[-1] != width:
            raise ValueError(f"candidate blocks must have shape [H, {width}] or [H, F, {width}]")
        if self.row_keys is not None:
            if self.blocks.ndim != 2 or self.row_keys.shape != (self.blocks.shape[0], 2):
                raise ValueError("flat candidate keys must have shape [R, 2] beside [R, 15] blocks")
            if self._keys_validated:
                return
            live = self.row_keys[:, 0] >= 0 if row_keys_host is None else None
            if row_keys_host is not None:
                bad_live = any(k >= 0 and (k > width - 1 or anchor < 0)
                               for k, anchor in row_keys_host)
                bad_pad = any(k < 0 and (k != -1 or anchor != -1)
                              for k, anchor in row_keys_host)
            elif kernel_enabled('tail_lean', '1') and self.row_keys.is_cuda:
                # serving: both checks read home in one trip (the same predicates, masked
                # in place of boolean indexing so no host sync sizes the selection)
                keys, pad = self.row_keys, (~live)[:, None]
                bad_live = (live & ((keys[:, 0] > width - 1) | (keys[:, 1] < 0))).any()
                bad_pad = (pad & (keys != -1)).any()
                bad_live, bad_pad = torch.stack((bad_live, bad_pad)).tolist()
            else:
                bad_live = bool(((self.row_keys[live, 0] > width - 1) | (self.row_keys[live, 1] < 0)).any())
                bad_pad = bool((self.row_keys[~live] != -1).any())
            if bad_live:
                raise ValueError("flat candidate keys must contain live (k=0..drafts, anchor>=0) pairs")
            if bad_pad:
                raise ValueError("flat candidate padding keys must be (-1, -1)")


@dataclass(frozen=True)
class CandidateDelivery:
    candidates: list[CandidateSet | None]
    draft_ms: list[float]
    coalesced: int
    gaps: list[int]


def unpack_candidate_delivery(buffers, *, private_wire=False, host_select=False, run_config=None):
    """Prepare host bookkeeping without mutating the target mailbox or metrics."""
    available, keys, blocks, draft_ms, telemetry = buffers[:5]
    topk = buffers[5:]
    if private_wire:
        controls = torch.cat((available[:, None], keys, telemetry), dim=1).tolist()
        all_times = draft_ms.tolist()
        times = [all_times[row] for row, values in enumerate(controls) if values[0]]
        coalesced = sum(values[3] for values in controls)
        gaps = [values[4] for values in controls]
        candidates = (unpack_candidate_sets(available, keys, blocks, row_keys=topk[0],
                                            private_wire=True, controls=controls,
                                            host_select=host_select, run_config=run_config)
                      if len(topk) == 1 else
                      unpack_candidate_sets(available, keys, blocks, *topk,
                                            private_wire=True, controls=controls,
                                            host_select=host_select, run_config=run_config))
        return CandidateDelivery(candidates, times, coalesced, gaps)
    times = draft_ms[available.to(torch.bool)].tolist()
    coalesced = int(telemetry[:, 0].sum().item())
    gaps = [int(value) for value in telemetry[:, 1].tolist()]
    candidates = (unpack_candidate_sets(available, keys, blocks, row_keys=topk[0], run_config=run_config)
                  if len(topk) == 1 else
                  unpack_candidate_sets(available, keys, blocks, *topk, run_config=run_config))
    return CandidateDelivery(candidates, times, coalesced, gaps)


@dataclass
class ProtocolCounters:
    hit: int = 0
    miss: int = 0
    late: int = 0


@dataclass(frozen=True)
class ProtocolDecision:
    kind: DecisionKind
    drafts: torch.Tensor | list[int]


class CandidateMailbox:
    """Nonblocking per-stream candidate storage keyed by protocol round."""

    def __init__(self) -> None:
        self._candidate_sets: dict[tuple[int, int], CandidateSet] = {}
        self.stale = 0

    def put(self, candidate_set: CandidateSet) -> None:
        key = (int(candidate_set.seq_id), int(candidate_set.round_id))
        self._candidate_sets[key] = candidate_set

    def has(self, *, seq_id: int, round_id: int) -> bool:
        """Report exact readiness without consuming the candidate."""
        return (int(seq_id), int(round_id)) in self._candidate_sets

    def take(self, *, seq_id: int, round_id: int) -> CandidateSet | None:
        """Return the exact candidate immediately and discard older arrivals."""
        seq_id = int(seq_id)
        round_id = int(round_id)
        stale_keys = [
            key
            for key in self._candidate_sets
            if key[0] == seq_id and key[1] < round_id
        ]
        for key in stale_keys:
            del self._candidate_sets[key]
        self.stale += len(stale_keys)
        return self._candidate_sets.pop((seq_id, round_id), None)


def pack_prompt_indices(seqs, *, device: torch.device) -> torch.Tensor:
    """Pack frozen dataset indices, falling back only for non-benchmark callers."""
    values = []
    for seq in seqs:
        prompt_index = getattr(seq, "jetspec_prompt_index", None)
        values.append(int(seq.seq_id if prompt_index is None else prompt_index))
    return torch.tensor(values, dtype=torch.int64, device=device)


def candidate_blocks_tensor(blocks, *, hypotheses=None) -> torch.Tensor:
    """Convert drafter ``Block`` objects to ``[H, guess + 14 drafts]`` ids."""
    if blocks and isinstance(blocks[0], (list, tuple)):
        rows = [candidate_blocks_tensor(branches) for branches in blocks]
        F = max(row.shape[0] for row in rows)
        H = hypotheses or max(branches[0].hypothesis for branches in blocks) + 1
        packed = torch.full((H,F,15), -1, dtype=torch.int64, device=rows[0].device)
        for branches,row in zip(blocks,rows):
            packed[branches[0].hypothesis,:row.shape[0]] = row
        return packed
    rows = []
    for block in blocks:
        if block.correction_guess is None or block.draft_ids.numel() != 14:
            raise ValueError("kth candidates require one guess and 14 draft ids")
        guess = torch.tensor(
            [block.correction_guess], dtype=torch.int64, device=block.draft_ids.device
        )
        rows.append(torch.cat([guess, block.draft_ids.to(torch.int64)]))
    if not rows:
        raise ValueError("candidate block list must be non-empty")
    stacked = torch.stack(rows)
    if hypotheses is not None:
        packed = stacked.new_full((hypotheses,15),-1)
        for block,row in zip(blocks,stacked):
            packed[block.hypothesis] = row
        return packed
    return stacked


class PosteriorBranches(list):
    """Published branch groups with a fixed wire and a real stage fallback."""
    def __init__(self, blocks, *, output_hypotheses):
        if not blocks or any(not group for group in blocks):
            raise ValueError('posterior needs a computed fallback block')
        F=len(blocks[0])
        mapping={int(group[0].hypothesis):group for group in blocks}
        if F not in (1,5) or len(mapping)!=len(blocks) or any(
                len(group)!=F or any(int(b.hypothesis)!=k for b in group)
                for k,group in mapping.items()):
            raise ValueError('posterior branch groups must have unique k and F=1 or F=5')
        requested=list(map(int,output_hypotheses))
        if len(set(requested))!=len(requested) or any(k<0 or k>14 for k in requested):
            raise ValueError('posterior publishes only unique k=0..14')
        if any(k not in mapping for k in requested):
            raise ValueError('posterior output requires a computed hypothesis')
        super().__init__([mapping[k] for k in requested])
        self.hypotheses=requested;self.posterior=True
        self.fallback=blocks[0][0]
        self.computed_hypotheses=len(blocks);self.branches=F
        if self:
            wire=candidate_blocks_tensor(self,hypotheses=16)
        else:
            wire=self.fallback.draft_ids.new_full((16,F,15),-1,dtype=torch.int64)
        self.wire=wire[:,0] if F==1 else wire


def oracle_candidate_blocks(blocks, *, hypotheses):
    """Preserve wire capacity while masking every unavailable hypothesis."""
    packed = candidate_blocks_tensor(blocks, hypotheses=hypotheses)
    if packed.ndim == 3:
        if packed.shape[1] != 1:
            raise ValueError('oracle wire must have one physical branch')
        packed = packed[:, 0]
    if hypotheses == 16:
        packed[15] = -1
    return packed


def select_candidate_block(blocks: torch.Tensor, *, true_k: int, correction: int,
                           row_keys: torch.Tensor | None = None) -> torch.Tensor:
    """Select the first surviving pin, or the rank-1 branch on a miss."""
    if row_keys is not None:
        from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import hit_lookup
        index = hit_lookup(row_keys, true_k, correction, pins=blocks[:, 0])
        return blocks[index] if index is not None else blocks.new_full((blocks.shape[-1],), -1)
    row = blocks[int(true_k)]
    if row.ndim == 1:
        return row
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import hit_lookup
    index = hit_lookup(None, true_k, correction, pins=row[:, 0])
    return row[index if index is not None else 0]


def select_candidate_block_indexed(candidate: CandidateSet, *, true_k: int,
                                   correction: int) -> torch.Tensor:
    """Use the private packet's first surviving keyed row when available."""
    index = getattr(candidate, "_row_index", None)
    if index is None:
        return select_candidate_block(candidate.blocks, true_k=true_k,
                                      correction=correction, row_keys=candidate.row_keys)
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import hit_lookup
    row = hit_lookup(candidate.row_keys, true_k, correction, index=index)
    if row is None:
        return candidate.blocks.new_full((candidate.blocks.shape[-1],), -1)
    return candidate.blocks[row]


def select_candidate_drafts_host(candidate_set: CandidateSet | None, *, seq_id: int,
                                 round_id: int, true_k: int, correction: int,
                                 miss_mode: Literal["ar", "verify"],
                                 counters: ProtocolCounters) -> tuple[ProtocolDecision, list[int] | None]:
    """Decide the private N=1 packet from its receiver-owned host rows."""
    if miss_mode not in {"ar", "verify"}:
        raise ValueError("miss_mode must be ar or verify")
    if candidate_set is None:
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        current_engine().budget.late(False)
        counters.late += 1
        return ProtocolDecision("late", torch.empty(0, dtype=torch.int64)), None
    rows = getattr(candidate_set, "_blocks_host", None)
    if rows is None or candidate_set.row_keys is None:
        raise ValueError("host selection requires private keyed candidate rows")
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import hit_lookup
    row = hit_lookup(candidate_set.row_keys, true_k, correction,
                     index=getattr(candidate_set, "_row_index", {}))
    block = rows[row] if row is not None else [-1] * candidate_set.blocks.shape[-1]
    if candidate_set.seq_id != int(seq_id) or candidate_set.round_id != int(round_id):
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        current_engine().budget.late(False)
        counters.late += 1
        return ProtocolDecision("late", torch.empty(0, dtype=torch.int64)), block
    if block[0] < 0:
        counters.miss += 1
        return ProtocolDecision("miss", torch.empty(0, dtype=candidate_set.blocks.dtype)), block
    if block[0] == int(correction):
        counters.hit += 1
        return ProtocolDecision("hit", torch.tensor(block[1:], dtype=candidate_set.blocks.dtype)), block
    counters.miss += 1
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    drafts = [] if current_engine().budget.ar_step("miss", miss_mode) else block[1:]
    return ProtocolDecision("miss", torch.tensor(drafts, dtype=candidate_set.blocks.dtype)), block


def select_candidate_drafts_list(candidate_set: CandidateSet | None, *, seq_id: int,
                                 round_id: int, true_k: int, correction: int,
                                 miss_mode: Literal["ar", "verify"],
                                 counters: ProtocolCounters) -> tuple[ProtocolDecision, list[int] | None]:
    """Return the N=1 private-wire decision without rematerializing a tensor."""
    if miss_mode not in {"ar", "verify"}:
        raise ValueError("miss_mode must be ar or verify")
    if candidate_set is None:
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        current_engine().budget.late(False)
        counters.late += 1
        return ProtocolDecision("late", []), None
    rows = getattr(candidate_set, "_blocks_host", None)
    if rows is None or candidate_set.row_keys is None:
        raise ValueError("host selection requires private keyed candidate rows")
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import hit_lookup
    row = hit_lookup(candidate_set.row_keys, true_k, correction,
                     index=getattr(candidate_set, "_row_index", {}))
    block = rows[row] if row is not None else [-1] * candidate_set.blocks.shape[-1]
    if candidate_set.seq_id != int(seq_id) or candidate_set.round_id != int(round_id):
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        current_engine().budget.late(False)
        counters.late += 1
        return ProtocolDecision("late", []), block
    if block[0] < 0:
        counters.miss += 1
        return ProtocolDecision("miss", []), block
    if block[0] == int(correction):
        counters.hit += 1
        return ProtocolDecision("hit", block[1:]), block
    counters.miss += 1
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    return ProtocolDecision("miss", [] if current_engine().budget.ar_step("miss", miss_mode) else block[1:]), block


def candidate_row_coverage(row_keys, *, true_k, correction):
    """Read the host candidate keys once for the three coverage counters."""
    keys = row_keys.tolist() if isinstance(row_keys, torch.Tensor) else row_keys
    live = [k for k, _ in keys if k >= 0]
    return sorted(set(live)), len(live), [int(true_k), int(correction)] in keys


class ChainSnapshotError(RuntimeError):
    """A chain AFTER_MISS probe met a key snapshot of another round (contract 5f (d))."""


@dataclass(frozen=True)
class ChainKeySnapshot:
    """Host copy of one published candidate set's exact keys, tagged by its round.

    ``live_keys`` holds the (k, anchor) pairs with both words >= 0, which is the
    device ring's ``valid`` key rule. ``legacy_pins`` is set only for an unkeyed
    candidate set (rows indexed by k). ``redraft`` marks the same-round second
    delivery, whose one row is keyed exactly by construction.
    """

    round_id: int
    live_keys: frozenset
    legacy_pins: tuple | None = None
    redraft: bool = False

    @classmethod
    def from_row_keys(cls, round_id, row_keys, *, redraft=False):
        keys = row_keys.tolist() if isinstance(row_keys, torch.Tensor) else row_keys
        return cls(int(round_id), frozenset((int(k), int(a)) for k, a in keys
                                            if k >= 0 and a >= 0), None, bool(redraft))

    @classmethod
    def from_candidate(cls, candidate, *, round_id, redraft=False):
        """The verifier's snapshot of a delivered set; an absent set has no live key."""
        if candidate is None:
            return cls(int(round_id), frozenset(), None, bool(redraft))
        if int(candidate.round_id) != int(round_id):
            raise ChainSnapshotError("chain candidate round differs from the probed round")
        if candidate.row_keys is not None:
            live_keys = getattr(candidate, "_live_keys", None)
            if live_keys is not None:
                return cls(int(round_id), live_keys, None, bool(redraft))
            return cls.from_row_keys(round_id, candidate.row_keys, redraft=redraft)
        pins = candidate.blocks[..., 0].tolist()
        return cls(int(round_id), frozenset(), tuple(
            tuple(row) if isinstance(row, list) else (row,) for row in pins), bool(redraft))


def chain_exact_hit(snapshot: ChainKeySnapshot, true_k: int, recovery: int) -> bool:
    """The exact one-ahead key rule: intake's ``index_for_pending`` ``exact``."""
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import hit_lookup
    return hit_lookup(snapshot.live_keys, true_k, recovery, pins=snapshot.legacy_pins,
                      legacy=snapshot.legacy_pins is not None) is not None


def chain_after_miss(snapshot: ChainKeySnapshot | None, *, round_id: int, true_k: int,
                     recovery: int, eos: bool = False) -> int:
    """AFTER_MISS for the command of ``round_id`` (5f): one function on both ranks.

    - Round 0 (5c-0 admission): the pending is the seeded admission row; no probe.
    - EOS (5e): no probe on the terminal command; nothing follows it.
    - AR miss (K / A = -1): a negative key never matches, as the device ``valid``
      rule requires both key words >= 0.
    - Redraft round: the second delivery must carry the exact key it was drafted for.
    The snapshot must be tagged with ``round_id``; a mismatch raises, never guesses.
    """
    from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import after_miss
    return after_miss(snapshot, round_id=round_id, true_k=true_k, recovery=recovery,
                      eos=eos, snapshot_error=ChainSnapshotError)


def select_candidate_drafts(
    candidate_set: CandidateSet | None,
    *,
    seq_id: int,
    round_id: int,
    true_k: int,
    correction: int,
    miss_mode: Literal["ar", "verify"],
    counters: ProtocolCounters,
    selected_block: torch.Tensor | None = None,
) -> ProtocolDecision:
    """Pick hypothesis ``k`` and apply the correction-guess gate."""
    if miss_mode not in {"ar", "verify"}:
        raise ValueError("miss_mode must be ar or verify")
    if (
        candidate_set is None
        or candidate_set.seq_id != int(seq_id)
        or candidate_set.round_id != int(round_id)
    ):
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        current_engine().budget.late(False)
        counters.late += 1
        return ProtocolDecision("late", torch.empty(0, dtype=torch.int64))
    if candidate_set.row_keys is None and not 0 <= int(true_k) < candidate_set.blocks.shape[0]:
        counters.miss += 1
        return ProtocolDecision("miss", torch.empty(0, dtype=torch.int64))

    block = selected_block
    if block is None:
        block = select_candidate_block(candidate_set.blocks, true_k=true_k, correction=correction,
                                       row_keys=candidate_set.row_keys)
    if int(block[0].item()) < 0:
        counters.miss += 1
        return ProtocolDecision("miss", block.new_empty((0,)))
    if int(block[0].item()) == int(correction):
        counters.hit += 1
        return ProtocolDecision("hit", block[1:])

    counters.miss += 1
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    drafts = block.new_empty((0,)) if current_engine().budget.ar_step("miss", miss_mode) else block[1:]
    return ProtocolDecision("miss", drafts)


def pack_candidate_sets(
    candidate_sets: list[CandidateSet | None],
    *,
    hypotheses: int,
    device: torch.device,
    branches: int = 1,
run_config=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack optional per-stream sets into fixed NCCL-friendly tensors."""
    batch_size = len(candidate_sets)
    available = torch.zeros(batch_size, dtype=torch.int64, device=device)
    keys = torch.full((batch_size, 2), -1, dtype=torch.int64, device=device)
    width = draft_width(run_config=run_config) + 1
    shape = (hypotheses, width) if branches == 1 else (hypotheses, branches, width)
    blocks = torch.full((batch_size, *shape), -1, dtype=torch.int64, device=device)
    for row, candidate_set in enumerate(candidate_sets):
        if candidate_set is None:
            continue
        if candidate_set.blocks.shape != shape:
            raise ValueError(
                f"candidate set must have shape {shape}, "
                f"got {tuple(candidate_set.blocks.shape)}"
            )
        available[row] = 1
        keys[row] = torch.tensor(
            [candidate_set.seq_id, candidate_set.round_id],
            dtype=torch.int64,
            device=device,
        )
        blocks[row] = candidate_set.blocks.to(device=device, dtype=torch.int64)
    return available, keys, blocks


def unpack_candidate_sets(
    available: torch.Tensor,
    keys: torch.Tensor,
    blocks: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    topk_logps: torch.Tensor | None = None,
    row_keys: torch.Tensor | None = None,
    *,
    private_wire: bool = False,
    controls: list[list[int]] | None = None,
    host_select: bool = False,
run_config=None) -> list[CandidateSet | None]:
    """Unpack candidate tensors without changing any token values."""
    if available.ndim != 1:
        raise ValueError("candidate availability must have shape [B]")
    batch_size = available.shape[0]
    if keys.shape != (batch_size, 2):
        raise ValueError("candidate keys must have shape [B, 2]")
    if blocks.ndim not in (3, 4) or blocks.shape[0] != batch_size or blocks.shape[-1] != draft_width(run_config=run_config) + 1:
        raise ValueError("candidate blocks must have shape [B, H, drafts+1] or [B, H, F, drafts+1]")
    if row_keys is not None and row_keys.shape != (batch_size, blocks.shape[1], 2):
        raise ValueError("candidate row keys must have shape [B, R, 2]")
    candidate_sets = []
    for row in range(batch_size):
        if (controls[row][0] if private_wire else int(available[row].item())) == 0:
            candidate_sets.append(None)
            continue
        if private_wire:
            key_list = None if row_keys is None else row_keys[row].tolist()
            candidate = CandidateSet(
                seq_id=controls[row][1], round_id=controls[row][2],
                blocks=blocks[row], row_keys=None if row_keys is None else row_keys[row],
                topk_ids=None if topk_ids is None else topk_ids[row],
                topk_logps=None if topk_logps is None else topk_logps[row].float(),
                row_keys_host=key_list, run_config=run_config)
            if key_list is not None:
                index = {}
                for offset, (key, pin) in enumerate(zip(key_list, blocks[row, :, 0].tolist())):
                    if key[0] >= 0 and pin >= 0:
                        index.setdefault(tuple(key), offset)
                object.__setattr__(candidate, "_row_index", index)
                if host_select:
                    object.__setattr__(candidate, "_blocks_host", blocks[row].tolist())
                    object.__setattr__(candidate, "_row_keys_host", key_list)
                    object.__setattr__(candidate, "_live_keys", frozenset(
                        (int(k), int(a)) for k, a in key_list if k >= 0 and a >= 0))
            candidate_sets.append(candidate)
        else:
            candidate_sets.append(
                CandidateSet(
                    seq_id=int(keys[row, 0].item()),
                    round_id=int(keys[row, 1].item()),
                    blocks=blocks[row].clone(),
                    row_keys=None if row_keys is None else row_keys[row].clone(),
                    topk_ids=None if topk_ids is None else topk_ids[row].clone(),
                    topk_logps=None if topk_logps is None else topk_logps[row].float().clone(),
                 run_config=run_config)
            )
    return candidate_sets
