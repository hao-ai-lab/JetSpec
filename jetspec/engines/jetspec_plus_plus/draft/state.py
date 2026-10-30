
from __future__ import annotations
from dataclasses import dataclass
import os
import time
import torch
from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import FrontInputs
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomePacket
from jetspec.engines.jetspec_plus_plus.wire.contract import TokenBuffers
from jetspec.engines.jetspec_plus_plus.wire.contract import outcome_message_views
from jetspec.engines.jetspec_plus_plus.wire.contract import rng_words

class PublishedRowRing:
    """Three reusable device snapshots, indexed by published round modulo 3."""

    def __init__(self, *, device: torch.device | str, rows: int, lanes: int, run_config=None):
        self.rows = torch.zeros(3, rows, lanes, dtype=torch.int64, device=device)
        self.p_top1 = torch.zeros(3, rows, lanes, dtype=torch.float32, device=device)
        self.passes_run = torch.zeros(3, rows, dtype=torch.int64, device=device)
        self.keys = torch.full((3, rows, 2), -1, dtype=torch.int64, device=device)
        self.rounds = torch.full((3,), -1, dtype=torch.int64, device=device)
        from jetspec.engines.jetspec_plus_plus.runtime.profile import capture_candidates
        if capture_candidates(run_config=run_config):
            self.draft_top16_ids = torch.full((3, rows, lanes, 16), -1, dtype=torch.long, device=device)
            self.draft_top16_probabilities = torch.zeros((3, rows, lanes, 16), device=device)
            self.draft_token_probabilities = torch.zeros((3, rows, lanes), device=device)
            self.draft_pinned = torch.zeros((3, rows, lanes), device=device, dtype=torch.bool)

    def publish_candidates(self, round_id, ids, probabilities, token_probabilities, pinned):
        """Optional graph-owned distribution snapshots in the same row order."""
        slot = torch.arange(3, device=self.rows.device) == round_id % 3
        self.draft_top16_ids.copy_(torch.where(slot[:, None, None, None], ids[None],
                                             self.draft_top16_ids))
        self.draft_top16_probabilities.copy_(torch.where(slot[:, None, None, None], probabilities[None],
                                                       self.draft_top16_probabilities))
        self.draft_token_probabilities.copy_(torch.where(slot[:, None, None], token_probabilities[None],
                                                       self.draft_token_probabilities))
        self.draft_pinned.copy_(torch.where(slot[:, None, None], pinned[None], self.draft_pinned))

    def select_pending_candidates(self, packet):
        index, _, valid = self.index_for_pending(packet)
        slot = packet[OutcomeOffset.ROUND].remainder(3).reshape(1)
        return tuple(torch.where(valid, value.index_select(0, slot).squeeze(0)
                                 .index_select(0, index).squeeze(0), torch.zeros_like(value[0, 0]))
                     for value in (self.draft_top16_ids, self.draft_top16_probabilities,
                                   self.draft_token_probabilities, self.draft_pinned))

    def publish(self, round_id: int | torch.Tensor, rows: torch.Tensor,
                keys: torch.Tensor, p_top1: torch.Tensor | None = None,
                passes_run: torch.Tensor | None = None) -> None:
        if rows.shape != self.rows.shape[1:] or keys.shape != self.keys.shape[1:]:
            raise ValueError("published row snapshot has wrong shape")
        if p_top1 is None:
            p_top1 = torch.zeros_like(self.p_top1[0])
        if (p_top1.shape != self.p_top1.shape[1:] or p_top1.dtype != torch.float32
                or p_top1.device != self.p_top1.device):
            raise ValueError("published top-1 snapshot must be float32[R, lanes]")
        if passes_run is None:
            passes_run = torch.zeros_like(self.passes_run[0])
        if (passes_run.shape != self.passes_run.shape[1:] or passes_run.dtype != torch.int64
                or passes_run.device != self.passes_run.device):
            raise ValueError("published pass labels must be int64[R]")
        if isinstance(round_id, torch.Tensor):
            if round_id.numel() != 1 or round_id.dtype != torch.int64 or round_id.device != self.rows.device:
                raise ValueError("published round must be an int64 device scalar")
            round_word = round_id
        else:
            round_word = torch.as_tensor(round_id, dtype=torch.int64, device=self.rows.device)
        slot = torch.arange(3, device=self.rows.device) == round_word.remainder(3)
        self.rows.copy_(torch.where(slot[:, None, None], rows[None], self.rows))
        self.p_top1.copy_(torch.where(slot[:, None, None], p_top1[None], self.p_top1))
        self.passes_run.copy_(torch.where(slot[:, None], passes_run[None], self.passes_run))
        self.keys.copy_(torch.where(slot[:, None, None], keys[None], self.keys))
        self.rounds.copy_(torch.where(slot, round_word, self.rounds))

    def empty(self, round_id: int | torch.Tensor) -> None:
        self.publish(round_id, torch.zeros_like(self.rows[0]),
                     torch.full_like(self.keys[0], -1),
                     torch.zeros_like(self.p_top1[0]),
                     torch.zeros_like(self.passes_run[0]))

    def publish_masked(self, round_id: int | torch.Tensor, rows: torch.Tensor,
                       keys: torch.Tensor, published: torch.Tensor,
                       p_top1: torch.Tensor | None = None,
                       passes_run: torch.Tensor | None = None) -> None:
        """Snapshot the wire rows, or advance the ring with an empty round."""
        if published.shape != () or published.device != self.rows.device:
            raise ValueError("published must be a device scalar")
        visible = published.to(torch.bool)
        self.publish(round_id, torch.where(visible, rows, torch.zeros_like(rows)),
                     torch.where(visible, keys, torch.full_like(keys, -1)),
                     None if p_top1 is None else torch.where(
                         visible, p_top1, torch.zeros_like(p_top1)),
                     None if passes_run is None else torch.where(
                         visible, passes_run, torch.zeros_like(passes_run)))

    def select(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row, _, found = self.select_with_top1(packet)
        return row, found

    def select_with_top1(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, top1, exact, _ = self.select_for_outcome(packet)
        return row, top1, exact

    def select_for_outcome(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor]:
        """Select the exact verified row, or the eager k-row for an AR miss."""
        index, matches, valid = self.index_for_outcome(packet)
        published_round = packet[OutcomeOffset.ROUND] - 1
        slot = published_round.remainder(3).reshape(1)
        rows = self.rows.index_select(0, slot).squeeze(0)
        top1 = self.p_top1.index_select(0, slot).squeeze(0)
        return (rows.index_select(0, index).squeeze(0),
                top1.index_select(0, index).squeeze(0), matches, valid)

    def index_for_outcome(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                torch.Tensor]:
        """Return the row verified in the preceding published round."""
        published_round = packet[OutcomeOffset.ROUND] - 1
        slot = published_round.remainder(3).reshape(1)
        keys = self.keys.index_select(0, slot).squeeze(0)
        valid = ((self.rounds.index_select(0, slot).squeeze(0) == published_round)
                 & (keys[:, 0] >= 0) & (keys[:, 1] >= 0))
        matches = (valid & (keys[:, 0] == packet[OutcomeOffset.K])
                   & (keys[:, 1] == packet[OutcomeOffset.A])
                   & (packet[OutcomeOffset.K] >= 0))
        same_k = valid & (keys[:, 0] == packet[OutcomeOffset.TRUE_K])
        fallback = torch.where(same_k.any(), same_k.to(torch.int64).argmax(),
                               valid.to(torch.int64).argmax())
        index = torch.where(matches.any(), matches.to(torch.int64).argmax(), fallback).reshape(1)
        return index, matches.any(), valid.any()

    def index_for_pending(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                torch.Tensor]:
        """Select the current candidate row that eager stages for the next round."""
        published_round = packet[OutcomeOffset.ROUND]
        slot = published_round.remainder(3).reshape(1)
        keys = self.keys.index_select(0, slot).squeeze(0)
        valid = ((self.rounds.index_select(0, slot).squeeze(0) == published_round)
                 & (keys[:, 0] >= 0) & (keys[:, 1] >= 0))
        same_k = valid & (keys[:, 0] == packet[OutcomeOffset.TRUE_K])
        exact = same_k & (keys[:, 1] == packet[OutcomeOffset.RECOVERY])
        fallback = torch.where(same_k.any(), same_k.to(torch.int64).argmax(),
                               valid.to(torch.int64).argmax())
        index = torch.where(exact.any(), exact.to(torch.int64).argmax(), fallback).reshape(1)
        return index, exact.any(), valid.any()

    def select_pending(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                             torch.Tensor, torch.Tensor,
                                                             torch.Tensor]:
        index, exact, valid = self.index_for_pending(packet)
        slot = packet[OutcomeOffset.ROUND].remainder(3).reshape(1)
        row = self.rows.index_select(0, slot).squeeze(0).index_select(0, index).squeeze(0)
        top1 = self.p_top1.index_select(0, slot).squeeze(0).index_select(0, index).squeeze(0)
        passes = self.passes_run.index_select(0, slot).squeeze(0).index_select(0, index).squeeze(0)
        return row, top1, passes, exact, valid


class ChainTapBuffers:
    """Sequence-owned fixed exact-tap prefix, seeded at prefill and extended in Seg A."""

    def __init__(self, *, device, capacity: int, lanes: int, hidden: int,
                 dtype=torch.bfloat16):
        # Inactive lanes write into distinct scratch rows after the live prefix.
        self.backing = torch.zeros(capacity + lanes, 5, hidden, device=device, dtype=dtype)
        self.storage = self.backing[:capacity]
        self.lanes = lanes
        self.lane = torch.arange(lanes, device=device)

    def seed(self, prompt_taps: torch.Tensor) -> None:
        if (prompt_taps.ndim != 3 or prompt_taps.shape[1:] != self.storage.shape[1:]
                or prompt_taps.shape[0] > self.storage.shape[0]
                or prompt_taps.device != self.storage.device
                or prompt_taps.dtype != self.storage.dtype):
            raise ValueError("prefill taps do not match static exact-tap storage")
        self.storage[:prompt_taps.shape[0]].copy_(prompt_taps)

    def append(self, packet: torch.Tensor, taps: torch.Tensor) -> None:
        if taps.shape != (self.lanes, *self.storage.shape[1:]) or taps.device != self.storage.device:
            raise ValueError("outcome tap block does not match static storage")
        length = packet[OutcomeOffset.L]
        count = packet[OutcomeOffset.TAP_COUNT]
        # The final recovery ID has no exact target tap until next round.
        start = length - count
        capacity = self.storage.shape[0]
        write = ((length <= capacity) & (start >= 0)
                 & (count >= 0) & (count <= self.lanes) & (self.lane < count))
        rows = torch.where(write, start + self.lane, capacity + self.lane)
        self.backing.index_copy_(0, rows, taps)

    def prefix_padded(self, rows: int) -> torch.Tensor:
        if rows > self.storage.shape[0]:
            raise ValueError("tap prefix bucket exceeds sequence capacity")
        return self.storage[:rows].reshape(rows, -1)

    def decoder_prefix(self, packet: torch.Tensor) -> torch.Tensor:
        """Exact tap-4 rows for every newly verified position."""
        length = packet[OutcomeOffset.L]
        count = packet[OutcomeOffset.TAP_COUNT]
        lane = self.lane
        positions = length - count + lane
        selected = self.storage.index_select(
            0, positions.clamp(0, self.storage.shape[0] - 1))[:, -1]
        valid = (lane < count) & (positions >= 0) & (positions < self.storage.shape[0])
        return torch.where(valid[:, None], selected, torch.zeros_like(selected))


def rng_rows_from_one_draw(capacity: int, width: int, seed: int, prompt_index: int):
    """Recover the shared CPU randint words, or return None if the rail differs."""
    def draw(high):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) ^ int(prompt_index))
        return torch.randint(high, (width,), generator=generator)

    low, high = 1 << 20, (1 << 20) - 1
    first, second = draw(low), draw(high)
    words = first + low * torch.remainder(second - first, high)
    if bool((words >= 1 << 32).any()):
        return None
    rows = torch.zeros(capacity + 1, width, dtype=torch.int64)
    committed = torch.arange(2, capacity + 1, dtype=torch.int64)[:, None]
    rows[2:] = torch.remainder(words[None, :], committed)
    for length in (2, capacity):
        if not torch.equal(rows[length], draw(length)):
            return None
    return rows


class ChainRngTable:
    """Rail-identical CPU RNG indices for every possible committed length.

    The seeded CPU generator resets for each length in the eager first-block
    build. Precompute these tiny index rows at admission; Seg A gathers the
    row selected by the device L without a host read or a pageable upload.
    """

    def __init__(self, *, capacity: int, lanes: int, seed: int,
                 prompt_index: int, device, run_config=None):
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
        self.run_config = as_run_config(run_config)
        if capacity < 2:
            raise ValueError("RNG table needs room for the query token")
        width = rng_words(lanes)
        one_draw = setting(self.run_config, 'engine.engine_settings.chain_admit_rng_one_draw', "0") == "1"
        check = one_draw and setting(self.run_config, 'engine.engine_settings.chain_admit_rng_one_draw_check', "0") == "1"
        rows = (rng_rows_from_one_draw(capacity, width, seed, prompt_index)
                if one_draw else None)
        fallback = one_draw and rows is None

        def loop_rows():
            result = torch.zeros(capacity + 1, width, dtype=torch.int64)
            for committed in range(2, capacity + 1):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed) ^ int(prompt_index))
                result[committed] = torch.randint(committed, (width,), generator=generator)
            return result

        if rows is None:
            rows = loop_rows()
        if check:
            loop = rows if fallback else loop_rows()
            mismatches = int((rows != loop).sum())
            if mismatches:
                rows, fallback = loop, True
            assert torch.equal(rows, loop)
            print(f"CHAIN_ADMIT_RNG_ONE_DRAW_CHECK mismatch={mismatches} fallback={int(fallback)}")
        if one_draw and torch.device(device).type == "cuda":
            self.rows = rows.pin_memory().to(device=device, non_blocking=True)
        else:
            self.rows = rows.to(device=device)

    def fill(self, rp: torch.Tensor) -> None:
        if rp.dtype != torch.int64 or rp.shape[0] < Offset.RNG_IDX + self.rows.shape[1]:
            raise ValueError("RNG table requires the static round params")
        committed = rp[Offset.L] + 1
        indices = self.rows.index_select(0, committed.clamp(0, self.rows.shape[0] - 1).reshape(1))
        rp[Offset.RNG_IDX:Offset.RNG_IDX + self.rows.shape[1]].copy_(indices[0])


@dataclass
class ChainLengthTracker:
    """Host key estimate from the latest completed async packet mirror."""

    capacity: int
    lanes: int
    L_known: int
    mirrored_round: int
    eos: bool = False

    def key_length(self, round_id: int) -> int:
        age = round_id - self.mirrored_round
        if age < 0:
            raise ValueError("chain round precedes the mirrored packet")
        return self.L_known + (age + 1) * self.lanes

    def bucket(self, round_id: int, *, step: int) -> int:
        if step not in (128, 256):
            raise ValueError("chain key step must be 128 or 256")
        length = self.key_length(round_id)
        return ((length + step - 1) // step) * step

    def at_capacity(self, round_id: int) -> bool:
        return self.key_length(round_id) > self.capacity

    def mirror(self, *, round_id: int, L: int, eos: bool) -> None:
        if round_id <= self.mirrored_round:
            return
        self.L_known, self.mirrored_round, self.eos = L, round_id, eos
