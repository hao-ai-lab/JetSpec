"""The serving environment contract: every knob the serving tree reads, and a startup check.

The names are frozen (launchers, banked cells and policy files spell them); this module
declares them, it does not rename them.  A knob is an ``os.environ`` name read under
the serving engine (``vendor/`` and tests excluded) whose prefix is one of
``SCANNED_PREFIXES``.  ``serve/tests/test_env_registry.py`` scans the tree and fails, naming
them, when a read name is missing here.

``validate_environment()`` runs once, from ``serve/bench/bench.py``, before the engine is
built.  It refuses configurations that would serve something else quietly:

  (a) a JETSPEC_*/E04_*/W270_* variable that is present but not registered (a typo means
      "off" everywhere else in the tree); ``JETSPEC_ALLOW_UNREGISTERED_ENV=1`` downgrades
      this one check to a printed warning;
  (b) a registered knob whose value is outside its allowed set (most sites compare with
      ``==`` and treat every other spelling as "off");
  (c) a knob that only acts under another flag, set while that flag is off.  This one is
      asked of a ``--jetspec`` run only: the banked AR baselines (``cell_w2*.sh`` mode ``ar``)
      export JETSPEC_KSET=one-k beside JETSPEC_ROWS=0 on purpose, and without ``--jetspec``
      the whole JETSPEC_* family is inert by declaration, not by accident.
      The block-head rules are the exception and are asked of every run, because
      ``--blockhead`` is never a ``--jetspec`` run: JETSPEC_BLOCKHEAD_GRAPH=1 needs
      JETSPEC_BLOCKHEAD set, and refuses TORCHDYNAMO_DISABLE=1 (the reading site raises).

N=1 JetSpec serving also requires the critical-path miss redraft.
Named paper recipes are historical records: dry resolution is allowed, while a live
N=1 launch points to the submission commit for reproduction. C > 1 is unchanged.

It changes no default and reads no tensor: a valid configuration is untouched.

Stdlib only, so it imports (and ``python -m jetspec.engines.jetspec_plus_plus.draft.env`` runs) without torch.
"""
from __future__ import annotations

import difflib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

REQUIRED = "REQUIRED"      # the reading site raises when the name is absent (os.environ[...] / _required_env)
NO_DEFAULT = "NO_DEFAULT"  # read with .get(name): absent is None, and every comparison then reads "off"

SCANNED_PREFIXES = ("JETSPEC_", "E04_", "W2", "W3", "PRED_", "SSD_")
VALIDATED_PREFIXES = ("JETSPEC_", "E04_", "W2", "W3", "PRED_")
ALLOW_UNREGISTERED = "JETSPEC_ALLOW_UNREGISTERED_ENV"

FLAG = ("0", "1")


@dataclass(frozen=True)
class Knob:
    name: str
    kind: str                # flag | enum | int | float | str | path | csv
    default: str             # the literal default of the first-executed site, or REQUIRED / NO_DEFAULT
    allowed: tuple = None    # closed value set, when the code compares the value with == / in
    reader: str = "serve"    # serve: read by the serving engine; launcher: set for a launcher
                             # or a sibling component and merely inherited by the serving process
    note: str = ""


class EnvironmentContractError(ValueError):
    """The process environment breaks the serving contract; the message lists every breach."""


def _knobs(*rows):
    table = {}
    for row in rows:
        knob = Knob(*row) if isinstance(row, tuple) else row
        if knob.name in table:
            raise AssertionError(f"duplicate registry entry {knob.name}")
        table[knob.name] = knob
    return table


def _flag(name, default="0", note=""):
    return Knob(name, "flag", default, FLAG, note=note)


def _only(name, *values, default=NO_DEFAULT, note=""):
    return Knob(name, "enum", default, tuple(values), note=note)


def _launcher(name, note):
    return Knob(name, "str", NO_DEFAULT, reader="launcher", note=note)


_E04_UNSET_OR_1 = "the reading site raises on any value but 1; unset is off"
# E04_LAB_SWAP: off, or any +-joined set of distinct swap modes in any order (tau_lab.swap_modes)
_E04_LAB_SWAP_MODES = ("slot1_pre_j0", "first_wrong_refresh", "pre_verify")
_E04_LAB_SWAPS = tuple("+".join(order) for size in range(1, len(_E04_LAB_SWAP_MODES) + 1)
                       for chosen in itertools.combinations(_E04_LAB_SWAP_MODES, size)
                       for order in itertools.permutations(chosen))


def _e04_lab_swaps_with(mode):
    return tuple(value for value in _E04_LAB_SWAPS if mode in value.split("+"))


# each swap mode's own margin knob (tau_lab.MODE_MARGINS)
_E04_LAB_MODE_MARGINS = (("slot1_pre_j0", "E04_LAB_SWAP_MARGIN_SLOT1"),
                         ("first_wrong_refresh", "E04_LAB_SWAP_MARGIN_REFRESH"),
                         ("pre_verify", "E04_LAB_SWAP_MARGIN_PREVERIFY"))

REGISTRY = _knobs(
    Knob("CRUCIBLE_TOOLCHAIN_KEY", "str", NO_DEFAULT, note="optional compile-pin toolchain identity"),
    Knob("TORCHINDUCTOR_CACHE_DIR", "path", NO_DEFAULT, note="compile cache path; required when recording tuning without a pin"),
    _flag("W446_KERNEL_PROFILE", NO_DEFAULT, note="roundtrace kernel profiling; absent disables profiling"),
    Knob("W632_HEAD_CAPTURE", "path", NO_DEFAULT, note="optional batch head capture output prefix"),
    _flag("W632_HEAD_FUSE", "1", note="batch head epilogue fusion under the existing graph guards"),
    _flag("JETSPEC_ANATOMY_LITE", "0", note="query-only round timing; 0 disables observation"),
    _flag("JETSPEC_DEVICE_HOST_STAMPS", note="opt-in device-pair host partition using existing step stamps; no device sync"),
    _flag("JETSPEC_COMPLETED_TAPS_DUMP_ONLY", note="collect completed tap CPU snapshots only when a dump is requested"),
    Knob("JETSPEC_SERVE_POLICY", "path", NO_DEFAULT,
         note="strict jetspec.serve_policy/1 JSON for default-off serving mechanisms"),
    # --- e04 opt-in policy flags (heads/, live_rows.py) ---------------------------------------
    _only("E04_DECODER_TAIL_SEED", "admission"),
    _flag("E04_JOINT_OFFERS", NO_DEFAULT),
    _only("E04_KPRED_POSTERIOR", "1", note=_E04_UNSET_OR_1),
    _only("E04_KPRED_RENORM", "1", note=_E04_UNSET_OR_1 + "; requires JETSPEC_KPRED_FRAME=wire"),
    _only("E04_KSCORE", "blend", "blend_dec"),
    _only("E04_RECORD_FLAGS", "1", note=_E04_UNSET_OR_1),
    _only("E04_RECORD_PTOP1", "1", note=_E04_UNSET_OR_1),
    _only("E04_RECORD_QSHADOW", "1", note=_E04_UNSET_OR_1),
    _flag("JETSPEC_CHAIN_WIRE_RECORD", note="post-replay chain calibration records; requires W280_RECORD and both E04 record flags"),
    _only("E04_ROWRULE", "joint"),
    # --- e04 tau lab (tau_lab.py): offline tau research on the eager N=1 rows path ----------
    _only("E04_LAB_REFRESH_CUTOFF", "draft", "decoder_f", "oracle", default="draft",
          note="the adapter refresh's k-hat: draft rule, min(draft, decoder first disagreement), "
               "or min(draft, reference k)"),
    _only("E04_LAB_SWAP", "off", *_E04_LAB_SWAPS, default="off",
          note="decoder top-1 swaps, off or a +-joined set of: J0 input slot 1, the slot after "
               "the refresh cutoff, the first disagreement of the final block"),
    Knob("E04_LAB_SWAP_MARGIN", "float", "0.0", note="the swap's probability margin"),
    Knob("E04_LAB_REF_TOKENS", "path", NO_DEFAULT,
         note="JSON {requests: {prompt index: output token ids}} of a knob-off T=0 read"),
    _only("E04_LAB_PASS_LABEL", "J2", note="every pass after J0 reads the J2 thresholds"),
    _flag("E04_LAB_REFRESH_EVERY", note="the adapter refresh (with the lab hooks) before every pass after J0"),
    _only("E04_LAB_EXTRA_PASSES", *(str(count) for count in range(9)), default="0",
          note="passes after the recipe's, each after a refresh; labels stay the recipe's last"),
    _only("E04_LAB_SHADOW_AR", *(str(count) for count in range(16)), default="0",
          note="slots 1..n decoded greedily by adapter + decoder before J0, then pinned"),
    _flag("E04_LAB_EXACT", note="the lab hooks act in the exact anchor0 round too (admission, miss "
                                "redraft), over the exact prefix with no tail"),
    _only("E04_LAB_SWAP_KHAT_MAX", *(str(count) for count in range(1, 16)),
          note="the first_wrong_refresh swap only on rows whose draft k-hat is below this"),
    _only("E04_LAB_SWAP_SLOT_MAX", *(str(count) for count in range(1, 16)),
          note="the first_wrong_refresh swap only at slots up to this"),
    *(Knob(name, "float", NO_DEFAULT, note=f"the {mode} swap's margin in place of E04_LAB_SWAP_MARGIN "
                                           f"(unset: E04_LAB_SWAP_MARGIN)")
      for mode, name in _E04_LAB_MODE_MARGINS),
    _only("E04_LAB_PREVERIFY_ROUNDS", *(str(count) for count in range(1, 16)), default="1",
          note="the pre_verify swap repeated up to n rounds per row, each after the adapter and decoder "
               "re-read the edited block"),
    _only("E04_LAB_ROWS_MIN", *(str(count) for count in range(1, 17)),
          note="after the joint rule, the top n pairs by joint score when fewer than n passed tau"),
    _flag("E04_LAB_SWAP_ROWS", note="after the joint rule, the pair (s - 1, the decoder's top offer) per "
                                    "decoder-written slot s of the verified block"),
    _only("E04_LAB_KSCORE_WRITTEN", "disagree", "draft_p", "both",
          note="the BLEND k score at the verified block's decoder-written slots: no argmax term, the "
               "draft model's p of the written token, or both"),
    _flag("E04_LAB_SEED_EXACT", note="a forced exact round whose round-ahead rows hit starts from the hit "
                                     "row's final block (the correction pinned, the row's lab pins dropped)"),
    _only("E04_LAB_SEED_PASSES", *(str(count) for count in range(5)), default="0",
          note="the drafter passes a seeded exact round runs from the seed before pre_verify (0: none)"),
    _only("E04_LAB_RECORD_TOPK", *(str(count) for count in range(1, 33)),
          note="records only: the drafter's and the decoder's top-n at every block slot, per refresh and "
               "pre_verify read"),
    Knob("E04_LAB_SWAP_RULE_FILE", "path", NO_DEFAULT,
         note="JSON parameters of the two-expert rule (both experts' top-n, a score, a slot x k-hat bias table, "
              "pin) that decides the first_wrong_refresh and pre_verify swaps (tau_lab.SwapRule)"),
    _only("E04_LAB_AFTER_SWAP_FILL", "keep", "mask", "noise", "context", "decoder", default="keep",
          note="the next pass's input at the unpinned slots after a first_wrong_refresh swap: the old tokens, "
               "the mask token, seeded uniform token ids, or the round's J0 context draws; decoder (tl11): the "
               "refresh decoder read's argmax at the slots it covers, unpinned"),
    Knob("E04_LAB_FILL_SEED", "int", "0", note="the seed of E04_LAB_AFTER_SWAP_FILL=noise"),
    _flag("E04_LAB_PIN_AGREED", note="the first_wrong_refresh refresh under decoder_f also pins slots 1..khat_used"),
    _only("E04_LAB_TAIL_PASSES", "0", "1", "2", default="0",
          note="drafter-only passes after the recipe's last (no refresh, no decoder read); labels stay the "
               "recipe's last"),
    # --- checkpoint, adapter and head paths --------------------------------------------------
    Knob("JETSPEC_ADAPTER_CONFIG", "path", NO_DEFAULT),
    Knob("JETSPEC_ADAPTER_HEAD", "path", NO_DEFAULT),
    Knob("JETSPEC_CHECKPOINT_DIR", "path", NO_DEFAULT),
    Knob("JETSPEC_DRAFTER_CORRECTION_HEAD", "path", ""),
    Knob("JETSPEC_EAGLE_DRAFT_PATH", "path", NO_DEFAULT,
         note="bench.py --eagle: the draft head directory, in place of the size table (the EAGLE-3 head loader)"),
    Knob("JETSPEC_HEADS_DIR", "path", REQUIRED, note="required once the predictor k set or the decoder anchor loads"),
    Knob("JETSPEC_PINS", "path", NO_DEFAULT, note="required by the c1_tap_aligned_h2h adapter"),
    Knob("JETSPEC_TARGET_MODEL_PATH", "path", REQUIRED, note="required when the anchor decoder loads"),
    Knob("JETSPEC_RUN_MANIFEST_DIR", "path", NO_DEFAULT, note="set by bench.py so the spawned draft process writes its run-manifest part"),
    Knob("JETSPEC_ROUNDTRACE", "path", NO_DEFAULT, reader="launcher",
         note="opt-in trace directory consumed by serve/roundtrace before serving starts"),
    Knob("JETSPEC_ROUNDTRACE_DEVICE", "flag", "0", FLAG, reader="launcher",
         note="enable buffered CUDA event pairs in the roundtrace recorder"),
    Knob("JETSPEC_ROUNDTRACE_EVENT_PAIRS", "int", "8192", reader="launcher",
         note="fail-closed per-rank roundtrace CUDA event-pair pool size"),
    Knob("JETSPEC_TARGET_RELEASE_WEIGHTS", "path", NO_DEFAULT),
    Knob("JETSPEC_MPS_DRAFT_PRIORITY", "enum", NO_DEFAULT, ("0", "1"),
         note="MPS priority for spawned async draft client only"),
    Knob("JETSPEC_MPS_DRAFT_ACTIVE_THREAD_PERCENTAGE", "int", NO_DEFAULT,
         note="MPS active thread percentage for spawned async draft client only"),
    Knob("JETSPEC_DATASET", "str", REQUIRED, note="keys the threshold bundle; required on predictor/decoder serving"),
    Knob("JETSPEC_TARGET_DTYPE", "enum", NO_DEFAULT, ("bf16", "fp32")),
    _flag("JETSPEC_HYBRID_FP32_LINEAR_BF16_STORAGE",
          note="FP32 hybrid projection compute with exact BF16 checkpoint storage; other source dtypes retain FP32 storage"),
    _flag("JETSPEC_HYBRID_FUSED_NORM", note="opt-in FLA RMSNorm for the Qwen3.5 text target"),
    _flag("JETSPEC_HYBRID_FAST_SNAPSHOT", note="opt-in single-launch hybrid GDN verify snapshots"),
    _flag("JETSPEC_HYBRID_VERIFY_TRITON", note="opt-in hybrid Triton paged attention qualification"),
    _flag("JETSPEC_HYBRID_VERIFY_FLASHINFER", note="opt-in 27B hybrid device pair with DSpark FlashInfer verification"),
    Knob("JETSPEC_BLOCK_SIZE_READ", "enum", "16", ("4", "8", "16", "32"),
         note="serving verify block; widths other than 16 read a head trained at block 16"),
    _only("JETSPEC_DRAFT_DTYPE", "bf16", "fp32",
          note="divergent default: None at draft_runner.py (only fp32 acts; unset serves the checkpoint "
               "dtype), 'bf16' at blockhead_serve.py (the block head is loaded in the named dtype)"),
    # --- block-head mode (blockhead_serve.py, speculator_blockhead.py; bench.py --blockhead) -
    Knob("JETSPEC_BLOCKHEAD", "path", NO_DEFAULT,
         note="a block-head checkpoint directory; set, it replaces the draft runner in the sync slot"),
    _flag("JETSPEC_BLOCKHEAD_GRAPH",
          note="read only while the block-head speculator is built; 1 refuses TORCHDYNAMO_DISABLE=1"),
    _flag("JETSPEC_BLOCKHEAD_ATTN_LEN"),
    _flag("JETSPEC_BLOCKHEAD_GLUE_LEAN"),
    _flag("JETSPEC_BLOCKHEAD_QKV_FUSED"),
    _flag("JETSPEC_BLOCKHEAD_LMHEAD_KN"),
    _flag("JETSPEC_BLOCKHEAD_BENCH", note="measure 200 extension and query graph replays at startup"),
    Knob("JETSPEC_BLOCKHEAD_DECISION_AUDIT", "path", NO_DEFAULT,
         note="compare each optimized draft with a stock graph on the same input"),
    _flag("JETSPEC_BLOCKHEAD_BATCH",
          note="opt-in one batched block-head forward for C>1, eager or graphed"),
    _flag("JETSPEC_BLOCKHEAD_EQUIV",
          note="diagnostic: replay the serial drafter on each batched input and record token differences"),
    _only("JETSPEC_BLOCKHEAD_TF32", "0", note="the reading site raises on any value but 0; unset leaves TF32 as found"),
    _only("JETSPEC_BLOCKHEAD_BLOCK", "4", "8",
          note="verify block (anchor included, k = block - 1); the head still runs its trained block and "
               "keeps the first k drafts; unset serves the trained block"),
    Knob("W277_BLOCKHEAD_PARITY_RECORD", "path", NO_DEFAULT,
         note="a diagnostic: the first 8 block-head rounds are saved under it; not prefix-refused"),
    # --- sync drafter mode (sync_drafter.py, speculator_sync_drafter.py; bench.py --jetspec-sync-drafter)
    Knob("JETSPEC_SYNC_DRAFTER", "path", NO_DEFAULT,
         note="a DFlash-class drafter checkpoint directory; set, it replaces the draft runner in the sync slot"),
    Knob("JETSPEC_SYNC_PASSES", "enum", "1", ("1", "2", "3", "4"),
         note="drafter forwards per round; above 1 needs a Jacobi-forcing (mask-free) drafter"),
    _flag("JETSPEC_SYNC_GRAPH", note="replay the drafter forward from CUDA graphs"),
    _flag("JETSPEC_SYNC_STATIC", default="1",
          note="1: a static context K/V cache extended by the committed rows; 0: rebuild the context every round"),
    # --- protocol ----------------------------------------------------------------------------
    Knob("JETSPEC_ANCHOR", "enum", "guessed", ("guessed", "oracle", "decoder")),
    Knob("JETSPEC_DRAFTS", "enum", REQUIRED, ("3", "7", "14", "15", "31")),
    Knob("JETSPEC_F", "enum", "1", ("1", "2", "3", "5")),
    _flag("JETSPEC_F_GRAPHS"),
    _flag("JETSPEC_FUSED_QKV"),
    _flag("JETSPEC_LMHEAD_FUSED"),
    _flag("JETSPEC_J0_LMHEAD_KN", note="N=1 draft-only dense KN vocab projection with K3 reduction"),
    _flag("JETSPEC_J0_GATEUP_PACKED", note="pack only the N=1 draft-row gate/up projections"),
    _flag("JETSPEC_J0_QKV_GLUE_FUSED", note="N=1 draft-row Q/K norm, RoPE and KV write in one launch"),
    _flag("JETSPEC_J0_NORM_GEMM_FUSED", note="BF16 N=1 lean draft-row RMSNorm GEMM prologue and gate/up epilogue"),
    _flag("JETSPEC_ROW_GEOMETRY_FUSED", note="N=1 BF16 row RoPE gathers and packed-mask update in one launch"),
    _flag("JETSPEC_DECODER_TOPK_PREFILTER"),
    _flag("JETSPEC_DECODER_GATEUP_PACKED", note="N>1 lean decoder gate/up projection uses one GEMM"),
    _flag("JETSPEC_DECODER_SINGLE_GATEUP_PACKED", note="N=1 lean decoder load-time gate/up packing"),
    _flag("JETSPEC_DECODER_SLOT_WRITE", note="N=1 merged decoder norm/RoPE writes K/V directly to slots"),
    _flag("JETSPEC_DECODER_GQA_VIEW", note="N=1 BF16 decoder attention broadcasts KV groups as views"),
    _flag("JETSPEC_DECODER_SELECT_RADIX", note="N=1 BF16 chain decoder selects via an original-probability radix cutoff"),
    Knob("JETSPEC_FUSE", "enum", "off", ("off", "norm_gemm", "mlp", "attn", "compile", "all")),
    _flag("JETSPEC_HOST_FAST"),
    _flag("JETSPEC_PERSEQ"),
    _flag("JETSPEC_VERIFY_PRESTAGE_MIXED"),
    _flag("JETSPEC_VERIFY_PRESTAGE_MIXED_CHECK"),
    _flag("JETSPEC_VERIFY_DEVICE"),
    _flag("JETSPEC_VERIFY_REFERENCE", note="8B N=1 T0 pair with reference host verification"),
    _only("JETSPEC_ENGINE", "eager", "reference", "eval", "fast",
         note="explicit engine preset; historical recipes retain their environment"),
    _flag("JETSPEC_REFERENCE_CANDIDATE_SOCKET", note="same-host reference candidate bytes over a Unix socket"),
    _flag("JETSPEC_REFERENCE_CANDIDATE_SPIN", note="dedicate one CPU reader to polling the reference candidate socket"),
    _flag("JETSPEC_REFERENCE_CANDIDATE_MAIN", note="receive and decode reference candidate bytes at main-thread intake"),
    _flag("JETSPEC_REFERENCE_CANDIDATE_SHADOW", note="compare every socket candidate byte with the original store"),
    _flag("JETSPEC_REFERENCE_HOST_SELECT", note="select private reference candidates on the main thread without CPU tensor churn"),
    _flag("JETSPEC_REFERENCE_OUTCOME_HOST", note="reuse reference CPU header storage and compute its digest with host integers"),
    _flag("JETSPEC_REFERENCE_COMMAND_SOCKET", note="deliver reference chain commands over the reverse candidate socket"),
    _flag("JETSPEC_REFERENCE_TAP_PREPARE", note="prepare full reference tap payload before host acceptance"),
    _flag("JETSPEC_VERIFY_OUTCOME_FUSED"),
    _flag("JETSPEC_VERIFY_OUTCOME_FUSED_CHECK"),
    _flag("SSD_VERIFY_DEVICE", note="N=1 block-head or sync-drafter target device verify"),
    _flag("SSD_VERIFY_SYNC_AUDIT", note="count host/device synchronizations in block-head device verify"),
    Knob("JETSPEC_VERIFY_DEVICE_TRANSPORT", "enum", "chain", ("chain", "legacy"),
         note="legacy uses the current draft rank request wire during verify-device reads"),
    _flag("JETSPEC_VERIFY_SYNC_AUDIT"),
    _flag("JETSPEC_VERIFY_HOST_CENSUS",
          note="a47: per-round host segment stamps of the N=1 device-verify round (verify_host_census)"),
    _flag("JETSPEC_VERIFY_STAGE_HOST"),
    _flag("JETSPEC_CANDIDATE_DECODE_LEAN"),
    _flag("JETSPEC_VERIFY_SELECT_LEAN"),
    _flag("JETSPEC_OUTCOME_WIRE_RING"),
    _flag("JETSPEC_OUTCOME_PACK_PRESYNC"),
    _flag("JETSPEC_OUTCOME_PACK_PRESYNC_CHECK"),
    _flag("JETSPEC_OUTCOME_PREPOST_NO_ACK",
          note="enqueue preposted NCCL outcomes without the redundant host receive-ready get"),
    _flag("JETSPEC_PAIR_OUTCOME_PUSH"),
    _flag("JETSPEC_HYBRID_PAIR", note="N=1 Qwen3.5 state and layer-63 KV pair extension"),
    _flag("JETSPEC_HYBRID_C1_LEAN", note="opt-in offset-norm/partial-RoPE/gated hybrid adapter glue for query and refresh"),
    _flag("JETSPEC_HYBRID_C1_FUSED", note="opt-in Triton offset-norm statistics, partial rotary/cache and SiLU fusion for hybrid query, refresh and prefix extension"),
    _flag("JETSPEC_HYBRID_C1_GLUON", note="default-off Blackwell adapter head norm/partial-RoPE/KV-write fusion for hybrid lean and fused paths"),
    _flag("JETSPEC_HYBRID_VERIFY_AUDIT", note="diagnostic host read of verify top-two logits and emitted-token argmax assertion"),
    _flag("JETSPEC_HYBRID_STATE_AUDIT", note="diagnostic host-synchronized SHA256 of transferred state"),
    _flag("JETSPEC_HYBRID_TRANSFER_TIMING", note="CUDA-event timing of hybrid outcome receive wait"),
    _flag("JETSPEC_PAIR_OUTCOME_PUSH_CHECK"),
    _flag("JETSPEC_VERIFY_SELECT_HOST"),
    _flag("JETSPEC_VERIFY_RECORD_LATE"),
    _flag("JETSPEC_VERIFY_POST_LEAN"),
    _flag("JETSPEC_CANDIDATE_RECV_AHEAD"),
    _flag("JETSPEC_VERIFY_SELECT_LIST"),
    _flag("JETSPEC_VERIFY_PUBLISH_FIRST"),
    _flag("JETSPEC_CHAIN_CMD_LOCAL_COUNTER"),
    _flag("JETSPEC_VERIFY_STAGE_EARLY"),
    _flag("JETSPEC_VERIFY_PLAN_REUSE"),
    _flag("JETSPEC_VERIFY_PLAN_REUSE_CHECK"),
    _flag("JETSPEC_VERIFY_PLAN_LEAN"),
    _flag("JETSPEC_VERIFY_STAGE_BLOB"),
    _flag("JETSPEC_VERIFY_STAGE_BLOB_FAST"),
    _flag("JETSPEC_VERIFY_STAGE_BLOB_FAST_CHECK"),
    _flag("JETSPEC_VERIFY_INPUT_DIRECT"),
    _flag("JETSPEC_VERIFY_INPUT_VIEW"),
    Knob("JETSPEC_VERIFY_KERNEL_PROFILE", "path", NO_DEFAULT,
         note="diagnostic: sampled verify-rank kernel and idle-gap JSONL"),
    Knob("JETSPEC_VERIFY_KERNEL_PROFILE_EVERY", "int", "20",
         note="profile every Nth verify round when JETSPEC_VERIFY_KERNEL_PROFILE is set"),
    Knob("JETSPEC_ROW_LAYOUT", "enum", "kth", ("kth", "anchor0")),
    _flag("JETSPEC_W241_ASYNC_ROWS"),
    _flag("JETSPEC_AUTO"),
    Knob("JETSPEC_AUTO_PROFILE", "enum", "default", ("default", "math")),
    Knob("JETSPEC_TREE_B", "enum", "0", ("0", "31", "63", "127")),
    Knob("JETSPEC_TREE_ALGO", "enum", "accum_logp", ("accum_logp", "top2gap_fanout")),
    _flag("JETSPEC_TREE_SPINE"),
    # --- rows --------------------------------------------------------------------------------
    _flag("JETSPEC_ROWS"),
    _flag("JETSPEC_ROWS_ASSEMBLY"),
    _flag("JETSPEC_ROWS_DEFER_HOME", default="1"),
    _flag("JETSPEC_ROWS_DEVICE_CHECKS", default="1"),
    Knob("JETSPEC_ROWS_CAP", "int", "128"),
    _flag("JETSPEC_ROWS_CTX_EARLY"),
    _flag("JETSPEC_ROWS_CTX_INCR"),
    _flag("JETSPEC_ROWS_FAST"),
    _flag("JETSPEC_ROWS_FUSED"),
    _flag("JETSPEC_ROWS_FUSED_GRAPH"),
    _flag("JETSPEC_CTX_KV_FUSED"),
    _flag("JETSPEC_CTX_KV_DIRECT", note="N=1 packed context KV retains inference storage for fused writes"),
    _flag("JETSPEC_CTX_KV_APPEND", note="N=1 segmented chain with exact host L and miss redraft"),
    _flag("JETSPEC_CTX_KV_APPEND_CHECK", note="compare appended context KV with full projection"),
    _flag("JETSPEC_JPASS_GLUE_LEAN"),
    _flag("JETSPEC_ROWS_PASS_LOOP", note="exact eager fused row layers launched from C++"),
    _flag("JETSPEC_PASS_LOOP_DEVICE", note="exact fixed-pass bookkeeping in C++; deferred pinned metadata"),
    _flag("JETSPEC_ROWS_GRAPHS", note="divergent default: '0' at draft_runner.py, None at the heads sites"),
    _flag("JETSPEC_ROW_STAGE_GRAPH"),
    _flag("JETSPEC_ROWS_K3"),
    _flag("JETSPEC_K3_HAZARD_SPLIT"),
    _flag("JETSPEC_FRONT_STATIC"),
    Knob("JETSPEC_ROWS_RULE_CAP", "int", NO_DEFAULT, note="defaults to JETSPEC_ROWS_CAP"),
    Knob("JETSPEC_CHAIN_GRAPH", "enum", "0", ("0", "seg")),
    Knob("JETSPEC_CHAIN_RECORD", "path", NO_DEFAULT),
    Knob("JETSPEC_CHAIN_RECORD_ROUNDS", "int", "300"),
    Knob("JETSPEC_CHAIN_RECORD_SKIP_REQUESTS", "int", "0"),
    Knob("JETSPEC_CHAIN_REFERENCE_HEAD", "str", NO_DEFAULT),
    _flag("JETSPEC_CHAIN_WARM_REACHABLE"),
    _flag("JETSPEC_CHAIN_WARM_SINGLE_KEY",
          note="capture only the ceiling prefix key; shorter prefixes use the existing exact-length mask"),
    _flag("JETSPEC_CHAIN_CAPTURE_STAMPS"),
    Knob("JETSPEC_CHAIN_WARM_READ", "str", NO_DEFAULT,
         note="serve_read prompt lengths and token budget for reachable chain warm-up"),
    _flag("JETSPEC_CHAIN_ADMIT_BATCH"),
    _flag("JETSPEC_CHAIN_ADMIT_GRAPH"),
    _flag("JETSPEC_CHAIN_ADMIT_GRAPH_CHECK"),
    _flag("JETSPEC_CHAIN_ADMIT_TIMING"),
    _flag("JETSPEC_CHAIN_ADMIT_RNG_ONE_DRAW"),
    _flag("JETSPEC_CHAIN_ADMIT_RNG_ONE_DRAW_CHECK"),
    _flag("JETSPEC_CHAIN_ADMIT_HOST_LEAN"),
    _flag("JETSPEC_CHAIN_ADMIT_HOST_LEAN_CHECK"),
    _flag("JETSPEC_CHAIN_HANDOFF"),
    _flag("JETSPEC_CHAIN_HANDOFF_TIMING"),
    _flag("JETSPEC_CHAIN_PREPOST"),
    _flag("JETSPEC_CHAIN_PREPOST_TIMING"),
    _flag("JETSPEC_CHAIN_TAIL_TRACE",
          note="diagnostic: one CHAIN_TAIL_TRACE line per draft round with absolute host stamps"),
    _flag("JETSPEC_CHAIN_HOST_LEAN"),
    _flag("JETSPEC_CHAIN_POST_LEAN"),
    Knob("JETSPEC_CHAIN_KERNEL_PROFILE", "path", NO_DEFAULT,
         note="diagnostic: append per-kernel device time of sampled chain rounds to this JSONL"),
    Knob("JETSPEC_CHAIN_KERNEL_PROFILE_EVERY", "int", "20",
         note="profile every Nth chain round when JETSPEC_CHAIN_KERNEL_PROFILE is set"),
    Knob("JETSPEC_CHAIN_FIXED_CONFIG", "str", NO_DEFAULT,
         note="J<m>-R<n>; segmented capture retains its fixed inventory"),
    Knob("JETSPEC_CHAIN_J_SWITCH_L", "int", NO_DEFAULT,
         note="length bucket where fixed J2-R0 or J3-R0 switches to J1-R0; multiple of 128"),
    _flag("JETSPEC_CHAIN_C1_KEY_BUCKET"),
    _flag("JETSPEC_CHAIN_REFRESH_KEY_BUCKET"),
    _flag("JETSPEC_CHAIN_DECODER_KEY_BUCKET"),
    _flag("JETSPEC_DECODER_GLUE_FUSED"),
    _flag("JETSPEC_HYBRID_DECODER_REFRESH_GATES", note="draft hybrid refresh sigmoid and SiLU gate products"),
    _flag("JETSPEC_HYBRID_DECODER_REFRESH_STATE_LEAN", note="draft refresh writes only selected row states and omits unused FLA final states"),
    _flag("JETSPEC_HYBRID_DECODER_REFRESH_STAGING", note="row-isolated GDN conv staging and persistent compact vocab tiles"),
    _flag("JETSPEC_HYBRID_CONTEXT_DIRTY", note="hybrid N=1 dirty-slot context append with fused KV layout writes"),
    _flag("JETSPEC_DECODER_QKV_FUSED"),
    _flag("JETSPEC_DECODER_PREFIX_MERGE"),
    _flag("JETSPEC_DECODER_LMHEAD_KN"),
    Knob("JETSPEC_CHAIN_SEGA_GLUE", "enum", "0", ("0", "1", "2")),
    _flag("JETSPEC_CHAIN_INTAKE_FUSED"),
    _flag("JETSPEC_CHAIN_GLUE2_PROFILE"),
    _flag("JETSPEC_CHAIN_ATTN3"),
    _flag("JETSPEC_C1_ATTN_LEN"),
    _flag("JETSPEC_DECODER_ATTN_LEN"),
    _flag("JETSPEC_DECODER_ATTN_LEN_LONG"),
    _flag("JETSPEC_JPASS_ATTN_LEN"),
    Knob("JETSPEC_CHAIN_TAU_TABLE", "str", NO_DEFAULT,
         note="JSON tau for five J/R labels or ten J/R-C4/C8 labels for launch throughput"),
    _flag("JETSPEC_CHAIN_GRAPH_CHECK"),
    _flag("JETSPEC_CHAIN_ADMIT_GC", "1",
          note="full gc.collect + empty_cache at each chain admission (0: skip while capture-once keeps the graph set)"),
    _flag("JETSPEC_CHAIN_MISS_REDRAFT"),
    _flag("JETSPEC_CHAIN_MISS_REDRAFT_TIMING"),
    _flag("JETSPEC_CHAIN_GATHER_VEC"),
    _flag("JETSPEC_CHAIN_MR_GLUE_LEAN"),
    _flag("JETSPEC_CHAIN_PUBLISH_FUSED"),
    _flag("JETSPEC_C1_KV_SPLIT_LEAN"),
    _flag("JETSPEC_C1_SPAN_KV_FUSED"),
    _flag("JETSPEC_C1_APPEND_FOLD", note="requires C1_SPAN_KV_FUSED, FUSED_QKV, C1_GRAPH_LAYOUT=live, C1_ROPE_NORM, C1_KV_SPLIT_LEAN, FRONT_STATIC"),
    _flag("JETSPEC_C1_PREP_LEAN"),
    Knob("JETSPEC_CHAIN_MISS_REDRAFT_MAX_MS", "float", "10.0",
         note="positive timeout; inf waits for the same-round redraft without a cutoff"),
    _flag("JETSPEC_REFINE_K3"),
    _flag("JETSPEC_ROWS_STATIC"),
    _flag("JETSPEC_ROWS_STATIC_CHAIN"),
    _only("JETSPEC_ROW_SELECT", "budget"),
    Knob("JETSPEC_ROW_SELECT_EXP", "csv", "1,1"),
    _flag("JETSPEC_ROW_SELECT_LENGTH", NO_DEFAULT,
          note="divergent default: None at heads/lac.py (acts), '0' at live_rows.py (record only)"),
    _only("JETSPEC_G_ROW_BUCKETS", "pow2", "cap", note="only pow2 acts; cap names the unset behaviour"),
    # --- k set -------------------------------------------------------------------------------
    Knob("JETSPEC_KSET", "enum", NO_DEFAULT,
         ("one-k", "top-m", "all-k", "posterior", "posterior-topm", "predictor"),
         note="parsed only under JETSPEC_ROWS=1; predictor requires decoder BLEND or "
              "JETSPEC_LEGACY_KPRED_FORWARD=1"),
    _only("JETSPEC_KSET_AFTER_MISS", "zero", "off", note="only zero acts"),
    Knob("JETSPEC_KSET_CAP", "int", "16"),
    Knob("JETSPEC_KSET_M", "int", "7"),
    Knob("JETSPEC_KSET_MAX", "int", "15"),
    Knob("JETSPEC_KSET_ORDER", "csv", NO_DEFAULT, note="defaults to 0..hypotheses-1"),
    Knob("JETSPEC_KSET_Q", "float", "0",
         note="divergent default: '0' at live_rows.py (required under KSET=posterior), None at the draft_runner.py record"),
    Knob("JETSPEC_KPRED_COVERAGE", "str", "0.99", note="a threshold-bundle key, compared as a string"),
    _only("JETSPEC_KPRED_FRAME", "wire"),
    _flag("JETSPEC_LEGACY_KPRED_FORWARD", "0",
          note="deprecated 1 runs the standalone k predictor and requires JETSPEC_KSET=predictor; "
               "0 is the decoder-BLEND default"),
    # --- anchor decoder ----------------------------------------------------------------------
    Knob("JETSPEC_ANCHOR_CAP", "int", "16",
         note="divergent default: '16' at the offer sites, None where E04_ROWRULE=joint requires 32"),
    Knob("JETSPEC_ANCHOR_COVERAGE", "str", "0.99", note="a threshold-bundle key, compared as a string"),
    _flag("JETSPEC_ANCHOR_DREJ"),
    _only("JETSPEC_ANCHOR_SELECT", "topk", "off", note="only topk acts"),
    _flag("JETSPEC_ANCHOR_SELECT_GRAPH"),
    _flag("JETSPEC_ANCHOR_SELECT_FUSED"),
    _flag("JETSPEC_ANCHOR_SELECT_EAGER_FUSED", note="fully eager anchor metadata fusion; no graph or torch.compile"),
    _only("JETSPEC_ANCHOR_SELECT_HOME", "one"),
    Knob("JETSPEC_DECODER_CACHE_CAPACITY", "int", "800"),
    _only("JETSPEC_DECODER_FRAME", "wire"),
    _flag("JETSPEC_DECODER_GRAPH_LEAN", NO_DEFAULT),
    _flag("JETSPEC_DECODER_GRAPH_WINDOW", NO_DEFAULT),
    _flag("JETSPEC_DECODER_LOOP", NO_DEFAULT),
    _flag("JETSPEC_DECODER_PREFETCH", NO_DEFAULT),
    _flag("JETSPEC_DECODER_TABLES", NO_DEFAULT),
    _flag("JETSPEC_HEADS_FAST", NO_DEFAULT),
    _flag("JETSPEC_E04_LEAN"),
    Knob("JETSPEC_CORRECTION_HEAD_PREDICTOR", "enum", "wire", ("wire", "bare")),
    # --- miss redraft, deadline, tail --------------------------------------------------------
    _flag("JETSPEC_MISS_REDRAFT"),
    Knob("JETSPEC_MISS_REDRAFT_ANCHOR", "enum", "none", ("none", "decoder")),
    _flag("JETSPEC_MISS_REDRAFT_FORCE"),
    Knob("JETSPEC_MISS_REDRAFT_PASSES", "enum", "1", ("1", "2", "3", "4")),
    _flag("JETSPEC_DRAFT_DEADLINE"),
    _only("JETSPEC_DRAFT_SPLIT", "verify_gpu", default=""),
    Knob("JETSPEC_DRAFT_SPLIT_FRAC", "float", "0.5"),
    Knob("JETSPEC_DRAFT_DEADLINE_MARGIN_MS", "float", "1.0"),
    _flag("JETSPEC_TAIL_LEAN"),
    _flag("JETSPEC_TAIL_PREISSUE"),
    _flag("JETSPEC_TAIL_PUBLISH"),
    _flag("JETSPEC_REQUEST_LEAN"),
    # --- adapter refresh ---------------------------------------------------------------------
    Knob("JETSPEC_ADAPTER_REFRESH", "enum", "off", ("off", "after_j0", "after_j0_j1")),
    _flag("JETSPEC_REFRESH_GRAPH"),
    _only("JETSPEC_EVAL_MODE", "off", "pipelined", "critical-path", default="off"),
    _only("JETSPEC_EVAL_J3_LABEL", "J3", "J2", default="J3",
          note="eval-only depth-four label comparison; fixed J/R depth is unchanged"),
    Knob("JETSPEC_EVAL_ROWS", "int", "16"),
    Knob("JETSPEC_EVAL_DRAFT_DELAY_MS", "float", "0",
         note="eval-only host delay before chain replay for timing-independence gates"),
    _only("JETSPEC_EVAL_REFRESH_MODE", "full", "states_only", default="full"),
    _only("JETSPEC_EVAL_CAPTURE_ROUND_CANDIDATES", "0", "1", "decoder", default="0",
          note="eval-only capture: 1 = final-pass draft top-16; decoder = also each decoder window and its prefix taps"),
    _flag("JETSPEC_REFRESH_FIXED"),
    _flag("JETSPEC_KEYLEN_BUCKET"),
    _flag("JETSPEC_REFRESH_INJECT_LEAN"),
    _flag("JETSPEC_REFRESH_INJECT_LEAN2", note="N=1 fixed INCLUSIVE refresh packed K/V and fused cache write"),
    _flag("JETSPEC_REFRESH_INJECT_MASK_FUSED", note="BF16 N=1 lean refresh cache write also updates packed mask"),
    _flag("JETSPEC_REFRESH_SWAP_GLUE_FUSED", note="BF16 lean swap gathers last-stream taps and writes metadata directly"),
    _flag("JETSPEC_REFRESH_INJECT_INCLUSIVE", note="N=1 fixed-layout refresh includes its anchor and cutoff entry"),
    _flag("JETSPEC_REFRESH_INJECT_PACK"),
    Knob("JETSPEC_REFRESH_KHAT_CAP2", "int", NO_DEFAULT),
    Knob("JETSPEC_REFRESH_KHAT_T", "float", "0.1"),
    Knob("JETSPEC_REFRESH_KHAT_T2", "float", NO_DEFAULT),
    _flag("JETSPEC_REFRESH_LEAN"),
    _flag("JETSPEC_REFRESH_DECODER_LOOP", note="eager refresh decoder layers in C++"),
    _flag("JETSPEC_REFRESH_ADAPTER_LOOP", note="lean adapter refresh layers in C++"),
    _flag("JETSPEC_REFRESH_GEMM_TUNE", note="SM80 refresh-only no-split-K GEMM plans"),
    _flag("JETSPEC_REFRESH_KV_PACK", note="refresh-only load-time K|V packing"),
    _flag("JETSPEC_REFRESH_GATEUP_PACKED", note="dense refresh-only load-time gate|up packing"),
    _flag("JETSPEC_DRAFT_STREAMS", note="N=1 BF16 chain context KV overlaps decoder and row selection"),
    _flag("JETSPEC_REFRESH_STREAMS", note="N=1 BF16 refresh KV projection overlaps swap decoder"),
    _flag("JETSPEC_C1_GATEUP_PACKED", note="dense N=1 C1 query load-time gate|up packing"),
    _flag("JETSPEC_REFRESH_ATTN_LEAN", note="fixed-refresh device-length GQA; no dense bias"),
    _flag("JETSPEC_REFRESH_ATTN_TUNE", note="refresh-only split-KV query tile tuning"),
    _flag("JETSPEC_ADAPTER_ATTN_LAYOUT_FUSED", note="BF16 N=1 adapter attention writes projection-ready output strides"),
    _flag("JETSPEC_ADAPTER_QUERY_LAYOUT_FUSED", note="BF16 N=1 QK writer emits token-major queries for attention output layout"),
    _flag("JETSPEC_REFRESH_PREP_LEAN", note="fused fixed-refresh layout, gathers and scatter"),
    Knob("JETSPEC_REFRESH_PACK", "enum", "0", ("0", "1", "2")),
    _flag("JETSPEC_REFRESH_REUSE0"),
    _flag("JETSPEC_REFRESH_SWAP",
          note="decoder first-disagreement swap at the N=1 graphed chain's refresh"),
    _flag("JETSPEC_REFRESH_SWAP_LEAN",
          note="compact BF16 refresh head with reference decoder arithmetic"),
    _flag("JETSPEC_REFRESH_SWAP_LEAN_CHECK", note="on/off consumed-decision audit tensors"),
    _flag("JETSPEC_REFRESH_SWAP_WRITE_FUSED", note="compact decision and conditional draft-id write in one launch"),
    _flag("JETSPEC_REFRESH_SWAP_GLUE", note="refresh decoder prefix-in-place GQA and fused prelude"),
    # --- C1 adapter kernels ------------------------------------------------------------------
    _flag("JETSPEC_ADAPTER_EW_FUSED"),
    _flag("JETSPEC_ADAPTER_GEMM_TUNE", note="SM100 BF16 M64 packed adapter gate/up graph-tuned cuBLASLt plan"),
    _flag("JETSPEC_ADAPTER_ATTN_SPLIT", note="SM100 BF16 fast5 adapter cuDNN GQA on unique cache heads with live device mask"),
    _flag("JETSPEC_ATTN_SM80", note="requires CUDA sm8x, bf16, N=1; split-KV for keys above 512"),
    Knob("JETSPEC_ATTN_SM100", "enum", "0", ("0", "1", "flashinfer")),
    _flag("JETSPEC_SLOT_ATTN_CHECK", note="compare captured slot attention with gather SDPA"),
    _flag("JETSPEC_ADAPTER_FUSE_CHECK", note="compare each captured adapter block with separate K/V projections"),
    Knob("JETSPEC_ADAPTER_GRAPH_DUMP", "path", "", note="diagnostic CUDA graph dump directory"),
    _flag("JETSPEC_C1_APPEND"),
    _flag("JETSPEC_C1_COMPILE"),
    _flag("JETSPEC_C1_COMPILE_KVWRITE"),
    Knob("JETSPEC_C1_COMPILE_MODE", "str", "max-autotune-no-cudagraphs", note="handed to torch.compile"),
    Knob("JETSPEC_C1_COMPILE_PIN", "path", ""),
    _flag("JETSPEC_C1_COMPILE_PIN_RECORD", note="record tuning choices for compile-pin sealing"),
    Knob("JETSPEC_C1_COMPILE_PIN_SHA", "str", "", note="required with JETSPEC_C1_COMPILE_PIN"),
    Knob("JETSPEC_C1_FUSED", "enum", "0", ("0", "1", "2")),
    Knob("JETSPEC_C1_GEMM", "enum", "aten", ("aten", "direct")),
    _flag("JETSPEC_C1_GLUE_FUSED"),
    Knob("JETSPEC_C1_QK_WARPS", "enum", "16", ("4", "8", "16"), note="experimental 16-row query Q/K CUDA launch geometry"),
    _flag("JETSPEC_C1_QK_FUSED", note="experimental dense query Q/K postprocessing in one certified CUDA launch"),
    _flag("JETSPEC_REFRESH_QK_FUSED", note="experimental refresh Q/K postprocessing in one certified CUDA launch"),
    _flag("JETSPEC_C1_GRAPH"),
    Knob("JETSPEC_C1_GRAPH_LAYOUT", "enum", "full", ("full", "live")),
    _flag("JETSPEC_C1_KVCACHE"),
    _flag("JETSPEC_C1_MULTI_SLOT"),
    _flag("JETSPEC_BATCH_DRAFT_GRAPH", note="N>1 draft chain under CUDA graphs per batch bucket"),
    _flag("JETSPEC_CTX_PREFIX_RESIDENT", note="N>1 context graph keeps prefix K/V in bucket cache"),
    _flag("JETSPEC_CTX_KV_ONE_GEMM", note="N>1 context graph projects all layer K/V weights at once"),
    _flag("JETSPEC_PACKED_LIVE_ROWS", note="N>1 adapter and context project packed committed rows"),
    _flag("JETSPEC_BATCH_DRAFT_GRAPH_FRONT", note="N>1 adapter and decoder front captured too"),
    _flag("JETSPEC_DEVICE_PUBLISH", note="N>1 keyed rows publish into the static candidate packet"),
    _flag("JETSPEC_DEVICE_PUBLISH_CHECK", note="compare the direct candidate packet with legacy packing"),
    _flag("JETSPEC_BATCH_ROWS_LIVE", default="1",
          note="under JETSPEC_BATCH_DRAFT_GRAPH=1: the Jacobi passes run over the live rows only"),
    Knob("JETSPEC_BATCH_ROWS_QUANTUM", "int", NO_DEFAULT,
         note="JETSPEC_BATCH_ROWS_LIVE: the pass's live row count rounds up to a multiple of it "
              "(unset: bucket rows / 16, at least 2)"),
    _flag("JETSPEC_BATCH_ROWS_ASSEMBLY_GRAPH", default="1",
          note="under JETSPEC_BATCH_DRAFT_GRAPH=1: the row assembly is one captured call per C"),
    _flag("JETSPEC_BATCH_ROWS_ASSEMBLY_FUSED", default="1",
          note="JETSPEC_BATCH_ROWS_ASSEMBLY_GRAPH on CUDA: the BLEND float64 chain as three Triton kernels"),
    _flag("JETSPEC_BATCH_ROWS_LIVE_CHECK",
          note="diagnostic: also run the fixed row-cap passes each round and count live-row mismatches (W470_ROWS_CHECK)"),
    _flag("JETSPEC_BATCH_ROWS_ASSEMBLY_CHECK",
          note="diagnostic: also run the eager stage each round and count mismatches (W470_ASSEMBLY_CHECK)"),
    Knob("JETSPEC_C1_MULTI_SLOTS", "int", "32"),
    _flag("JETSPEC_C1_KVNORM_BATCH"),
    _flag("JETSPEC_C1_LEAN", note="divergent default: '0' at c1_fused.py/drafter.py, None at heads/decoder.py"),
    _flag("JETSPEC_C1_GLUON", note="opt-in Blackwell dense head norm/full RoPE/direct KV writes in C1 and refresh lean loops"),
    _flag("JETSPEC_C1_LOOP"),
    _flag("JETSPEC_C1_NATIVE"),
    _only("JETSPEC_C1_NORM", "wide"),
    _flag("JETSPEC_C1_PRELOOP"),
    _flag("JETSPEC_C1_ROPE_NORM"),
    _flag("JETSPEC_C1_ROPE_TABLE"),
    _flag("JETSPEC_C1_SHARED_PREFIX"),
    # --- diagnostics -------------------------------------------------------------------------
    _flag("JETSPEC_DEVICE_SPLIT"),
    _flag("JETSPEC_DECODER_BATCH_PREFETCH"),
    Knob("JETSPEC_DEBUG_DUMP_DIR", "path", REQUIRED, note="required only on the debug dump path"),
    Knob("JETSPEC_LOGIT_DUMP_DIR", "path", NO_DEFAULT),
    _flag("JETSPEC_LOGIT_DUMP_EAGLE",
          note="0: the EAGLE target prefill returns conditioning, not logits, and the lossless dumper skips it "
               "(the fork's behaviour); 1: the prefill returns both and the dumper considers it (verifier.py)"),
    Knob("JETSPEC_LOGIT_REFERENCE", "path", NO_DEFAULT),
    Knob("JETSPEC_LOGIT_TARGETS", "csv", ""),
    Knob("JETSPEC_SYNC_DUMP_DIR", "path", NO_DEFAULT),
    # --- the check's own escape hatch --------------------------------------------------------
    _flag(ALLOW_UNREGISTERED, note="1 downgrades the unregistered-name refusal to a warning; nothing else"),
    # --- SSD_* (the fork's engine; registered and value-checked, names not prefix-refused) ---
    Knob("SSD_ATTENTION_BACKEND", "enum", "sdpa", ("flashinfer", "sdpa", "triton"),
         note="divergent default: 'sdpa' at layers/attention.py, None at tree_serve.py and the bench.py label"),
    Knob("SSD_CUDA_ARCH", "str", "9.0"),
    Knob("SSD_CUTE_DSL_CACHE_ROOT", "path", NO_DEFAULT),
    Knob("SSD_DATASET_DIR", "path", REQUIRED),
    Knob("SSD_DRAFT_MODEL", "path", NO_DEFAULT, note="defaults to a snapshot under SSD_HF_CACHE"),
    _flag("SSD_EAGER_VERIFY_WRAPPER"),
    _flag("SSD_EAGER_VERIFY_CAUSAL_N1"),
    Knob("SSD_RANK_CPU_AFFINITY", "str", NO_DEFAULT,
         note="one or two disjoint inclusive CPU ranges for serving ranks"),
    Knob("SSD_CPU_PIN_PLAN", "str", NO_DEFAULT,
         note="serve_read resolved per-rank CPU and NUMA placement"),
    Knob("SSD_EAGLE3_8B", "path", NO_DEFAULT, note="defaults to a snapshot under SSD_HF_CACHE"),
    Knob("SSD_EAGLE3_QWEN_32B", "path", NO_DEFAULT, note="defaults to a snapshot under SSD_HF_CACHE"),
    Knob("SSD_EAGLE3_SPECFORGE_70B", "path", NO_DEFAULT, note="defaults to a snapshot under SSD_HF_CACHE"),
    Knob("SSD_HF_CACHE", "path", REQUIRED),
    Knob("SSD_DIST_PORT", "int", "1223"),
    _flag("SSD_PROFILE"),
    _flag("SSD_PROFILE_DRAFT"),
    _flag("SSD_PROFILE_TARGET"),
    Knob("SSD_RANK_DEVICE_MAP", "csv", NO_DEFAULT),
    Knob("SSD_SPLIT_DEDICATED_UUID", "str", NO_DEFAULT),
    _only("SSD_SHARED_GPU_DIST", "gloo"),
    Knob("SSD_SHARED_GPU_KV_UTILIZATION", "float", NO_DEFAULT),
    Knob("SSD_SHM_NAME", "str", "jetspec.engines.jetspec_plus_plus.serving.api", note="per-run shared-memory segment name; LLMEngine sets ssd_<pid>"),
    Knob("SSD_TARGET_MODEL", "path", NO_DEFAULT, note="defaults to a snapshot under SSD_HF_CACHE"),
    _flag("SSD_VERIFY_MASK_CACHE"),
    _flag("SSD_VERIFY_PLAN_CACHE"),
    _flag("SSD_VERIFY_CAUSAL_CHECK"),
    # --- historical serving switches still read by the backend ------------------------------
    # These names predate the JETSPEC_/SSD_ convention.  They are part of the same fail-closed
    # contract: a misspelling must not silently select another serving condition.
    _flag("W240_WARMUP"),
    Knob("JETSPEC_WARMUP_PREFIX_LENGTHS", "str", "[]",
         note="serve_read's pinned prompt lengths for pre-decode graph/JIT warmup"),
    _flag("W258B_TRACE", NO_DEFAULT),
    Knob("W259_C1_CACHE_CAPACITY", "int", "768"),
    Knob("W264_CELL_NAME", "str", REQUIRED,
         note="required only by the verifier debug-dump path"),
    _flag("W264_INDEPENDENT_RECORDS"),
    _flag("W278_HEAD_PROFILE", NO_DEFAULT),
    _flag("W278_OFFER_DUMP", NO_DEFAULT),
    Knob("W278_RECORD_INPUTS_DIR", "path", NO_DEFAULT),
    Knob("W278_RECORD_INPUTS_LIMIT", "int", "240"),
    Knob("W280_RECORD", "path", NO_DEFAULT),
    Knob("PRED_SERVE_READ_RECORDS", "path", NO_DEFAULT, reader="launcher",
         note="repo-native serve reader population recorder inherited by spawned children"),
    Knob("TRITON_CACHE_DIR", "path", NO_DEFAULT, reader="launcher",
         note="repo-native serve reader persistent Triton specialization cache"),
    _launcher("PYTHONOPTIMIZE", "strips asserts; per-round host work now follows config.verbose"),
    *(_launcher(name, "headline serving CPU thread cap") for name in (
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "W240_SPEC_THREADS",
        "W240_THREAD_CAPS",
    )),
    _flag("JETSPEC_DRAFT_PIN_CHECK", "1"),
    _flag("W321_EVENTS", NO_DEFAULT),
    Knob("W323_D15_LOG", "path", NO_DEFAULT),
    _flag("W324_MARKS", NO_DEFAULT),
    _flag("W327_PUBLISH_CHECK", NO_DEFAULT),
    Knob("W328_DEFER_ADAPTER", "str", NO_DEFAULT),
    Knob("W328_DEFER_TAPS", "str", NO_DEFAULT),
    Knob("W328_DIGEST", "str", NO_DEFAULT),
    Knob("W328_DIGEST_TAPS", "str", NO_DEFAULT),
    # --- launcher names: nothing under serve/ reads them, the serving process inherits them ---
    Knob("JETSPEC_EVAL_LIVECODEBENCH_DIR", "path", NO_DEFAULT, reader="launcher",
         note="the eval prompt loader reads the six LiveCodeBench JSONL files"),
    # The a100 cell launchers (experiments/a100/cell_*.sh, the reader's cell_w280 launcher env)
    # export W270_* beside the JETSPEC_* knobs they translate into.
    *(_launcher(name, "a100 cell launcher") for name in (
        "W270_ANCHOR", "W270_ANCHOR_CAP", "W270_ANCHOR_COVERAGE", "W270_ANCHOR_DREJ", "W270_AUTO",
        "W270_CALIBRATION_SOURCE", "W270_CELL_NAME", "W270_COUNT", "W270_DATASET", "W270_DEVICES",
        "W270_FUSE", "W270_GRAPHS", "W270_HOST_FAST", "W270_INDEPENDENT_RECORDS",
        "W270_KERNEL_REPEAT_DIR", "W270_KPRED_COVERAGE", "W270_KSET", "W270_KSET_CAP", "W270_KSET_Q",
        "W270_MODE", "W270_NUMA_NODE", "W270_PREPARE", "W270_PROFILE_DIR", "W270_REGIME",
        "W270_ROWS_CAP", "W270_SEAT_AUTHORITY", "W270_STACK", "W270_STOP_AFTER_CELL",
        "W270_STOP_BEFORE_LAUNCH", "W270_TOKENS", "W270_TRACE", "W270_TRACE_DIR", "W270_TREE_ALGO",
        "W270_TREE_B", "W270_TREE_SPINE")),
    *(_launcher(name, "predictor/reader e04 launcher") for name in (
        "E04_ALLOWED_SEATS", "E04_CELLS", "E04_JOBS", "E04_LOGS", "E04_PY", "E04_RECORDS", "E04_ROOT",
        "E04_RUN", "E04_TOOLING", "E04_TRACES", "E04_WORK", "E04_WORK_BASE", "E04_WORK_SERVE",
        "E04_WORKSTATION")),
    *(_launcher(name, "adapter/kit_fast and predictor tooling") for name in (
        "JETSPEC_H2H_ROOT", "JETSPEC_KIT_ARMS_ROOT", "JETSPEC_KIT_BASE_ROOT",
        "JETSPEC_KIT_COMPARATOR_ROOT", "JETSPEC_KIT_CONTAINER_IMAGE", "JETSPEC_KIT_CONTAINER_PYTHON",
        "JETSPEC_KIT_CONTAINER_ROOT", "JETSPEC_KIT_CPU_PYTHON", "JETSPEC_KIT_DATA_ROOT",
        "JETSPEC_KIT_DENSE_COUNTS_PATH", "JETSPEC_KIT_INIT_DRAFT_ROOT", "JETSPEC_KIT_PYLIBS_ROOT",
        "JETSPEC_KIT_REPO", "JETSPEC_KIT_REPO_SUBDIR", "JETSPEC_KIT_STUBS_ROOT",
        "JETSPEC_KIT_TARGET_MODEL_ROOT", "JETSPEC_KIT_WORK_ROOT", "JETSPEC_SPECFORGE_ROOT",
        "JETSPEC_SPEC_FORGE")),
)


@dataclass(frozen=True)
class Requires:
    """``name`` acts only while ``flag`` holds one of ``values``; set otherwise, it is inert."""
    name: str
    flag: str
    values: tuple
    when: tuple = None   # None: whenever ``name`` is present; else only for these values of ``name``
    why: str = ""
    always: bool = False  # False: asked of a ``--jetspec`` run only; True: asked of every run


# JETSPEC_AUTO=1 installs JETSPEC_ROWS=1 (routing.apply_auto_route) in every process before
# the rows path reads it, so the rows requirements below are waived under it.
_ROWS_WAIVER = ("JETSPEC_AUTO", "1")

REQUIREMENTS = (
    Requires("JETSPEC_CTX_KV_DIRECT", "JETSPEC_CTX_KV_FUSED", ("1",), when=("1",),
             why="direct context writes use the existing packed projection and postprocess"),
    *(Requires("JETSPEC_DECODER_SLOT_WRITE", name, ("1",), when=("1",),
               why="direct slots require the merged packed decoder") for name in (
                   "JETSPEC_DECODER_PREFIX_MERGE", "JETSPEC_DECODER_QKV_FUSED",
                   "JETSPEC_DECODER_GRAPH_LEAN")),
    Requires("JETSPEC_DRAFT_STREAMS", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="stream fork/join belongs to the fixed N=1 chain"),
    *(Requires("JETSPEC_REFRESH_STREAMS", name, (value,), when=("1",),
               why="refresh fork projects the pre-swap superset and commits the final cutoff")
      for name, value in (("JETSPEC_CHAIN_GRAPH", "seg"),
                          ("JETSPEC_REFRESH_SWAP", "1"),
                          ("JETSPEC_REFRESH_INJECT_LEAN2", "1"))),
    *(Requires("JETSPEC_C1_GATEUP_PACKED", name, (value,), when=("1",),
               why="C1 packing selects the dense lean Python query loop") for name, value in (
                   ("JETSPEC_C1_LEAN", "1"), ("JETSPEC_C1_NORM", "wide"),
                   ("JETSPEC_FUSED_QKV", "1"))),
    Requires("JETSPEC_DECODER_SINGLE_GATEUP_PACKED", "JETSPEC_DECODER_GRAPH_LEAN", ("1",), when=("1",),
             why="single-sequence packing belongs to the lean decoder body"),
    *(Requires("JETSPEC_DECODER_SELECT_RADIX", name, (value,), when=("1",),
               why="radix selection uses the N=1 chain's existing probability and tie kernels")
      for name, value in (("JETSPEC_CHAIN_GRAPH", "seg"),
                          ("JETSPEC_ANCHOR_SELECT_FUSED", "1"),
                          ("JETSPEC_DECODER_FRAME", "wire"))),
    Requires("JETSPEC_REFRESH_GEMM_TUNE", "JETSPEC_REFRESH_LEAN", ("1",), when=("1",),
             why="GEMM tuning belongs only to the lean refresh body"),
    Requires("JETSPEC_REFRESH_KV_PACK", "JETSPEC_REFRESH_LEAN", ("1",), when=("1",),
             why="packed projections belong to the lean refresh body"),
    Requires("JETSPEC_REFRESH_GATEUP_PACKED", "JETSPEC_REFRESH_LEAN", ("1",), when=("1",),
             why="packed gate/up belongs to the lean refresh body"),
    Requires("JETSPEC_REFRESH_KV_PACK", "JETSPEC_C1_NORM", ("wide",), when=("1",),
             why="the lean refresh uses wide norm kernels"),
    *(Requires("JETSPEC_REFRESH_INJECT_LEAN2", name, (value,), when=("1",),
               why="lean2 selects the N=1 fixed INCLUSIVE refresh route") for name, value in (
                   ("JETSPEC_REFRESH_FIXED", "1"), ("JETSPEC_REFRESH_INJECT_INCLUSIVE", "1"),
                   ("JETSPEC_CHAIN_GRAPH", "seg"))),
    Requires("JETSPEC_CTX_KV_APPEND", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="append selects fixed captured N=1 chain context graphs"),
    Requires("JETSPEC_CTX_KV_APPEND", "JETSPEC_CHAIN_MISS_REDRAFT", ("1",), when=("1",),
             why="exact host L and miss invalidation use the redraft command"),
    Requires("JETSPEC_CTX_KV_APPEND_CHECK", "JETSPEC_CTX_KV_APPEND", ("1",), when=("1",),
             why="the context cache check needs the appended cache"),
    Requires("JETSPEC_REFRESH_INJECT_INCLUSIVE", "JETSPEC_REFRESH_FIXED", ("1",), when=("1",),
             why="inclusive injection changes the fixed-layout refresh"),
    Requires("JETSPEC_REFRESH_INJECT_INCLUSIVE", "JETSPEC_CHAIN_FIXED_CONFIG", ("J1-R1", "J2-R1", "J3-R1", "J2-R2", "J3-R3"), when=("1",),
             why="inclusive injection is gated on the N=1 R1 chain"),
    Requires("JETSPEC_VERIFY_OUTCOME_FUSED", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="the fused outcome consumes the N=1 CUDA device accept result"),
    Requires("JETSPEC_VERIFY_OUTCOME_FUSED_CHECK", "JETSPEC_VERIFY_OUTCOME_FUSED", ("1",), when=("1",),
             why="the shadow outcome comparison requires the fused path"),
    *(Requires("JETSPEC_PAIR_OUTCOME_PUSH", name, ("1",), when=("1",),
               why="pair outcome push uses the N=1 device chain outcome") for name in (
                   "JETSPEC_VERIFY_DEVICE", "JETSPEC_OUTCOME_WIRE_RING",
                   "JETSPEC_OUTCOME_PACK_PRESYNC", "JETSPEC_CHAIN_PREPOST",
                   "JETSPEC_CHAIN_CMD_LOCAL_COUNTER", "JETSPEC_CHAIN_HOST_LEAN")),
    Requires("JETSPEC_PAIR_OUTCOME_PUSH_CHECK", "JETSPEC_PAIR_OUTCOME_PUSH", ("1",),
             when=("1",), why="the push check compares the old outcome path"),
    Requires("JETSPEC_CANDIDATE_RECV_AHEAD", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="receive ahead requires the N=1 device-verify store receiver"),
    Requires("JETSPEC_REFERENCE_CANDIDATE_SOCKET", "JETSPEC_VERIFY_REFERENCE", ("1",), when=("1",),
             why="candidate socket keeps the reference pair's existing wire bytes"),
    Requires("JETSPEC_REFERENCE_COMMAND_SOCKET", "JETSPEC_REFERENCE_CANDIDATE_MAIN", ("1",), when=("1",),
             why="reverse command delivery requires the private main-thread reference socket"),
    Requires("JETSPEC_REFERENCE_COMMAND_SOCKET", "JETSPEC_CHAIN_CMD_LOCAL_COUNTER", ("1",), when=("1",),
             why="reverse command delivery keeps both ranks' ordered local command IDs"),
    *(Requires("JETSPEC_REFERENCE_TAP_PREPARE", name, ("1",), when=("1",),
               why="early tap preparation uses the full N=1 reference outcome ring") for name in (
                   "JETSPEC_VERIFY_REFERENCE", "JETSPEC_OUTCOME_WIRE_RING")),
    Requires("JETSPEC_REFERENCE_OUTCOME_HOST", "JETSPEC_VERIFY_REFERENCE", ("1",), when=("1",),
             why="host outcome storage belongs to the N=1 reference engine"),
    Requires("JETSPEC_REFERENCE_HOST_SELECT", "JETSPEC_REFERENCE_CANDIDATE_MAIN", ("1",), when=("1",),
             why="host list selection requires private main-thread reference candidate decoding"),
    Requires("JETSPEC_REFERENCE_CANDIDATE_SHADOW", "JETSPEC_REFERENCE_CANDIDATE_SOCKET", ("1",), when=("1",),
             why="the shadow compares the socket and store payloads"),
    Requires("JETSPEC_REFERENCE_CANDIDATE_SPIN", "JETSPEC_REFERENCE_CANDIDATE_SOCKET", ("1",), when=("1",),
             why="the polling reader requires the reference candidate socket"),
    Requires("JETSPEC_REFERENCE_CANDIDATE_MAIN", "JETSPEC_REFERENCE_CANDIDATE_SOCKET", ("1",), when=("1",),
             why="main-thread intake requires the reference candidate socket"),
    Requires("JETSPEC_REFERENCE_CANDIDATE_MAIN", "JETSPEC_REFERENCE_CANDIDATE_SPIN", ("0",), when=("1",),
             why="main-thread intake has no polling reader"),
    Requires("JETSPEC_REFERENCE_CANDIDATE_MAIN", "JETSPEC_ANCHOR", ("decoder",), when=("1",),
             why="main-thread intake preserves the private N=1 decoder wire"),
    Requires("JETSPEC_CANDIDATE_RECV_AHEAD", "JETSPEC_ROWS", ("1",), when=("1",),
             why="receive ahead requires keyed candidate rows"),
    Requires("JETSPEC_CANDIDATE_RECV_AHEAD", "JETSPEC_ROWS_FAST", ("1",), when=("1",),
             why="receive ahead requires CPU fast candidate delivery"),
    Requires("JETSPEC_VERIFY_SELECT_LIST", "JETSPEC_VERIFY_SELECT_HOST", ("1",), when=("1",),
             why="list selection reads the private host rows"),
    Requires("JETSPEC_VERIFY_PUBLISH_FIRST", "JETSPEC_OUTCOME_PACK_PRESYNC", ("1",), when=("1",),
             why="publication must use the packet already packed before sync"),
    Requires("JETSPEC_VERIFY_PUBLISH_FIRST", "JETSPEC_VERIFY_POST_LEAN", ("1",), when=("1",),
             why="publication reads mirrored host words once"),
    Requires("JETSPEC_VERIFY_PUBLISH_FIRST", "JETSPEC_VERIFY_RECORD_LATE", ("1",), when=("1",),
             why="deferred metrics flush after the replay record"),
    Requires("JETSPEC_OUTCOME_PACK_PRESYNC", "JETSPEC_OUTCOME_WIRE_RING", ("1",), when=("1",),
             why="early outcome pack borrows an ordered ring slot"),
    Requires("JETSPEC_OUTCOME_PACK_PRESYNC", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="the device verifier supplies the pre-sync hook"),
    Requires("JETSPEC_OUTCOME_PACK_PRESYNC_CHECK", "JETSPEC_OUTCOME_PACK_PRESYNC", ("1",), when=("1",),
             why="the check compares an early pack with the post-sync pack"),
    Requires("JETSPEC_VERIFY_SELECT_HOST", "JETSPEC_CANDIDATE_DECODE_LEAN", ("1",), when=("1",),
             why="host selection reads the private decoded row lists"),
    Requires("JETSPEC_VERIFY_SELECT_HOST", "JETSPEC_VERIFY_SELECT_LEAN", ("1",), when=("1",),
             why="host selection belongs to the N=1 lean decision path"),
    Requires("JETSPEC_VERIFY_SELECT_HOST", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="host selection belongs to the N=1 device verify path"),
    Requires("JETSPEC_VERIFY_RECORD_LATE", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="the device verifier supplies the post-launch hook"),
    Requires("JETSPEC_VERIFY_POST_LEAN", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="post-verify lean reads the N=1 mirrored packet"),
    Requires("JETSPEC_CHAIN_ADMIT_BATCH", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="batched admission belongs to the segmented chain"),
    Requires("JETSPEC_CHAIN_ADMIT_RNG_ONE_DRAW", "JETSPEC_CHAIN_ADMIT_BATCH", ("1",), when=("1",),
             why="one-draw RNG belongs to the batched chain admission"),
    Requires("JETSPEC_CHAIN_ADMIT_RNG_ONE_DRAW_CHECK", "JETSPEC_CHAIN_ADMIT_RNG_ONE_DRAW", ("1",), when=("1",),
             why="the RNG check compares the one-draw and loop tables"),
    Requires("JETSPEC_CHAIN_ADMIT_HOST_LEAN", "JETSPEC_CHAIN_ADMIT_BATCH", ("1",), when=("1",),
             why="host-lean belongs to the batched chain admission"),
    Requires("JETSPEC_CHAIN_ADMIT_HOST_LEAN_CHECK", "JETSPEC_CHAIN_ADMIT_HOST_LEAN", ("1",), when=("1",),
             why="the host check compares the host and device round-0 packets"),
    Requires("JETSPEC_CHAIN_ADMIT_GRAPH", "JETSPEC_CHAIN_ADMIT_BATCH", ("1",), when=("1",),
             why="admission graph extends the batched admission"),
    Requires("JETSPEC_CHAIN_ADMIT_GRAPH_CHECK", "JETSPEC_CHAIN_ADMIT_GRAPH", ("1",), when=("1",),
             why="admission graph check compares the replay with the eager adapter"),
    Requires("JETSPEC_REFRESH_ATTN_LEAN", "JETSPEC_REFRESH_FIXED", ("1",), when=("1",),
             why="length metadata belongs to the fixed lean refresh"),
    Requires("JETSPEC_REFRESH_ATTN_TUNE", "JETSPEC_REFRESH_ATTN_LEAN", ("1",), when=("1",),
             why="tuning consumes the fixed-refresh device attention metadata"),
    Requires("JETSPEC_REFRESH_PREP_LEAN", "JETSPEC_REFRESH_FIXED", ("1",), when=("1",),
             why="fused preparation belongs to the fixed refresh"),
    Requires("JETSPEC_REFRESH_PREP_LEAN", "JETSPEC_REFRESH_ATTN_LEAN", ("1",), when=("1",),
             why="fused preparation supplies device attention metadata"),
    Requires("JETSPEC_REFRESH_PREP_LEAN", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="fused refresh preparation remains chain-only until eager qualification"),
    Requires("JETSPEC_REFRESH_FIXED", "JETSPEC_REFRESH_PACK", ("2",), when=("1",),
             why="fixed refresh consumes the packed correction path"),
    Requires("JETSPEC_REFRESH_FIXED", "JETSPEC_REFRESH_LEAN", ("1",), when=("1",),
             why="fixed refresh consumes the lean adapter path"),
    Requires("JETSPEC_REFRESH_FIXED", "JETSPEC_C1_NORM", ("wide",), when=("1",),
             why="fixed refresh consumes wide C1 normalization"),
    Requires("JETSPEC_ROWS_STATIC_CHAIN", "JETSPEC_ROWS_ASSEMBLY", ("1",), when=("1",),
             why="the chain rows body reads assembled live rows"),
    Requires("JETSPEC_ROWS_STATIC_CHAIN", "JETSPEC_ROWS_GRAPHS", ("1",), when=("1",),
             why="the chain rows body uses the graphed rows pass"),
    Requires("JETSPEC_ROWS_K3", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="the K3 kernel uses the static chain rows segment"),
    Requires("JETSPEC_ROWS_K3", "JETSPEC_DECODER_FRAME", ("wire",), when=("1",),
             why="the K3 BLEND kernel reads wire-aligned draft ids"),
    Requires("JETSPEC_LMHEAD_FUSED", "JETSPEC_REFINE_K3", ("1",), when=("1",),
             why="the fused head feeds K3 partials instead of full logits"),
    Requires("JETSPEC_LMHEAD_FUSED", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="the fused head is wired into the segmented J passes"),
    Requires("JETSPEC_J0_LMHEAD_KN", "JETSPEC_LMHEAD_FUSED", ("1",), when=("1",),
             why="the draft-only KN alternative replaces the fused segmented head"),
    *(Requires("JETSPEC_J0_QKV_GLUE_FUSED", required, ("1",), when=("1",),
               why="QKV glue consumes the packed N=1 lean row projection")
      for required in ("JETSPEC_JPASS_GLUE_LEAN", "JETSPEC_ROWS_FUSED_GRAPH", "JETSPEC_FUSED_QKV")),
    *(Requires("JETSPEC_J0_GATEUP_PACKED", required, values, when=("1",),
               why="draft-row packing requires the N=1 fused segmented pass")
      for required, values in (("JETSPEC_CHAIN_GRAPH", ("seg",)),
                               ("JETSPEC_JPASS_GLUE_LEAN", ("1",)),
                               ("JETSPEC_ROWS_FUSED_GRAPH", ("1",)))),
    Requires("JETSPEC_FRONT_STATIC", "JETSPEC_DRAFTS", ("15", "31"), when=("1",),
             why="the front chain serves b16 and b32 blocks"),
    Requires("JETSPEC_FRONT_STATIC", "JETSPEC_C1_GRAPH_LAYOUT", ("live",), when=("1",),
             why="the fixed C1 query uses the live cache layout"),
    Requires("JETSPEC_FRONT_STATIC", "JETSPEC_C1_FUSED", ("2",), when=("1",),
             why="the live C1 query requires fused projection"),
    Requires("JETSPEC_FRONT_STATIC", "JETSPEC_C1_LEAN", ("1",), when=("1",),
             why="the live C1 query uses the lean kernel path"),
    Requires("JETSPEC_FRONT_STATIC", "JETSPEC_DECODER_GRAPH_LEAN", ("1",), when=("1",),
             why="the decoder window body uses lean kernels"),
    Requires("JETSPEC_FRONT_STATIC", "JETSPEC_DECODER_GRAPH_WINDOW", ("1",), when=("1",),
             why="the decoder window body shares graph K/V storage"),
    Requires("JETSPEC_DECODER_PREFIX_MERGE", "JETSPEC_FRONT_STATIC", ("1",), when=("1",),
             why="the merged decoder reads the fixed prefix and window rows"),
    Requires("JETSPEC_DECODER_PREFIX_MERGE", "JETSPEC_DECODER_GRAPH_LEAN", ("1",), when=("1",),
             why="the merged decoder uses the lean layer parts"),
    Requires("JETSPEC_DECODER_PREFIX_MERGE", "JETSPEC_DECODER_GRAPH_WINDOW", ("1",), when=("1",),
             why="the merged decoder shares window graph K/V storage"),
    Requires("JETSPEC_DECODER_PREFIX_MERGE", "JETSPEC_DECODER_QKV_FUSED", ("1",), when=("1",),
             why="the merged decoder reads one packed QKV matrix per layer"),
    Requires("JETSPEC_DECODER_PREFIX_MERGE", "JETSPEC_DECODER_GLUE_FUSED", ("1",), when=("1",),
             why="the merged decoder uses the staged prefix and window glue"),
    Requires("JETSPEC_CHAIN_GRAPH_CHECK", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="shadow checking requires segmented chain capture"),
    Requires("JETSPEC_CHAIN_RECORD", "JETSPEC_CHAIN_GRAPH", ("seg",),
             why="draft recording requires segmented chain capture"),
    Requires("JETSPEC_CHAIN_RECORD_ROUNDS", "JETSPEC_CHAIN_RECORD", None,
             why="the record limit requires an output directory"),
    Requires("JETSPEC_CHAIN_RECORD_SKIP_REQUESTS", "JETSPEC_CHAIN_RECORD", None,
             why="record warmup count requires an output directory"),
    Requires("JETSPEC_CHAIN_REFERENCE_HEAD", "JETSPEC_CHAIN_RECORD", None,
             why="reference provenance requires a draft fixture"),
    Requires("JETSPEC_CHAIN_WARM_REACHABLE", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="reachable warm-up applies only to segmented chain capture"),
    Requires("JETSPEC_CHAIN_WARM_SINGLE_KEY", "JETSPEC_CHAIN_WARM_REACHABLE", ("1",),
             when=("1",), why="the single ceiling key requires reachable chain warm-up"),
    Requires("JETSPEC_CHAIN_CAPTURE_STAMPS", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="capture stamps apply only to segmented chain capture"),
    Requires("JETSPEC_CHAIN_HANDOFF", "JETSPEC_CHAIN_FIXED_CONFIG",
             ("J1-R0", "J2-R0", "J3-R0", "J1-R1", "J2-R1", "J3-R1", "J2-R2", "J3-R3"),
             when=("1",), why="handoff pre-stages a fixed chain cell"),
    Requires("JETSPEC_CHAIN_HANDOFF", "JETSPEC_CHAIN_GRAPH", ("seg",),
             when=("1",), why="handoff requires segmented chain replay"),
    Requires("JETSPEC_CHAIN_HANDOFF_TIMING", "JETSPEC_CHAIN_GRAPH", ("seg",),
             when=("1",), why="handoff timing reads segmented chain events"),
    Requires("JETSPEC_CHAIN_PREPOST", "JETSPEC_CHAIN_FIXED_CONFIG",
             ("J1-R0", "J2-R0", "J3-R0", "J1-R1", "J2-R1", "J3-R1", "J2-R2", "J3-R3"),
             when=("1",), why="prepost defers bookkeeping for a fixed chain cell"),
    Requires("JETSPEC_CHAIN_PREPOST", "JETSPEC_CHAIN_GRAPH", ("seg",),
             when=("1",), why="prepost requires segmented chain replay"),
    Requires("JETSPEC_CHAIN_PREPOST_TIMING", "JETSPEC_CHAIN_GRAPH", ("seg",),
             when=("1",), why="prepost timing measures both segmented-chain arms"),
    Requires("JETSPEC_CHAIN_TAIL_TRACE", "JETSPEC_CHAIN_GRAPH", ("seg",),
             when=("1",), why="the tail trace stamps segmented chain rounds"),
    Requires("JETSPEC_CHAIN_HOST_LEAN", "JETSPEC_CHAIN_GRAPH", ("seg",),
             when=("1",), why="the host control upload feeds a captured chain"),
    Requires("JETSPEC_CHAIN_POST_LEAN", "JETSPEC_CHAIN_GRAPH", ("seg",),
             when=("1",), why="post work overlaps a segmented chain replay"),
    Requires("JETSPEC_CHAIN_C1_KEY_BUCKET", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="the C1 key bucket is a per-chain-key capture shape"),
    Requires("JETSPEC_CHAIN_REFRESH_KEY_BUCKET", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="the refresh key bucket is a per-chain-key capture shape"),
    Requires("JETSPEC_CHAIN_REFRESH_KEY_BUCKET", "JETSPEC_REFRESH_FIXED", ("1",), when=("1",),
             why="the refresh key bucket needs fixed-layout refresh"),
    Requires("JETSPEC_CHAIN_DECODER_KEY_BUCKET", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="the decoder window bucket is a per-chain-key capture shape"),
    Requires("JETSPEC_CHAIN_SEGA_GLUE", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1", "2"),
             why="the Seg A glue is a segmented chain body"),
    Requires("JETSPEC_CHAIN_INTAKE_FUSED", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="fused intake runs inside the segmented chain capture"),
    Requires("JETSPEC_CHAIN_ATTN3", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="ATTN3 exact-L plans require the segmented chain outcome"),
    Requires("JETSPEC_CHAIN_J_SWITCH_L", "JETSPEC_CHAIN_FIXED_CONFIG", ("J2-R0", "J3-R0"),
             why="length switching replays J1-R0 from an R0 capture"),
    Requires("JETSPEC_CHAIN_MISS_REDRAFT", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="miss redraft uses the segmented chain"),
    Requires("JETSPEC_CHAIN_MISS_REDRAFT", "JETSPEC_CHAIN_FIXED_CONFIG", ("J1-R0", "J2-R0", "J3-R0", "J1-R1", "J2-R1", "J3-R1", "J2-R2", "J3-R3"), when=("1",),
             why="the J0-only miss segment resumes the hypothesis passes"),
    Requires("JETSPEC_CHAIN_MISS_REDRAFT_TIMING", "JETSPEC_CHAIN_MISS_REDRAFT", ("1",), when=("1",),
             why="miss timing prints only when the redraft path is active"),
    Requires("JETSPEC_CHAIN_MISS_REDRAFT_MAX_MS", "JETSPEC_CHAIN_MISS_REDRAFT", ("1",),
             why="the dedicated redraft valve is read only during redraft"),
    Requires("JETSPEC_REFRESH_SWAP_LEAN", "JETSPEC_REFRESH_SWAP", ("1",), when=("1",),
             why="the lean read serves only the refresh swap"),
    Requires("JETSPEC_REFRESH_SWAP_WRITE_FUSED", "JETSPEC_REFRESH_SWAP_LEAN", ("1",), when=("1",),
             why="fused decision/write consumes the compact decoder result"),
    Requires("JETSPEC_REFRESH_SWAP_GLUE", "JETSPEC_REFRESH_SWAP", ("1",), when=("1",),
             why="the decoder glue serves only the refresh swap"),
    Requires("JETSPEC_REFRESH_SWAP_LEAN_CHECK", "JETSPEC_REFRESH_SWAP_LEAN", ("1",), when=("1",),
             why="the decision audit compares the lean read with the reference refresh swap"),
    Requires("JETSPEC_REFRESH_SWAP", "JETSPEC_CHAIN_FIXED_CONFIG", ("J1-R1", "J2-R1", "J3-R1", "J2-R2", "J3-R3"),
             when=("1",), why="the swap acts at the adapter refresh; an R0 config has none"),
    Requires("JETSPEC_REFRESH_SWAP", "JETSPEC_ADAPTER_REFRESH", ("after_j0",), when=("1",),
             why="the chain's one refresh is after J0"),
    Requires("JETSPEC_REFRESH_SWAP", "JETSPEC_ANCHOR", ("decoder",), when=("1",),
             why="the swap reads the anchor decoder"),
    Requires("JETSPEC_REFINE_K3", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="refinement K3 is the segmented chain pass body"),
    Requires("JETSPEC_CHAIN_TAU_TABLE", "JETSPEC_CHAIN_GRAPH", ("seg",),
             why="tau optimization selects segmented chain launches"),
    Requires("JETSPEC_CHAIN_GRAPH", "JETSPEC_ROWS_ASSEMBLY", ("1",), when=("seg",),
             why="the chain consumes device row-stage outputs"),
    Requires("JETSPEC_CHAIN_GRAPH", "JETSPEC_ROW_LAYOUT", ("anchor0",), when=("seg",),
             why="the chain uses the anchor0 row geometry"),
    Requires("JETSPEC_CHAIN_GRAPH", "JETSPEC_VERIFY_DEVICE", ("1",), when=("seg",),
             why="the chain intake consumes the fixed device outcome envelope"),
    Requires("JETSPEC_CHAIN_GRAPH", "JETSPEC_FRONT_STATIC", ("1",), when=("seg",),
             why="the chain calls the static C1 and decoder bodies"),
    Requires("JETSPEC_CHAIN_GRAPH", "JETSPEC_REFRESH_FIXED", ("1",), when=("seg",),
             why="the chain's refresh has a fixed captured shape"),
    Requires("SSD_VERIFY_DEVICE", "JETSPEC_BLOCKHEAD", None, when=("1",), always=True,
             why="the N=1 device verify path needs JETSPEC_BLOCKHEAD or JETSPEC_SYNC_DRAFTER"),
    Requires("SSD_VERIFY_SYNC_AUDIT", "SSD_VERIFY_DEVICE", ("1",), when=("1",), always=True,
             why="sync audit applies only to block-head device verify"),
    Requires("JETSPEC_VERIFY_PRESTAGE_MIXED_CHECK", "JETSPEC_VERIFY_PRESTAGE_MIXED", ("1",),
             when=("1",), why="mixed prestage checks require the staged path"),
    Requires("JETSPEC_VERIFY_DEVICE", "JETSPEC_ROWS", ("1",), when=("1",),
             why="device verify requires live candidate rows"),
    Requires("JETSPEC_VERIFY_DEVICE", "JETSPEC_ROWS_FAST", ("1",), when=("1",),
             why="device verify requires CPU fast candidate delivery"),
    Requires("JETSPEC_VERIFY_DEVICE", "JETSPEC_CHAIN_GRAPH", ("seg",), when=("1",),
             why="device verify sends host commands and a fixed outcome envelope"),
    Requires("JETSPEC_VERIFY_HOST_CENSUS", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="the host census stamps the N=1 device-verify round"),
    Requires("JETSPEC_DEVICE_HOST_STAMPS", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="device host stamps require device verification"),
    Requires("JETSPEC_DEVICE_HOST_STAMPS", "JETSPEC_ANATOMY_LITE", ("1",), when=("1",),
             why="device host stamps extend the existing anatomy observer"),
    Requires("JETSPEC_VERIFY_STAGE_HOST", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="host staging applies to the N=1 device-verify round"),
    Requires("JETSPEC_VERIFY_STAGE_EARLY", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="early planning applies to the N=1 device-verify round"),
    Requires("JETSPEC_VERIFY_PLAN_REUSE", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="plan reuse applies to the N=1 device-verify graph"),
    Requires("JETSPEC_VERIFY_PLAN_REUSE_CHECK", "JETSPEC_VERIFY_PLAN_REUSE", ("1",), when=("1",),
             why="reuse comparison requires plan reuse"),
    Requires("JETSPEC_VERIFY_PLAN_LEAN", "JETSPEC_VERIFY_PLAN_REUSE", ("1",), when=("1",),
             why="lean plan writes require guarded N=1 plan reuse"),
    Requires("JETSPEC_VERIFY_PLAN_LEAN", "JETSPEC_VERIFY_STAGE_BLOB", ("1",), when=("1",),
             why="the aliased KV length is carried by the staging blob"),
    Requires("JETSPEC_VERIFY_STAGE_BLOB", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="blob staging applies to the N=1 device-verify graph"),
    Requires("JETSPEC_VERIFY_STAGE_BLOB", "JETSPEC_VERIFY_STAGE_HOST", ("1",), when=("1",),
             why="blob staging uses pinned host staging"),
    Requires("JETSPEC_VERIFY_STAGE_BLOB_FAST", "JETSPEC_VERIFY_STAGE_BLOB", ("1",), when=("1",),
             why="fast blob staging uses the N=1 staging blob"),
    Requires("JETSPEC_VERIFY_STAGE_BLOB_FAST_CHECK", "JETSPEC_VERIFY_STAGE_BLOB_FAST", ("1",), when=("1",),
             why="fast blob comparison requires fast blob staging"),
    Requires("JETSPEC_VERIFY_INPUT_DIRECT", "JETSPEC_VERIFY_DEVICE", ("1",), when=("1",),
             why="direct proposal input applies to the N=1 device-verify graph"),
    Requires("JETSPEC_VERIFY_INPUT_VIEW", "JETSPEC_VERIFY_INPUT_DIRECT", ("1",), when=("1",),
             why="cached proposal input requires direct graph input"),
    Requires("JETSPEC_TAIL_PUBLISH", "JETSPEC_ANCHOR", ("decoder",), when=("1",),
             why="early publish requires the decoder anchor"),
    Requires("JETSPEC_TAIL_PUBLISH", "JETSPEC_ROWS", ("1",), when=("1",),
             why="early publish requires live rows"),
    Requires("JETSPEC_TAIL_PUBLISH", "JETSPEC_ROWS_FAST", ("1",), when=("1",),
             why="early publish requires optimized rows"),
    Requires("E04_LAB_REF_TOKENS", "E04_LAB_REFRESH_CUTOFF", ("oracle",),
             why="the reference is read only by the oracle refresh cutoff (tau_lab.py)"),
    Requires("E04_LAB_SWAP_MARGIN", "E04_LAB_SWAP", _E04_LAB_SWAPS,
             why="the margin gates a swap only"),
    Requires("E04_LAB_REFRESH_CUTOFF", "JETSPEC_ADAPTER_REFRESH", ("after_j0", "after_j0_j1"),
             when=("decoder_f", "oracle"), why="the cutoff is the adapter refresh's"),
    Requires("E04_LAB_SWAP", "JETSPEC_ADAPTER_REFRESH", ("after_j0", "after_j0_j1"),
             when=_e04_lab_swaps_with("first_wrong_refresh"), why="the swap runs at the adapter refresh"),
    Requires("E04_LAB_REFRESH_CUTOFF", "JETSPEC_ANCHOR", ("decoder",), when=("decoder_f",),
             why="rule F reads the anchor decoder"),
    Requires("E04_LAB_SWAP", "JETSPEC_ANCHOR", ("decoder",), when=_E04_LAB_SWAPS,
             why="every swap reads the anchor decoder"),
    Requires("E04_LAB_REFRESH_EVERY", "JETSPEC_ADAPTER_REFRESH", ("after_j0", "after_j0_j1"),
             when=("1",), why="it adds adapter refreshes"),
    Requires("E04_LAB_EXTRA_PASSES", "JETSPEC_ADAPTER_REFRESH", ("after_j0", "after_j0_j1"),
             when=tuple(str(count) for count in range(1, 9)), why="each extra pass follows a refresh"),
    Requires("E04_LAB_SHADOW_AR", "JETSPEC_ANCHOR", ("decoder",),
             when=tuple(str(count) for count in range(1, 16)), why="the shadow is the anchor decoder"),
    Requires("E04_LAB_EXACT", "JETSPEC_ROW_LAYOUT", ("anchor0",), when=("1",),
             why="the exact round is the anchor0 layout's (_exact_anchor0_rows)"),
    *(Requires(name, "E04_LAB_SWAP", _e04_lab_swaps_with("first_wrong_refresh"),
               why="the gate acts on the first_wrong_refresh swap only")
      for name in ("E04_LAB_SWAP_KHAT_MAX", "E04_LAB_SWAP_SLOT_MAX")),
    *(Requires(name, "E04_LAB_SWAP", _e04_lab_swaps_with(mode), why=f"the margin of the {mode} swap only")
      for mode, name in _E04_LAB_MODE_MARGINS),
    Requires("E04_LAB_PREVERIFY_ROUNDS", "E04_LAB_SWAP", _e04_lab_swaps_with("pre_verify"),
             why="it repeats the pre_verify swap"),
    Requires("E04_LAB_ROWS_MIN", "E04_ROWRULE", ("joint",),
             why="the floor ranks the joint rule's pair scores (tau_lab.joint_rows)"),
    Requires("E04_LAB_SWAP_ROWS", "E04_ROWRULE", ("joint",), when=("1",),
             why="the rows are added after the joint rule's (tau_lab.joint_rows)"),
    Requires("E04_LAB_KSCORE_WRITTEN", "E04_KSCORE", ("blend_dec",), when=("disagree", "both"),
             why="the argmax indicator is an E04_KSCORE=blend_dec term"),
    Requires("E04_LAB_KSCORE_WRITTEN", "E04_KSCORE", ("blend", "blend_dec"), when=("draft_p",),
             why="it replaces the BLEND k score's draft term"),
    Requires("E04_LAB_SEED_EXACT", "E04_LAB_EXACT", ("1",), when=("1",),
             why="the seeded round is the lab's exact round (_exact_anchor0_rows)"),
    *(Requires("E04_LAB_SEED_EXACT", name, ("1",), when=("1",),
               why="only a forced miss redraft follows a round-ahead hit")
      for name in ("JETSPEC_MISS_REDRAFT", "JETSPEC_MISS_REDRAFT_FORCE")),
    Requires("E04_LAB_SEED_PASSES", "E04_LAB_SEED_EXACT", ("1",),
             why="it counts the seeded round's drafter passes"),
    Requires("E04_LAB_RECORD_TOPK", "JETSPEC_ADAPTER_REFRESH", ("after_j0", "after_j0_j1"),
             why="it records at the adapter refresh (tau_lab.refresh_hook)"),
    Requires("E04_LAB_SWAP_RULE_FILE", "E04_LAB_SWAP",
             tuple(value for value in _E04_LAB_SWAPS
                   if {"first_wrong_refresh", "pre_verify"} & set(value.split("+"))),
             why="the rule decides the first_wrong_refresh and pre_verify swaps only (tau_lab.swap_rule)"),
    Requires("E04_LAB_SWAP_RULE_FILE", "JETSPEC_DRAFTS", ("15",),
             why="its bias table buckets block slots 1..15 and k-hat 0..15"),
    Requires("E04_LAB_AFTER_SWAP_FILL", "E04_LAB_SWAP", _e04_lab_swaps_with("first_wrong_refresh"),
             when=("mask", "noise", "context", "decoder"), why="it fills after the first_wrong_refresh swap"),
    Requires("E04_LAB_FILL_SEED", "E04_LAB_AFTER_SWAP_FILL", ("noise",), why="it seeds the noise fill"),
    Requires("E04_LAB_PIN_AGREED", "E04_LAB_SWAP", _e04_lab_swaps_with("first_wrong_refresh"), when=("1",),
             why="it pins at the first_wrong_refresh refresh"),
    Requires("E04_LAB_PIN_AGREED", "E04_LAB_REFRESH_CUTOFF", ("decoder_f",), when=("1",),
             why="the agreed run is decoder_f's khat_used"),
    Requires("JETSPEC_MISS_REDRAFT_FORCE", "JETSPEC_MISS_REDRAFT", ("1",), when=("1",),
             why="forced redrafting only runs with miss redraft"),
    Requires("JETSPEC_MISS_REDRAFT_PASSES", "JETSPEC_MISS_REDRAFT", ("1",),
             why="pass count is read only during miss redraft"),
    Requires("JETSPEC_MISS_REDRAFT_ANCHOR", "JETSPEC_MISS_REDRAFT", ("1",),
             why="anchor choice is read only during miss redraft"),
    Requires("JETSPEC_DRAFT_DEADLINE_MARGIN_MS", "JETSPEC_DRAFT_DEADLINE", ("1",),
             why="margin is read only with the draft deadline"),
    Requires("JETSPEC_KSET", "JETSPEC_ROWS", ("1",),
             why="live_rows.resolve_kset_count and live_hypotheses return before reading it"),
    Requires("JETSPEC_KSET", "JETSPEC_LEGACY_KPRED_FORWARD", ("1",), when=("predictor",),
             why="the predictor k set without decoder BLEND needs the legacy forward"),
    Requires("JETSPEC_LEGACY_KPRED_FORWARD", "JETSPEC_KSET", ("predictor",), when=("1",),
             why="the legacy forward would otherwise be enabled without selecting its mass"),
    Requires("JETSPEC_ROWS_FAST", "JETSPEC_ROWS", ("1",), when=("1",),
             why="every read sits under JETSPEC_ROWS=1 (draft_runner.py, speculator_async.py)"),
    Requires("JETSPEC_F_GRAPHS", "JETSPEC_ROWS", ("1",), when=("1",),
             why="read only while building the rows path (draft_runner.py)"),
    # ``values=None``: the flag is a path, so the rule asks that it is set, not what it spells.
    Requires("JETSPEC_BLOCKHEAD_GRAPH", "JETSPEC_BLOCKHEAD", None, when=("1",), always=True,
             why="blockhead_graph_requested is called only by the block-head speculator (speculator_blockhead.py)"),
    *(Requires(name, required, values, when=("1",), always=True,
               why="the optimized block head runs only in its N=1 graph")
      for name in ("JETSPEC_BLOCKHEAD_ATTN_LEN", "JETSPEC_BLOCKHEAD_GLUE_LEAN",
                   "JETSPEC_BLOCKHEAD_QKV_FUSED", "JETSPEC_BLOCKHEAD_LMHEAD_KN",
                   "JETSPEC_BLOCKHEAD_BENCH")
      for required, values in (("JETSPEC_BLOCKHEAD", None), ("JETSPEC_BLOCKHEAD_GRAPH", ("1",)))),
    Requires("JETSPEC_BLOCKHEAD_DECISION_AUDIT", "JETSPEC_BLOCKHEAD", None, always=True),
    Requires("JETSPEC_BLOCKHEAD_DECISION_AUDIT", "JETSPEC_BLOCKHEAD_GRAPH", ("1",), always=True),
    Requires("JETSPEC_BLOCKHEAD_BATCH", "JETSPEC_BLOCKHEAD", None, when=("1",), always=True,
             why="the batched drafter is constructed only in block-head mode"),
    Requires("JETSPEC_BLOCKHEAD_BLOCK", "JETSPEC_BLOCKHEAD", None, always=True,
             why="read only while the block-head checkpoint contract is resolved (blockhead_serve.py)"),
    Requires("JETSPEC_BLOCKHEAD_EQUIV", "JETSPEC_BLOCKHEAD_BATCH", ("1",), when=("1",),
             why="serial replay is only defined on the batched block-head path"),
)


def require_verify_device_batch(max_num_seqs: int, environ=None) -> None:
    """Refuse the device path when a batch can contain more than one sequence."""
    environ = os.environ if environ is None else environ
    if environ.get("JETSPEC_VERIFY_REFERENCE", "0") == "1" and max_num_seqs != 1:
        raise ValueError("JETSPEC_VERIFY_REFERENCE=1 requires N=1")
    if environ.get("JETSPEC_VERIFY_DEVICE", "0") == "1" and max_num_seqs != 1:
        raise ValueError("JETSPEC_VERIFY_DEVICE=1 requires N=1")
    if environ.get("JETSPEC_VERIFY_OUTCOME_FUSED", "0") == "1" and max_num_seqs != 1:
        raise ValueError("JETSPEC_VERIFY_OUTCOME_FUSED=1 requires N=1")
    if environ.get("SSD_VERIFY_DEVICE", "0") == "1" and max_num_seqs != 1:
        raise ValueError("SSD_VERIFY_DEVICE=1 requires N=1")
    if environ.get("JETSPEC_PAIR_OUTCOME_PUSH", "0") == "1" and max_num_seqs != 1:
        raise ValueError("JETSPEC_PAIR_OUTCOME_PUSH=1 requires N=1")


@dataclass(frozen=True)
class Refuses:
    """``name`` holding one of ``when`` refuses ``other`` holding one of ``values``; asked of every run."""
    name: str
    when: tuple
    other: str
    values: tuple
    why: str = ""


REFUSALS = (
    Refuses("JETSPEC_PAIR_OUTCOME_PUSH", ("1",), "JETSPEC_OUTCOME_PREPOST_NO_ACK", ("1",),
            why="peer outcome slots replace the NCCL no-ACK receive protocol"),
    Refuses("JETSPEC_PAIR_OUTCOME_PUSH", ("1",), "JETSPEC_REFERENCE_COMMAND_SOCKET", ("1",),
            why="peer outcome polling keeps TCP prefill/exit boundaries"),
    Refuses("JETSPEC_VERIFY_OUTCOME_FUSED", ("1",), "SSD_VERIFY_DEVICE", ("1",),
            why="the fused outcome belongs to the JetSpec pair, not block-head device verify"),
    Refuses("JETSPEC_PAIR_OUTCOME_PUSH", ("1",), "JETSPEC_VERIFY_SYNC_AUDIT", ("1",),
            why="outcome push requires the pre-sync pack without sync auditing"),
    Refuses("JETSPEC_PAIR_OUTCOME_PUSH", ("1",), "JETSPEC_DRAFT_SPLIT", ("verify_gpu",),
            why="outcome push requires a dedicated draft GPU"),
    Refuses("JETSPEC_PAIR_OUTCOME_PUSH", ("1",), "SSD_SHARED_GPU_DIST", ("gloo",),
            why="outcome push requires a device-visible peer buffer"),
    Refuses("JETSPEC_PAIR_OUTCOME_PUSH", ("1",), "JETSPEC_MPS_DRAFT_PRIORITY", ("0", "1"),
            why="outcome push requires a dedicated draft GPU"),
    Refuses("JETSPEC_BLOCKHEAD_GRAPH", ("1",), "TORCHDYNAMO_DISABLE", ("1",),
            why="blockhead_serve.blockhead_graph_requested raises: the graphed block head refuses the eager regime"),
)


def _effective(environ, name):
    value = environ.get(name)
    if value is not None:
        return value
    default = REGISTRY[name].default
    return None if default in (REQUIRED, NO_DEFAULT) else default


def context_append_enabled(environ=None):
    environ = os.environ if environ is None else environ
    return (environ.get('JETSPEC_CTX_KV_APPEND', '0') == '1'
            or environ.get('JETSPEC_HYBRID_CONTEXT_DIRTY', '0') == '1')


def unregistered_names(environ):
    return sorted(name for name in environ
                  if name.startswith(VALIDATED_PREFIXES) and name not in REGISTRY)




def frozen_paper_recipes():
    """Explicit historical serving records; these names are never live N=1 recipes."""
    path = Path(__file__).resolve().parents[1] / "configs" / "frozen-paper-recipes.txt"
    return json.loads(path.read_text(encoding="utf-8"))["recipes"]


def require_recipe_launch(recipe_name, *, max_num_seqs=1, resolve_only=False):
    if max_num_seqs != 1 or recipe_name is None:
        return None
    record = frozen_paper_recipes().get(recipe_name)
    if record is not None and not resolve_only:
        raise EnvironmentContractError(
            f"historical paper recipe {recipe_name!r} is dry-resolve only; "
            f"use submission commit {record['submission_commit']} for reproduction")
    return record


def validate_lane_regime(environ, regime=None):
    """Whole-lane checks; legacy in-lane anchor-selection graphs are grandfathered."""
    regime = regime or ('eager' if environ.get('TORCHDYNAMO_DISABLE') == '1' else None)
    if regime not in (None, 'eager', 'graphed'):
        raise EnvironmentContractError(f'unknown engine regime {regime!r}')
    if regime == 'eager':
        graph_flags = ('JETSPEC_C1_GRAPH', 'JETSPEC_ROWS_GRAPHS', 'JETSPEC_F_GRAPHS',
                       'JETSPEC_VERIFY_DEVICE', 'JETSPEC_FRONT_STATIC',
                       'JETSPEC_DECODER_GRAPH_WINDOW', 'JETSPEC_ROWS_FUSED_GRAPH',
                       'JETSPEC_BLOCKHEAD_GRAPH', 'JETSPEC_SYNC_GRAPH',
                       'JETSPEC_BATCH_DRAFT_GRAPH', 'JETSPEC_ROW_STAGE_GRAPH')
        enabled = [name for name in graph_flags if environ.get(name) == '1']
        if environ.get('JETSPEC_CHAIN_GRAPH') == 'seg':
            enabled.append('JETSPEC_CHAIN_GRAPH')
        if enabled:
            raise EnvironmentContractError('regime eager refuses whole-lane graphs: ' + ', '.join(enabled))
    if (regime == 'graphed' and environ.get('JETSPEC_REFRESH_SWAP') == '1'
            and environ.get('JETSPEC_CHAIN_GRAPH', '0') != 'seg'):
        raise EnvironmentContractError('regime graphed refuses the eager refresh-swap executor')


def validate_environment(environ=None, *, jetspec=True, max_num_seqs=1, warn=print,
                         recipe_name=None, resolve_only=False, regime=None):
    """Refuse an environment that would serve something other than what it spells.

    ``jetspec``: whether this run serves JetSpec (bench.py ``--jetspec``); check (c) needs it.
    ``max_num_seqs`` is the actual serving concurrency, not the prompt count.
    A frozen ``recipe_name`` is valid only with ``resolve_only=True`` at N=1.
    Returns the registered knobs present in ``environ`` (name -> value).  Raises
    ``EnvironmentContractError`` naming every breach at once.
    """
    environ = os.environ if environ is None else environ
    problems = []
    is_eval = environ.get('JETSPEC_EVAL_MODE', 'off') != 'off'
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import assert_uncut; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import FIXED_CONFIGS
    from jetspec.engines.jetspec_plus_plus.draft.block_geometry import block_width
    width = block_width(environ)
    try:
        assert_uncut(environ)
    except ValueError as error:
        problems.append(str(error))
    if (environ.get("JETSPEC_HYBRID_FP32_LINEAR_BF16_STORAGE", "0") == "1"
            and environ.get("JETSPEC_TARGET_DTYPE") != "fp32"):
        problems.append("JETSPEC_HYBRID_FP32_LINEAR_BF16_STORAGE=1 requires JETSPEC_TARGET_DTYPE=fp32")
    config = environ.get('JETSPEC_CHAIN_FIXED_CONFIG')
    eager_swap = (environ.get('JETSPEC_CHAIN_GRAPH', '0') == '0'
                  and environ.get('JETSPEC_REFRESH_SWAP', '0') == '1')
    schedule = None
    if config is not None:
        from jetspec.engines.jetspec_plus_plus.draft.eager_schedule import parse_schedule
        try:
            schedule = parse_schedule(config)
            if eager_swap:
                if schedule.r < 1:
                    problems.append(f'JETSPEC_CHAIN_FIXED_CONFIG={config}: R0 config has no refresh')
                validate_eager_labels(environ, schedule, recipe=recipe_name)
        except ValueError as error:
            problems.append(str(error))
    if environ.get('JETSPEC_HYBRID_CONTEXT_DIRTY', '0') == '1':
        if environ.get('JETSPEC_HYBRID_PAIR', '0') != '1':
            problems.append('JETSPEC_HYBRID_CONTEXT_DIRTY=1 requires JETSPEC_HYBRID_PAIR=1')
        # Apply the existing append contract without changing the caller's map.
        environ = dict(environ, JETSPEC_CTX_KV_APPEND='1')
    try:
        validate_lane_regime(environ, regime)
    except EnvironmentContractError as error:
        problems.append(str(error))
    if (environ.get('JETSPEC_CHAIN_FIXED_CONFIG') is not None
            and environ.get('JETSPEC_CHAIN_GRAPH', '0') != 'seg'
            and not (environ.get('JETSPEC_CHAIN_GRAPH', '0') == '0'
                     and environ.get('JETSPEC_REFRESH_SWAP', '0') == '1')):
        problems.append('JETSPEC_CHAIN_FIXED_CONFIG requires JETSPEC_CHAIN_GRAPH=seg or eager refresh swap')
    if (environ.get('JETSPEC_REFRESH_SWAP') == '1'
            and environ.get('JETSPEC_CHAIN_GRAPH', '0') == '0'):
        for name, required in (('JETSPEC_ROWS', '1'), ('JETSPEC_ROWS_ASSEMBLY', '1'),
                               ('JETSPEC_ROW_LAYOUT', 'anchor0'),
                               ('JETSPEC_REFRESH_FIXED', '1')):
            if environ.get(name) != required:
                problems.append(f'eager refresh swap requires {name}={required}')
        widths = ('15', '31') if environ.get('JETSPEC_HYBRID_PAIR') == '1' else ('15',)
        if environ.get('JETSPEC_DRAFTS') not in widths:
            problems.append('eager refresh swap requires JETSPEC_DRAFTS=' + ' or '.join(widths))
        if width == 32 and environ.get('JETSPEC_ROWS_K3') == '1':
            problems.append('b32 eager refresh swap requires the Torch row rule (JETSPEC_ROWS_K3=0)')
        if environ.get('JETSPEC_ROWS_RULE_CAP') not in ('4', '8', '16'):
            problems.append('eager refresh swap requires JETSPEC_ROWS_RULE_CAP=4, 8 or 16')
        try:
            if int(environ.get('JETSPEC_ROWS_CAP', '128')) < int(environ.get('JETSPEC_ROWS_RULE_CAP', '4')):
                problems.append('eager refresh swap row bucket exceeds JETSPEC_ROWS_CAP')
        except ValueError:
            problems.append('eager refresh swap requires integer row capacities')
    elif (environ.get('JETSPEC_REFRESH_SWAP') == '1'
          and environ.get('JETSPEC_CHAIN_GRAPH', '0') != 'seg'):
        problems.append('JETSPEC_REFRESH_SWAP=1 requires JETSPEC_CHAIN_GRAPH=seg')

    if 'JETSPEC_SERVE_POLICY' in environ:
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import load_policy; from jetspec.engines.jetspec_plus_plus.draft.serve_policy import validate_policy
        try:
            validate_policy(load_policy(environ), environ, jetspec=jetspec,
                            max_num_seqs=max_num_seqs)
        except ValueError as error:
            problems.append(str(error))

    historical = require_recipe_launch(recipe_name, max_num_seqs=max_num_seqs,
                                       resolve_only=resolve_only)
    if jetspec and max_num_seqs == 1 and historical is None:
        from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
        fallback = budget_type(environ).required_redraft(environ)
        if environ.get(fallback, "0") != "1":
            problems.append(f"N=1 JetSpec serving requires {fallback}=1; "
                            "a miss must redraft from the known true anchor on the critical path")

    if environ.get("JETSPEC_DEVICE_HOST_STAMPS") == "1" and max_num_seqs != 1:
        problems.append("JETSPEC_DEVICE_HOST_STAMPS=1 requires N=1")

    unknown = unregistered_names(environ)
    if unknown:
        listing = "; ".join(_suggest(name) for name in unknown)
        if environ.get(ALLOW_UNREGISTERED) == "1":
            warn(f"WARNING {ALLOW_UNREGISTERED}=1: unregistered environment names are ignored by "
                 f"the serving tree: {listing}")
        else:
            problems.append(
                f"unregistered environment names (nothing in serve/ reads them, so they are inert): {listing}. "
                f"Register the name in jetspec/engines/jetspec_plus_plus/draft/env.py, or set {ALLOW_UNREGISTERED}=1 to proceed.")

    for name in sorted(environ):
        knob = REGISTRY.get(name)
        if knob is not None and knob.allowed is not None and environ[name] not in knob.allowed:
            problems.append(f"{name}={environ[name]!r} is outside its allowed set {list(knob.allowed)}"
                            + (f" ({knob.note})" if knob.note else ""))

    switch_L = environ.get("JETSPEC_CHAIN_J_SWITCH_L")
    if switch_L is not None and (not switch_L.isascii() or not switch_L.isdecimal()
                                 or int(switch_L) % 128):
        problems.append("JETSPEC_CHAIN_J_SWITCH_L must be a nonnegative multiple of 128")

    waived = environ.get(_ROWS_WAIVER[0]) == _ROWS_WAIVER[1]
    for rule in REQUIREMENTS:
        if not (jetspec or rule.always):
            continue
        if rule.name not in environ or (rule.when is not None and environ[rule.name] not in rule.when):
            continue
        if (rule.name == "JETSPEC_CHAIN_GRAPH" and rule.flag == "JETSPEC_VERIFY_DEVICE"
                and environ.get("JETSPEC_VERIFY_REFERENCE", "0") == "1"):
            continue
        if (rule.name == "JETSPEC_PAIR_OUTCOME_PUSH"
                and rule.flag in ("JETSPEC_VERIFY_DEVICE", "JETSPEC_OUTCOME_PACK_PRESYNC")
                and environ.get("JETSPEC_VERIFY_REFERENCE", "0") == "1"):
            # Host acceptance precedes publication; only delivery changes.
            continue
        if rule.flag == "JETSPEC_ROWS" and waived:
            continue
        if (is_eval and config in FIXED_CONFIGS
                and rule.flag == 'JETSPEC_CHAIN_FIXED_CONFIG'
                and rule.name in ('JETSPEC_CHAIN_HANDOFF', 'JETSPEC_CHAIN_PREPOST',
                                  'JETSPEC_CHAIN_MISS_REDRAFT', 'JETSPEC_REFRESH_SWAP',
                                  'JETSPEC_REFRESH_INJECT_INCLUSIVE')):
            # Eval captures its fixed schedule, including J0-only and R0 cells.
            # Refresh kernels may be available without being replayed in R0.
            continue
        if rule.flag == "JETSPEC_CHAIN_FIXED_CONFIG" and schedule is not None:
            if rule.name in ("JETSPEC_REFRESH_SWAP", "JETSPEC_REFRESH_INJECT_INCLUSIVE"):
                if schedule.r >= 1:
                    continue
            elif rule.name == "JETSPEC_CHAIN_J_SWITCH_L":
                if schedule.j >= 2 and schedule.r == 0:
                    continue
            else:
                continue
        if (rule.name == "JETSPEC_CHAIN_MISS_REDRAFT"
                and rule.flag == "JETSPEC_CHAIN_FIXED_CONFIG"
                and rule.flag not in environ):
            # The dynamic launch policy selects the same captured J1/J2/J3 bodies.
            continue
        if (rule.name == "JETSPEC_CTX_KV_APPEND" and rule.flag == "JETSPEC_CHAIN_MISS_REDRAFT"
                and schedule is not None and (schedule.r >= 1 or is_eval)
                and environ.get("JETSPEC_CHAIN_PREPOST") == "1"):
            # R1's existing prepost command supplies exact host L. Refresh writes
            # separate KV slots; the next Seg A hides them and dirties prefix+tail.
            continue
        if (rule.name == "SSD_VERIFY_DEVICE" and rule.flag == "JETSPEC_BLOCKHEAD"
                and environ.get("JETSPEC_SYNC_DRAFTER")):
            continue
        if (rule.name == "JETSPEC_KSET" and rule.flag == "JETSPEC_LEGACY_KPRED_FORWARD"
                and _effective(environ, rule.flag) == "0"
                and environ.get("E04_KSCORE") == "blend_dec"
                and environ.get("E04_ROWRULE") == "joint"
                and environ.get("JETSPEC_KSET_AFTER_MISS") == "zero"):
            continue
        if rule.values is None:
            if not _effective(environ, rule.flag):
                problems.append(
                    f"{rule.name}={environ[rule.name]!r} is inert unless {rule.flag} is set: {rule.why}")
        elif _effective(environ, rule.flag) not in rule.values:
            problems.append(
                f"{rule.name}={environ[rule.name]!r} is inert unless {rule.flag} is "
                f"{' or '.join(rule.values)} (it is {environ.get(rule.flag, 'unset')!r}): {rule.why}")

    for rule in REFUSALS:
        if environ.get(rule.name) in rule.when and environ.get(rule.other) in rule.values:
            problems.append(
                f"{rule.name}={environ[rule.name]!r} refuses {rule.other}={environ[rule.other]!r}: {rule.why}")

    if environ.get("JETSPEC_HYBRID_VERIFY_FLASHINFER", "0") == "1":
        if not jetspec or max_num_seqs != 1:
            problems.append("JETSPEC_HYBRID_VERIFY_FLASHINFER=1 requires N=1 JetSpec")
        if regime is not None and regime != "graphed":
            problems.append("JETSPEC_HYBRID_VERIFY_FLASHINFER=1 requires graphed verification")
        for name, expected in (("JETSPEC_HYBRID_PAIR", "1"),
                               ("JETSPEC_VERIFY_DEVICE", "1"),
                               ("JETSPEC_VERIFY_REFERENCE", "0"),
                               ("SSD_VERIFY_DEVICE", "0"),
                               ("SSD_ATTENTION_BACKEND", "flashinfer"),
                               ("JETSPEC_HYBRID_VERIFY_TRITON", "0")):
            if _effective(environ, name) != expected:
                problems.append(f"JETSPEC_HYBRID_VERIFY_FLASHINFER=1 requires {name}={expected}")

    if environ.get("JETSPEC_VERIFY_REFERENCE", "0") == "1":
        if not jetspec or max_num_seqs != 1:
            problems.append("JETSPEC_VERIFY_REFERENCE=1 requires N=1 JetSpec")
        if regime is not None and regime != "graphed":
            problems.append("JETSPEC_VERIFY_REFERENCE=1 requires the graphed reference engine")
        if _effective(environ, "JETSPEC_VERIFY_DEVICE_TRANSPORT") != "chain":
            problems.append("JETSPEC_VERIFY_REFERENCE=1 requires JETSPEC_VERIFY_DEVICE_TRANSPORT=chain")
        if environ.get("SSD_ATTENTION_BACKEND", "flashinfer") != "flashinfer":
            problems.append("JETSPEC_VERIFY_REFERENCE=1 requires SSD_ATTENTION_BACKEND=flashinfer")
        for name, expected in (("JETSPEC_CHAIN_GRAPH", "seg"),
                               ("JETSPEC_VERIFY_DEVICE", "0"),
                               ("JETSPEC_ROWS", "1"), ("JETSPEC_ROWS_FAST", "1")):
            if _effective(environ, name) != expected:
                problems.append(f"JETSPEC_VERIFY_REFERENCE=1 requires {name}={expected}")
        # These alter target forward/planning, accept, or post-accept mechanics.
        # Candidate decoding, prepost/no-ACK and the wire ring are transport.
        forbidden = (
            "SSD_VERIFY_DEVICE", "SSD_VERIFY_PLAN_CACHE", "SSD_VERIFY_MASK_CACHE",
            "JETSPEC_VERIFY_SELECT_LEAN", "JETSPEC_VERIFY_SELECT_HOST",
            "JETSPEC_VERIFY_RECORD_LATE", "JETSPEC_VERIFY_POST_LEAN",
            "JETSPEC_VERIFY_STAGE_HOST", "JETSPEC_VERIFY_STAGE_EARLY",
            "JETSPEC_VERIFY_STAGE_BLOB", "JETSPEC_VERIFY_STAGE_BLOB_FAST",
            "JETSPEC_VERIFY_INPUT_DIRECT", "JETSPEC_VERIFY_INPUT_VIEW",
            "JETSPEC_VERIFY_PLAN_REUSE", "JETSPEC_VERIFY_PLAN_LEAN",
            "JETSPEC_VERIFY_PRESTAGE_MIXED", "JETSPEC_VERIFY_OUTCOME_FUSED",
            "JETSPEC_VERIFY_PUBLISH_FIRST", "JETSPEC_OUTCOME_PACK_PRESYNC",
            "JETSPEC_CANDIDATE_RECV_AHEAD", "JETSPEC_VERIFY_HOST_CENSUS",
        )
        for name in forbidden:
            if environ.get(name, "0") == "1":
                problems.append(f"JETSPEC_VERIFY_REFERENCE=1 refuses {name}=1")

    if environ.get("JETSPEC_VERIFY_OUTCOME_FUSED") == "1":
        if max_num_seqs != 1:
            problems.append("JETSPEC_VERIFY_OUTCOME_FUSED=1 requires N=1")
        if environ.get("JETSPEC_BLOCKHEAD"):
            problems.append("JETSPEC_VERIFY_OUTCOME_FUSED=1 refuses the block-head path")

    if max_num_seqs != 1:
        if environ.get("JETSPEC_CTX_KV_DIRECT") == "1":
            problems.append("JETSPEC_CTX_KV_DIRECT=1 requires N=1")
        if environ.get("JETSPEC_DECODER_SLOT_WRITE") == "1":
            problems.append("JETSPEC_DECODER_SLOT_WRITE=1 requires N=1")
        if environ.get("JETSPEC_J0_LMHEAD_KN") == "1":
            problems.append("JETSPEC_J0_LMHEAD_KN=1 requires N=1")
        if environ.get("JETSPEC_J0_GATEUP_PACKED") == "1":
            problems.append("JETSPEC_J0_GATEUP_PACKED=1 requires N=1")
        if environ.get("JETSPEC_J0_QKV_GLUE_FUSED") == "1":
            problems.append("JETSPEC_J0_QKV_GLUE_FUSED=1 requires N=1")
        if environ.get("JETSPEC_DRAFT_STREAMS") == "1":
            problems.append("JETSPEC_DRAFT_STREAMS=1 requires N=1")
        if environ.get("JETSPEC_REFRESH_STREAMS") == "1":
            problems.append("JETSPEC_REFRESH_STREAMS=1 requires N=1")
        if environ.get("JETSPEC_C1_GATEUP_PACKED") == "1":
            problems.append("JETSPEC_C1_GATEUP_PACKED=1 requires N=1")
        if environ.get("JETSPEC_ADAPTER_GEMM_TUNE") == "1":
            problems.append("JETSPEC_ADAPTER_GEMM_TUNE=1 requires N=1")
        if environ.get("JETSPEC_ADAPTER_ATTN_SPLIT") == "1":
            problems.append("JETSPEC_ADAPTER_ATTN_SPLIT=1 requires N=1")
        if environ.get("JETSPEC_DECODER_SINGLE_GATEUP_PACKED") == "1":
            problems.append("JETSPEC_DECODER_SINGLE_GATEUP_PACKED=1 requires N=1")
        if environ.get("JETSPEC_DECODER_SELECT_RADIX") == "1":
            problems.append("JETSPEC_DECODER_SELECT_RADIX=1 requires N=1")
        if environ.get("JETSPEC_REFRESH_GEMM_TUNE") == "1":
            problems.append("JETSPEC_REFRESH_GEMM_TUNE=1 requires N=1")
        if environ.get("JETSPEC_REFRESH_ATTN_TUNE") == "1":
            problems.append("JETSPEC_REFRESH_ATTN_TUNE=1 requires N=1")
        if environ.get("JETSPEC_REFRESH_KV_PACK") == "1":
            problems.append("JETSPEC_REFRESH_KV_PACK=1 requires N=1")
        if environ.get("JETSPEC_REFRESH_GATEUP_PACKED") == "1":
            problems.append("JETSPEC_REFRESH_GATEUP_PACKED=1 requires N=1")
        if environ.get("JETSPEC_REFRESH_INJECT_LEAN2") == "1":
            problems.append("JETSPEC_REFRESH_INJECT_LEAN2=1 requires N=1")
        if environ.get("JETSPEC_CTX_KV_APPEND") == "1":
            problems.append("JETSPEC_CTX_KV_APPEND=1 requires N=1")
        if environ.get("JETSPEC_REFRESH_INJECT_INCLUSIVE") == "1":
            problems.append("JETSPEC_REFRESH_INJECT_INCLUSIVE=1 requires N=1")
        if environ.get("JETSPEC_REFRESH_SWAP") == "1":
            problems.append("JETSPEC_REFRESH_SWAP=1 requires N=1")
        for name in ("JETSPEC_BLOCKHEAD_ATTN_LEN", "JETSPEC_BLOCKHEAD_GLUE_LEAN",
                     "JETSPEC_BLOCKHEAD_QKV_FUSED", "JETSPEC_BLOCKHEAD_LMHEAD_KN",
                     "JETSPEC_BLOCKHEAD_BENCH"):
            if environ.get(name) == "1":
                problems.append(f"{name}=1 requires N=1")
        if environ.get("JETSPEC_BLOCKHEAD_DECISION_AUDIT"):
            problems.append("JETSPEC_BLOCKHEAD_DECISION_AUDIT requires N=1")

    if environ.get("JETSPEC_CHAIN_WIRE_RECORD") == "1":
        if not environ.get("W280_RECORD"):
            problems.append("JETSPEC_CHAIN_WIRE_RECORD=1 requires W280_RECORD to name the record destination")
        if environ.get("JETSPEC_DRAFT_DEADLINE") == "1":
            problems.append("JETSPEC_CHAIN_WIRE_RECORD=1 refuses JETSPEC_DRAFT_DEADLINE=1: "
                            "recording runs with the deadline off so recorded rounds equal served rounds")
        if not jetspec or environ.get("JETSPEC_CHAIN_GRAPH") != "seg":
            problems.append("JETSPEC_CHAIN_WIRE_RECORD=1 requires the segmented JetSpec chain")
        for name in ("E04_RECORD_PTOP1", "E04_RECORD_QSHADOW"):
            if environ.get(name) != "1":
                problems.append(f"JETSPEC_CHAIN_WIRE_RECORD=1 requires {name}=1")

    if jetspec and environ.get("JETSPEC_CHAIN_GRAPH") == "seg":
        if max_num_seqs != 1:
            problems.append("JETSPEC_CHAIN_GRAPH=seg requires N=1")
        if "JETSPEC_ROWS_FUSED" in environ:
            problems.append("JETSPEC_CHAIN_GRAPH=seg requires JETSPEC_ROWS_FUSED unset")
        if environ.get("JETSPEC_DRAFTS") not in ("15", "31"):
            problems.append("JETSPEC_CHAIN_GRAPH=seg requires JETSPEC_DRAFTS=15 or 31")
        if (environ.get("JETSPEC_HYBRID_PAIR") == "1"
                and width == 32
                and environ.get("JETSPEC_ROWS_K3") == "1"):
            problems.append("b32 segmented chain requires the Torch row rule (JETSPEC_ROWS_K3=0)")
        if environ.get("JETSPEC_ROWS_RULE_CAP") not in ("4", "8", "16"):
            problems.append("JETSPEC_CHAIN_GRAPH=seg requires JETSPEC_ROWS_RULE_CAP=4, 8 or 16")

    if environ.get("JETSPEC_PAIR_OUTCOME_PUSH") == "1":
        if max_num_seqs != 1:
            problems.append("JETSPEC_PAIR_OUTCOME_PUSH=1 requires N=1")
        if environ.get("JETSPEC_CHAIN_RECORD"):
            problems.append("JETSPEC_PAIR_OUTCOME_PUSH=1 refuses JETSPEC_CHAIN_RECORD")
        if (environ.get("JETSPEC_MPS_DRAFT_ACTIVE_THREAD_PERCENTAGE") or
                environ.get("CUDA_MPS_PIPE_DIRECTORY") or environ.get("CUDA_MPS_LOG_DIRECTORY")):
            problems.append("JETSPEC_PAIR_OUTCOME_PUSH=1 refuses the draft MPS client")

    if environ.get("JETSPEC_HYBRID_PAIR") == "1":
        if environ.get("JETSPEC_REFRESH_KV_PACK") == "1":
            problems.append("JETSPEC_REFRESH_KV_PACK=1 refuses the hybrid adapter "
                            "(refresh uses the gated partial-RoPE module route, not the packed body)")
        if environ.get("JETSPEC_REFRESH_GATEUP_PACKED") == "1":
            problems.append("JETSPEC_REFRESH_GATEUP_PACKED=1 refuses the hybrid adapter")
        if environ.get("JETSPEC_C1_GATEUP_PACKED") == "1":
            problems.append("JETSPEC_C1_GATEUP_PACKED=1 refuses the hybrid adapter")
        if environ.get("JETSPEC_REFRESH_SWAP_GLUE") == "1":
            problems.append("JETSPEC_REFRESH_SWAP_GLUE=1 refuses the hybrid decoder "
                            "(GDN state carry and gated D256 partial-RoPE attention)")
        if not jetspec or max_num_seqs != 1:
            problems.append("JETSPEC_HYBRID_PAIR=1 requires N=1 JetSpec")
        if environ.get("JETSPEC_DRAFTS") not in ("15", "31"):
            problems.append("JETSPEC_HYBRID_PAIR=1 requires JETSPEC_DRAFTS=15 or 31")
        requirements = ()
        if environ.get('JETSPEC_CHAIN_GRAPH', '0') == '0':
            requirements += (("JETSPEC_REFRESH_SWAP", "1"),)
        else:
            requirements += (("JETSPEC_CHAIN_GRAPH", "seg"),
                             ("JETSPEC_VERIFY_DEVICE", "0" if environ.get("JETSPEC_VERIFY_REFERENCE") == "1" else "1"))
        for name, required in requirements:
            if environ.get(name) != required:
                problems.append(f"JETSPEC_HYBRID_PAIR=1 requires {name}={required}")
    if environ.get("JETSPEC_HYBRID_STATE_AUDIT") == "1" and environ.get("JETSPEC_HYBRID_PAIR") != "1":
        problems.append("JETSPEC_HYBRID_STATE_AUDIT=1 requires JETSPEC_HYBRID_PAIR=1")
    if environ.get("JETSPEC_HYBRID_TRANSFER_TIMING") == "1" and environ.get("JETSPEC_HYBRID_PAIR") != "1":
        problems.append("JETSPEC_HYBRID_TRANSFER_TIMING=1 requires JETSPEC_HYBRID_PAIR=1")
    # The audit reads the N=1 device verify's logits, so the Qwen3-8B chain pair carries it too.
    if (environ.get("JETSPEC_HYBRID_VERIFY_AUDIT") == "1" and environ.get("JETSPEC_HYBRID_PAIR") != "1"
            and not (max_num_seqs == 1 and (environ.get("JETSPEC_VERIFY_DEVICE") == "1"
                                           or environ.get("SSD_VERIFY_DEVICE") == "1"
                                           or environ.get("JETSPEC_VERIFY_REFERENCE") == "1"))):
        problems.append("JETSPEC_HYBRID_VERIFY_AUDIT=1 requires JETSPEC_HYBRID_PAIR=1 or the N=1 "
                        "device verify (JETSPEC_VERIFY_DEVICE=1 or SSD_VERIFY_DEVICE=1)")

    if problems:
        raise EnvironmentContractError(
            "serving environment refused:\n  - " + "\n  - ".join(problems))
    return {name: environ[name] for name in sorted(environ) if name in REGISTRY}


def summary():
    serve = [knob for knob in REGISTRY.values() if knob.reader == "serve"]
    return {
        "names": len(REGISTRY),
        "serve": len(serve),
        "launcher": len(REGISTRY) - len(serve),
        "required": sum(knob.default == REQUIRED for knob in serve),
        "no_default": sum(knob.default == NO_DEFAULT for knob in serve),
        "allowed_sets": sum(knob.allowed is not None for knob in REGISTRY.values()),
        "requirements": len(REQUIREMENTS),
        "refusals": len(REFUSALS),
    }


if __name__ == "__main__":
    for knob in REGISTRY.values():
        print(f"{knob.name:36s} {knob.reader:8s} {knob.kind:5s} default={knob.default!s:12.40s} "
              f"allowed={'' if knob.allowed is None else ','.join(knob.allowed)}")
    print("REGISTRY " + " ".join(f"{key}={value}" for key, value in summary().items()))
