"""Host keys and validation of explicitly supplied compatibility launch records.

Serving declarations are a data-only record beside the retired binding table.
Ranks consume config; compatibility callers supply their values explicitly.
"""
from __future__ import annotations
import difflib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REQUIRED = "REQUIRED"
NO_DEFAULT = "NO_DEFAULT"
FLAG = ("0", "1")

@dataclass(frozen=True)
class Knob:
    name: str
    kind: str
    default: str
    allowed: tuple = None
    reader: str = "serve"
    note: str = ""

class EnvironmentContractError(ValueError):
    """An explicit launch record breaks the serving contract."""

@dataclass(frozen=True)
class Requires:
    name: str
    flag: str
    values: tuple
    when: tuple = None
    why: str = ""
    always: bool = False

@dataclass(frozen=True)
class Refuses:
    name: str
    when: tuple
    other: str
    values: tuple
    why: str = ""

@lru_cache(maxsize=1)
def _legacy_contract():
    path = Path(__file__).resolve().parents[1] / 'legacy_env.json'
    record = json.loads(path.read_text())['contract']
    def tuples(row):
        return {key: tuple(value) if isinstance(value, list) else value
                for key, value in row.items()}
    return record, {name: Knob(**tuples(row)) for name, row in record['registry'].items()}, tuple(
        Requires(**tuples(row)) for row in record['requirements']), tuple(
        Refuses(**tuples(row)) for row in record['refusals'])

def legacy_registry():
    """Defaults and contracts retained for historical recipes and launch adapters."""
    return _legacy_contract()[1]

def _name(category, field):
    from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name
    return environment_name(category, field)

REGISTRY = {name: Knob(name, 'str', NO_DEFAULT, reader='launcher') for name in (
    'JETSPEC_HOST', 'CUDA_VISIBLE_DEVICES', 'JETSPEC_SHM_NAME',
    'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
    'JETSPEC_RESOLVED_CONFIG', 'JETSPEC_RESOLVED_CONFIG_SHA256', 'SSD_DIST_PORT',
)}
SCANNED_PREFIXES = tuple(_legacy_contract()[0]['scanned_prefixes'])
VALIDATED_PREFIXES = tuple(_legacy_contract()[0]['validated_prefixes'])
ALLOW_UNREGISTERED = _legacy_contract()[0]['allow_unregistered']
_ROWS_WAIVER = tuple(_legacy_contract()[0]['rows_waiver'])
REQUIREMENTS = _legacy_contract()[2]
REFUSALS = _legacy_contract()[3]


def require_verify_device_batch(max_num_seqs: int, environ=None, *, run_config=None) -> None:
    """Refuse the device path when a batch can contain more than one sequence."""
    if max_num_seqs == 1:
        return
    environ = {} if environ is None else environ
    if run_config is not None:
        from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
        values = config_values(run_config)
        fields = ('verify_reference', 'jetspec_verify_device', 'verify_outcome_fused',
                  'ssd_verify_device', 'pair_outcome_push')
        environ = {_name('engine', 'engine_settings.' + field):
                   values.get('engine.engine_settings.' + field, '0') for field in fields}
    if environ.get(_name('engine', 'engine_settings.verify_reference'), "0") == "1" and max_num_seqs != 1:
        raise ValueError("JETSPEC_VERIFY_REFERENCE=1 requires N=1")
    if environ.get(_name('engine', 'engine_settings.jetspec_verify_device'), "0") == "1" and max_num_seqs != 1:
        raise ValueError("JETSPEC_VERIFY_DEVICE=1 requires N=1")
    if environ.get(_name('engine', 'engine_settings.verify_outcome_fused'), "0") == "1" and max_num_seqs != 1:
        raise ValueError("JETSPEC_VERIFY_OUTCOME_FUSED=1 requires N=1")
    if environ.get(_name('engine', 'engine_settings.ssd_verify_device'), "0") == "1" and max_num_seqs != 1:
        raise ValueError("SSD_VERIFY_DEVICE=1 requires N=1")
    if environ.get(_name('engine', 'engine_settings.pair_outcome_push'), "0") == "1" and max_num_seqs != 1:
        raise ValueError("JETSPEC_PAIR_OUTCOME_PUSH=1 requires N=1")


def _effective(environ, name):
    value = environ.get(name)
    if value is not None:
        return value
    default = legacy_registry()[name].default
    return None if default in (REQUIRED, NO_DEFAULT) else default


def context_append_enabled(environ=None, *, run_config=None):
    if run_config is not None:
        from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
        values = config_values(run_config)
        return (values.get('engine.engine_settings.ctx_kv_append', '0') == '1'
                or values.get('engine.engine_settings.hybrid_context_dirty', '0') == '1')
    environ = {} if environ is None else environ
    return (environ.get(_name('engine', 'engine_settings.ctx_kv_append'), '0') == '1'
            or environ.get(_name('engine', 'engine_settings.hybrid_context_dirty'), '0') == '1')


def unregistered_names(environ):
    from jetspec.engines.jetspec_plus_plus.profiles import RETIRED_SELECTORS
    retired = {'JETSPEC_' + name.upper() for name in RETIRED_SELECTORS}
    return sorted(name for name in environ
                  if name.startswith(VALIDATED_PREFIXES) and name not in legacy_registry()
                  and name not in REGISTRY
                  and name not in retired)


def _suggest(name):
    close = difflib.get_close_matches(name, legacy_registry(), n=3, cutoff=0.6)
    return f"{name} (did you mean {', '.join(close)}?)" if close else f"{name} (no close registered name)"


def frozen_paper_recipes():
    """Explicit historical serving records; these names are never live N=1 recipes."""
    from jetspec.engines.jetspec_plus_plus.read.legacy_recipes import frozen_paper_recipes as read_frozen_paper_recipes
    return read_frozen_paper_recipes()


def require_recipe_launch(recipe_name, *, max_num_seqs=1, resolve_only=False):
    if max_num_seqs != 1 or recipe_name is None:
        return None
    from jetspec.engines.jetspec_plus_plus.read.legacy_recipes import FROZEN_PAPER_NAMES
    if recipe_name not in FROZEN_PAPER_NAMES:
        return None
    record = frozen_paper_recipes().get(recipe_name)
    if record is not None and not resolve_only:
        raise EnvironmentContractError(
            f"historical paper recipe {recipe_name!r} is dry-resolve only; "
            f"use submission commit {record['submission_commit']} for reproduction")
    return record


def validate_lane_regime(environ, regime=None):
    """Whole-lane checks; legacy in-lane anchor-selection graphs are grandfathered."""
    regime = regime or ('eager' if environ.get(_name('engine', 'engine_settings.torchdynamo_disable')) == '1' else None)
    if regime not in (None, 'eager', 'graphed'):
        raise EnvironmentContractError(f'unknown engine regime {regime!r}')
    if regime == 'eager':
        graph_flags = (_name('engine', 'engine_settings.c1_graph'), _name('engine', 'engine_settings.rows_graphs'), _name('engine', 'engine_settings.f_graphs'),
                       _name('engine', 'engine_settings.jetspec_verify_device'), _name('engine', 'engine_settings.front_static'),
                       _name('engine', 'engine_settings.decoder_graph_window'), _name('engine', 'engine_settings.rows_fused_graph'),
                       _name('engine', 'engine_settings.blockhead_graph'), _name('engine', 'engine_settings.sync_graph'),
                       _name('engine', 'engine_settings.batch_draft_graph'), _name('engine', 'engine_settings.row_stage_graph'))
        enabled = [name for name in graph_flags if environ.get(name) == '1']
        if environ.get(_name('engine', 'engine_settings.chain_graph')) == 'seg':
            enabled.append(_name('engine', 'engine_settings.chain_graph'))
        if enabled:
            raise EnvironmentContractError('regime eager refuses whole-lane graphs: ' + ', '.join(enabled))
    if (regime == 'graphed' and environ.get(_name('engine', 'engine_settings.refresh_swap')) == '1'
            and environ.get(_name('engine', 'engine_settings.chain_graph'), '0') != 'seg'):
        raise EnvironmentContractError('regime graphed refuses the eager refresh-swap executor')


def inert_requirements(environ, *, jetspec=True):
    """The Requires rules that ``environ`` sets but cannot act on, after the documented exemptions."""
    from jetspec.engines.jetspec_plus_plus.jetspec.eval_profile import FIXED_CONFIGS
    is_eval = environ.get(_name('engine', 'engine_settings.eval_mode'), 'off') != 'off'
    config = environ.get(_name('method', 'method_settings.chain_fixed_config'))
    schedule = None
    if config is not None:
        from jetspec.engines.jetspec_plus_plus.draft.method.schedule import parse_schedule
        try:
            schedule = parse_schedule(config)
        except ValueError:
            pass
    waived = environ.get(_ROWS_WAIVER[0]) == _ROWS_WAIVER[1]
    for rule in REQUIREMENTS:
        if not (jetspec or rule.always):
            continue
        if rule.name not in environ or (rule.when is not None and environ[rule.name] not in rule.when):
            continue
        if (rule.name == _name('engine', 'engine_settings.chain_graph') and rule.flag == _name('engine', 'engine_settings.jetspec_verify_device')
                and environ.get(_name('engine', 'engine_settings.verify_reference'), "0") == "1"):
            continue
        if (rule.name == _name('engine', 'engine_settings.pair_outcome_push')
                and rule.flag in (_name('engine', 'engine_settings.jetspec_verify_device'), _name('engine', 'engine_settings.outcome_pack_presync'))
                and environ.get(_name('engine', 'engine_settings.verify_reference'), "0") == "1"):
            # Host acceptance precedes publication; only delivery changes.
            continue
        if rule.flag == _name('engine', 'engine_settings.rows') and waived:
            continue
        if (is_eval and config in FIXED_CONFIGS
                and rule.flag == _name('method', 'method_settings.chain_fixed_config')
                and rule.name in (_name('engine', 'engine_settings.chain_handoff'), _name('engine', 'engine_settings.chain_prepost'),
                                  _name('engine', 'engine_settings.chain_miss_redraft'), _name('engine', 'engine_settings.refresh_swap'),
                                  _name('engine', 'engine_settings.refresh_inject_inclusive'))):
            # Eval captures its fixed schedule, including J0-only and R0 cells.
            # Refresh kernels may be available without being replayed in R0.
            continue
        if rule.flag == _name('method', 'method_settings.chain_fixed_config') and schedule is not None:
            if rule.name in (_name('engine', 'engine_settings.refresh_swap'), _name('engine', 'engine_settings.refresh_inject_inclusive')):
                if schedule.r >= 1:
                    continue
            elif rule.name == _name('engine', 'engine_settings.chain_j_switch_l'):
                if schedule.j >= 2 and schedule.r == 0:
                    continue
            else:
                continue
        if (rule.name == _name('engine', 'engine_settings.chain_miss_redraft')
                and rule.flag == _name('method', 'method_settings.chain_fixed_config')
                and rule.flag not in environ):
            # The dynamic launch policy selects the same captured J1/J2/J3 bodies.
            continue
        if (rule.name == _name('engine', 'engine_settings.ctx_kv_append') and rule.flag == _name('engine', 'engine_settings.chain_miss_redraft')
                and schedule is not None and (schedule.r >= 1 or is_eval)
                and environ.get(_name('engine', 'engine_settings.chain_prepost')) == "1"):
            # R1's existing prepost command supplies exact host L. Refresh writes
            # separate KV slots; the next Seg A hides them and dirties prefix+tail.
            continue
        if (rule.name == _name('engine', 'engine_settings.ssd_verify_device') and rule.flag == _name('model', 'model_settings.blockhead')
                and environ.get(_name('model', 'model_settings.sync_drafter'))):
            continue
        if (rule.name == _name('method', 'method_settings.kset') and rule.flag == _name('engine', 'engine_settings.legacy_kpred_forward')
                and _effective(environ, rule.flag) == "0"
                and environ.get(_name('method', 'method_settings.kscore')) == "blend_dec"
                and environ.get(_name('method', 'method_settings.rowrule')) == "joint"
                and environ.get(_name('method', 'method_settings.kset_after_miss')) == "zero"):
            continue
        if rule.values is None:
            if not _effective(environ, rule.flag):
                yield rule
        elif _effective(environ, rule.flag) not in rule.values:
            yield rule


def validate_environment(environ=None, *, jetspec=True, max_num_seqs=1, warn=print,
                         recipe_name=None, resolve_only=False, regime=None):
    """Refuse an environment that would serve something other than what it spells.

    ``jetspec``: whether this run serves JetSpec (bench.py ``--jetspec``); check (c) needs it.
    ``max_num_seqs`` is the actual serving concurrency, not the prompt count.
    A frozen ``recipe_name`` is valid only with ``resolve_only=True`` at N=1.
    Returns accepted knobs present in ``environ``, including retired defaults (name -> value). Raises
    ``EnvironmentContractError`` naming every breach at once.
    """
    from jetspec.engines.jetspec_plus_plus.profiles import RETIRED_SELECTORS, refuse_retired
    environ = {} if environ is None else environ
    refuse_retired(environ)
    problems = []
    from jetspec.engines.jetspec_plus_plus.jetspec.eval_profile import assert_uncut
    from jetspec.engines.jetspec_plus_plus.draft.block_geometry import block_width
    width = block_width(environ)
    try:
        assert_uncut(environ)
    except ValueError as error:
        problems.append(str(error))
    if (environ.get(_name('engine', 'engine_settings.hybrid_fp32_linear_bf16_storage'), "0") == "1"
            and environ.get(_name('model', 'model_settings.target_dtype')) != "fp32"):
        problems.append("JETSPEC_HYBRID_FP32_LINEAR_BF16_STORAGE=1 requires JETSPEC_TARGET_DTYPE=fp32")
    config = environ.get(_name('method', 'method_settings.chain_fixed_config'))
    eager_swap = (environ.get(_name('engine', 'engine_settings.chain_graph'), '0') == '0'
                  and environ.get(_name('engine', 'engine_settings.refresh_swap'), '0') == '1')
    schedule = None
    if config is not None:
        from jetspec.engines.jetspec_plus_plus.draft.method.schedule import parse_schedule
        try:
            schedule = parse_schedule(config)
            if eager_swap:
                if schedule.r < 1:
                    problems.append(f'JETSPEC_CHAIN_FIXED_CONFIG={config}: R0 config has no refresh')
                from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import validate_eager_labels
                validate_eager_labels(environ, schedule, recipe=recipe_name)
        except ValueError as error:
            problems.append(str(error))
    if environ.get(_name('engine', 'engine_settings.hybrid_context_dirty'), '0') == '1':
        if environ.get(_name('engine', 'engine_settings.hybrid_pair'), '0') != '1':
            problems.append('JETSPEC_HYBRID_CONTEXT_DIRTY=1 requires JETSPEC_HYBRID_PAIR=1')
        # Apply the existing append contract without changing the caller's map.
        environ = dict(environ, JETSPEC_CTX_KV_APPEND='1')
    try:
        validate_lane_regime(environ, regime)
    except EnvironmentContractError as error:
        problems.append(str(error))
    # The C>1 batched draft round takes J from the pass count; the fixed config is only its label.
    batch_round = (max_num_seqs != 1
                   and environ.get(_name('engine', 'engine_settings.batch_draft_graph')) == '1')
    if (environ.get(_name('method', 'method_settings.chain_fixed_config')) is not None
            and not batch_round
            and environ.get(_name('engine', 'engine_settings.chain_graph'), '0') != 'seg'
            and not (environ.get(_name('engine', 'engine_settings.chain_graph'), '0') == '0'
                     and environ.get(_name('engine', 'engine_settings.refresh_swap'), '0') == '1')):
        problems.append('JETSPEC_CHAIN_FIXED_CONFIG requires JETSPEC_CHAIN_GRAPH=seg or eager refresh swap')
    if (environ.get(_name('engine', 'engine_settings.refresh_swap')) == '1'
            and environ.get(_name('engine', 'engine_settings.chain_graph'), '0') == '0'):
        for name, required in ((_name('engine', 'engine_settings.rows'), '1'), (_name('engine', 'engine_settings.rows_assembly'), '1'),
                               (_name('engine', 'engine_settings.row_layout'), 'anchor0'),
                               (_name('engine', 'engine_settings.refresh_fixed'), '1')):
            if environ.get(name) != required:
                problems.append(f'eager refresh swap requires {name}={required}')
        widths = ('15', '31') if environ.get(_name('engine', 'engine_settings.hybrid_pair')) == '1' else ('15',)
        if environ.get(_name('method', 'method_settings.drafts')) not in widths:
            problems.append('eager refresh swap requires JETSPEC_DRAFTS=' + ' or '.join(widths))
        if width == 32 and environ.get(_name('engine', 'engine_settings.rows_k3')) == '1':
            problems.append('b32 eager refresh swap requires the Torch row rule (JETSPEC_ROWS_K3=0)')
        if environ.get(_name('method', 'method_settings.rows_rule_cap')) not in ('4', '8', '16'):
            problems.append('eager refresh swap requires JETSPEC_ROWS_RULE_CAP=4, 8 or 16')
        try:
            if int(environ.get(_name('engine', 'engine_settings.rows_cap'), '128')) < int(environ.get(_name('method', 'method_settings.rows_rule_cap'), '4')):
                problems.append('eager refresh swap row bucket exceeds JETSPEC_ROWS_CAP')
        except ValueError:
            problems.append('eager refresh swap requires integer row capacities')
    elif (environ.get(_name('engine', 'engine_settings.refresh_swap')) == '1'
          and environ.get(_name('engine', 'engine_settings.chain_graph'), '0') != 'seg'):
        problems.append('JETSPEC_REFRESH_SWAP=1 requires JETSPEC_CHAIN_GRAPH=seg')

    if _name('engine', 'engine_settings.serve_policy') in environ:
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import load_policy
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import validate_policy
        try:
            validate_policy(load_policy(environ), environ, jetspec=jetspec,
                            max_num_seqs=max_num_seqs)
        except ValueError as error:
            problems.append(str(error))

    historical = require_recipe_launch(recipe_name, max_num_seqs=max_num_seqs,
                                       resolve_only=resolve_only)
    if jetspec and max_num_seqs == 1 and historical is None:
        from jetspec.engines.jetspec_plus_plus.cutoffs import budget_type
        fallback = budget_type(environ).required_redraft(environ)
        if environ.get(fallback, "0") != "1":
            problems.append(f"decision-214: N=1 JetSpec serving requires {fallback}=1; "
                            "a miss must redraft from the known true anchor on the critical path")

    if environ.get(_name('engine', 'engine_settings.device_host_stamps')) == "1" and max_num_seqs != 1:
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
                f"Register the name in engine/jetspec.engines.jetspec_plus_plus/jetspec/env.py, or set {ALLOW_UNREGISTERED}=1 to proceed.")

    for name in sorted(environ):
        knob = legacy_registry().get(name)
        if knob is not None and knob.allowed is not None and environ[name] not in knob.allowed:
            problems.append(f"{name}={environ[name]!r} is outside its allowed set {list(knob.allowed)}"
                            + (f" ({knob.note})" if knob.note else ""))

    switch_L = environ.get(_name('engine', 'engine_settings.chain_j_switch_l'))
    if switch_L is not None and (not switch_L.isascii() or not switch_L.isdecimal()
                                 or int(switch_L) % 128):
        problems.append("JETSPEC_CHAIN_J_SWITCH_L must be a nonnegative multiple of 128")

    for rule in inert_requirements(environ, jetspec=jetspec):
        # The chain miss-redraft valves belong to the N=1 chain; the batched round never reads them.
        if batch_round and rule.flag == _name('engine', 'engine_settings.chain_miss_redraft'):
            continue
        if rule.values is None:
            problems.append(
                f"{rule.name}={environ[rule.name]!r} is inert unless {rule.flag} is set: {rule.why}")
        else:
            problems.append(
                f"{rule.name}={environ[rule.name]!r} is inert unless {rule.flag} is "
                f"{' or '.join(rule.values)} (it is {environ.get(rule.flag, 'unset')!r}): {rule.why}")

    for rule in REFUSALS:
        if environ.get(rule.name) in rule.when and environ.get(rule.other) in rule.values:
            problems.append(
                f"{rule.name}={environ[rule.name]!r} refuses {rule.other}={environ[rule.other]!r}: {rule.why}")

    if environ.get(_name('engine', 'engine_settings.hybrid_verify_flashinfer'), "0") == "1":
        if not jetspec or max_num_seqs != 1:
            problems.append("JETSPEC_HYBRID_VERIFY_FLASHINFER=1 requires N=1 JetSpec")
        if regime is not None and regime != "graphed":
            problems.append("JETSPEC_HYBRID_VERIFY_FLASHINFER=1 requires graphed verification")
        for name, expected in ((_name('engine', 'engine_settings.hybrid_pair'), "1"),
                               (_name('engine', 'engine_settings.jetspec_verify_device'), "1"),
                               (_name('engine', 'engine_settings.verify_reference'), "0"),
                               (_name('engine', 'engine_settings.ssd_verify_device'), "0"),
                               (_name('engine', 'engine_settings.attention_backend'), "flashinfer"),
                               (_name('engine', 'engine_settings.hybrid_verify_triton'), "0")):
            if _effective(environ, name) != expected:
                problems.append(f"JETSPEC_HYBRID_VERIFY_FLASHINFER=1 requires {name}={expected}")

    if environ.get(_name('engine', 'engine_settings.verify_reference'), "0") == "1":
        if not jetspec or max_num_seqs != 1:
            problems.append("JETSPEC_VERIFY_REFERENCE=1 requires N=1 JetSpec")
        if regime is not None and regime != "graphed":
            problems.append("JETSPEC_VERIFY_REFERENCE=1 requires the graphed reference engine")
        if _effective(environ, _name('engine', 'engine_settings.verify_device_transport')) != "chain":
            problems.append("JETSPEC_VERIFY_REFERENCE=1 requires JETSPEC_VERIFY_DEVICE_TRANSPORT=chain")
        if environ.get(_name('engine', 'engine_settings.attention_backend'), "flashinfer") != "flashinfer":
            problems.append("JETSPEC_VERIFY_REFERENCE=1 requires SSD_ATTENTION_BACKEND=flashinfer")
        for name, expected in ((_name('engine', 'engine_settings.chain_graph'), "seg"),
                               (_name('engine', 'engine_settings.jetspec_verify_device'), "0"),
                               (_name('engine', 'engine_settings.rows'), "1"), (_name('engine', 'engine_settings.rows_fast'), "1")):
            if _effective(environ, name) != expected:
                problems.append(f"JETSPEC_VERIFY_REFERENCE=1 requires {name}={expected}")
        # These alter target forward/planning, accept, or post-accept mechanics.
        # Candidate decoding, prepost/no-ACK and the wire ring are transport.
        forbidden = (
            _name('engine', 'engine_settings.ssd_verify_device'), _name('engine', 'engine_settings.verify_plan_cache'), _name('engine', 'engine_settings.verify_mask_cache'),
            _name('engine', 'engine_settings.verify_select_lean'), _name('engine', 'engine_settings.verify_select_host'),
            _name('engine', 'engine_settings.verify_record_late'), _name('engine', 'engine_settings.verify_post_lean'),
            _name('engine', 'engine_settings.verify_stage_host'), _name('engine', 'engine_settings.verify_stage_early'),
            _name('engine', 'engine_settings.verify_stage_blob'), _name('engine', 'engine_settings.verify_stage_blob_fast'),
            _name('engine', 'engine_settings.verify_input_direct'), _name('engine', 'engine_settings.verify_input_view'),
            _name('engine', 'engine_settings.verify_plan_reuse'), _name('engine', 'engine_settings.verify_plan_lean'),
            _name('engine', 'engine_settings.verify_prestage_mixed'), _name('engine', 'engine_settings.verify_outcome_fused'),
            _name('engine', 'engine_settings.verify_publish_first'), _name('engine', 'engine_settings.outcome_pack_presync'),
            _name('engine', 'engine_settings.candidate_recv_ahead'), _name('engine', 'engine_settings.verify_host_census'),
        )
        for name in forbidden:
            if environ.get(name, "0") == "1":
                problems.append(f"JETSPEC_VERIFY_REFERENCE=1 refuses {name}=1")

    if environ.get(_name('engine', 'engine_settings.verify_outcome_fused')) == "1":
        if max_num_seqs != 1:
            problems.append("JETSPEC_VERIFY_OUTCOME_FUSED=1 requires N=1")
        if environ.get(_name('model', 'model_settings.blockhead')):
            problems.append("JETSPEC_VERIFY_OUTCOME_FUSED=1 refuses the block-head path")

    if max_num_seqs != 1:
        if environ.get(_name('engine', 'engine_settings.ctx_kv_direct')) == "1":
            problems.append("JETSPEC_CTX_KV_DIRECT=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.decoder_slot_write')) == "1":
            problems.append("JETSPEC_DECODER_SLOT_WRITE=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.j0_lmhead_kn')) == "1":
            problems.append("JETSPEC_J0_LMHEAD_KN=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.j0_gateup_packed')) == "1":
            problems.append("JETSPEC_J0_GATEUP_PACKED=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.j0_qkv_glue_fused')) == "1":
            problems.append("JETSPEC_J0_QKV_GLUE_FUSED=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.draft_streams')) == "1":
            problems.append("JETSPEC_DRAFT_STREAMS=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.refresh_streams')) == "1":
            problems.append("JETSPEC_REFRESH_STREAMS=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.c1_gateup_packed')) == "1":
            problems.append("JETSPEC_C1_GATEUP_PACKED=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.adapter_gemm_tune')) == "1":
            problems.append("JETSPEC_ADAPTER_GEMM_TUNE=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.decoder_single_gateup_packed')) == "1":
            problems.append("JETSPEC_DECODER_SINGLE_GATEUP_PACKED=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.decoder_select_radix')) == "1":
            problems.append("JETSPEC_DECODER_SELECT_RADIX=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.refresh_gemm_tune')) == "1":
            problems.append("JETSPEC_REFRESH_GEMM_TUNE=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.refresh_attn_tune')) == "1":
            problems.append("JETSPEC_REFRESH_ATTN_TUNE=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.refresh_kv_pack')) == "1":
            problems.append("JETSPEC_REFRESH_KV_PACK=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.refresh_gateup_packed')) == "1":
            problems.append("JETSPEC_REFRESH_GATEUP_PACKED=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.refresh_inject_lean2')) == "1":
            problems.append("JETSPEC_REFRESH_INJECT_LEAN2=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.ctx_kv_append')) == "1":
            problems.append("JETSPEC_CTX_KV_APPEND=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.refresh_inject_inclusive')) == "1":
            problems.append("JETSPEC_REFRESH_INJECT_INCLUSIVE=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.refresh_swap')) == "1":
            problems.append("JETSPEC_REFRESH_SWAP=1 requires N=1")
        for name in (_name('engine', 'engine_settings.blockhead_attn_len'), _name('engine', 'engine_settings.blockhead_glue_lean'),
                     _name('engine', 'engine_settings.blockhead_qkv_fused'), _name('engine', 'engine_settings.blockhead_lmhead_kn'),
                     _name('engine', 'engine_settings.blockhead_bench')):
            if environ.get(name) == "1":
                problems.append(f"{name}=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.blockhead_decision_audit')):
            problems.append("JETSPEC_BLOCKHEAD_DECISION_AUDIT requires N=1")

    if environ.get(_name('engine', 'engine_settings.chain_wire_record')) == "1":
        if not environ.get(_name('engine', 'engine_settings.w280_record')):
            problems.append("JETSPEC_CHAIN_WIRE_RECORD=1 requires W280_RECORD to name the record destination")
        if environ.get(_name('engine', 'engine_settings.draft_deadline')) == "1":
            problems.append("JETSPEC_CHAIN_WIRE_RECORD=1 refuses JETSPEC_DRAFT_DEADLINE=1: "
                            "recording runs with the deadline off so recorded rounds equal served rounds")
        if not jetspec or environ.get(_name('engine', 'engine_settings.chain_graph')) != "seg":
            problems.append("JETSPEC_CHAIN_WIRE_RECORD=1 requires the segmented JetSpec chain")
        for name in (_name('engine', 'engine_settings.record_ptop1'), _name('engine', 'engine_settings.record_qshadow')):
            if environ.get(name) != "1":
                problems.append(f"JETSPEC_CHAIN_WIRE_RECORD=1 requires {name}=1")

    if jetspec and environ.get(_name('engine', 'engine_settings.chain_graph')) == "seg":
        if max_num_seqs != 1:
            problems.append("JETSPEC_CHAIN_GRAPH=seg requires N=1")
        if _name('engine', 'engine_settings.rows_fused') in environ:
            problems.append("JETSPEC_CHAIN_GRAPH=seg requires JETSPEC_ROWS_FUSED unset")
        if environ.get(_name('method', 'method_settings.drafts')) not in ("15", "31"):
            problems.append("JETSPEC_CHAIN_GRAPH=seg requires JETSPEC_DRAFTS=15 or 31")
        if (environ.get(_name('engine', 'engine_settings.hybrid_pair')) == "1"
                and width == 32
                and environ.get(_name('engine', 'engine_settings.rows_k3')) == "1"):
            problems.append("b32 segmented chain requires the Torch row rule (JETSPEC_ROWS_K3=0)")
        if environ.get(_name('method', 'method_settings.rows_rule_cap')) not in ("4", "8", "16"):
            problems.append("JETSPEC_CHAIN_GRAPH=seg requires JETSPEC_ROWS_RULE_CAP=4, 8 or 16")

    if environ.get(_name('engine', 'engine_settings.pair_outcome_push')) == "1":
        if max_num_seqs != 1:
            problems.append("JETSPEC_PAIR_OUTCOME_PUSH=1 requires N=1")
        if environ.get(_name('engine', 'engine_settings.chain_record')):
            problems.append("JETSPEC_PAIR_OUTCOME_PUSH=1 refuses JETSPEC_CHAIN_RECORD")
        if (environ.get(_name('hardware', 'hardware_settings.mps_draft_active_thread_percentage')) or
                environ.get("CUDA_MPS_PIPE_DIRECTORY") or environ.get("CUDA_MPS_LOG_DIRECTORY")):
            problems.append("JETSPEC_PAIR_OUTCOME_PUSH=1 refuses the draft MPS client")

    if environ.get(_name('engine', 'engine_settings.hybrid_pair')) == "1":
        if environ.get(_name('engine', 'engine_settings.refresh_kv_pack')) == "1":
            problems.append("JETSPEC_REFRESH_KV_PACK=1 refuses the hybrid adapter "
                            "(refresh uses the gated partial-RoPE module route, not the packed body)")
        if environ.get(_name('engine', 'engine_settings.refresh_gateup_packed')) == "1":
            problems.append("JETSPEC_REFRESH_GATEUP_PACKED=1 refuses the hybrid adapter")
        if environ.get(_name('engine', 'engine_settings.c1_gateup_packed')) == "1":
            problems.append("JETSPEC_C1_GATEUP_PACKED=1 refuses the hybrid adapter")
        if environ.get(_name('engine', 'engine_settings.refresh_swap_glue')) == "1":
            problems.append("JETSPEC_REFRESH_SWAP_GLUE=1 refuses the hybrid decoder "
                            "(GDN state carry and gated D256 partial-RoPE attention)")
        if not jetspec:
            problems.append("JETSPEC_HYBRID_PAIR=1 requires JetSpec")
        if environ.get(_name('method', 'method_settings.drafts')) not in ("15", "31"):
            problems.append("JETSPEC_HYBRID_PAIR=1 requires JETSPEC_DRAFTS=15 or 31")
        requirements = ()
        if max_num_seqs != 1:
            # C>1 serves the batched draft round on the F3 slots: host verify, no swap.
            requirements += ((_name('engine', 'engine_settings.batch_draft_graph'), "1"),
                             (_name('engine', 'engine_settings.perseq'), "0"))
            if environ.get(_name('engine', 'engine_settings.chain_graph'), '0') != '0':
                problems.append("JETSPEC_HYBRID_PAIR=1 at C>1 requires JETSPEC_CHAIN_GRAPH unset")
            if environ.get(_name('method', 'method_settings.tree_b'), '0') != '0':
                problems.append("JETSPEC_HYBRID_PAIR=1 at C>1 requires no tree budget (JETSPEC_TREE_B=0)")
        elif environ.get(_name('engine', 'engine_settings.chain_graph'), '0') == '0':
            requirements += ((_name('engine', 'engine_settings.refresh_swap'), "1"),)
        else:
            requirements += ((_name('engine', 'engine_settings.chain_graph'), "seg"),
                             (_name('engine', 'engine_settings.jetspec_verify_device'), "0" if environ.get(_name('engine', 'engine_settings.verify_reference')) == "1" else "1"))
        for name, required in requirements:
            if environ.get(name) != required:
                problems.append(f"JETSPEC_HYBRID_PAIR=1 requires {name}={required}")
    if environ.get(_name('engine', 'engine_settings.hybrid_state_audit')) == "1" and environ.get(_name('engine', 'engine_settings.hybrid_pair')) != "1":
        problems.append("JETSPEC_HYBRID_STATE_AUDIT=1 requires JETSPEC_HYBRID_PAIR=1")
    if environ.get(_name('engine', 'engine_settings.hybrid_transfer_timing')) == "1" and environ.get(_name('engine', 'engine_settings.hybrid_pair')) != "1":
        problems.append("JETSPEC_HYBRID_TRANSFER_TIMING=1 requires JETSPEC_HYBRID_PAIR=1")
    # serving: the audit reads the N=1 device verify's logits, so the Qwen3-8B chain pair carries it too.
    if (environ.get(_name('engine', 'engine_settings.hybrid_verify_audit')) == "1" and environ.get(_name('engine', 'engine_settings.hybrid_pair')) != "1"
            and not (max_num_seqs == 1 and (environ.get(_name('engine', 'engine_settings.jetspec_verify_device')) == "1"
                                           or environ.get(_name('engine', 'engine_settings.ssd_verify_device')) == "1"
                                           or environ.get(_name('engine', 'engine_settings.verify_reference')) == "1"))):
        problems.append("JETSPEC_HYBRID_VERIFY_AUDIT=1 requires JETSPEC_HYBRID_PAIR=1 or the N=1 "
                        "device verify (JETSPEC_VERIFY_DEVICE=1 or SSD_VERIFY_DEVICE=1)")

    if problems:
        raise EnvironmentContractError(
            "serving environment refused:\n  - " + "\n  - ".join(problems))
    retired = {'JETSPEC_' + name.upper() for name in RETIRED_SELECTORS}
    return {name: environ[name] for name in sorted(environ)
            if name in legacy_registry() or name in retired}


def summary():
    serve = [knob for knob in REGISTRY.values() if knob.reader == "serve"]
    return {
        "names": len(REGISTRY),
        "serve": len(serve),
        "launcher": len(REGISTRY) - len(serve),
        "required": sum(knob.default == REQUIRED for knob in serve),
        "no_default": sum(knob.default == NO_DEFAULT for knob in serve),
        "allowed_sets": sum(knob.allowed is not None for knob in REGISTRY.values()),
        "requirements": 0,
        "refusals": 0,
    }


if __name__ == "__main__":
    for knob in REGISTRY.values():
        print(f"{knob.name:36s} {knob.reader:8s} {knob.kind:5s} default={knob.default!s:12.40s} "
              f"allowed={'' if knob.allowed is None else ','.join(knob.allowed)}")
    print("REGISTRY " + " ".join(f"{key}={value}" for key, value in summary().items()))
