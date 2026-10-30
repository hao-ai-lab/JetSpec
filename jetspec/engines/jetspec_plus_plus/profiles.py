"""Frozen kernel and transport selections; compile pins refer to model artifacts."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from jetspec.engines.jetspec_plus_plus.draft.method.model_spec import AttentionSpec


TRANSPORT_SWITCHES = frozenset((
    'draft_streams', 'refresh_streams', 'pair_outcome_push',
    'reference_candidate_socket', 'reference_candidate_spin',
    'reference_candidate_main', 'reference_candidate_shadow',
    'reference_host_select', 'reference_outcome_host',
    'reference_command_socket', 'reference_tap_prepare',
))


# Compatibility refusals only: these experiments belong to no frozen profile.
RETIRED_SELECTORS = dict.fromkeys((
    'j0_norm_gemm_fused', 'row_geometry_fused', 'decoder_gateup_packed',
    'chain_attn3', 'decoder_attn_len_long', 'refresh_swap_glue_fused',
    'adapter_attn_layout_fused', 'adapter_attn_split', 'ctx_kv_one_gemm',
    'packed_live_rows'), '0')
RETIRED_SELECTORS['fuse'] = 'off'


def refuse_retired(environ):
    """Accept old defaults, but never silently run a retired experiment."""
    for variant, default in RETIRED_SELECTORS.items():
        name = 'JETSPEC_' + variant.upper()
        if name in environ and environ[name] != default:
            raise ValueError(f'{name}={environ[name]!r}: retired variant {variant}; '
                             f'only its old default {default!r} is accepted')


@dataclass(frozen=True)
class Profile:
    kernels: frozenset
    transport: frozenset = frozenset()
    compile_pin: str = None

    def __post_init__(self):
        object.__setattr__(self, 'kernels', frozenset(self.kernels))
        object.__setattr__(self, 'transport', frozenset(self.transport))
        names = [item.partition('=')[0] for item in self.kernels]
        if len(names) != len(set(names)):
            raise ValueError('a profile cannot select two variants of one kernel')

    def value(self, name, default=None):
        """Return the selected variant by membership; absent flag means disabled."""
        prefix = name + '='
        return next((item[len(prefix):] for item in self.kernels if item.startswith(prefix)), default)


def legacy_names(kernels):
    """The replay boundary can compare a frozen selection to old settings dumps."""
    result = {}
    for item in kernels:
        name, value = item.split('=', 1)
        key = name.upper() if name.startswith(('ssd_', 'w')) else 'JETSPEC_' + name.upper()
        result[key] = value
    return result


def _kernel_switch(name):
    if any(token in name for token in ('check', 'audit', 'profile', 'dump', 'timing')):
        return False
    # Opt-in verify cuts are engine settings read beside the profile, never members.
    if name.startswith('verify_cut_'):
        return False
    return (any(token in name for token in (
        'fused', 'fuse', 'gluon', 'lean', 'packed', 'attn', 'lmhead_kn',
        'gemm_tune', 'qkv', 'k3', 'rope', 'kvnorm', 'slot_write', 'gqa_view',
        'select_radix', 'prefix_merge', 'kv_pack', 'kv_one_gemm'))
        or name in ('ssd_attention_backend', 'c1_gemm', 'c1_norm',
                    'refresh_pack', 'chain_sega_glue'))


def legacy_selector_names():
    """Registry names captured by the profile adapter, including stream options."""
    from jetspec.engines.jetspec_plus_plus.runtime.env import legacy_registry as _legacy_registry
    REGISTRY = _legacy_registry()
    kernels = frozenset(key for key, knob in REGISTRY.items()
                        if knob.kind in ('flag', 'enum')
                        and _kernel_switch(key.removeprefix('JETSPEC_').lower()))
    streams = frozenset('JETSPEC_' + name.upper() for name in TRANSPORT_SWITCHES)
    return kernels | streams


def select_legacy(environ, mode='current'):
    """Capture legacy selectors once at the adapter, before either rank is spawned."""
    from jetspec.engines.jetspec_plus_plus.runtime.env import REQUIRED
    from jetspec.engines.jetspec_plus_plus.runtime.env import NO_DEFAULT
    from jetspec.engines.jetspec_plus_plus.runtime.env import legacy_registry as _legacy_registry
    REGISTRY = _legacy_registry()
    refuse_retired(environ)
    chosen, streams = set(), set()
    for key in legacy_selector_names():
        name = key.removeprefix('JETSPEC_').lower()
        value = environ.get(key, REGISTRY[key].default)
        if name in TRANSPORT_SWITCHES:
            if value == '1':
                streams.add(name)
            continue
        if value in (REQUIRED, NO_DEFAULT, None, '', '0', 'off'):
            continue
        # The existing environment contract validates launch inputs. Capture here
        # also serves its budget lookup, before that contract finishes validation.
        chosen.add(name + '=' + value)
    kernels = frozenset(chosen)
    streams = frozenset(streams)
    if mode in ('pipelined', 'critical-path', 'on'):
        for name in ('paper-eval-a100-8b', 'paper-eval-a100-27b', 'paper-eval-a100-27b-b32'):
            if PROFILES[name].kernels == kernels and not streams:
                return name + '@' + mode
    for name, profile in PROFILES.items():
        if (name.startswith(('recipe-', 'table3-')) and profile.kernels == kernels
                and profile.transport - {'pair', 'lockstep', 'eager'} == streams):
            return name + '@' + mode
    # Standalone legacy/qualification callers can select a tested subcombination.
    # Its serialized identity contains the whole immutable selection, not process state.
    return 'legacy:' + ','.join(sorted(kernels)) + ';' + ','.join(sorted(streams)) + '@' + mode


def select_config(resolved, mode='current'):
    """Capture compatibility engine leaves; explicit named profiles need no switches."""
    from jetspec.engines.jetspec_plus_plus.launch_env import execution_settings
    settings = execution_settings(resolved)
    refuse_retired({'JETSPEC_' + name.upper(): settings[name]
                   for name in RETIRED_SELECTORS if name in settings})
    explicit = resolved['engine'].get('profile', 'current')
    if explicit != 'current' and explicit in PROFILES:
        if mode in ('pipelined', 'critical-path', 'on') and mode != execution(explicit):
            return explicit + '@' + mode
        return explicit
    from jetspec.engines.jetspec_plus_plus.runtime.env import legacy_registry as _legacy_registry
    REGISTRY = _legacy_registry()
    environment = {}
    for key in REGISTRY:
        name = key.removeprefix('JETSPEC_').lower()
        field = 'attention_backend' if name == 'ssd_attention_backend' else name
        if field in settings:
            environment[key] = settings[field]
    engine = resolved['engine']
    if (explicit == 'current' and engine['kind'] == 'fast' and engine.get('cutoffs', True)
            and not engine.get('recipe') and resolved['hardware']['gpu'] == 'a100'
            and 'model' in resolved and 'method' in resolved):
        from jetspec.engines.jetspec_plus_plus.launch.entry import serving_recipe
        from jetspec.engines.jetspec_plus_plus.launch_env import environment_name
        qualified = serving_recipe(resolved)['environment']
        if qualified.get(environment_name('engine', 'engine_settings.hybrid_verify_triton')) == '1':
            # The immutable engine and launch overlay must select the same route.
            for field in ('attention_backend', 'verify_plan_lean'):
                key = environment_name('engine', 'engine_settings.' + field)
                environment[key] = qualified[key]
    selected = select_legacy(environment, mode)
    return selected


def kernel_value(name, default=None, *, environ=None):
    """Read only the Engine's frozen membership; environ is a legacy adapter input."""
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    profile = get_profile(current_engine(environ).profile)
    # Default-on flags are captured during selection, so absence always means off.
    from jetspec.engines.jetspec_plus_plus.runtime.env import legacy_registry as _legacy_registry
    REGISTRY = _legacy_registry()
    key = next(iter(legacy_names({name + '=1'})))
    if key in REGISTRY and REGISTRY[key].kind == 'flag':
        default = '0'
    return profile.value(name, default)


def kernel_enabled(name, value='1', *, environ=None):
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    kernels = get_profile(current_engine(environ).profile).kernels
    if value in ('0', 'off'):
        return not any(item.startswith(name + '=') for item in kernels)
    return name + '=' + value in kernels


def transport_enabled(name, *, environ=None, run_config=None):
    from jetspec.engines.jetspec_plus_plus.engine import current_engine, _installed
    if _installed is None and run_config is not None and environ is None:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        environ = {'JETSPEC_' + switch.upper(): setting(
            run_config, 'engine.engine_settings.' + switch, '0')
            for switch in TRANSPORT_SWITCHES}
    return name in get_profile(current_engine(environ).profile).transport


# One base selection; shape restrictions and unqualified paths are independent.
BASE_KERNEL_SET = frozenset({
    'adapter_gemm_tune=1',
    'adapter_query_layout_fused=1',
    'anchor_select_fused=1',
    'attn_sm100=1',
    'batch_rows_assembly_fused=1',
    'c1_attn_len=1',
    'c1_fused=2',
    'c1_gateup_packed=1',
    'c1_gemm=direct',
    'c1_glue_fused=1',
    'c1_gluon=1',
    'c1_kv_split_lean=1',
    'c1_kvnorm_batch=1',
    'c1_lean=1',
    'c1_norm=wide',
    'c1_prep_lean=1',
    'c1_qk_fused=1',
    'c1_rope_norm=1',
    'c1_rope_table=1',
    'c1_span_kv_fused=1',
    'candidate_decode_lean=1',
    'chain_admit_host_lean=1',
    'chain_host_lean=1',
    'chain_intake_fused=1',
    'chain_mr_glue_lean=1',
    'chain_post_lean=1',
    'chain_publish_fused=1',
    'chain_sega_glue=2',
    'ctx_kv_fused=1',
    'decoder_attn_len=1',
    'decoder_glue_fused=1',
    'decoder_gqa_view=1',
    'decoder_graph_lean=1',
    'decoder_lmhead_kn=1',
    'decoder_prefix_merge=1',
    'decoder_qkv_fused=1',
    'decoder_select_radix=1',
    'decoder_single_gateup_packed=1',
    'decoder_slot_write=1',
    'e04_lean=1',
    'fused_qkv=1',
    'hybrid_c1_fused=1',
    'hybrid_c1_gluon=1',
    'hybrid_c1_lean=1',
    'hybrid_decoder_refresh_state_lean=1',
    'j0_gateup_packed=1',
    'j0_lmhead_kn=1',
    'j0_qkv_glue_fused=1',
    'jpass_attn_len=1',
    'jpass_glue_lean=1',
    'k3_hazard_split=1',
    'lmhead_fused=1',
    'refine_k3=1',
    'refresh_attn_lean=1',
    'refresh_attn_tune=1',
    'refresh_gateup_packed=1',
    'refresh_inject_lean2=1',
    'refresh_inject_lean=1',
    'refresh_inject_mask_fused=1',
    'refresh_kv_pack=1',
    'refresh_lean=1',
    'refresh_pack=2',
    'refresh_prep_lean=1',
    'refresh_qk_fused=1',
    'refresh_swap_lean=1',
    'refresh_swap_write_fused=1',
    'request_lean=1',
    'rows_fused_graph=1',
    'rows_k3=1',
    'ssd_attention_backend=flashinfer',
    'tail_lean=1',
    'verify_outcome_fused=1',
    'verify_plan_lean=1',
    'verify_post_lean=1',
    'verify_select_lean=1',
    'w632_head_fuse=1',
})

# These paths have not been qualified with recurrent target state. In particular,
# verify_plan_lean requires the FlashInfer verify route; D256 defaults to SDPA.
KERNEL_QUALIFICATION_EXCEPTIONS = MappingProxyType({
    'candidate_decode_lean': 'not qualified with recurrent target state',
    'chain_admit_host_lean': 'not qualified with recurrent target state',
    'chain_post_lean': 'not qualified with recurrent target state',
    'j0_lmhead_kn': 'not qualified with recurrent target state',
    'jpass_attn_len': 'not qualified with recurrent target state',
    'k3_hazard_split': 'not qualified with recurrent target state',
    'verify_outcome_fused': 'not qualified with recurrent target state',
    'verify_plan_lean': 'not qualified with recurrent target state',
    'verify_post_lean': 'not qualified with recurrent target state',
    'verify_select_lean': 'not qualified with recurrent target state',
})

# verify_cut_lean27b qualifies these verify paths bitwise with recurrent target state at
# the 27B attention shape. verify_plan_reuse is an engine setting, not a member. The
# verify_plan_lean exception stays: it also needs verify_stage_host and verify_stage_blob.
# verify_plan_reuse is not qualified: the D256 FlashInfer 0.6.18 wrapper is JIT-built, and the
# reuse guard refuses JIT (B200 run 2026-10-17).
VERIFY_CUT_LEAN27B = frozenset(('verify_outcome_fused', 'verify_post_lean'))


def verify_cut_lean27b(name, owner, runner):
    """Opt-in verify_cut_lean27b: name is qualified for this recurrent target runner."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    if (name not in VERIFY_CUT_LEAN27B
            or setting(owner, 'engine.engine_settings.verify_cut_lean27b', '0') != '1'):
        return False
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    from jetspec.engines.jetspec_plus_plus.models.state import has_state
    if current_engine().kind != 'fast' or not has_state(runner):
        return False
    attention = runner.model_spec.attention
    return (attention.head_dim == 256 and attention.rotary_dim == 64
            and attention.output_gate and attention.norm_offset == 1
            and attention.norm_round == 'after_weight')

# Existing wrappers retain their architecture fallback (for example attn_sm100
# falls back on sm80). Membership does not assert native Gluon availability.
_DENSE_SHAPE_KERNELS = frozenset((
    'adapter_gemm_tune',
    'adapter_query_layout_fused',
    'attn_sm100',
    'c1_attn_len',
    'c1_gateup_packed',
    'c1_gluon',
    'c1_kv_split_lean',
    'c1_qk_fused',
    'c1_span_kv_fused',
    'ctx_kv_fused',
    'decoder_attn_len',
    'decoder_gqa_view',
    'decoder_lmhead_kn',
    'decoder_prefix_merge',
    'decoder_qkv_fused',
    'decoder_select_radix',
    'decoder_single_gateup_packed',
    'decoder_slot_write',
    'refresh_attn_lean',
    'refresh_attn_tune',
    'refresh_gateup_packed',
    'refresh_inject_lean2',
    'refresh_inject_mask_fused',
    'refresh_kv_pack',
    'refresh_prep_lean',
    'refresh_qk_fused',
))
_STATE_SHAPE_KERNELS = frozenset((
    'draft_fuse_a256',
    'hybrid_c1_fused',
    'hybrid_c1_gluon',
    'hybrid_c1_lean',
    'hybrid_decoder_refresh_state_lean',
))


def _qualified_kernel_set(attention, has_state):
    dense = (attention.head_dim == 128 and attention.rotary_dim == 128
             and not attention.output_gate and attention.norm_offset == 0
             and attention.norm_round == 'before_weight')
    state = (has_state and attention.head_dim == 256 and attention.rotary_dim == 64
             and attention.output_gate and attention.norm_offset == 1
             and attention.norm_round == 'after_weight')
    def supported(item):
        name = item.partition('=')[0]
        if name in _DENSE_SHAPE_KERNELS:
            return dense
        if name in _STATE_SHAPE_KERNELS:
            return state
        return not (has_state and name in KERNEL_QUALIFICATION_EXCEPTIONS)
    return frozenset(item for item in BASE_KERNEL_SET if supported(item))


def kernel_set(spec, architecture):
    """Filter the qualified base against ModelSpec, preserving wrapper fallbacks.

    No arithmetic is generalized here. Architecture-specific native dispatch
    remains in each existing wrapper; current sm80/sm100 memberships coincide.
    Sealed profiles bypass this policy and retain their recorded selections.
    """
    if architecture not in ('sm80', 'sm100'):
        raise ValueError(f'unqualified kernel architecture {architecture!r}')
    return _qualified_kernel_set(spec.attention, spec.has_state)


# Named current profiles are historical qualifications of these two shapes.
# New model callers use kernel_set(ModelSpec, architecture), never a model name.
_CURRENT_ATTENTION_SHAPES = {
    '8b': (AttentionSpec(32, 8, 128, 128, False, 0, 'before_weight'), False),
    '27b': (AttentionSpec(q_heads=24, kv_heads=4, head_dim=256, rotary_dim=64,
                         output_gate=True, norm_offset=1, norm_round='after_weight'), True),
}
PROFILE_QUALIFICATION = MappingProxyType({
    'current-gb200-8b': 'not yet measured; not a Table 3 baseline',
    'current-gb200-27b': 'not yet measured; not a Table 3 baseline',
})

# Frozen selections, sealed from the recipe settings dumps and Table 3 basis manifests.
_DATA = json.loads((Path(__file__).parent / 'config/profiles.json').read_text())
_KERNEL_SETS = {
    **{name: frozenset(items) for name, items in _DATA['kernel_sets'].items()},
    **{f'current-{gpu}-{name}': _qualified_kernel_set(*shape)
       for gpu in ('a100', 'b200', 'gb200')
       for name, shape in _CURRENT_ATTENTION_SHAPES.items()},
}
PROFILES = MappingProxyType({
    name: Profile(_KERNEL_SETS[row['kernels']], frozenset(row['transport']), row['compile_pin'])
    for name, row in _DATA['profiles'].items()})
PAPER_RECIPES = MappingProxyType(_DATA['paper_recipes'])
_TABLE3_EXECUTION = _DATA['table3_execution']


def require_live_execution(name):
    """Historical eager profiles describe sealed reads, never a new launch."""
    if execution(name) in ('eager', 'eager-timed'):
        raise ValueError(f'{name}: retired eager execution')


def get_profile(name):
    """Resolve a named profile; legacy snapshots remain self-contained at spawn."""
    name = name.split('@', 1)[0]
    if name in ('eager', 'eager-timed', 'pipelined', 'critical-path', 'on', 'uncut'):
        return PROFILES['current']
    if name.startswith('legacy:'):
        kernels, _, transport = name[7:].partition(';')
        return Profile(frozenset(filter(None, kernels.split(','))),
                       frozenset(filter(None, transport.split(','))))
    return PROFILES[name]


def execution(name):
    """Compatibility execution labels are distinct from kernel membership."""
    if '@' in name:
        return name.split('@', 1)[1]
    if name.startswith('paper-eval-') and name in PAPER_RECIPES:
        if name.endswith(('-dspark', '-dflash', '-domino', '-jetspec-sync')):
            return 'eager'
    return _TABLE3_EXECUTION.get(name, 'pipelined' if name.startswith('paper-eval-') else name)
