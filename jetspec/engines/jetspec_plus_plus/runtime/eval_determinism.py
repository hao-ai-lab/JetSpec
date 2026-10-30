"""Eval compiler policy: fixed heuristics, no measured configuration selection.

PyTorch's deterministic mode chooses one Triton reduction/pointwise configuration
before consulting an autotune cache. The guard checks that every compiled kernel
actually receives one configuration and records it in the read manifest.
"""

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
import json
import os
from jetspec.engines.jetspec_plus_plus.runtime.config_values import config_values
from jetspec.engines.jetspec_plus_plus.launch_env import environment_name

COMPILER_FLAGS = ('max_autotune', 'max_autotune_gemm', 'max_autotune_pointwise',
                  'coordinate_descent_tuning')
WORKSPACE = ':4096:8'


def install(rank, *, run_config=None):
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    if not current_engine().deterministic:
        return None
    for knob in ('refresh_gemm_tune', 'adapter_gemm_tune'):
        if kernel_enabled(knob):
            label = 'JETSPEC_' + knob.upper()
            raise ValueError(f'eval has no fixed configuration for {label}=1')
    if config_values(run_config).get('engine.engine_settings.c1_compile', '0') == '1':
        raise ValueError('eval has no fixed configuration for ' +
                         environment_name('engine', 'engine_settings.c1_compile') + '=1')
    import torch
    from torch._inductor import config
    from torch._inductor.runtime import triton_heuristics as heuristics
    import jetspec.engines.jetspec_plus_plus.runtime.run_manifest as run_manifest
    if config_values(run_config).get('engine.engine_settings.cublas_workspace_config') != WORKSPACE:
        if torch.cuda.is_initialized():
            raise RuntimeError('eval workspace policy must be installed before CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = WORKSPACE
    torch.use_deterministic_algorithms(True)
    for name in COMPILER_FLAGS:
        setattr(config, name, False)
    config.triton.autotune_pointwise = False
    policy = dict(schema='jetspec.eval_determinism/1', deterministic_algorithms=True,
                  compiler_flags={name: False for name in COMPILER_FLAGS},
                  autotune_pointwise=False, cublas_workspace_config=WORKSPACE)
    run_manifest.record_value('eval_determinism', policy, run_config=run_config)
    # Adaptation: compiler guards are local to this serving package.
    original = heuristics.CachingAutotuner.run
    if getattr(original, '_jetspec_compiler_pin', False):
        return True

    def fixed(tuner, *args, **kwargs):
        # Kernel construction can happen in a compiler worker or an FX-cache
        # loader before installation. Check the actual executable in this rank.
        if getattr(tuner, '_jetspec_compiler_checked', False):
            return original(tuner, *args, **kwargs)
        configs = heuristics.unique_configs(tuner.configs if tuner.configs is not None
            else [launcher.config for launcher in tuner.launchers])
        if len(configs) != 1:
            raise RuntimeError(f'eval kernel has multiple autotune configurations: {len(configs)}')
        metadata = tuner.inductor_meta
        if not metadata.get('are_deterministic_algorithms_enabled'):
            raise RuntimeError('eval kernel was compiled without the deterministic configuration policy')
        if metadata.get('coordinate_descent_tuning'):
            raise RuntimeError('eval kernel enables timed coordinate-descent configuration selection')
        selected = configs[0]
        if getattr(selected, 'found_by_coordesc', False):
            raise RuntimeError('eval kernel configuration was selected by timed coordinate descent')
        row = dict(kernel=metadata.get('kernel_name'), size_hints=tuner.size_hints,
                   heuristic=str(tuner.heuristic_type), config=dict(kwargs=dict(selected.kwargs),
                   num_warps=selected.num_warps, num_stages=selected.num_stages,
                   maxnreg=getattr(selected, 'maxnreg', None)))
        run_manifest.observe(f'eval_kernel_configs_rank_{rank}', json.dumps(row, sort_keys=True), run_config=run_config)
        tuner._jetspec_compiler_checked = True
        return original(tuner, *args, **kwargs)

    fixed._jetspec_compiler_pin = True
    heuristics.CachingAutotuner.run = fixed
    print('EVAL_DETERMINISM_POLICY ' + json.dumps(policy, sort_keys=True), flush=True)
    return True
