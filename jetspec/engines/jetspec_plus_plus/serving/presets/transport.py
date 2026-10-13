"""Rank placement shared by all engine presets and historical recipes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Lockstep:
    placement = 'one_gpu'
    devices = 1
    waits_for_candidate = True


@dataclass(frozen=True)
class Pair:
    placement = 'pair'
    devices = 2
    waits_for_candidate = False


@dataclass(frozen=True)
class MPS:
    placement = 'one_gpu_mps'
    devices = 1
    waits_for_candidate = True


def resolve_placement(placement, devices, *, shared_kv_cap=None):
    selected = [value.strip() for value in devices.split(',') if value.strip()]
    if len(selected) != len(set(selected)):
        raise ValueError('devices must be distinct')
    if placement in ('pair', 'tp2'):
        if len(selected) != 2:
            raise ValueError(f'{placement} requires two devices')
        environment = {'CUDA_VISIBLE_DEVICES': ','.join(selected),
                       'SSD_RANK_DEVICE_MAP': '0,1'}
    else:
        if len(selected) != 1:
            raise ValueError(f'{placement} requires one device')
        environment = {'CUDA_VISIBLE_DEVICES': selected[0], 'SSD_RANK_DEVICE_MAP': '0,0',
                       'SSD_SHARED_GPU_DIST': 'gloo'}
        if shared_kv_cap is not None:
            environment['SSD_SHARED_GPU_KV_UTILIZATION'] = shared_kv_cap
    return dict(placement=placement, visible_devices=selected, environment=environment)
