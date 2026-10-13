"""Optional logical-rank to local-CUDA-device placement."""
import os


def cuda_device_for_rank(rank):
    """Resolve a logical distributed rank without changing the default mapping."""
    value = os.environ.get("SSD_RANK_DEVICE_MAP")
    if value is None:
        return int(rank)
    try:
        devices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("SSD_RANK_DEVICE_MAP must contain CUDA device indices") from error
    if not devices or any(device < 0 for device in devices) or int(rank) >= len(devices):
        raise ValueError("SSD_RANK_DEVICE_MAP must cover every logical rank")
    return devices[int(rank)]


def gpu_memory_utilization(default):
    """Cap each rank's KV reservation when ranks intentionally share a GPU."""
    value = os.environ.get("SSD_SHARED_GPU_KV_UTILIZATION")
    if value is None:
        return float(default)
    try:
        utilization = float(value)
    except ValueError as error:
        raise ValueError("SSD_SHARED_GPU_KV_UTILIZATION must be a fraction") from error
    if not 0 < utilization <= 1:
        raise ValueError("SSD_SHARED_GPU_KV_UTILIZATION must be within (0, 1]")
    return min(float(default), utilization)
