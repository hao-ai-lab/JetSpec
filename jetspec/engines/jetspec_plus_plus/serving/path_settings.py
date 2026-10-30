"""Resolve serving paths from the explicit launch snapshot."""
import os
from types import SimpleNamespace

from jetspec.engines.jetspec_plus_plus.serving.settings import setting

# Unresolved dataclass defaults; Config resolves them from its received record.
DEFAULT_TARGET = None
DEFAULT_DRAFT = None


def _required_env(run_config, path: str, note: str) -> str:
    from jetspec.engines.jetspec_plus_plus.launch_env import environment_name
    value = setting(run_config, path)
    if value:
        return value
    category, field = path.split('.', 1)
    raise RuntimeError(f"Missing required env var {environment_name(category, field)}. {note}")


def install_architecture(run_config):
    """Publish the launch architecture before the first kernel compilation."""
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST',
                          setting(run_config, 'hardware.hardware_settings.cuda_arch', '9.0'))


def resolve_paths(run_config):
    cache = _required_env(run_config, 'hardware.hardware_settings.hf_cache',
                          'Set it to your HuggingFace cache hub directory (for example: /path/to/huggingface/hub).')
    dataset = _required_env(run_config, 'hardware.hardware_settings.dataset_dir',
                            'Set it to your processed dataset directory (for example: /path/to/processed_datasets).')
    return SimpleNamespace(
        HF_CACHE_DIR=cache, DATASET_DIR=dataset,
        DEFAULT_TARGET=setting(run_config, 'engine.engine_settings.target_model',
            f'{cache}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'),
        DEFAULT_DRAFT=setting(run_config, 'engine.engine_settings.draft_model',
            f'{cache}/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'),
        EAGLE3_SPECFORGE_70B=setting(run_config, 'model.model_settings.eagle3_specforge_70b',
            f'{cache}/models--lmsys--SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge'),
        EAGLE3_YUHUILI_8B=setting(run_config, 'model.model_settings.eagle3_8b',
            f'{cache}/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B'),
        EAGLE3_QWEN_32B=setting(run_config, 'model.model_settings.eagle3_qwen_32b',
            f'{cache}/models--RedHatAI--Qwen3-32B-speculator.eagle3'),
        DATASET_PATHS={
            'humaneval': f'{dataset}/humaneval/humaneval_data_10000.jsonl',
            'alpaca': f'{dataset}/alpaca/alpaca_data_10000.jsonl',
            'c4': f'{dataset}/c4/c4_data_10000.jsonl',
            'gsm': f'{dataset}/gsm8k/gsm8k_data_10000.jsonl',
            'ultrafeedback': f'{dataset}/ultrafeedback/ultrafeedback_data_10000.jsonl',
        },
    )
