"""Load the text-only portion of a Qwen3.5 checkpoint."""

from glob import glob
from pathlib import Path

from safetensors import safe_open
from tqdm import tqdm

from jetspec.engines.jetspec_plus_plus.models.loader import default_weight_loader


def load_hybrid_text(model, path: str):
    loaded = set()
    files = sorted(glob(str(Path(path) / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors shards in {path}")
    for file in tqdm(files, desc="Loading hybrid text files"):
        with safe_open(file, framework="pt", device="cpu") as weights:
            for source_name in weights.keys():
                if source_name.startswith("model.language_model."):
                    target_name = "model." + source_name.removeprefix("model.language_model.")
                elif source_name == "lm_head.weight":
                    target_name = source_name
                else:
                    continue  # vision tower and native MTP are not served
                parameter = model.get_parameter(target_name)
                weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
                weight_loader(parameter, weights.get_tensor(source_name))
                loaded.add(target_name)
    missing = set(dict(model.named_parameters())) - loaded
    if missing:
        raise RuntimeError(f"hybrid text checkpoint is missing parameters: {sorted(missing)}")
