import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)






def load_safetensors_model(model: nn.Module, path: str, packed_modules_mapping: dict):
    """Load model weights from safetensors files"""
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    for file in tqdm(safetensor_files, desc="Loading model files"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_model(model: nn.Module, path: str, target_path: str = None, target_hidden_size: int = None):
    print(f"[load_model] loading model from {path}")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Check if this is an EAGLE3 draft model
    is_eagle = 'eagle' in path.lower()
    
    if is_eagle:
        load_eagle_model(model, path, packed_modules_mapping, target_path=target_path, target_hidden_size=target_hidden_size)
    else:
        load_safetensors_model(model, path, packed_modules_mapping)

    print(f"[load_model] finished loading model from {path}")
