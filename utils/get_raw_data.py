import torch.nn as nn
from tqdm import tqdm
from utils.weight_initialize import init_weights_kaiming

def get_raw_data(model: nn.Module, target_modules: list):
    update_vector = []
    meta_information = []
    for k, v in tqdm(model.state_dict().items()):
        if not any(k.endswith(f"{t}.weight") for t in target_modules):
            continue

        parts = k.split(".")
        layer_indices = [i for i, p in enumerate(parts) if p.isdigit()]
        layer_indice = int(parts[layer_indices[0]]) if layer_indices else None
        if layer_indice is None:
            raise ValueError(f"cannot detect layer_id for key: {k}")

        target_module = next((t for t in target_modules if k.endswith(f"{t}.weight")), None)
        if target_module is None:
            raise ValueError(f"cannot detect target_module for key: {k}")

        rand_model = init_weights_kaiming(v)

        update_vector.append(v - rand_model)
        meta_information.append({
            "layer_id": layer_indice,
            "target_module": target_module,
            "in_features_size": v.size(1),
            "out_features_size": v.size(0)
        })

    return update_vector, meta_information