import math
import torch
from typing import Dict


def clipping_noise(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    l2_norm = torch.norm(tensor, p=2)
    print(f"L2 norm of noise: {l2_norm}")
    print(f"Threshold: {threshold}")
    scale = min(1.0, threshold / l2_norm)
    return tensor * scale

def add_gaussian_noise_to_tensor(
    tensors: Dict[str, torch.Tensor],
    std: float,
    noise_boundary: float,
    base_seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    noises = {}
    items = list(tensors.items())  # stable order for seed assignment
    for layer_idx, (name, tensor) in enumerate(items):
        torch.manual_seed(base_seed + layer_idx)
        t_dev = tensor.to(device)
        noise = torch.normal(0.0, std, size=t_dev.shape, device=device, dtype=t_dev.dtype)
        noises[name] = noise.cpu()
        del t_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined_l2 = math.sqrt(sum(torch.norm(n, p=2).item() ** 2 for n in noises.values()))
    scale = min(1.0, noise_boundary / combined_l2) if combined_l2 > 0 else 1.0
    return {name: tensors[name] + noises[name] * scale for name in tensors}

def add_gaussian_noise_to_model_dict(
    model: list[torch.Tensor], 
    std:float,
    noise_boundary:float,
    base_seed: int, 
    device: torch.device,
) -> list[tuple[str, tuple[torch.Tensor, torch.Tensor]]]:
    noises = []

    for i, weight in enumerate(model):
        torch.manual_seed(base_seed + i)
        weight_dev = weight.to(device)
        noise = torch.normal(0.0, std, size=weight_dev.shape, device=device, dtype=weight_dev.dtype)
        noises.append(noise.cpu())

        del weight_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined_l2 = math.sqrt(
        sum(torch.norm(noise, p=2).item() ** 2 for noise in noises)
    )
    scale = min(1.0, noise_boundary / combined_l2) if combined_l2 > 0 else 1.0
    return [
        (weight + noises[idx] * scale)
        for idx, weight in enumerate(model)
    ]