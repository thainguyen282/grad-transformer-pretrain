import math
import torch
from typing import List

def clipping_noise(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    l2_norm = torch.norm(tensor, p=2)
    print(f"L2 norm of noise: {l2_norm}")
    print(f"Threshold: {threshold}")
    scale = min(1.0, threshold / l2_norm)
    return tensor * scale

def add_gaussian_noise_to_tensor(tensor: torch.Tensor, std: float, noise_boundary: float, random_seed: int) -> torch.Tensor:
    torch.manual_seed(random_seed)
    noise = torch.normal(0.0, std, size=tensor.shape, device=tensor.device, dtype=tensor.dtype)
    noise  = clipping_noise(noise, noise_boundary)
    return tensor + noise

def add_gaussian_noise_to_tensor_list(
    tensors: List[torch.Tensor],
    std: float,
    noise_boundary: float,
    base_seed: int,
    device: torch.device,
) -> List[torch.Tensor]:
    noises = []
    for layer_idx, tensor in enumerate(tensors):
        torch.manual_seed(base_seed + layer_idx)
        t_dev = tensor.to(device)
        noise = torch.normal(0.0, std, size=t_dev.shape, device=device, dtype=t_dev.dtype)
        noises.append(noise.cpu())
        del t_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined_l2 = math.sqrt(sum(torch.norm(n, p=2).item() ** 2 for n in noises))
    print(f"Combined L2 norm of noise: {combined_l2}")
    print(f"Noise boundary: {noise_boundary}")
    scale = min(1.0, noise_boundary / combined_l2) if combined_l2 > 0 else 1.0



    return [tensor + noise * scale for tensor, noise in zip(tensors, noises)]