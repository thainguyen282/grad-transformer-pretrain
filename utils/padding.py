import torch
import torch.nn.functional as F

def extract_proj_name(weight_key: str) -> str:
    return weight_key.split(".")[-2]

def pad_tensor(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    pad_dims = []
    for current, target in zip(reversed(t.shape), reversed(target_shape)):  
        pad_dims += [0, target - current]
    return F.pad(t, pad_dims, mode="constant", value=0)

def add_padding(
    update_vector: list[tuple[str, tuple[torch.Tensor, torch.Tensor]]], 
    hidden_state: int,
    rank: int
) -> list[tuple[str, torch.Tensor]]:
    padded = []

    for k, (lora_A, lora_B) in update_vector:
        proj = extract_proj_name(k)
        max_in, max_out = (3584, 3584)
        target_A = (max_in, rank)
        target_B = (rank, max_out)

        base = k.removesuffix(".weight")
        lora_A_path = f"{base}.lora_A.weight"
        lora_B_path = f"{base}.lora_B.weight"
        padded.append((lora_A_path, pad_tensor(lora_A, target_A)))
        padded.append((lora_B_path, pad_tensor(lora_B, target_B)))
    return padded


# def add_padding(
#     update_vector: list[tuple[str, tuple[torch.Tensor, torch.Tensor]]], 
#     target_shapes: dict[str, tuple[int, int]],
#     rank: int
# ) -> list[tuple[str, torch.Tensor]]:
#     padded = []

#     for k, (lora_A, lora_B) in update_vector:
#         proj = extract_proj_name(k)
#         if proj not in target_shapes:
#             print(f"{proj} in update vector but cnot include in config file")
#             continue
#         max_in, max_out = target_shapes[proj]
#         target_A = (max_in, rank)
#         target_B = (rank, max_out)

#         base = k.removesuffix(".weight")
#         lora_A_path = f"{base}.lora_A.weight"
#         lora_B_path = f"{base}.lora_B.weight"
#         padded.append((lora_A_path, pad_tensor(lora_A, target_A)))
#         padded.append((lora_B_path, pad_tensor(lora_B, target_B)))
#     return padded