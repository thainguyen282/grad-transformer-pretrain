import torch.nn as nn
import torch

# def init_weights_kaiming(module):
#     if isinstance(module, nn.Linear):
#         nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         nn.init.normal_(module.weight, mean=0.0, std=0.02)
#         if module.padding_idx is not None:
#             nn.init.zeros_(module.weight[module.padding_idx])

#     elif isinstance(module, nn.LayerNorm):
#         nn.init.ones_(module.weight)
#         nn.init.zeros_(module.bias)

def init_weights_kaiming(tensor):
    t = tensor.clone()
    if t.dim() >= 2:
        nn.init.kaiming_normal_(t, mode="fan_in", nonlinearity="linear")
    else:
        nn.init.zeros_(t)
    return t

def init_lora_kaiming(tensor, rank):
    in_dim, out_dim = tensor.shape

    std = (1.0 / (in_dim * rank)) ** 0.25

    A = torch.randn(in_dim, rank) * std
    B = torch.randn(rank, out_dim) * std

    return A, B
