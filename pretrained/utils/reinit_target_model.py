import argparse
import math
import torch
from torch.nn import init
from peft.utils import _get_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def reinit_pretrained_weights(model: torch.nn.Module, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)

    key_list = [key for key, _ in moedel.named_modules()]
    for key in key_list:
        if not any(key.endswith(target) for target in TARGET_MODULES):
            continue

        _, target, _ = _get_submodules(model, key)

        if hasattr(target, 'weight') and target.weight is not None:
            init.kaiming_normal_(
                    target.weight,
                    mode='fan_in',
                    nonlinearity='relu')
            print(f"Reinit base weight: {key}  shape={tuple(target.weight.shape)}")

        if hasattr(target, 'bias') and target.bias is not None:
            init.zeros_(target.bias)
            print(f"Reinit bias       : {key}")

    return model


def save_reinitialized_model(model: torch.nn.Module, output_dir: str, tokenizer=None):

    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")

def build_reinitalized_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.pretrain_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    reinit_pretrained_weights(model, seed=args.seed)
    output_dir = os.path.join(args.save_dir, "random_init_model", args.pretrain_model_path)
    save_reinitialized_model(model, output_dir, tokenizer=tokenizer)
    return model
