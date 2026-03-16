import argparse
import math
import torch
from torch.nn import init
from peft.utils import _get_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]


def reinit_pretrained_weights(model: torch.nn.Module, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)

    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        if not any(key.endswith(target) for target in TARGET_MODULES):
            continue

        _, target, _ = _get_submodules(model, key)

        if hasattr(target, 'weight') and target.weight is not None:
            init.kaiming_uniform_(target.weight, a=math.sqrt(5))
            print(f"Reinit base weight: {key}  shape={tuple(target.weight.shape)}")

        if hasattr(target, 'bias') and target.bias is not None:
            fan_in = target.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(target.bias, -bound, bound)
            print(f"Reinit bias       : {key}")

    return model


def save_reinitialized_model(model: torch.nn.Module, output_dir: str, tokenizer=None):

    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinitialize pretrained weights and save model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the reinitialized model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map="auto")

    reinit_pretrained_weights(model, seed=args.seed)
    save_reinitialized_model(model, args.output_dir, tokenizer=tokenizer)
