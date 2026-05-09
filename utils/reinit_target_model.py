import os
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import init
from peft.utils import _get_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer

# Matches ``rvkd_inference_pretrain`` default LoRA line (no ``lm_head``). Override via
# ``--lora_target_modules`` on the CLI or pass ``target_modules`` into the helpers.
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Backward-compatible alias
TARGET_MODULES = list(DEFAULT_LORA_TARGET_MODULES)


def _resolve_target_modules(
    args=None,
    target_modules: Optional[Sequence[str]] = None,
) -> List[str]:
    if target_modules is not None:
        return list(target_modules)
    if args is not None:
        tm = getattr(args, "lora_target_modules", None)
        if tm:
            return list(tm)
    return list(DEFAULT_LORA_TARGET_MODULES)


def reinit_pretrained_weights(
    model: nn.Module,
    seed: int = None,
    *,
    target_modules: Optional[Sequence[str]] = None,
    args=None,
):
    """
    Re-initialize only ``nn.Linear.weight`` for modules LoRA would attach to:
    same rule as PEFT (``module_name.endswith(target)`` for each target string).
    Biases and all other parameters stay at pretrained values.
    """
    mods = _resolve_target_modules(args=args, target_modules=target_modules)
    if seed is not None:
        torch.manual_seed(seed)

    for key in (k for k, _ in model.named_modules()):
        if not any(key.endswith(t) for t in mods):
            continue
        _, module, _ = _get_submodules(model, key)
        if not isinstance(module, nn.Linear) or module.weight is None:
            continue
        init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

    return model


def save_reinitialized_model(model: torch.nn.Module, output_dir: str, tokenizer=None):
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")


def build_reinitalized_model(args):
    output_dir = os.path.join(args.save_dir, "random_init_model", args.pretrain_model_path)
    if os.path.exists(output_dir):
        return AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="cpu")
    model = AutoModelForCausalLM.from_pretrained(args.pretrain_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    reinit_pretrained_weights(model, seed=args.seed)
    save_reinitialized_model(model, output_dir, tokenizer=tokenizer)
    return model
    