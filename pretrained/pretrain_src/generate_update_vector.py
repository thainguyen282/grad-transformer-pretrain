import math
import os
import sys
import copy
import json
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import yaml
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import rich

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.gaussian_noise_injection import add_gaussian_noise_to_tensor_list
from loraxs_utils.initialization_utils import find_and_initialize
from utils.convert_gradients import parse_key, group_by_layer_and_merge, flatten_and_merge

def log_rich(message: str, label: str | None = None, newline: bool = False):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "\n" if newline else ""
    if label:
        rich.print(
            f"{prefix}[bold cyan]{ts}[/bold cyan] "
            f"[bold magenta][{label}][/bold magenta] "
            f"[green]{message}[/green]"
        )
    else:
        rich.print(f"{prefix}[bold cyan]{ts}[/bold cyan] [green]{message}[/green]")

def init_weights_kaiming(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def verify_models_are_distinct(base: torch.Tensor, noisy_tensors: list, label: str):
    log_rich("Verifying weight differences across noisy variants ...", label=label, newline=True)
    all_passed = True
    for i, noisy in enumerate(noisy_tensors):
        for j in range(i + 1, len(noisy_tensors)):
            diff = (noisy - noisy_tensors[j]).abs().max().item()
            if diff == 0.0:
                all_passed = False
    if all_passed:
        log_rich(
            f"All {len(noisy_tensors)} noisy variants are distinct from the base and from each other. ✓",
            label=label,
        )
    else:
        raise ValueError(
            f"[{label}] One or more noisy variants are identical to the base or to another variant. "
        )

def get_update_vector(dw_list, merge_option, model_name, print_layers, inv_matrices, verbose=False):
    delta_lora_grads = {}
    for (key, lora_a_inv, lora_b_inv), delta_w in zip(inv_matrices, dw_list):
        dev = lora_b_inv.device
        delta_w = delta_w.to(device=dev, dtype=lora_b_inv.dtype)
        matrix_r = (lora_b_inv @ delta_w) @ lora_a_inv
        if verbose:
            log_rich(f"{key}: {matrix_r.shape}")
        grad_key = f"{key}.default_lora_latent_mapping.weight"
        delta_lora_grads[grad_key] = matrix_r.cpu()
        del matrix_r
        del delta_w
        torch.cuda.empty_cache()
    if merge_option == 'by_layer':
        sorted_items = sorted(delta_lora_grads.items(), key=lambda x: parse_key(x[0]))
        return group_by_layer_and_merge(
            sorted_items, model_name, merge_option, print_layers=print_layers if verbose else False,
        )
    elif merge_option == 'flatten':
        return flatten_and_merge(delta_lora_grads)
    else:
        raise ValueError(f"Unsupported merge_option: {merge_option}")

def build_and_save_delta(
    model_path: str,
    save_dir: str,
    lora_config: LoraConfig,
    peft_config_dict: dict,
    reconstr_config: dict,
    adapter_name: str,
    merge_option: str,
    model_name: str,
    print_layers: bool,
    num_noisy_samples: int,
    delta_noise_std: float,
    seed: int,
    label: str,
    noise_boundary: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_rich(f"Loading model from [yellow]{model_path}[/yellow] ...", label=label, newline=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map='auto',
    #     torch_dtype=torch.float16,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
    )
    pretrained_state = {
        k: v.bfloat16()
        for k, v in model.state_dict().items()
        if any(f".{t}." in k for t in lora_config.target_modules)
    }
    del model
    torch.cuda.empty_cache()
    config = AutoConfig.from_pretrained(model_path)
    log_rich("Building random-init model on CPU ...", label=label)
    with init_empty_weights():  
        random_init_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    random_init_model = random_init_model.to_empty(device="cpu")
    random_init_model.apply(init_weights_kaiming)
    log_rich("Random-init model ready.", label=label)
    delta_weights = []
    for key, module in random_init_model.named_modules():
        if not any(key.endswith(t) for t in lora_config.target_modules):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        weight_name = f"{key}.weight"
        if weight_name not in pretrained_state:
            continue
        p = pretrained_state[weight_name]
        r = module.weight.detach()
        p = p.to(device=device, dtype=torch.bfloat16)
        r = r.to(device=device, dtype=torch.bfloat16)
        delta = p - r
        delta_weights.append(delta.cpu().to(torch.bfloat16))
        del p, r
        torch.cuda.empty_cache()
    del pretrained_state
    # initialize lora 
    random_init_model.to(device)
    random_init_model = get_peft_model(random_init_model, lora_config)
    find_and_initialize(
        random_init_model, peft_config_dict, adapter_name=adapter_name,
        reconstr_type='svd', reconstruct_config=reconstr_config,
    )
    inv_loraxs_weights = [
        (key, module)
        for key, module in random_init_model.named_modules()
        if any(key.endswith(t) for t in lora_config.target_modules) and hasattr(module, 'lora_Ainv') and hasattr(module, 'lora_Binv')
    ]
    inv_matrices = [
        (key, target.lora_Ainv.detach().bfloat16().cpu(), target.lora_Binv.detach().bfloat16().cpu())
        for key, target in inv_loraxs_weights
    ]
    
    del random_init_model
    torch.cuda.empty_cache()
    # Then move to GPU once the model is freed
    inv_matrices = [
        (key, a.to(device), b.to(device))
        for key, a, b in inv_matrices
    ]


    os.makedirs(save_dir, exist_ok=True)

    update_vector = get_update_vector(delta_weights, merge_option, model_name, print_layers, inv_matrices, verbose=True)
    base_path = os.path.join(save_dir, "gradients_base.pt")
    torch.save(update_vector, base_path)
    log_rich(f"Saved base delta tensor -> [yellow]{base_path}[/yellow]", label=label)
    
    log_rich("Starting to generate noisy variants...")
    num_layers = len(delta_weights)
    for i in range(num_noisy_samples):
        noisy_delta_weights = add_gaussian_noise_to_tensor_list(
            delta_weights, delta_noise_std, noise_boundary, base_seed=seed + i * num_layers, device=device,
        )
        noisy_update_vector = get_update_vector(noisy_delta_weights, merge_option, model_name, print_layers, inv_matrices, verbose=False)
        out_path = os.path.join(save_dir, f"model_{i+1}.pt")
        torch.save(noisy_update_vector, out_path)
        log_rich(f"Saved noisy variant {i+1} -> [yellow]{out_path}[/yellow]", label=label)
        del noisy_delta_weights, noisy_update_vector
        torch.cuda.empty_cache()
    log_rich(f"Saved {num_noisy_samples} noisy variants in [yellow]{save_dir}[/yellow]", label=label)
    del delta_weights
    del inv_matrices
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return update_vector

def run_generate_update_vector(args): 
    model_config_path = os.path.join(project_root, args.model_config)
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    models = model_config["models"]
    log_rich(f"Loaded {len(models)} model(s) from [yellow]{model_config_path}[/yellow]", newline=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    lora_config = LoraConfig(
        r=args.loraxs_rank,
        lora_alpha=256, # 2048
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            'lm_head',
        ],
        bias="none",    
        task_type=TaskType.CAUSAL_LM,
    )
    adapter_name = "default"
    peft_config_dict = {adapter_name: lora_config}

    reconstr_config_path = os.path.join(project_root, "configs", "reconstruct_config.yaml")
    with open(reconstr_config_path, 'r') as f:
        reconstr_config = yaml.load(f, Loader=yaml.FullLoader)
    reconstr_config['svd']['rank'] = args.loraxs_rank

    shared_kwargs = dict(
        lora_config=lora_config,
        peft_config_dict=peft_config_dict,
        reconstr_config=reconstr_config,
        adapter_name=adapter_name,
        merge_option=args.merge_option,
        model_name=args.model_name,
        print_layers=args.print_layers,
        num_noisy_samples=args.num_noisy_samples,
        delta_noise_std=args.delta_noise_std,
        seed=args.seed,
    )

    for entry in models:
        model_path = entry["path"]
        noise_boundary = entry["noise_boundary"]
        label = os.path.basename(model_path)
        log_rich(f"{'='*60}", newline=True)
        log_rich(f"Processing model: [yellow]{model_path}[/yellow] (noise_boundary={noise_boundary})")
        log_rich(f"{'='*60}")
        build_and_save_delta(
            model_path=model_path,
            save_dir=os.path.join("saves", model_path),
            label=label,
            noise_boundary=noise_boundary,
            **shared_kwargs,
        )
        torch.cuda.empty_cache()

    log_rich("Done. All tensors saved to disk.", newline=True)
