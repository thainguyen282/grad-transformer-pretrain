import os
import sys
import copy
import json
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import yaml
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from rich.console import Console
from utils.logging import log_build_start, log_rich
from lora_utils.svd_utils import get_linear_rec_svd
# from utils.get_update_vector import lora_update_vector, loraxs_update_vector
from utils.convert_gradients import group_by_layer_and_merge, flatten_and_merge

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from utils.gaussian_noise_injection import add_gaussian_noise_to_tensor
from lora_utils.initialization_utils import find_and_initialize
# from utils.convert_gradients import (
#     parse_key,
#     group_by_layer_and_merge,
#     flatten_and_merge,
# )
from utils.weight_initialize import init_weights_kaiming, init_lora_kaiming
from utils.padding import add_padding
import gc

"""
This file goal to create the data structure of the method
Crawl the target module of each layer and its meta information
Create two data structure 
1. List of all target module projection layer 
2. List of all meta information respective to the list of above projection layer
"""

def compute_target_shapes(models: list[dict]) -> dict[str, tuple[int, int]]:
    proj_names = models[0]["proj_shapes"].keys()
    return {
        proj: (
            max(m["proj_shapes"][proj][0] for m in models),
            max(m["proj_shapes"][proj][1] for m in models),
        )
        for proj in proj_names
    }

def generate_update_vector(args, console: Console):
    target_modules=[ 
        "q_proj", "k_proj", "v_proj", "o_proj",
        # "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            # "lm_head"
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_empty_device = "cpu"


    model_config_path = os.path.join(project_root, args.model_config)
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    models = model_config["models"]
    log_rich(
        f"Loaded [yellow]{len(models)}[/yellow] model(s) from [yellow]{model_config_path}[/yellow]",
        console,
        newline=True,
    )


    unified_format_shape = compute_target_shapes(models)

    for entry in models:
        model_path = entry["path"]
        noise_boundary = entry["noise_boundary"]
        label = os.path.basename(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        ### extract target module and meta information

        update_vector = []
        for k, v in tqdm(model.state_dict().items()):
            if not any(k.endswith(f"{t}.weight") for t in target_modules):
                continue

            svd = get_linear_rec_svd(
                v,
                rank=args.lora_rank,
                n_iter=10,
                random_state=42
            )[1:3]

            kai = init_lora_kaiming(v, args.lora_rank)

            update_vector.append((k, (svd[0] - kai[0], svd[1] - kai[1])))

        ### Group and Padding 

        # for i in range(args.num_noisy_samples):
        #     noise_model = add_gaussian_noise_to_dict(update_vector)
            

        padded_update_vector = add_padding(update_vector, hidden_state=3584, rank=args.lora_rank)
        if args.merge_option == 'by_layer':
            merged_tensor = group_by_layer_and_merge(
                padded_update_vector, 
                args.model_name, 
                args.merge_option,
            )
            print(merged_tensor.shape)
        elif args.merge_option == 'flatten':
            merged_tensor = flatten_and_merge(padded_update_vector)
        
        save_path = os.path.join(args.save_dir, f"gradients/{label}/gradient_base.pt")
        print(f"save base gradient to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(merged_tensor, save_path)

        del model, padded_update_vector, merged_tensor
        torch.cuda.empty_cache()

    log_rich("Done. All tensors saved to disk.", console, newline=True)
