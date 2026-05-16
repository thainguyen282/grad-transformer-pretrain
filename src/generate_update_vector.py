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

from utils.gaussian_noise_injection import add_gaussian_noise_to_tensor, add_gaussian_noise_to_model_dict
from lora_utils.initialization_utils import find_and_initialize
from utils.get_raw_data import get_raw_data
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

    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    models = model_config["models"]
    log_rich(
        f"Loaded [yellow]{len(models)}[/yellow] model(s) from [yellow]{args.model_config}[/yellow]",
        console,
        newline=True,
    )

    for entry in models:
        model_path = entry["path"]
        noise_boundary = entry["noise_boundary"]
        label = os.path.basename(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )

        ### extract target module and meta information

        update_vector, meta_information = get_raw_data(model, target_modules)
        dir = os.path.join(args.save_dir, model_path)
        os.makedirs(dir, exist_ok=True)
        torch.save(meta_information, os.path.join(dir, "meta_information.pt"))
        torch.save(update_vector, os.path.join(dir, "gradient_base.pt"))

        for i in tqdm(range(args.num_noisy_samples), desc="Sampling Data"):
            sample_update_vector = add_gaussian_noise_to_model_dict(
                model=update_vector,
                std=args.std,
                noise_boundary=noise_boundary,
                base_seed=args.seed + i,   # shift seed per sample for diversity
                device=device,
            )

            torch.save(sample_update_vector, os.path.join(dir, f"sample_{i}.pt"))
            tqdm.write(f"Done generate sample {i}")
   



        ### Group and Padding 

        # for i in range(args.num_noisy_samples):
        #     noise_model = add_gaussian_noise_to_dict(update_vector)
            
        # padded_update_vector = add_padding(update_vector, target_size=3584)


    #     if args.merge_option == 'by_layer':
    #         merged_tensor = group_by_layer_and_merge(
    #             padded_update_vector, 
    #             args.model_name, 
    #             args.merge_option,
    #         )
    #         print(merged_tensor.shape)
    #     elif args.merge_option == 'flatten':
    #         merged_tensor = flatten_and_merge(padded_update_vector)
        
    #     save_path = os.path.join(args.save_dir, f"gradients/{label}/gradient_base.pt")
    #     print(f"save base gradient to {save_path}")
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     torch.save(merged_tensor, save_path)

    #     del model, padded_update_vector, merged_tensor
    #     torch.cuda.empty_cache()

    # log_rich("Done. All tensors saved to disk.", console, newline=True)
