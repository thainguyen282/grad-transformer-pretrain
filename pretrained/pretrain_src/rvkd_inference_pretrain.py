import os
import sys
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import argparse
import random
import torch
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import logging
import evaluate
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from utils.transformer import Embedding2EmbeddingT5
from utils.convert_gradients import split_tensor_to_lora, parse_key, split_tensor_to_lora_xs
from tqdm import tqdm
from loraxs_utils.initialization_utils import find_and_initialize
from utils.reinit_target_model import build_reinitalized_model
from utils.evaluate_model import evaluate

def apply_gradients(peft_model, lora_gradients, device="cuda"):
    for name, param in peft_model.named_parameters():
        if param.requires_grad and name in lora_gradients:
            grad = lora_gradients[name].to(device)
            if 'layers.27.self_attn.q_proj.lora_A' in name:
                if len(lora_gradients.keys()) == 144:
                    print("Applied to small")
                else:
                    print("Applied to large")
                print(grad)
            param.data = param.data + grad.to(param.device).to(param.dtype)
    return peft_model

def apply_gradients_xs(peft_model, lora_gradients_xs, device="cuda"):
    all_param_names = set(name for name, _ in peft_model.named_parameters())
    for name, param in peft_model.named_parameters():
        if param.requires_grad and name in lora_gradients_xs:
            # name will be like 'base_model.model.model.layers.0.self_attn.q_proj.lora_latent_mapping'
            module_prefix = name.rsplit('.', 1)[0]
            has_lora_A = any(n.startswith(module_prefix) and 'lora_A' in n for n in all_param_names)
            has_lora_B = any(n.startswith(module_prefix) and 'lora_B' in n for n in all_param_names)
            if has_lora_A and has_lora_B:
                print(f"[xs] {name}: block also has lora_A and lora_B")
            grad = lora_gradients_xs[name].to(device)
            # computation of grad if required
            # grad = grad @ lora_A @ lora_B
            param.data = grad.to(param.dtype)
    return peft_model

def run_rvkd_inference_pretrain(args):
    args_dict = vars(args)
    for k, v in args_dict.items():
        print(f"  {k}: {v}")
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    accelerator = Accelerator(
        log_with="wandb", 
        mixed_precision="bf16",
    )
    device = accelerator.device
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)

    accelerator.init_trackers("evaluate-rvkd-pretrain", config=args_dict, init_kwargs={"wandb": {"name": "rvkd_inference_pretrain"}})
    if args.quantize:
        raise NotImplementedError("Quantization is currently not supported for pretraining")
    else:
        large_model = build_reinitalized_model(args)
    
    large_lora_config = LoraConfig(
        r=args.loraxs_rank,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    set_seed(seed)
    large_model = get_peft_model(large_model, large_lora_config)
    adapter_name = "default"
    peft_config_dict = {adapter_name: large_lora_config}

    reconstr_config_path = os.path.join(project_root, "configs", "reconstruct_config.yaml")
    with open(reconstr_config_path, 'r') as f:
        reconstr_config = yaml.load(f, Loader=yaml.FullLoader)
    reconstr_config['svd']['rank'] = args.loraxs_rank

    find_and_initialize(large_model, peft_config_dict,
        adapter_name=adapter_name, reconstr_type='svd', reconstruct_config=reconstr_config)
    large_model.to(device)

    print("The following large weights are tuned:")
    for name, param in large_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
            print(name, param[0])
            break
    
    transform_model = Embedding2EmbeddingT5(input_dim=args.dim, output_dim=args.dim, base_model=args.base_model_path)
    transform_model.load_state_dict(torch.load(args.transform_model_path, weights_only=True))
    transform_model.to(device)
    transform_model.to(torch.float32)
    
    small_lora_gradients = torch.load(args.small_model_gradient_path, weights_only=True).to(device)
    small_lora_gradients = torch.unsqueeze(small_lora_gradients, 0)

    large_gradients = transform_model(small_lora_gradients, use_teacher_forcing=False, L_out=args.l_out)
    large_gradients = torch.squeeze(large_gradients, dim=0)
    large_lora_keys = [name for name, param in large_model.named_parameters() if param.requires_grad]
    predicted_large_lora_gradients = split_tensor_to_lora_xs(tensor=large_gradients, keys=large_lora_keys, model_name=args.large_model_path, lora_rank=args.loraxs_rank)
    del small_lora_gradients
    del transform_model


    with torch.no_grad():
        large_model.eval()
        predicted_large_model = apply_gradients_xs(large_model, predicted_large_lora_gradients)
        result = evaluate(predicted_large_model, args)
        print(f"Predicted large model: {result} on {args.dataset_path}")   
        torch.save(predicted_large_model.state_dict(), args.save_path)
