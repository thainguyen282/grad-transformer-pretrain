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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from utils.transformer import Embedding2EmbeddingT5
from utils.convert_gradients import split_tensor_to_lora, parse_key, split_tensor_to_lora_xs
from tqdm import tqdm
from lora_utils.initialization_utils import find_and_initialize
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
        # large_model = build_reinitalized_model(args)
        large_model = AutoModelForCausalLM.from_pretrained(
            args.pretrain_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    large_model.config.tie_word_embeddings = False
    large_model.tie_weights = False

    
    large_lora_config = LoraConfig(
        r=args.loraxs_rank,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    large_model = get_peft_model(large_model, large_lora_config)
    adapter_name = "default"
    peft_config_dict = {adapter_name: large_lora_config}

    reconstr_config_path = os.path.join(project_root, "configs", "reconstruct_config.yaml")
    with open(reconstr_config_path, 'r') as f:
        reconstr_config = yaml.load(f, Loader=yaml.FullLoader)
    reconstr_config['svd']['rank'] = args.loraxs_rank

    find_and_initialize(large_model, peft_config_dict,
        adapter_name=adapter_name, reconstr_type='svd', reconstruct_config=reconstr_config)
    
    payload = torch.load(args.loraxs_model_path, map_location="cpu")
    lora_sd = payload["state_dict"]
    dtype = next(large_model.parameters()).dtype
    dev = next(large_model.parameters()).device
    lora_sd = {k: v.to(device=dev, dtype=dtype) for k, v in lora_sd.items()}
    missing, unexpected = large_model.load_state_dict(lora_sd, strict=False)

    if missing:
        print("Missing keys:")
        for k in missing:
            print(f"  {k}")

    if unexpected:
        print("Unexpected keys:")
        for k in unexpected:
            print(f"  {k}")
    
    transform_model = Embedding2EmbeddingT5(input_dim=args.dim, output_dim=args.dim, base_model=args.base_model_path)
    transform_model.load_state_dict(torch.load(args.transform_model_path, weights_only=True))
    transform_model.to(device)
    transform_model.to(torch.bfloat16)

    small_lora_gradients = torch.load(args.small_model_gradient_path, weights_only=True).to(
        device=device, dtype=torch.bfloat16
    )
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
        tok = AutoTokenizer.from_pretrained(args.pretrain_model_path, padding_side="left")
        _p = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
            "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            "What is 2+2? Answer briefly."
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        enc = tok(_p, return_tensors="pt")
        mdev = next(predicted_large_model.parameters()).device
        input_ids = enc["input_ids"].to(mdev)
        attn = enc["attention_mask"].to(mdev)
        plen = input_ids.shape[1]
        gen = predicted_large_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=min(args.max_new_tokens, 256),
            bos_token_id=tok.bos_token_id,
            pad_token_id=tok.pad_token_id,
            eos_token_id=[tok.eos_token_id, tok.pad_token_id],
            use_cache=True,
        )
        print("=== single-prompt preview ===")
        print(tok.decode(gen[0, plen:].cpu(), skip_special_tokens=True))
        print("=== end preview ===")

        # result = evaluate(predicted_large_model, args)
        # print(f"Predicted large model: {result} on {args.dataset_path}")
        torch.save(predicted_large_model.state_dict(), os.join(args.save_dir, "new_model", args.pretrain_model_path))

