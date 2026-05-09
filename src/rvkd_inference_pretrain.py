import os
import sys
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from datetime import datetime 

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from utils.transformer import Embedding2EmbeddingT5
from utils.convert_gradients import split_tensor_to_lora_xs
from lora_utils.initialization_utils import find_and_initialize
from utils.reinit_target_model import DEFAULT_LORA_TARGET_MODULES, build_reinitalized_model
from utils.evaluate_model import evaluate_pretrain_model, build_pretrain_val_dataloader
from utils.logging import log_build_start, log_rich
from rich.console import Console

from peft.tuners.lora import Linear as LoraLinear
from tqdm import tqdm
import numpy as np

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
            module_prefix = name.rsplit('.', 2)[0]
            has_lora_A = any(n.startswith(module_prefix) and 'lora_A' in n for n in all_param_names)
            has_lora_B = any(n.startswith(module_prefix) and 'lora_B' in n for n in all_param_names)
            if has_lora_A and has_lora_B:
                print(f"[xs] {name}: block also has lora_A and lora_B")
            weight = lora_gradients_xs[name].to(device)
            # computation of grad if required
            # grad = grad @ lora_A @ lora_B
            param.data = weight.to(param.dtype)
    return peft_model

def run_rvkd_inference_pretrain(args, console: Console):
    args_dict = vars(args)
    accelerator = Accelerator(
        log_with="wandb", 
        mixed_precision="bf16",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)

    run_name = f"rvkd_inference_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    accelerator.init_trackers(
        "evaluate-rvkd-pretrain",
        config=args_dict,
        init_kwargs={"wandb": {"name": run_name}}
    )

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
    del transform_model
    del small_lora_gradients
    torch.cuda.empty_cache()

    if args.quantize:
        raise NotImplementedError("Quantization is currently not supported for pretraining")
    else:
        large_model = build_reinitalized_model(args)
        large_model = large_model.cpu()
        # large_model = AutoModelForCausalLM.from_pretrained(
        #     args.pretrain_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        # )
    large_model.config.tie_word_embeddings = False
    large_model.tie_weights = False

    
    lora_targets = (
        list(args.lora_target_modules)
        if getattr(args, "lora_target_modules", None)
        else list(DEFAULT_LORA_TARGET_MODULES)
    )
    large_lora_config = LoraConfig(
        r=args.loraxs_rank,
        lora_alpha=args.loraxs_rank,
        target_modules=lora_targets,
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
    
    # for name, param in large_model.named_parameters():
    #     if 'base_layer.weight' in name and any(t in name for t in lora_targets):
    #         param.data.zero_()
    #         print(f"Clear {name}")
    payload = torch.load(args.loraxs_model_path, map_location="cpu")
    lora_sd = payload["state_dict"]
    dtype = next(large_model.parameters()).dtype
    dev = next(large_model.parameters()).device
    lora_sd = {k: v.to(device=dev, dtype=dtype) for k, v in lora_sd.items()}
    missing, unexpected = large_model.load_state_dict(lora_sd, strict=False)
    del payload, lora_sd

    # if missing:
    #     print("Missing keys:")
    #     for k in missing:
    #         print(f"  {k}")

    original_weight = {}
    
    large_lora_keys = [name for name, param in large_model.named_parameters() if param.requires_grad]
    predicted_large_lora_gradients = split_tensor_to_lora_xs(tensor=large_gradients, keys=large_lora_keys, model_name=args.large_model_path, lora_rank=args.loraxs_rank)

    with torch.no_grad():
        large_model.eval()
        predicted_large_model = apply_gradients_xs(large_model, predicted_large_lora_gradients, device="cpu")
        del predicted_large_lora_gradients
        predicted_large_model = predicted_large_model.to(device=device, dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        tok = AutoTokenizer.from_pretrained(args.pretrain_model_path, padding_side="left")

        _p = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
            "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            "Hello how are you today? Answer briefly."
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
        log_build_start(description="=== single-prompt preview ===", console=console)
        print(tok.decode(gen[0, plen:].cpu(), skip_special_tokens=True))
        log_rich("=== end preview ===", console=console)

        log_build_start(description="Evaluating ...", console=console)
        val_dataloader = build_pretrain_val_dataloader(args, tok, for_perplexity=True)
        predicted_large_model.eval()
        eval_losses = []
        for eval_batch in tqdm(val_dataloader, desc="Perplexity eval"):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            outputs = predicted_large_model(**eval_batch)
            eval_losses.append(outputs.loss.item())

        avg_eval_loss = float(np.mean(eval_losses))
        perplexity = float(np.exp(avg_eval_loss))
        print(f"Eval loss (NLL): {avg_eval_loss:.6f}")
        print(f"Perplexity: {perplexity:.3f}")

        if accelerator.is_main_process:
            accelerator.log(
                {"eval_loss": avg_eval_loss, "perplexity": perplexity},
                step=0,
            )

