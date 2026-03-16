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
from src.transformer import Embedding2EmbeddingT5
from convert_gradients import split_tensor_to_lora, parse_key, split_tensor_to_lora_xs
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from eval_math import *
from src.loraxs_utils.initialization_utils import find_and_initialize
from pretrained.config import RvkdInferencePretrainConfig

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

# def evaluate_model(tokenizer, model, val_dataloader, dataset_path, ans_template, ref_template, max_new_tokens):
#         generated_sequences = []
#         label_sequences = []
#         device = model.device
#         for eval_batch in tqdm(val_dataloader):
#             input_ids = eval_batch["input_ids"].to(device)
#             attention_mask = eval_batch["attention_mask"].to(device)
#             label_ids = eval_batch["labels"].to(device)

#             generated_tokens = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=max_new_tokens,
#                 bos_token_id=tokenizer.bos_token_id,
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
#                 use_cache=True,
#             )
            
#             generated_tokens = generated_tokens[:, input_ids.shape[1]:]
#             generated_sequences.extend(generated_tokens.cpu())
#             label_sequences.extend(label_ids.cpu())

#         predictions = pad_sequence(generated_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
#         labels = pad_sequence(label_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

#         result, records_table = compute_metrics((predictions, labels), dataset_path, ans_template, ref_template)
#         return result, records_table

def main(args):
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

    accelerator.init_trackers("evaluate-rvkd-pretrain", config=args_dict, init_kwargs={"wandb": {"name": args.save_fname}})
    rouge = evaluate.load("rouge", keep_in_memory=True)
    if args.quantize:
        raise NotImplementedError("Quantization is currently not supported for pretraining")
    else:
        large_model = AutoModelForCausalLM.from_pretrained(
            args.large_model_path,
            torch_dtype=torch.float32,
            device_map="auto",
        )
    
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
    
    transform_model = Embedding2EmbeddingT5(input_dim=args.input_dim, output_dim=args.output_dim, base_model=args.base_model_path)
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
        # result = evaluate_model(predicted_large_model, val_dataloader, args.dataset_path, args.ans_template, args.ref_template, args.max_new_tokens)
        # print(f"Predicted large model: {result}")   
        print(predicted_large_model)
        torch.save(predicted_large_model.state_dict(), args.save_path)

def run_rvkd_inference_pretrain_from_config(cfg: RvkdInferencePretrainConfig | None = None):
    if cfg is None:
        raise ValueError("cfg is required")

    args = argparse.Namespace(
        save_fname=cfg.save_fname,
        small_model_gradient_path=cfg.small_model_gradient_path,
        large_model_path=cfg.large_model_path,
        transform_model_path=cfg.transform_model_path,
        seed=cfg.seed,
        loraxs_rank=cfg.loraxs_rank,
        num_samples_per_val_dataset=cfg.num_samples_per_val_dataset,
        max_length_eval=cfg.max_length_eval,
        max_new_tokens=cfg.max_new_tokens,
        init_small_path=cfg.init_small_path,
        init_large_path=cfg.init_large_path,
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        l_out=cfg.l_out,
        small_tune_option=cfg.small_tune_option,
        large_tune_option=cfg.large_tune_option,
        base_model_path=cfg.base_model_path,
        quantize=cfg.quantize,
        save_path=cfg.save_path,
        save_fname=cfg.save_fname,
    )

    main(args) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain a model using update vectors"
    )
    # parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_fname', type=str)
    parser.add_argument('--small_model_gradient_path', type=str)
    parser.add_argument('--large_model_path', type=str)
    parser.add_argument('--transform_model_path', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--loraxs_rank', type=int, default=256)
    parser.add_argument('--num_samples_per_val_dataset', type=int)
    parser.add_argument('--max_length_eval', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--init_small_path', type=str)
    parser.add_argument('--init_large_path', type=str)
    parser.add_argument('--input_dim', type=int, default=None)
    parser.add_argument('--output_dim', type=int, default=None)
    parser.add_argument('--l_out', type=int, default=None)
    parser.add_argument('--small_tune_option', type=str, default=None)
    parser.add_argument('--large_tune_option', type=str, default=None)
    parser.add_argument('--base_model_path', type=str)
    # parser.add_argument('ans_template', type=str, required=True)
    # parser.add_argument('ref_template', type=str)
    # parser.add_argument('--delta', type=float, default=1e-5)
    # parser.add_argument('--dp_eps', type=float, default=None)
    # parser.add_argument('--clipping_threshold', type=float, default=1.0)
    # parser.add_argument('--mean', type=float, default=None)
    # parser.add_argument('--noise_std',type=float)
    # parser.add_argument('--noise_option',type=str)
    parser.add_argument('--quantize',default=False)
    # parser.add_argument('--random_gradients_scale',type=float,default=None)
    args = parser.parse_args()
    main(args)