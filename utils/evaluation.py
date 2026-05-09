import argparse
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.nn.utils.rnn import pad_sequence
import evaluate
import copy

from dataset import CustomDataset, random_sample
from datasets import Dataset
from eval_math import *
from prompter_utils import Prompter, generate_and_tokenize_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--gradient_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--input_key',
        type=str,
        default='problem'
    )

    parser.add_argument(
        '--target_key',
        type=str,
        default='solution'
    )

    parser.add_argument(
        '--num_samples_per_val_dataset',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--lora_rank',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--max_length_eval',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--tune_option',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--ans_template',
        type=str,
        default='####',
    )

    parser.add_argument(
        '--ref_template',
        type=str,
        default='####',
    )

    args = parser.parse_args()
    args_dict = vars(args)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    print(f"Set random seed to {seed}")

    for k, v in args_dict.items():
        print(k, v)

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="bf16",
    )

    accelerator.init_trackers("evaluate-rvkd", 
        config=args_dict,
        init_kwargs={
            "wandb": {
                "name": f"{args.dataset_path}_{args.model_path}",
            }
        }
    )

    rouge = evaluate.load("rouge", keep_in_memory=True)

    def compute_metrics(
            eval_pred, 
            dataset_path, 
            ans_template='####', 
            ref_template='####'
        ):
        if 'dialogsum' in dataset_path.lower() or 'samsum' in dataset_path.lower():
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            print(f"*****\n{decoded_preds[0]}\n====\n{decoded_labels[0]}\n*****")
            print(f"*****\n{decoded_preds[1]}\n====\n{decoded_labels[1]}\n*****")  
            print(f"*****\n{decoded_preds[2]}\n====\n{decoded_labels[2]}\n*****")          
            result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)
        else:
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            acc, records_table = evaluate_math_reasoning_accuracy(
                predictions=decoded_preds, 
                references=decoded_labels,
                ans_template=ans_template,
                ref_template=ref_template,
                table=False
            )
            result = {"accuracy": round(acc * 100, 2)}
        
        records_table = None
        
        return result, records_table

    def evaluate_model(model, val_dataloader, dataset_path, ans_template, ref_template):
        generated_sequences = []
        label_sequences = []
        for eval_batch in tqdm(val_dataloader):
            input_ids = eval_batch["input_ids"].to(device)
            attention_mask = eval_batch["attention_mask"].to(device)
            label_ids = eval_batch["labels"].to(device)

            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                use_cache=True,
            )
            
            generated_tokens = generated_tokens[:, input_ids.shape[1]:]
            generated_sequences.extend(generated_tokens.cpu())
            label_sequences.extend(label_ids.cpu())

        predictions = pad_sequence(generated_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(label_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

        result, records_table = compute_metrics((predictions, labels), dataset_path, ans_template, ref_template)
        return result, records_table

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

    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    set_seed(seed)
    model = get_peft_model(model, lora_config)
    model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
            print(name, param[0])
            break

    dataset = CustomDataset(path=args.dataset_path)
    train_dataset, val_dataset, test_dataset = dataset.split_dataset()

    if val_dataset == None:
        val_dataset = test_dataset

    if args.num_samples_per_val_dataset:
        val_dataset = random_sample(val_dataset, args.num_samples_per_val_dataset)
    
    if 'aqua_rat' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_aquarat, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'gsm8k' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_gsm8k, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'math_qa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_mathqa, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'svamp' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_svamp, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'hotpot_qa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_hotpotqa, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'commonsenseqa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_commonsenseqa, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'drop' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_drop, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'dialogsum' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_dialogsum, batched=True, load_from_cache_file=False)
    elif 'samsum' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_samsum, batched=True, load_from_cache_file=False)

    prompter = Prompter()
    val_dataset = Dataset.from_dict(generate_and_tokenize_prompt(val_dataset, prompter, tokenizer))
    tokenized_val_dataset = val_dataset.map(
        dataset.preprocess_function_causal,
        batched=True,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False,
        fn_kwargs={
            'prefix': "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
            'postfix': "<|im_end|>\n<|im_start|>assistant\n",
            'input_key': 'instruction',
            'target_key': 'output',
            'max_length': args.max_length_eval,
            'padding': True,
            'truncation': True,
            'tokenizer': tokenizer,
            'mode': 'inference'
        },
    )

    optimizer = AdamW(model.parameters())
    val_dataloader = DataLoader(tokenized_val_dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator)
    lora_gradients = torch.load(args.gradient_path, weights_only=True)

    with torch.no_grad():
        # print("Model before fine-tuned:")
        # model.eval()
        # result = evaluate_model(model, val_dataloader)
        print("Layers tuned:")
        predicted_model = apply_gradients(copy.deepcopy(model), lora_gradients)
        print('=============================')
        print(f"Model fine-tuned:")
        predicted_model.eval()
        result = evaluate_model(predicted_model, val_dataloader, args.dataset_path, args.ans_template, args.ref_template)
        print(result)