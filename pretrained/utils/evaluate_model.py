import os
import random
import torch
import numpy as np
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset

from dataset import CustomDataset, random_sample
from eval_math import evaluate_math_reasoning_accuracy
from src.utils import Prompter, generate_and_tokenize_prompt

def compute_metrics(eval_pred, dataset_path, ans_template="####", ref_template="####", tokenizer=None, rouge=None):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    if "dialogsum" in dataset_path.lower() or "samsum" in dataset_path.lower():
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result["gen_len"] = np.mean(
            [np.count_nonzero(p != tokenizer.pad_token_id) for p in predictions]
        )
    else:
        acc, _ = evaluate_math_reasoning_accuracy(
            predictions=decoded_preds,
            references=decoded_labels,
            ans_template=ans_template,
            ref_template=ref_template,
            table=False,
        )
        result = {"accuracy": round(acc * 100, 2)}

    return result

def evaluate_model(model, val_dataloader, tokenizer=None, device=None, max_new_tokens=None, dataset_path=None, ans_template=None, ref_template=None, rouge=None):
    generated_sequences, label_sequences = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["labels"].to(device)

            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                use_cache=True,
            )
            generated_tokens = generated_tokens[:, input_ids.shape[1]:]
            generated_sequences.extend(generated_tokens.cpu())
            label_sequences.extend(label_ids.cpu())

    predictions = pad_sequence(
        generated_sequences, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = pad_sequence(
        label_sequences, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return compute_metrics((predictions, labels), dataset_path=dataset_path, ans_template=ans_template, ref_template=ref_template, tokenizer=tokenizer, rouge=rouge)


def evaluate(model, args):
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    pretrained_model_path = os.path.join(args.save_dir, "random_init_model", args.pretrained_model_name)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    rouge = evaluate.load("rouge", keep_in_memory=True)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, padding_side="left")

    dataset = CustomDataset(path=args.dataset_path)
    _, val_dataset, test_dataset = dataset.split_dataset()
    if val_dataset is None:
        val_dataset = test_dataset
    if args.num_samples:
        val_dataset = random_sample(val_dataset, args.num_samples)

    if 'aqua_rat' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_aquarat,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'gsm8k' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_gsm8k,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'math_qa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_mathqa,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'svamp' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_svamp,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'hotpot_qa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_hotpotqa,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'commonsenseqa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_commonsenseqa,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'drop' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_drop,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'dialogsum' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_dialogsum,
            batched=True,
            load_from_cache_file=False
        )
    elif 'samsum' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_samsum,
            batched=True,
            load_from_cache_file=False
        )
    elif 'gpqa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_gpqa,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'theorem' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_theoremqa,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'hendrycks_math' in args.dataset_path.lower() or 'math' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_hendrycks_math,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'human_eval' in args.dataset_path.lower() or 'humaneval' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_humaneval,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'mbpp' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_mbpp,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )
    elif 'mmlu' in args.dataset_path.lower() or 'indommlu' in args.dataset_path.lower() or 'rummmlu' in args.dataset_path.lower():
        val_dataset = val_dataset.map(
            dataset.format_mmlu,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={'mode': 'inference'}
        )

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

    val_dataloader = DataLoader(tokenized_val_dataset, batch_size=8, collate_fn=default_data_collator)

    result = evaluate_model(model, val_dataloader, tokenizer=tokenizer, device=device, max_new_tokens=args.max_new_tokens, dataset_path=args.dataset_path, ans_template=args.ans_template, ref_template=args.ref_template, rouge=rouge)
    return result