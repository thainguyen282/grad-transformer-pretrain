import os
import time
import random
import logging

import yaml
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
    get_scheduler,
)
from trl import DataCollatorForCompletionOnlyLM

from utils.dataset import CustomDataset, random_sample
from datasets import Dataset
from utils.eval_math import *
from src.loraxs_utils.initialization_utils import find_and_initialize
from prompter_utils import (
    Prompter,
    generate_and_tokenize_prompt,
    make_contiguous_,
    report_noncontiguous_params,
)


def run_finetune_guardrail_qwen_loraxs(args):
    """
    Fine-tune Qwen on GuardRail using LoRA-XS initialized via SVD reconstruction.
    """
    torch.set_printoptions(threshold=float("inf"))

    # Repo root: pretrained/finetune_src -> ../..
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    reconstruct_config_path = os.path.join(project_root, "configs", "reconstruct_config.yaml")
    reconstruct_config_dir = os.path.dirname(reconstruct_config_path)
    if not os.path.exists(reconstruct_config_path):
        raise FileNotFoundError(
            f"Missing reconstruct config: {reconstruct_config_path} (dir exists: {os.path.exists(reconstruct_config_dir)})"
        )

    start_time = time.time()

    # Set random seed for initialization
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    safe_model_path = args.model_path.split("/")[1]
    safe_dataset_path = args.dataset_path.split("/")[1]

    args_dict = vars(args)
    if args.save_fname:
        save_fname = args.save_fname
    else:
        if args.finetune_split:
            save_fname = (
                f"{safe_model_path}_{safe_dataset_path}_{args.tune_option}_split_{args.finetune_split}_{args.split_option}_seed_{args.seed}"
            )
        elif args.train_indices_path:
            path = [x.split("/")[-1][-5] for x in args.train_indices_path]
            prefix = args.train_indices_path[0].split("/")[-1].replace(".pkl", "")
            save_fname = safe_model_path + "-" + prefix + "-" + "-".join(path)
        else:
            save_fname = f"{safe_model_path}_{safe_dataset_path}_{args.tune_option}_seed_{args.seed}"

    output_dir = os.path.join(project_root, "models", safe_dataset_path, save_fname)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir}")

    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)

    accelerator.init_trackers(
        "seckt",
        config=args_dict,
        init_kwargs={"wandb": {"name": save_fname}},
    )

    # Print the arguments
    print("Parameter configuration:")
    for k, v in args_dict.items():
        print(f"{k}: {v}")

    # -------------------------------------------------------------------------
    # Model loading + LoRA-XS initialization
    if args.quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
        )
        # Big memory saver for training
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        # Required for stable k-bit finetuning
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")

    lora_config = LoraConfig(
        r=args.loraxs_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    set_seed(seed)
    model = get_peft_model(model, lora_config)

    with open(reconstruct_config_path, "r") as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)

    adapter_name = "default"  # assume a single LoRA adapter per module
    peft_config_dict = {adapter_name: lora_config}
    reconstr_config["svd"]["rank"] = args.loraxs_rank

    find_and_initialize(
        model,
        peft_config_dict,
        adapter_name=adapter_name,
        reconstr_type="svd",
        writer=None,
        reconstruct_config=reconstr_config,
    )

    bad = report_noncontiguous_params(model)
    if len(bad) > 0:
        print("Making model parameters/buffers contiguous...")
        make_contiguous_(model)
        bad2 = report_noncontiguous_params(model)
        if len(bad2) > 0:
            raise RuntimeError(
                "Still have non-contiguous parameters after make_contiguous_(); "
                "check LoRA-XS initialization to ensure nn.Parameter(t.contiguous())."
            )

    print("The following weights are tuned:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
            print(name, param[0])
            break

    model.to(device)

    # -------------------------------------------------------------------------
    # Dataset prep
    dataset = CustomDataset(path=args.dataset_path)
    train_dataset, val_dataset, test_dataset = dataset.split_dataset()

    if args.train_indices_path:
        import pickle

        train_idx = []
        test_idx = []

        train_dataset_tmp = train_dataset

        for indices in args.train_indices_path:
            with open(indices, "rb") as file:
                idx = pickle.load(file)
            train_idx += idx["train"]
            print(len(train_idx))

        if args.test_indices_path:
            for indices in args.test_indices_path:
                with open(indices, "rb") as file:
                    idx = pickle.load(file)
                test_idx += idx["test"]

            train_dataset = train_dataset_tmp.select(train_idx)
            val_dataset = train_dataset_tmp.select(test_idx)
        else:
            # If no test indices are provided, treat validation as a subset of the train indices.
            train_dataset = train_dataset_tmp.select(train_idx)
            val_dataset = train_dataset_tmp.select(train_idx)

    if args.finetune_split:
        if args.split_option == "w2s":
            train_dataset = train_dataset.select(range(0, int(len(train_dataset) * args.finetune_split)))
            print(f"W2S: Keep the first {args.finetune_split * 100}% of the training set")
        else:
            train_dataset = train_dataset.select(
                range(int(len(train_dataset) * args.finetune_split), len(train_dataset))
            )
            print(f"Cut the first {args.finetune_split * 100}% of the training set")

    if args.num_samples_per_train_dataset:
        train_dataset = random_sample(train_dataset, n_samples=args.num_samples_per_train_dataset)

    if val_dataset is None:
        val_dataset = test_dataset

    if args.num_samples_per_val_dataset:
        val_dataset = random_sample(val_dataset, n_samples=args.num_samples_per_val_dataset)

    print(f"Final length train set: {len(train_dataset)}, validation set: {len(val_dataset)}")

    train_dataset = train_dataset.map(
        dataset.format_guardrail,
        batched=True,
        load_from_cache_file=False,
        fn_kwargs={"mode": "inference"},
    )
    val_dataset = val_dataset.map(
        dataset.format_guardrail,
        batched=True,
        load_from_cache_file=False,
        fn_kwargs={"mode": "inference"},
    )

    prompter = Prompter()

    train_dataset = Dataset.from_dict(generate_and_tokenize_prompt(train_dataset, prompter, tokenizer))
    val_dataset = Dataset.from_dict(generate_and_tokenize_prompt(val_dataset, prompter, tokenizer))

    tokenized_train_dataset = train_dataset.map(
        dataset.preprocess_function_causal,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        fn_kwargs={
            "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
            "postfix": "<|im_end|>\n<|im_start|>assistant\n",
            "max_length": args.max_length,
            "padding": True,
            "truncation": True,
            "tokenizer": tokenizer,
            "input_key": "instruction",
            "target_key": "output",
            "mode": "train",
        },
    )

    tokenized_val_dataset_for_loss = val_dataset.map(
        dataset.preprocess_function_causal,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        fn_kwargs={
            "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
            "postfix": "<|im_end|>\n<|im_start|>assistant\n",
            "max_length": args.max_length,
            "padding": True,
            "truncation": True,
            "tokenizer": tokenizer,
            "input_key": "instruction",
            "target_key": "output",
            "mode": "train",
        },
    )

    tokenized_val_dataset = val_dataset.map(
        dataset.preprocess_function_causal,
        batched=True,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False,
        fn_kwargs={
            "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
            "postfix": "<|im_end|>\n<|im_start|>assistant\n",
            "max_length": args.max_length_eval,
            "padding": True,
            "truncation": True,
            "tokenizer": tokenizer,
            "input_key": "instruction",
            "target_key": "output",
            "mode": "inference",
        },
    )

    # -------------------------------------------------------------------------
    # Training setup
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant",
        tokenizer=tokenizer,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc, records_table = evaluate_math_reasoning_accuracy(
            predictions=decoded_preds,
            references=decoded_labels,
            ans_template="####",
            ref_template="####",
            table=True,
        )
        result = {"accuracy": round(acc * 100, 2)}
        return [result, records_table]

    def evaluate_model(model_, val_dataloader_):
        generated_sequences = []
        label_sequences = []

        for eval_batch in tqdm(val_dataloader_):
            input_ids = eval_batch["input_ids"].to(device)
            attention_mask = eval_batch["attention_mask"].to(device)
            label_ids = eval_batch["labels"].to(device)

            generated_tokens = model_.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                use_cache=True,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )

            generated_tokens = generated_tokens.sequences[:, input_ids.shape[1] :]
            generated_sequences.extend(generated_tokens.cpu())
            label_sequences.extend(label_ids.cpu())

        predictions = pad_sequence(
            generated_sequences, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = pad_sequence(label_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

        result_, records_table_ = compute_metrics((predictions, labels))
        return result_, records_table_

    def evaluate_loss(model_, val_dataloader_):
        model_.eval()
        losses = []
        for eval_batch in val_dataloader_:
            with torch.no_grad():
                outputs = model_(**eval_batch)
                loss = outputs.loss
            losses.append(loss.item())
        return np.mean(losses)

    train_dataloader = DataLoader(
        tokenized_train_dataset,
        batch_size=args.finetune_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    val_dataloader = DataLoader(
        tokenized_val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    val_dataloader_for_loss = DataLoader(
        tokenized_val_dataset_for_loss,
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator,
    )

    optimizer = AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)

    num_training_steps = args.num_training_steps
    if args.scheduler:
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=args.num_training_steps * args.warmup_ratio,
            num_training_steps=int(args.num_training_steps / args.gradient_accumulation_steps),
        )

    if args.scheduler:
        model, optimizer, train_dataloader, val_dataloader_for_loss, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader_for_loss, scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader_for_loss = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader_for_loss
        )

    model.to(device)
    model.train()

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    step = 0
    _ = time.time()  # keep variable name parity; not used

    initial_lora_params = {
        name: param.clone().detach().cpu()
        for name, param in model.module.named_parameters()
        if param.requires_grad
    }

    initial_save_path = os.path.join(output_dir, "lora_initial.pt")
    torch.save(initial_lora_params, initial_save_path)

    best_metric = -1
    if accelerator.is_main_process:
        with torch.no_grad():
            eval_loss = evaluate_loss(model, val_dataloader_for_loss)
            best_eval_loss = eval_loss
            result, records_table = evaluate_model(model, val_dataloader)
            print(f"Step 0 Eval loss: {eval_loss:.6f}")
            print(f"Step 0 AQuA-RAT: {result}")
            accelerator.log({f"AQuA-RAT_{k}": v for k, v in result.items()}, step=0)
            accelerator.log({"eval_loss": eval_loss}, step=0)
            accelerator.log({f"AQuA-RAT_records_table": records_table}, step=0)
            best_metric = result["accuracy"]
            accelerator.log({"best_metric": best_metric}, step=step)

    accelerator.wait_for_everyone()

    print("TRAINING STARTED...")

    # Early stopping state (eval loss)
    best_eval_loss = float("inf")
    early_stop_counter = 0
    stop_training = False

    while step <= num_training_steps:
        avg_train_loss = 0

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                progress_bar.update(1)
                step += 1
                model.train()
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if args.scheduler:
                    scheduler.step()
                avg_train_loss += loss.item()

                if args.scheduler:
                    accelerator.log({"learning_rate": scheduler.get_last_lr()[0]}, step=step)

                if step % 100 == 0 and step != 0:
                    avg_train_loss /= 100
                    accelerator.log({"train_loss": avg_train_loss}, step=step)
                    avg_train_loss = 0

                    eval_loss = evaluate_loss(model, val_dataloader_for_loss)
                    print(f"Step {step} Eval loss: {eval_loss:.6f}")
                    accelerator.log({"eval_loss": eval_loss}, step=step)

                    # Early stopping on eval loss
                    if args.early_stopping_patience is not None:
                        improved = (best_eval_loss - eval_loss) > args.early_stopping_min_delta
                        if improved:
                            best_eval_loss = eval_loss
                            early_stop_counter = 0
                            accelerator.log({"best_eval_loss": best_eval_loss}, step=step)
                        else:
                            early_stop_counter += 100
                            accelerator.log({"early_stop_counter": early_stop_counter}, step=step)
                            if early_stop_counter >= args.early_stopping_patience:
                                print(
                                    f"Early stopping triggered at step {step}: "
                                    f"no eval loss improvement for {args.early_stopping_patience} checks."
                                )
                                stop_training = True

                if step >= num_training_steps or (stop_training and step != 0):
                    stop_training = True
                    if accelerator.is_main_process:
                        model.eval()
                        with torch.no_grad():
                            result, records_table = evaluate_model(model, val_dataloader)
                            print(f"Step {step} AQuA-RAT: {result}")
                            accelerator.log({f"AQuA-RAT_{k}": v for k, v in result.items()}, step=step)
                            accelerator.log(
                                {f"AQuA-RAT_records_table": records_table},
                                step=step,
                            )

                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                output_dir,
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save,
                            )

                            lora_grads = {
                                name: param.clone().detach().cpu()
                                for name, param in model.module.named_parameters()
                                if param.requires_grad
                            }

                            delta_lora_grads = {
                                name: (lora_grads[name] - initial_lora_params[name])
                                for name in lora_grads.keys()
                            }

                            delta_save_path = os.path.join(output_dir, "final_lora_gradients.pt")
                            torch.save(delta_lora_grads, delta_save_path)

                            print(
                                f"Latest model saved at step {step} to {delta_save_path}; {result}"
                            )

                            if result["accuracy"] > best_metric:
                                best_metric = result["accuracy"]
                                best_delta_save_path = os.path.join(
                                    output_dir, "best_lora_gradients.pt"
                                )
                                torch.save(delta_lora_grads, best_delta_save_path)
                                print(
                                    f"Best model saved at step {step} to {best_delta_save_path}; {result}"
                                )

                        accelerator.log({"best_metric": best_metric}, step=step)

                    accelerator.wait_for_everyone()
                    break

        if stop_training is True:
            break

    accelerator.end_training()

    # Avoid unused warnings for `start_time` (kept for parity with original script).
    _elapsed_s = time.time() - start_time

