import os
import sys
import json
import argparse
import logging
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, os.path.abspath(project_root))
from utils.transformer import Embedding2EmbeddingT5
from utils.custom_dataloader import build_dataloader, IndexDataset


def masked_mse_loss(pred, target, mask):
    mask_f = mask.unsqueeze(-1).to(dtype=pred.dtype)
    valid = mask_f.sum()
    if valid.item() == 0:
        raise ValueError("Mask is all False!!!")
    
    se = (pred - target).pow(2) * mask_f 
    return se.sum() / (valid * pred.size(-1)) # average of loss weight

def run_train_grad_transformer(args, console):
    args_dict = vars(args)
    if torch.cuda.is_available():
        mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        mixed_precision = "no"
    accelerator = Accelerator(log_with="wandb", mixed_precision=mixed_precision)
    device = accelerator.device
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)

    accelerator.init_trackers("embedding2embedding", config=args_dict)
    wandb.init(project="embedding2embedding", config=args_dict)
    model_load_path = args.save_dir
    model_pairs = [("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct")]

    model = Embedding2EmbeddingT5(
        input_dim=args.dim,
        output_dim=args.dim,
        freeze_t5=args.freeze_t5,
        base_model=args.base_model_path,
    )
    # model.to(device, dtype=torch.bfloat16)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    best_val_loss = float('inf')
    if args.patience:
        patience = args.patience
        patience_counter = 0
    step = 0
    steps_per_epoch = 0
    subset_dataset_size = (args.num_noisy_samples ** 2)
    if args.exclude_pretrain:
        model_pairs = [
        (m1, m2) for (m1, m2) in model_pairs
        if args.pretrain_model_path not in (m1, m2)
    ]

    dataset_size = len(model_pairs) * subset_dataset_size

    train_dataset_size = int(dataset_size * args.split)
    steps_per_epoch = max(1, train_dataset_size // args.batch_size)

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(dataset_size, generator=generator)

    train_indices = []
    val_indices = []


    for i in range(dataset_size):
        subset_id = perm[i] // subset_dataset_size
        data_id = perm[i] - (subset_id*subset_dataset_size)
        if data_id >= args.split*subset_dataset_size:
            val_indices.append(perm[i].item())
        else:
            train_indices.append(perm[i].item())

    print(args.batch_size)

    train_dataset = IndexDataset(train_indices, model_pairs, model_load_path, args.num_noisy_samples)
    train_loader =  DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        num_workers=0,
        shuffle=True,
    )
    val_dataset = IndexDataset(val_indices, model_pairs, model_load_path, args.num_noisy_samples)
    val_loader =  DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        num_workers=0,
        shuffle=True,
    )  
    
    if args.eval_per:
        eval_per = args.eval_per
    else:
        eval_per = int(steps_per_epoch)

    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    print(f"Evaluate every {eval_per} steps")
    
    print('TRAINING STARTED…')
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        total_loss = 0.0
        layer_loss = [0] * args.l_out
        
        # for xb, yb, encoder_mask, decoder_mask in tqdm(train_loader, desc="Training"):
        for xb, yb, embedding_mask_x, embedding_mask_y in tqdm(train_loader, desc="Training"):
            print(xb.dtype)
            print(yb.dtype)
            print(embedding_mask_x.dtype)
            print(embedding_mask_y.dtype)
            print("run into training step")
            step += 1
            optimizer.zero_grad()
            # pred = model(x=xb, y=yb, L_out=args.l_out, use_teacher_forcing=True,
            #                 encoder_attention_mask=encoder_mask.long(), decoder_attention_mask=decoder_mask.long()) # (batch_size, L_out, embedding_size))
            pred = model(x=xb, y=yb, embedding_mask_x=embedding_mask_x, embedding_mask_y=embedding_mask_y, L_out = args.l_out, use_teacher_forcing=True) # (batch_size, L_out, embedidng_size)
            loss = masked_mse_loss(pred, yb, decoder_mask)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            if step % eval_per == 0:
                model.eval()
                val_loss_sum = torch.tensor(0.0, device=accelerator.device)
                val_count = torch.tensor(0.0, device=accelerator.device)
                layer_loss_sum = torch.zeros(args.l_out, device=accelerator.device)
                layer_count = torch.zeros(args.l_out, device=accelerator.device)

                with torch.no_grad():
                    for xb, yb, encoder_mask, decoder_mask in tqdm(val_loader, desc="Validating"):
                        pred = model(
                            x=xb, y=yb, L_out=args.l_out, use_teacher_forcing=False,
                            encoder_attention_mask=encoder_mask,
                            decoder_attention_mask=decoder_mask
                        )
                        loss = masked_mse_loss(pred, yb, decoder_mask)
                        bs = xb.size(0)
                        val_loss_sum += loss  * bs
                        val_count += bs

                        for layer in range(args.l_out):
                            layer_mask = decoder_mask[:, layer]
                            if layer_mask.any():
                                layer_loss_sum[layer] += masked_mse_loss(
                                    pred[:, layer:layer+1, :],
                                    yb[:, layer:layer+1, :],
                                    decoder_mask[:, layer:layer+1],
                                ) * layer_mask.sum()
                                layer_count[layer] += layer_mask.sum()
                                
                # Reduce across all processes
                val_loss_sum = accelerator.reduce(val_loss_sum, reduction="sum")
                val_count = accelerator.reduce(val_count, reduction="sum")
                layer_loss_sum = accelerator.reduce(layer_loss_sum, reduction="sum")
                layer_count = accelerator.reduce(layer_count, reduction="sum")

                avg_val_loss = (val_loss_sum / val_count).item()
                avg_layer_loss = (layer_loss_sum / layer_count).tolist()
                avg_train_loss = total_loss / eval_per

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    accelerator.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=step)
                    for i, v in enumerate(avg_layer_loss):
                        accelerator.log({f"val_loss_layer_{i+1}": v}, step=step)
                    print(f"Step [{step}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

                    if avg_val_loss < best_val_loss:
                        # Define checkpoint directory
                        checkpoint_dir = os.path.join(args.save_dir, f"checkpoint_{step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)  # create it if it doesn't exist

                        save_path = os.path.join(checkpoint_dir, args.save_path)
                        unwrapped_model = accelerator.unwrap_model(model)
                        print("Meet better checkpoint -> saving")
                        torch.save(unwrapped_model.state_dict(), save_path)
                        
                        best_val_loss = avg_val_loss
                        if args.patience:
                            patience_counter = 0

                        # Manage old checkpoints
                        checkpoint_dirs = sorted(
                            [d for d in os.listdir(args.save_dir) if d.startswith("checkpoint_")],
                            key=lambda x: int(x.split("_")[1])
                        )
                        if len(checkpoint_dirs) > 3:
                            oldest_checkpoint = os.path.join(args.save_dir, checkpoint_dirs[0])
                            shutil.rmtree(oldest_checkpoint)

                        # Save a copy as best_checkpoint
                        best_checkpoint_dir = os.path.join(args.save_dir, "best_checkpoint")
                        os.makedirs(best_checkpoint_dir, exist_ok=True)
                        best_save_path = os.path.join(best_checkpoint_dir, args.save_path)
                        shutil.copyfile(save_path, best_save_path)
                    else:
                        if args.patience:
                            patience_counter += eval_per

                total_loss = 0
                accelerator.wait_for_everyone()
                model.train()

            if args.patience:
                if patience_counter >= patience * eval_per:
                    print(f"Early stopping triggered at step {step+1}")
                    break
            
        
            accelerator.log({f"Epoch": (step - 1) / steps_per_epoch}, step=step+1)
        
            if args.patience:
                if patience_counter >= patience * eval_per:
                    print(f"Early stopping triggered at step {step+1}")
                    break
                        
    accelerator.end_training()

