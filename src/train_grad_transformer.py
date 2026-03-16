import os
import sys
import json
import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
project_root = os.path.join(os.path.dirname(__file__), '../..')
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(project_root))
sys.path.insert(0, os.path.abspath(src_dir))
from src.transformer import Embedding2EmbeddingT5
from config import TrainGradTransformerConfig


def masked_mse_loss(pred, target, mask):
    mask_expanded = mask.unsqueeze(-1).expand_as(pred)
    diff = (pred - target) ** 2
    return diff[mask_expanded].mean()


def variable_lout_collate_fn(batch, l_out):
    x_batch, y_batch = zip(*batch)
    x_batch = torch.stack(x_batch, dim=0)

    output_dim = y_batch[0].shape[-1]
    B = len(y_batch)
    y_padded = torch.zeros(B, l_out, output_dim, dtype=y_batch[0].dtype)
    mask = torch.zeros(B, l_out, dtype=torch.bool)

    for i, y in enumerate(y_batch):
        l = min(y.shape[0], l_out)
        y_padded[i, :l, :] = y[:l]
        mask[i, :l] = True

    return x_batch, y_padded, mask


class InMemoryGradientDataset(Dataset):
    def __init__(self, small_gradients, large_gradients, indices=None):
        assert len(small_gradients) == len(large_gradients), (
            f"small_gradients ({len(small_gradients)}) and "
            f"large_gradients ({len(large_gradients)}) must have the same length"
        )
        self.small_gradients = small_gradients
        self.large_gradients = large_gradients
        self.dataset_size = len(small_gradients)
        self.indices = list(range(self.dataset_size)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        try:
            return self.small_gradients[actual_idx], self.large_gradients[actual_idx]
        except IndexError:
            print(f"Index out of range: {actual_idx} (dataset size: {self.dataset_size})")
            raise

def create_delta_dataset(
    small_gradients, large_gradients, split=0.8, batch_size=16, seed=42,
    num_samples_per_val_dataset=None, shuffle_train=True, l_out=None,
):
    dataset_size = len(small_gradients) ** 2

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=generator)

    small_list, large_list = [], []
    for i in range(len(small_gradients)):
        for j in range(len(large_gradients)):
            small_list.append(small_gradients[i])
            large_list.append(large_gradients[j])

    collate = (lambda batch: variable_lout_collate_fn(batch, l_out)) if l_out is not None else None

    if split < 1.0:
        train_size = int(dataset_size * split)
        train_idx = perm[:train_size].tolist()
        val_idx = perm[train_size:].tolist()
        if num_samples_per_val_dataset:
            val_idx = val_idx[:num_samples_per_val_dataset]
            print(f"Limited validation set to {len(val_idx)} samples")
        train_loader = DataLoader(
            InMemoryGradientDataset(small_list, large_list, indices=train_idx),
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
            shuffle=shuffle_train,
        )
        val_loader = DataLoader(
            InMemoryGradientDataset(small_list, large_list, indices=val_idx),
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
            shuffle=False,
        )
        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            InMemoryGradientDataset(small_gradients, large_gradients),
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
            shuffle=shuffle_train,
        )
        return train_loader, None

def main(args):
    args_dict = vars(args)
    for k, v in args_dict.items():
        print(f"  {k}: {v}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    accelerator = Accelerator(log_with="wandb", mixed_precision="bf16")
    device = accelerator.device
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)
    if args.save_fname:
        accelerator.init_trackers(
            "embedding2embedding", config=args_dict,
            init_kwargs={"wandb": {"name": args.save_fname}},
        )
    else:
        accelerator.init_trackers("embedding2embedding", config=args_dict)
    # model_load_path = os.path.join(project_root, "saves")
    model_load_path = "saves"

    model_pairs = []
    model_pairs_config_path = os.path.join(project_root, args.model_pairs_config)
    with open(model_pairs_config_path, 'r') as f:
        config = json.load(f)
        model_pairs = [
            (pair['small_model_path'], pair['large_model_path'])
            for pair in config.get('model_pairs', [])
        ]
    print(f"Loaded {len(model_pairs)} model pairs from {model_pairs_config_path}")

    ### load sample
    base_small = torch.load(
        os.path.join(model_load_path, args.small_model_path, "gradients_base.pt"),
        weights_only=True,
    )
    base_large = torch.load(
        os.path.join(model_load_path, args.large_model_path, "gradients_base.pt"),
        weights_only=True,
    )
    input_dim = base_small.shape[-1]
    output_dim = base_large.shape[-1]
    print(f"input_dim={input_dim}, output_dim={output_dim}")

    model = Embedding2EmbeddingT5(
        input_dim=input_dim,
        output_dim=output_dim,
        freeze_t5=args.freeze_t5,
        base_model=args.base_model_path,
    )

    if not model_pairs:
        model_pairs = [(args.small_model_path, args.large_model_path)]

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on: {device}")

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    best_val_loss = float('inf')
    if args.patience:
        patience = args.patience
        patience_counter = 0
    
    step = 0
    steps_per_epoch = 0
    dataset_size = len(model_pairs) * (args.num_noisy_samples ** 2)
    train_dataset_size = int(dataset_size * args.split)
    steps_per_epoch = max(1, train_dataset_size // args.batch_size)
    if args.eval_per:
        eval_per = args.eval_per
    else:
        eval_per = int(steps_per_epoch)

    print(f"Evaluate every {eval_per} steps")
    
    print('TRAINING STARTED…')
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        total_loss = 0.0
        layer_loss = [0] * args.l_out
        
        for small_model_path, large_model_path in tqdm(model_pairs, desc="Loading model pairs"):
            small_gradients, large_gradients = [], []
            for i in range(args.num_noisy_samples):
                small_gradients.append(
                    torch.load(
                        os.path.join(model_load_path, small_model_path, f"model_{i+1}.pt"),
                        weights_only=True,
                    )
                )
                large_gradients.append(
                    torch.load(
                        os.path.join(model_load_path, large_model_path, f"model_{i+1}.pt"),
                        weights_only=True,
                    )
                )
            
            temp_dataset_size = len(small_gradients) ** 2
            temp_train_loader, temp_val_loader = create_delta_dataset(
                small_gradients, large_gradients,
                split=args.split, batch_size=args.batch_size, seed=args.seed,
                num_samples_per_val_dataset=args.num_samples_per_val_dataset,
                shuffle_train=True, l_out=args.l_out,
            )
            temp_train_loader, temp_val_loader = accelerator.prepare(
                temp_train_loader, temp_val_loader
            )
            for xb, yb, mask in tqdm(temp_train_loader, desc="Training"):
                step += 1
                optimizer.zero_grad()
                print(xb.shape, yb.shape, mask.shape)
                pred = model(x=xb, y=yb, L_out=args.l_out, use_teacher_forcing=True,
                             decoder_attention_mask=mask.long()) # (batch_size, L_out, embedding_size))
                loss = masked_mse_loss(pred, yb, mask)
                accelerator.backward(loss)
                if args.clip is not None and args.clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                total_loss += loss.item()
                if step % eval_per == 0:
                    model.eval()

                    val_loss_sum = torch.tensor(0.0, device=accelerator.device)
                    val_count = torch.tensor(0.0, device=accelerator.device)

                    layer_loss_sum = torch.zeros(args.l_out, device=accelerator.device)
                    layer_count = torch.zeros(args.l_out, device=accelerator.device)

                    with torch.no_grad():
                        for xb, yb, mask in tqdm(temp_val_loader, desc="Validating"):
                            pred = model(x=xb, y=yb, L_out=args.l_out, use_teacher_forcing=False)
                            loss = masked_mse_loss(pred, yb, mask)

                            bs = xb.size(0)
                            val_loss_sum += loss * bs
                            val_count += bs

                            for layer in range(args.l_out):
                                layer_mask = mask[:, layer]
                                if layer_mask.any():
                                    layer_loss_sum[layer] += masked_mse_loss(
                                        pred[:, layer:layer+1, :],
                                        yb[:, layer:layer+1, :],
                                        mask[:, layer:layer+1],
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

                    if accelerator.is_main_process:
                        accelerator.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=step)
                        for i, v in enumerate(avg_layer_loss):
                            accelerator.log({f"val_loss_layer_{i+1}": v}, step=step)
                        print(f"Step [{step}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

                        if avg_val_loss < best_val_loss:
                            unwrapped_model = accelerator.unwrap_model(model)
                            torch.save(unwrapped_model.state_dict(), args.save_path)
                            best_val_loss = avg_val_loss
                            if args.patience:
                                patience_counter = 0
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
                
            
            # accelerator.log({f"tflops": tflops}, step=step+1)
            accelerator.log({f"Epoch": (step - 1) / steps_per_epoch}, step=step+1)
            
            if args.patience:
                if patience_counter >= patience * eval_per:
                    print(f"Early stopping triggered at step {step+1}")
                    break
                        
        accelerator.end_training()

def run_train_grad_transformer_from_config(cfg: TrainGradTransformerConfig | None = None):
    if cfg is None:
        raise ValueError("cfg is required")

    args = argparse.Namespace(
        small_model_path=cfg.small_model_path,
        large_model_path=cfg.large_model_path,
        model_pairs_config=cfg.model_pairs_config,
        num_noisy_samples=cfg.num_noisy_samples,
        num_samples_per_val_dataset=cfg.num_samples_per_val_dataset,
        split=cfg.split,
        base_model_path=cfg.base_model_path,
        freeze_t5=cfg.freeze_t5,
        l_out=cfg.l_out,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        clip=cfg.clip,
        eval_per=cfg.eval_per,
        patience=cfg.patience,
    )

    main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an Embedding2EmbeddingT5 model to map small-model gradients to large-model gradients."
    )

    parser.add_argument("--small_model_path", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--large_model_path", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--model_pairs_config", type=str, default="configs/model_pairs_config.json")
    parser.add_argument("--num_noisy_samples", type=int, default=20)
    parser.add_argument("--num_samples_per_val_dataset", type=int, default=None)
    parser.add_argument("--split", type=float, default=0.8)

    parser.add_argument("--base_model_path", type=str, default="google/flan-t5-large")
    parser.add_argument("--freeze_t5", action="store_true")
    parser.add_argument("--l_out", type=int, default=40)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--eval_per", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_path", type=str, default=os.path.join(project_root, "saves", "Qwen-3B-7B.pt"))
    parser.add_argument("--save_fname", type=str, default=None)

    args = parser.parse_args()
    main(args)

