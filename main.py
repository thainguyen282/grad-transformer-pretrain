from config import parse_args
from src.generate_update_vector import generate_update_vector
from src.rvkd_inference_pretrain import run_rvkd_inference_pretrain
from src.train_grad_transformer import run_train_grad_transformer
from config import print_args
import random
import torch
import numpy as np
from accelerate.utils import set_seed
from rich.console import Console
import os
console = Console(highlight=False)

def run_pipeline(args, console):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if args.train_option == "pretrain":
        os.makedirs(args.save_dir, exist_ok=True)
        if args.pretrain_stage in ("1", "all"):
            # Step 1: Generate update vectors
            generate_update_vector(args, console)
        if args.pretrain_stage in ("2", "all"):
            # Step 2: Train gradient transformer
            run_train_grad_transformer(args, console)
        if args.pretrain_stage in ("3", "all"):
            # Step 3: RVKD inference pretrain
            run_rvkd_inference_pretrain(args, console)
    else:
        raise ValueError(f"Unsupported train_option")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    run_pipeline(args, console)
