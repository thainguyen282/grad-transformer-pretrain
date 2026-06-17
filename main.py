import os
from config import parse_args
from rich.console import Console
from utils.utils import seed_everything
from utils.logging import log_rich, print_args
from runs.run_pretrain import run as run_pretrain

# from src.generate_update_vector import generate_update_vector
# from src.rvkd_inference_pretrain import run_rvkd_inference_pretrain
# from src.train_grad_transformer import run_train_grad_transformer


def run_pipeline(args, console):

    if args.mode == "train":
        run_pretrain(args, console)


if __name__ == "__main__":
    console = Console(highlight=False)
    args = parse_args()
    seed_everything(args.seed)
    print_args(args, console=console)
    run_pipeline(args, console)
