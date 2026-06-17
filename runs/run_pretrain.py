import os
from argparse import Namespace
from data.GTAData import GTADataset
from rich.console import Console
from utils.logging import log_rich, log_build_start


def run(args: Namespace, console: Console):

    log_rich(
        message="Running GTA training process for pretraining models",
        console=console,
    )

    # Build dataset
    dataset = GTADataset(
        args=args,
        model_dict_path=args.path_to_model_dict,
        save_dir=args.data_save_dir,
        console=console,
        num_samples=args.num_samples_per_train_dataset,
    )
    if not dataset.processed:
        dataset.preprocessing()
