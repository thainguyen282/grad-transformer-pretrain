import os
from argparse import Namespace
from data.GTAData import GTADataset
from rich.console import Console
from utils.logging import log_rich, log_build_start


def run(args: Namespace, console: Console):

    log_rich(
        message="Running GTA training process for pretraining models",
        console=console,
        new_line=True,
    )

    # Build dataset
    dataset = GTADataset(
        args=args,
        model_dict_path=args.path_to_model_dict,
        save_dir=args.data_save_dir,
        console=console,
    )
    if not dataset.processed:
        dataset.preprocessing()
