import os
import torch
from argparse import Namespace
from data.GTAData import GTADataset, GTA_collate_fn
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

    # Build dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: GTA_collate_fn(batch, padding_size=args.padding_size),
    )

    if args.debug:
        # take one batch and print the shapes of the tensors
        batch = dataset[0]
        source_model_dicts, target_model_dicts = batch
        log_rich(
            message=f"Source model dicts: {source_model_dicts.keys()}",
            console=console,
        )
        log_rich(
            message=f"Target model dicts: {target_model_dicts.keys()}",
            console=console,
        )
        for key in source_model_dicts.keys():
            log_rich(
                message=f"Source model dicts[{key}]: {source_model_dicts[key].shape}",
                console=console,
            )
        for key in target_model_dicts.keys():
            log_rich(
                message=f"Target model dicts[{key}]: {target_model_dicts[key].shape}",
                console=console,
            )
