import argparse
import os
from rich.console import Console
from rich.table import Table

def add_train_option_args(parser):
    parser.add_argument(
        "--train_option",
        type=str,
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="Which stage to run from pretrained/main.py",
    )
    parser.add_argument(
        "--pretrain_stage",
        type=str,
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which step(s) to run when --train_option=pretrain.",
    )

def add_general_args(parser):
    parser.add_argument(
        '--pretrain_model_path', type=str, default="Qwen2.5-7B-Instruct",
        help='Hugging Face model id or path for pretraining.',
    )
    parser.add_argument(
        '--lora_rank', type=int, default=256,
        help='LoRA-XS adapter rank.',
    )
    parser.add_argument(
        '--base_model_path', type=str, default="google/flan-t5-large",
        help='Base T5 model id or path (grad transformer / encoder side).',
    )
    parser.add_argument(
        '--l_out', type=int, default=288,
        help='Output sequence length (T5 decoder).',
    )
    parser.add_argument(
        '--dim', type=int, default=917504,
        help='Flattened delta / embedding dimension.',
    )
    parser.add_argument(
        '--num_noisy_samples', type=int, default=4,
        help='Number of noisy variants to generate per model.',
    )
    parser.add_argument(
        '--quantize', action='store_true',
        help='Enable quantization where supported.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility.',
    )
    parser.add_argument(
        '--model_name', type=str, default="qwen",
        help='Short name tag for logging and outputs.',
    )
    parser.add_argument(
        '--print_layers', action='store_true',
        help='Print model layer names and exit.',
    )
    parser.add_argument(
        '--save_dir', type=str, default="saves/gradients",
        help='Directory for checkpoints and artifacts.',
    )

def add_update_vector_args(parser):
    parser.add_argument(
        '--model_config', type=str, default=os.path.join("configs", "model_test_2048.json"),
        help='JSON listing models; each entry needs "path" and "noise_boundary".',
    )
    parser.add_argument(
        '--merge_option', type=str, choices=['by_layer', 'flatten'], default='by_layer',
        help='How to structure merged updated vectors: per-layer or flattened.',
    )
    parser.add_argument(
        '--std', type=float, default=0.1, 
        help='Standard deviation of gaussian noise to sample more data'
    )

def add_train_grad_transformer_args(parser):
    parser.add_argument(
        '--model_pairs_config', type=str, default=os.path.join("configs", "model_pairs_config_smallscale.json"),
        help='JSON config for (small, large) model pairs used in training.',
    )
    parser.add_argument(
        '--exclude_pretrain', action='store_true', 
        help='option to include target model in training set'
    )
    parser.add_argument(
        "--split", type=float, default=0.9,
        help='Train fraction of the dataset.',
    )
    parser.add_argument(
        '--num_epochs', type=int, default=3,
        help='Number of training epochs.',
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Minibatch size.',
    )
    parser.add_argument(
        '--patience', type=int, default=None,
        help='Early-stopping patience (epochs without improvement).',
    )
    parser.add_argument(
        '--lr', type=float, default=3e-5,
        help='Learning rate.',
    )
    parser.add_argument(
        '--freeze_t5', action='store_true',
        help='Freeze T5 weights; train only other parameters.',
    )
    parser.add_argument(
        '--eval_per', type=int, default=2,
        help='Run validation every N epochs.',
    )
    parser.add_argument(
        '--save_path', type=str, default="Qwen-1.5B-3B-7B.pt",
        help='Path to save the trained grad-transformer checkpoint.',
    )

    
def add_rvkd_inference_pretrain_args(parser):
    parser.add_argument(
        '--small_model_gradient_path', type=str, default="saves/Qwen/Qwen2.5-1.5B-Instruct/gradients_base.pt",
        help='Path to small-model gradient tensor (.pt).',
    )
    parser.add_argument(
        '--large_model_path', type=str, default="Qwen/Qwen2.5-32B-Instruct",
        help='Hugging Face id or path to the large teacher model.',
    )
    parser.add_argument(
        '--transform_model_path', type=str, default="saves/Qwen-1.5B-3B-7B-14B-32B.pt",
        help='Path to the trained grad-transformer weights.',
    )
    parser.add_argument(
        '--loraxs_model_path', type=str, default="saves/Qwen/Qwen2.5-32B-Instruct/random_init_lora_ab.pt", 
        help='Path to saved lora A, lora B of random initialized model used to constuct pretrain weight with r'
    )
    parser.add_argument(
        '--num_init_samples',
        type=int,
        default=10,
        help='Number of independent base-model re-inits (seed, SVD state); LoRA-XS predictions are averaged.',
    )
    parser.add_argument(
        '--transform_mc_draws',
        type=int,
        default=1,
        help='Monte Carlo draws through the grad-transformer (single T5 on GPU); combine with --transform_input_noise_std.',
    )
    parser.add_argument(
        '--transform_input_noise_std',
        type=float,
        default=0.0,
        help='Std of Gaussian noise added to small-model gradient embedding before the transformer (0 = identical draws unless num_init_samples>1).',
    )
    parser.add_argument(
        '--avg_random_init_samples',
        type=int,
        default=1,
        help='If >1, build the large base by averaging K independent random re-inits of q/k/v/o/mlps (fp32 mean), then one fresh pretrained load with those means.',
    )
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        help=(
            'Module name suffixes for LoRA and for random re-init (default: q/k/v/o/gate/up/down proj). '
            'Example: include lm_head to match step-1 update vectors that use lm_head.'
        ),
    )

def add_evaluate_args(parser):
    parser.add_argument(
        '--dataset_path', type=str, default="deepmind/aqua_rat",
        help='Hugging Face dataset name or path for evaluation.',
    )
    parser.add_argument(
        '--chat_template_path', type=str, default="/project/phan/tqn/grad-transformer/templates/alpaca", 
        help='chat tempalte formatted in json file'
    )
    parser.add_argument(
        '--ans_template', type=str, default="####",
        help='Substring marking the answer span in model output.',
    )
    parser.add_argument(
        '--ref_template', type=str, default="####",
        help='Substring marking the reference answer in labels.',
    )
    parser.add_argument(
        '--max_length_eval', type=int, default=4096,
        help='Max input length (tokens) for evaluation.',
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=4096,
        help='Max new tokens to generate per example.',
    )

def add_finetune_args(parser):
    parser.add_argument("--model_path", type=str, default=None, help="Base model to fine-tune (e.g. Qwen/...).")

    parser.add_argument("--train_indices_path", nargs="+", type=str, default=None)
    parser.add_argument("--test_indices_path", nargs="+", type=str, default=None)

    parser.add_argument("--num_samples_per_train_dataset", type=int, default=None)
    parser.add_argument("--num_samples_per_val_dataset", type=int, default=None)

    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", action="store_true")

    parser.add_argument("--finetune_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_lm_head", action="store_true")
    parser.add_argument("--tune_option", type=str, default=None)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_fname", type=str, default=None)

    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)

    parser.add_argument("--finetune_split", type=float, default=None)
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--noise_multiplier", type=float, default=0.0)
    parser.add_argument("--target_delta", type=float, default=None)
    parser.add_argument("--split_option", type=str, default="none")

    # Currently only used for logging / future extensions.
    parser.add_argument("--evaluate_every", type=int, default=1000)

console = Console()

ARG_GROUPS = {
    "Train Option": add_train_option_args,
    "General": add_general_args,
    "Update Vector": add_update_vector_args,
    "Grad Transformer Training": add_train_grad_transformer_args,
    "RVKD Inference (Pretrain)": add_rvkd_inference_pretrain_args,
    "Evaluation": add_evaluate_args,
    "Finetuning": add_finetune_args,
}


def extract_group_keys(add_fn):
    """Create a temp parser to extract arg names from a function."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    add_fn(parser)
    return [a.dest for a in parser._actions if a.dest != "help"]


def print_args(args):
    data = vars(args)
    used_keys = set()

    for group_name, fn in ARG_GROUPS.items():
        keys = extract_group_keys(fn)

        table = Table(title=group_name)
        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        has_rows = False
        for k in keys:
            if k in data:
                table.add_row(k, str(data[k]))
                used_keys.add(k)
                has_rows = True

        if has_rows:
            console.print(table)

    # catch anything unexpected
    remaining = [k for k in data if k not in used_keys]
    if remaining:
        table = Table(title="Other")
        table.add_column("Argument", style="cyan")
        table.add_column("Value", style="magenta")

        for k in remaining:
            table.add_row(k, str(data[k]))

        console.print(table)


def parse_args():
    parser = argparse.ArgumentParser()
    arg_grp = parser.add_argument_group(title="Model generation")
    add_train_option_args(arg_grp)
    add_general_args(arg_grp)
    add_update_vector_args(arg_grp)
    add_train_grad_transformer_args(arg_grp)
    add_rvkd_inference_pretrain_args(arg_grp)
    add_evaluate_args(arg_grp)
    add_finetune_args(arg_grp)
    return parser.parse_args()
