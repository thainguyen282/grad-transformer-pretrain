import argparse
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def add_train_option_args(parser):
    parser.add_argument(
        "--train_option",
        type=str,
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="Which stage to run from pretrained/main.py",
    )

def add_general_args(parser):
    parser.add_argument(
        '--pretrain_model_path', type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help='Hugging Face model id or path for pretraining.',
    )
    parser.add_argument(
        '--loraxs_rank', type=int, default=256,
        help='LoRA-XS adapter rank.',
    )
    parser.add_argument(
        '--base_model_path', type=str, default="google/flan-t5-large",
        help='Base T5 model id or path (grad transformer / encoder side).',
    )
    parser.add_argument(
        '--l_out', type=int, default=40,
        help='Output sequence length (T5 decoder).',
    )
    parser.add_argument(
        '--dim', type=int, default=458752,
        help='Flattened delta / embedding dimension.',
    )
    parser.add_argument(
        '--num_noisy_samples', type=int, default=100,
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
        '--save_dir', type=str, default="saves",
        help='Directory for checkpoints and artifacts.',
    )

def add_update_vector_args(parser):
    parser.add_argument(
        '--model_config', type=str, default=os.path.join(PROJECT_ROOT, "configs", "model_1.json"),
        help='JSON listing models; each entry needs "path" and "noise_boundary".',
    )
    parser.add_argument(
        '--delta_noise_std', type=float, default=0.1,
        help='Std of Gaussian noise added to the merged delta tensor.',
    )
    parser.add_argument(
        '--merge_option', type=str, choices=['by_layer', 'flatten'], default='by_layer',
        help='How to structure merged update vectors: per-layer or flattened.',
    )

def add_train_grad_transformer_args(parser):
    parser.add_argument(
        '--model_pairs_config', type=str, default=os.path.join(PROJECT_ROOT, "configs", "model_pairs_config.json"),
        help='JSON config for (small, large) model pairs used in training.',
    )
    parser.add_argument(
        "--split", type=float, default=0.95,
        help='Train fraction of the dataset (0–1).',
    )
    parser.add_argument(
        '--num_epochs', type=int, default=3,
        help='Number of training epochs.',
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Minibatch size.',
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early-stopping patience (epochs without improvement).',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help='Learning rate.',
    )
    parser.add_argument(
        '--freeze_t5', action='store_true',
        help='Freeze T5 weights; train only other parameters.',
    )
    parser.add_argument(
        '--eval_per', type=int, default=10,
        help='Run validation every N epochs.',
    )
    parser.add_argument(
        '--save_path', type=str, default="saves/Qwen-1.5B-3B-7B.pt",
        help='Path to save the trained grad-transformer checkpoint.',
    )

    
def add_rvkd_inference_pretrain_args(parser):
    parser.add_argument(
        '--small_model_gradient_path', type=str, default="saves/Qwen2.5-1.5B-Instruct/gradients_base.pt",
        help='Path to small-model gradient tensor (.pt).',
    )
    parser.add_argument(
        '--large_model_path', type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help='Hugging Face id or path to the large teacher model.',
    )
    parser.add_argument(
        '--transform_model_path', type=str, default="saves/Qwen-1.5B-3B-7B.pt",
        help='Path to the trained grad-transformer weights.',
    )

def add_evaluate_args(parser):
    parser.add_argument(
        '--dataset_path', type=str, default="deepmind/aqua_rat",
        help='Hugging Face dataset name or path for evaluation.',
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
