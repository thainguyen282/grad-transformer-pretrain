import argparse


def add_general_args(parser):
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="General mode: train or test (evaluate). 'train' runs the pretraining pipeline; 'test' runs evaluation on a specified dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default_exp",
        help="Name for the experiment; used in logging and saving artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )


def add_training_data_args(parser):
    parser.add_argument(
        "--path_to_model_dict",
        type=str,
        default="Qwen2.5-7B-Instruct",
        help="Json file listing models meta information, including model path and noise boundary for generating update vector.",
    )
    parser.add_argument(
        "--max_layer_out",
        type=int,
        default=288,
        help="Maximum number of layers (out-tokens) for GTA to generate update vector",
    )
    parser.add_argument(
        "--padding_size",
        type=int,
        default=2048,
        help="Size to pad the update vector to (square padding_size x padding_size); should be >= max_layer_out and >= max model width in model_dict.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable quantization where supported for development process.",
    )
    parser.add_argument(
        "--data_save_dir",
        type=str,
        default="./results/data",
        help="Directory to save generated training data (update vectors, model info, etc.).",
    )
    parser.add_argument(
        "--leave_out_target_model",
        action="store_true",
        help="Option to exclude the target model from the GTA training set (i.e. only train on other models' update vectors).",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.01,
        help="Scale of noise to add to the GTA-predicted update vector for augmentation.",
    )
    parser.add_argument(
        "--shuffle_data",
        action="store_true",
        help="Whether to shuffle the training data (update vectors) during GTA training.",
    )
    parser.add_argument(
        "--meta_embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Hugging Face model name for generating meta-embeddings as input features for GTA.",
    )


def add_testing_data_args(parser):
    parser.add_argument(
        "--testing_source_model_path",
        type=str,
        default=None,
        help="Hugging Face id or path to the source model to evaluate (e.g. Qwen/...). This should not be None for testing process.",
    )
    parser.add_argument(
        "--testing_target_model_path",
        type=str,
        default=None,
        help="Hugging Face id or path to the target model to evaluate (e.g. Qwen/...). This should not be None for testing process.",
    )


def add_GTA_args(parser):
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Total batch size across all devices (will be divided by number of GPUs).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--freeze_GTA_LM_module",
        action="store_true",
        help="Whether to freeze the LM module of GTA during training and only update the CNN laers.",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=100,
        help="Run validation every N steps.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./results/GTA_model",
        help="Path to save the trained GTA model.",
    )


def add_evaluate_args(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="deepmind/aqua_rat",
        help="Hugging Face dataset name or path for evaluation.",
    )
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default="/project/phan/tqn/grad-transformer/templates/alpaca",
        help="chat tempalte formatted in json file",
    )
    parser.add_argument(
        "--ans_template",
        type=str,
        default="####",
        help="Substring marking the answer span in model output.",
    )
    parser.add_argument(
        "--ref_template",
        type=str,
        default="####",
        help="Substring marking the reference answer in labels.",
    )
    parser.add_argument(
        "--max_length_eval",
        type=int,
        default=4096,
        help="Max input length (tokens) for evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Max new tokens to generate per example.",
    )


def add_finetune_args(parser):
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Base model to fine-tune (e.g. Qwen/...).",
    )

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


ARG_GROUPS = {
    "General": add_general_args,
    "Training Data": add_training_data_args,
    "Testing Data": add_testing_data_args,
    "GTA Training": add_GTA_args,
    "Evaluation": add_evaluate_args,
    "Fine-tuning": add_finetune_args,
}


def parse_args():
    parser = argparse.ArgumentParser()
    arg_grp = parser.add_argument_group(title="Model generation")
    for add_fn in ARG_GROUPS.values():
        add_fn(arg_grp)
    return parser.parse_args()
