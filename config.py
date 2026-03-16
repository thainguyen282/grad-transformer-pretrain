import os
from dataclasses import dataclass


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class GenerateUpdateVectorConfig:

    model_config: str = os.path.join("configs", "model_1.json")
    seed: int = 42
    model_name: str = "qwen"
    merge_option: str = "by_layer"
    print_layers: bool = False
    loraxs_rank: int = 256
    delta_noise_std: float = 0.1
    num_noisy_samples: int = 100

DEFAULT_GENERATE_UPDATE_VECTOR_CONFIG = GenerateUpdateVectorConfig()

@dataclass
class TrainGradTransformerConfig:

    model_dim: int = 458752 # loraxs_rank ^ 2 * 7 
    model_pairs_config: str = "configs/model_pairs_config.json"
    num_noisy_samples: int = 100
    num_samples_per_val_dataset: int = 100
    split: float = 0.8
    base_model_path: str = "google/flan-t5-large"
    freeze_t5: bool = False
    l_out: int = 40
    num_epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-4
    clip: float | None = None
    eval_per: int | None = None
    patience: int | None = None
    seed: int = 42
    save_path: str = os.path.join(PROJECT_ROOT, "saves", "Qwen-1B-3B-7B.pt")
    save_fname: str | None = None


DEFAULT_TRAIN_GRAD_TRANSFORMER_CONFIG = TrainGradTransformerConfig()

@dataclass
class RvkdInferencePretrainConfig:

    save_fname: str | None = None
    small_model_gradient_path: str | None = None
    large_model_path: str | None = None
    transform_model_path: str | None = None
    seed: int | None = None
    loraxs_rank: int = 256
    num_samples_per_val_dataset: int | None = None
    max_length_eval: int | None = None
    max_new_tokens: int | None = None
    init_small_path: str | None = None
    init_large_path: str | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    l_out: int | None = None
    small_tune_option: str | None = None
    large_tune_option: str | None = None
    base_model_path: str | None = None
    quantize: bool = False
    save_path: str | None = None


DEFAULT_RVKD_INFERENCE_PRETRAIN_CONFIG = RvkdInferencePretrainConfig()

