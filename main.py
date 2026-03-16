import os
from dataclasses import replace

from pretrained.config import (
    DEFAULT_GENERATE_UPDATE_VECTOR_CONFIG,
    GenerateUpdateVectorConfig,
    DEFAULT_TRAIN_GRAD_TRANSFORMER_CONFIG,
    TrainGradTransformerConfig,
    DEFAULT_RVKD_INFERENCE_PRETRAIN_CONFIG,
    RvkdInferencePretrainConfig,
)

from pretrained.src.generate_update_vector import run_generate_update_vector_from_config
from pretrained.src.train_grad_transformer import run_train_grad_transformer_from_config
from pretrained.src.rvkd_inference_pretrain import run_rvkd_inference_pretrain_from_config


def run_pipeline(
    gen_cfg: GenerateUpdateVectorConfig | None = None,
    train_cfg: TrainGradTransformerConfig | None = None,
    inference_cfg: RvkdInferencePretrainConfig | None = None,
):
    if gen_cfg is None:
        gen_cfg = DEFAULT_GENERATE_UPDATE_VECTOR_CONFIG
    print("\n=== Step 1: Generating update vectors ===")
    run_generate_update_vector_from_config(gen_cfg)

    if train_cfg is None:
        train_cfg = DEFAULT_TRAIN_GRAD_TRANSFORMER_CONFIG
    print("\n=== Step 2: Training gradient transformer ===")
    run_train_grad_transformer_from_config(train_cfg)

    if inference_cfg is None:
        inference_cfg = DEFAULT_RVKD_INFERENCE_PRETRAIN_CONFIG
    if inference_cfg.transform_model_path is None:
        inference_cfg = replace(inference_cfg, transform_model_path=train_cfg.save_path)

    print("\n=== Step 3: RVKD inference pretrain ===")
    run_rvkd_inference_pretrain_from_config(inference_cfg)


if __name__ == "__main__":
    run_pipeline()

