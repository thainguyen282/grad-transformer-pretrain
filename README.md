# grad-transformer-pretrain

Instructions for running the gradient-transformer pretraining pipeline: generate update vectors, train the gradient transformer, and run RVKD inference pretrain.

---

## Quick run (full pipeline)

Run all three steps in order with default configs:

```bash
# From the repository root (grad-transformer-pretrain/)
python main.py
```

`main.py` runs:

1. **Step 1:** Generate update vectors (noisy delta LoRA-XS tensors).
2. **Step 2:** Train the gradient transformer (map small-model → large-model gradients).
3. **Step 3:** RVKD inference pretrain using the trained transformer.

Defaults are defined in `config.py`. You can change behavior by editing that file or by running each step separately with the scripts below.

---

## Running each step / file

You can run each stage on its own. Use the **repository root** as the current directory and ensure it is on `PYTHONPATH` so that `config` and `src` import correctly.

### 1. Generate update vectors

Builds and saves noisy delta LoRA-XS tensors for the models listed in a config JSON.

**File:** `src/generate_update_vector.py`

**From repo root:**

```bash
python src/generate_update_vector.py [OPTIONS]
```

**Options (examples):**

| Option | Default | Description |
|--------|--------|-------------|
| `--model_config` | `configs/model_1.json` | Path to model config JSON (each entry: `path`, `noise_boundary`). |
| `--seed` | `42` | Random seed. |
| `--model_name` | `qwen` | Model name (e.g. for layer ordering). |
| `--merge_option` | `by_layer` | How to merge: `by_layer` or `flatten`. |
| `--print_layers` | (flag) | Print layer info. |
| `--loraxs_rank` | `256` | LoRA-XS rank. |
| `--delta_noise_std` | `0.1` | Std of Gaussian noise on merged delta. |
| `--num_noisy_samples` | `20` | Noisy variants per model. |

**Example:**

```bash
python src/generate_update_vector.py --model_config configs/model_1.json --num_noisy_samples 100 --loraxs_rank 256
```

---

### 2. Train gradient transformer

Trains the Embedding2EmbeddingT5 model to map small-model gradients to large-model gradients.

**File:** `src/train_grad_transformer.py`

**From repo root:**

```bash
python src/train_grad_transformer.py [OPTIONS]
```

**Options (examples):**

| Option | Default | Description |
|--------|--------|-------------|
| `--small_model_path` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Small model id/path. |
| `--large_model_path` | `Qwen/Qwen2.5-Coder-3B-Instruct` | Large model id/path. |
| `--model_pairs_config` | `configs/model_pairs_config.json` | JSON listing (small, large) pairs; overrides single pair if used. |
| `--num_noisy_samples` | `20` | Noisy samples per pair. |
| `--num_samples_per_val_dataset` | (optional) | Val set size. |
| `--split` | `0.8` | Train/val split ratio. |
| `--base_model_path` | `google/flan-t5-large` | T5 backbone for the transformer. |
| `--freeze_t5` | (flag) | Freeze T5 weights. |
| `--l_out` | `40` | Output sequence length. |
| `--num_epochs` | `3` | Training epochs. |
| `--batch_size` | `16` | Batch size. |
| `--lr` | `1e-4` | Learning rate. |
| `--clip` | (optional) | Gradient clipping max norm. |
| `--eval_per` | (optional) | Evaluation interval (steps). |
| `--patience` | (optional) | Early stopping patience (in eval steps). |
| `--seed` | `42` | Random seed. |
| `--save_path` | `saves/Qwen-3B-7B.pt` | Where to save the trained transformer. |
| `--save_fname` | (optional) | Extra filename suffix. |

**Example:**

```bash
python src/train_grad_transformer.py --model_pairs_config configs/model_pairs_config.json --num_epochs 5 --save_path saves/my_transform.pt
```

Gradient data is loaded from paths derived from `model_pairs_config` / model paths (e.g. under a shared `model_load_path` with `gradients_base.pt` and `model_1.pt`, …). Ensure Step 1 has been run so those files exist.

---

### 3. RVKD inference pretrain

Uses the trained gradient transformer for inference-time pretrain (RVKD).

**File:** `src/rvkd_inference_pretrain.py`

**From repo root:**

```bash
python src/rvkd_inference_pretrain.py --save_fname FNAME --small_model_gradient_path PATH --large_model_path PATH --transform_model_path PATH [OPTIONS]
```

**Important options:**

| Option | Description |
|--------|-------------|
| `--save_fname` | Output filename / run identifier. |
| `--small_model_gradient_path` | Path to small-model gradient data. |
| `--large_model_path` | Large model id/path. |
| `--transform_model_path` | Path to the trained transformer checkpoint (e.g. from Step 2). |

**Other options (examples):** `--seed`, `--loraxs_rank`, `--num_samples_per_val_dataset`, `--max_length_eval`, `--max_new_tokens`, `--init_small_path`, `--init_large_path`, `--input_dim`, `--output_dim`, `--l_out`, `--small_tune_option`, `--large_tune_option`, `--base_model_path`, `--quantize`, `--save_path`.

**Example:**

```bash
python src/rvkd_inference_pretrain.py --save_fname run1 --small_model_gradient_path path/to/small_grads --large_model_path Qwen/Qwen2.5-Coder-7B-Instruct --transform_model_path saves/Qwen-3B-7B.pt
```

When you run the full pipeline via `main.py`, Step 3 is called with defaults and `transform_model_path` is set to the checkpoint saved in Step 2.

---

## Utility scripts

These are supporting tools; they are not part of the default pipeline.

| File | Purpose |
|------|--------|
| `utils/convert_gradients.py` | Convert gradients for a model. Requires `--gradients_path`, `--model_name`, `--merge_option`; optional `--start_layer`, `--end_layer`, `--starting_indices_shadow_dataset`, etc. |
| `utils/reinit_target_model.py` | Reinitialize pretrained weights and save the model. Requires `--model_path`, `--output_dir`; optional `--seed`. |
| `utils/transformer.py` | Transformer training entrypoint (used internally by the pipeline). |
| `loraxs_utils/merge_adapter_to_base_model.py` | Merge LoRA adapter into the base model (CLI in that file). |

Run from repo root with `python utils/script_name.py` or `python loraxs_utils/script_name.py` as appropriate, and ensure the repo root is on `PYTHONPATH`.

---

## Config

Default dataclass configs live in **`config.py`**:

- `GenerateUpdateVectorConfig` / `DEFAULT_GENERATE_UPDATE_VECTOR_CONFIG`
- `TrainGradTransformerConfig` / `DEFAULT_TRAIN_GRAD_TRANSFORMER_CONFIG`
- `RvkdInferencePretrainConfig` / `DEFAULT_RVKD_INFERENCE_PRETRAIN_CONFIG`

Adjust these to change defaults for `main.py`, or override behavior by calling the scripts above with explicit arguments.

---

## Order of operations

1. **Generate update vectors** → produces gradient/noisy delta files expected by the trainer.
2. **Train gradient transformer** → produces a `.pt` checkpoint.
3. **RVKD inference pretrain** → uses that checkpoint and gradient paths.

For a single command that does all three with defaults, use **Quick run** with `main.py`.
