## Gradient-transformer pretrain (short guide)

This folder contains a 3‑step pipeline:

1. **Generate update vectors**
2. **Train the gradient transformer**
3. **RVKD inference pretrain**

The easiest way is to run everything with defaults via `main.py`.

---

## 1. Easiest way: run full pipeline

From the `pretrained/` repo root:

```bash
python main.py
```

`main.py`:

- parses all arguments from `config.py`
- then calls, in order:
  - `src/generate_update_vector.py`
  - `src/train_grad_transformer.py`
  - `src/rvkd_inference_pretrain.py`

Override any default by passing the same CLI flags you would give to the individual scripts (they are all defined in `config.py`).

---

## 2. Running Process

**General/Shared arguments**

- `--loraxs_rank` (int, default `256`): LoRA‑XS rank (also sets SVD rank).
- `--model_name` (str, default `qwen`): model family name for layer parsing.
- `--base_model_path` (str, default `google/flan-t5-large`): base T5 model (indirectly used via LoRA‑XS tools).
- `--seed` (int, default `42`), `--print_layers` (flag): reproducibility and optional verbose layer printing.
- `--quantize` (store true): quantize model or not
- `--save_dir` (str, default `saves`) directory for all save run


Step 1 - generate update vectors(`src/generate_update_vector.py`)
Runs LoRA‑XS reconstruction and saves noisy update vectors under `saves/…`.
**Core arguments**

- `--model_config` (str, default `configs/model.json`): JSON with a `models` list, each item containing:
  - `path`: model ID / path
  - `noise_boundary`: upper bound for noise scale
- `--merge_option` (`by_layer` | `flatten`, default `by_layer`): structure of the merged update tensor.
- `--delta_noise_std` (float, default `0.1`): Gaussian noise std added to the merged delta.
- `--num_noisy_samples` (int, default `100`): number of noisy variants per model.
---

Step 2 – train gradient transformer (`src/train_grad_transformer.py`)

Trains `Embedding2EmbeddingT5` to map small‑model gradients to large‑model gradients using the saved noisy update vectors.

**Dataset / pairing arguments**

- `--model_pairs_config` (str, default `configs/model_pairs_config.json`): JSON file with a `model_pairs` list, each item:
  - `small_model_path`
  - `large_model_path`
- `--num_noisy_samples` (int, default `100`): must match Step 1; used to construct all pairwise combinations.
- `--split` (float, default `0.8`): train/validation split ratio.

**Model shape / general arguments**

- `--dim` (int, default `458752`): gradient embedding dimension (input/output).
- `--l_out` (int, default `64`): output sequence length (number of layers).
- `--base_model_path` (str, default `google/flan-t5-large`): T5 backbone for the transformer.
- `--freeze_t5` (flag): if set, only train new layers on top of T5.
- `--loraxs_rank`, `--seed`: same meaning as in Step 1.

**Training arguments**

- `--num_epochs` (int, default `3`): training epochs.
- `--batch_size` (int, default `16`): batch size.
- `--lr` (float, default `1e-4`): learning rate.
- `--eval_per` (int, default `10`): evaluation interval in steps.
- `--patience` (int, default `10`): early stopping patience, measured in `eval_per` steps.
- `--save_path` (str, default `saves/Qwen-1B-3B-7B.pt`): where the best checkpoint is written.

---

Step 3 – RVKD inference pretrain (`src/rvkd_inference_pretrain.py`)

Loads the trained transformer and predicts large‑model LoRA‑XS updates from small‑model gradients, then applies them to the large model and saves the resulting weights.

**Core arguments**

- `--small_model_gradient_path` (str): path to the small‑model LoRA‑XS gradient tensor (e.g. `saves/<small_model>/gradients_base.pt` or a similar file from Step 1).
- `--large_model_path` (str): Hugging Face ID / path of the large target model.
- `--transform_model_path` (str): checkpoint from Step 2 (e.g. `saves/Qwen-1B-3B-7B.pt`).
- `--save_path` (str): path where the adapted large model weights will be saved (`state_dict`).

**Shape / decoding arguments**

- `--dim` (int, default `458752`): gradient embedding dimension, must match Step 2.
- `--l_out` (int, default `40`): number of output layers to predict.
- `--loraxs_rank` (int, default `256`): LoRA rank for the large model adapter.
- `--base_model_path` (str, default `google/flan-t5-large`): T5 backbone used inside the transformer.
---
