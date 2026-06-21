import os
import gc
import json
import torch
import random
from tqdm import tqdm
from argparse import Namespace
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer
from utils.logging import log_rich
from rich.console import Console


class GTADataset(Dataset):

    def __init__(
        self,
        args: Namespace,
        model_dict_path: str,
        num_samples: int,
        save_dir: str,
        console: Console,
        target_model: str = None,
        target_mode: bool = False,
        padding_size: int = 2048,
    ):
        self.num_samples = num_samples
        self.do_shuffle = args.shuffle_data
        self.meta_embedding_model = args.meta_embedding_model
        self.console = console
        self.processed = False
        self.target_model = target_model
        self.target_mode = target_mode
        self.padding_size = padding_size
        self.save_dir = save_dir

        if target_mode:
            assert (
                target_model is not None
            ), "Target model must be specified in target mode"
            self.console.log(
                f"Initializing GTA dataset in target mode with target model {target_model}"
            )

        if not os.path.exists(save_dir):
            log_rich(
                message=f"Creating save directory at: {save_dir}",
                console=console,
                label="INFO",
                newline=True,
            )
            os.makedirs(save_dir, exist_ok=True)

        with open(model_dict_path, "r") as f:
            self.model_dict = json.load(f)

        if self.target_mode:
            assert (
                self.target_model in self.model_dict
            ), f"Target model {self.target_model} not found in model dict"
            self.console.log(
                f"Filtering model dict to only include models smaller than target model {self.target_model}"
            )
            self.model_dict = {
                k: v for k, v in self.model_dict.items() if k != self.target_model
            }
            self.console.log(
                f"Filtered model dict to {len(self.model_dict)} models different from {self.target_model}"
            )

        if os.path.exists(
            os.path.join(
                save_dir,
                self.meta_embedding_model.split("/")[-1],
                "meta_info_dict.json",
            )
        ):
            with open(
                os.path.join(
                    save_dir,
                    self.meta_embedding_model.split("/")[-1],
                    "meta_info_dict.json",
                ),
                "r",
            ) as f:
                self.model_info_dict = json.load(f)
            self.processed = True

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get a source-target model pair for this sample
        source_model_name, target_model_name = self._get_source_target_pair(idx)
        source_model_dict = self._get_model_dict(source_model_name)
        target_model_dict = self._get_model_dict(target_model_name)

        # concate the meta info vector to a tensor, concat the weight to a tensor and create a mask tensor for the weight, and create a boolean tensor to indicate whether the weight has bias for both source and target model
        source_meta_info_vector = source_model_dict["model_meta_info"]
        source_meta_info_tensor = [source_meta_info_vector]
        source_weight_tensors = []
        source_mask_tensors = []
        source_has_bias_tensors = []
        for key in source_model_dict.keys():
            if key == "model_meta_info":
                continue
            source_meta_info_tensor.append(source_model_dict[key]["meta_info_vector"])
            source_weight_tensors.append(source_model_dict[key]["weight"])
            source_mask_tensors.append(source_model_dict[key]["mask"])
            source_has_bias_tensors.append(
                torch.tensor(source_model_dict[key]["has_bias"], dtype=torch.bool)
            )

        source_meta_info_tensor = torch.stack(source_meta_info_tensor)
        source_weight_tensor = torch.stack(source_weight_tensors)
        source_mask_tensor = torch.stack(source_mask_tensors)
        source_has_bias_tensor = torch.stack(source_has_bias_tensors)
        source_tensor_dict = {
            "meta_info": source_meta_info_tensor,  # (num_weights + 1, meta_info_dim)
            "weight": source_weight_tensor,  # (num_weights, padding_size, padding_size)
            "mask": source_mask_tensor,  # (num_weights, padding_size, padding_size)
            "has_bias": source_has_bias_tensor,  # (num_weights,)
        }

        target_meta_info_vector = target_model_dict["model_meta_info"]
        target_meta_info_tensor = [target_meta_info_vector]
        target_weight_tensors = []
        target_mask_tensors = []
        target_has_bias_tensors = []
        for key in target_model_dict.keys():
            if key == "model_meta_info":
                continue
            target_meta_info_tensor.append(target_model_dict[key]["meta_info_vector"])
            target_weight_tensors.append(target_model_dict[key]["weight"])
            target_mask_tensors.append(target_model_dict[key]["mask"])
            target_has_bias_tensors.append(
                torch.tensor(target_model_dict[key]["has_bias"], dtype=torch.bool)
            )

        target_meta_info_tensor = torch.stack(target_meta_info_tensor)
        target_weight_tensor = torch.stack(target_weight_tensors)
        target_mask_tensor = torch.stack(target_mask_tensors)
        target_has_bias_tensor = torch.stack(target_has_bias_tensors)
        target_tensor_dict = {
            "meta_info": target_meta_info_tensor,  # (num_weights + 1, meta_info_dim)
            "weight": target_weight_tensor,  # (num_weights, padding_size, padding_size)
            "mask": target_mask_tensor,  # (num_weights, padding_size, padding_size)
            "has_bias": target_has_bias_tensor,  # (num_weights,)
        }
        return source_tensor_dict, target_tensor_dict

    def _get_model_dict(self, model_name: str) -> dict:

        # get model meta information vector from the preprocessed meta information dict
        model_key = None
        for k, v in self.model_info_dict.items():
            if v["name"] == model_name:
                model_meta_dict_path = v["meta_info_vector_path"]
                noise = v["noise"] if "noise" in v else 0.0
                break
        model_meta_dict = torch.load(model_meta_dict_path)
        # model_meta_dict

        # add the weight to the model meta dict, if the weigh has bias, concatenate the bias into the weight and store a boolean flag to indicate that the bias is concatenated
        model = AutoModel.from_pretrained(model_name).to("cpu")
        for name, param in model.named_parameters(remove_duplicate=False):
            weight_name = name
            is_bias = False

            if "bias" in weight_name:
                weight_name = weight_name.replace(".bias", ".weight")
                is_bias = True

            weight_name = weight_name.replace(".weight", "")
            weight_meta_info_vector_name = f"{weight_name.replace('.', '_')}"
            if weight_meta_info_vector_name in model_meta_dict.keys():
                if is_bias:
                    if isinstance(model_meta_dict[weight_meta_info_vector_name], dict):
                        # Already has weight, concatenate the bias to the existing weight
                        model_meta_dict[weight_meta_info_vector_name]["weight"] = (
                            torch.cat(
                                [
                                    model_meta_dict[weight_meta_info_vector_name][
                                        "weight"
                                    ],
                                    torch.unsqueeze(param.data.cpu(), dim=1),
                                ],
                                dim=1,
                            )
                        )
                        model_meta_dict[weight_meta_info_vector_name]["has_bias"] = True
                    else:
                        model_meta_dict[weight_meta_info_vector_name] = {
                            "meta_info_vector": model_meta_dict[
                                weight_meta_info_vector_name
                            ],
                            "weight": param.data.cpu(),
                            "has_bias": True,
                            "has_weight": False,
                        }
                else:
                    if isinstance(model_meta_dict[weight_meta_info_vector_name], dict):
                        # Already has bias, concatenate the weight to the existing bias
                        model_meta_dict[weight_meta_info_vector_name]["weight"] = (
                            torch.cat(
                                [
                                    param.data.cpu(),
                                    torch.unsqueeze(
                                        model_meta_dict[weight_meta_info_vector_name][
                                            "weight"
                                        ],
                                        dim=1,
                                    ),
                                ],
                                dim=1,
                            )
                        )
                        model_meta_dict[weight_meta_info_vector_name][
                            "has_weight"
                        ] = True
                    else:
                        model_meta_dict[weight_meta_info_vector_name] = {
                            "meta_info_vector": model_meta_dict[
                                weight_meta_info_vector_name
                            ],
                            "weight": param.data.cpu(),
                            "has_bias": False,
                            "has_weight": True,
                        }

        # Padding the weight matrix to the max weight matrix size in the model and create a mask to indicate the original weight size
        for key in model_meta_dict.keys():
            if key == "model_meta_info":
                continue

            weight = model_meta_dict[key]["weight"]
            # add Gaussian noise to the weight if noise_scale > 0
            if noise > 0.0:
                weight += (1 + noise) * torch.randn_like(weight)

            padded_weight = torch.zeros(
                (self.padding_size, self.padding_size), dtype=weight.dtype
            )
            padded_weight[: weight.size(0), : weight.size(1)] = weight
            mask = torch.zeros((self.padding_size, self.padding_size), dtype=torch.bool)
            mask[: weight.size(0), : weight.size(1)] = True
            model_meta_dict[key]["weight"] = padded_weight
            model_meta_dict[key]["mask"] = mask

        return model_meta_dict

    def preprocessing(
        self,
    ) -> None:

        # create a model to extract embeddings for the meta information (e.g. model config details) using the specified meta_embedding_model
        save_path = os.path.join(
            self.save_dir, self.meta_embedding_model.split("/")[-1]
        )
        self.console.log(
            f"Preprocessing GTA dataset using {self.meta_embedding_model} model and saving meta information vectors to {save_path}"
        )
        emb_model = SentenceTransformer(self.meta_embedding_model).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_info_dict = {}

        for model_key in tqdm(
            self.model_dict.keys(), desc="Processing models for GTA dataset"
        ):
            model_name = self.model_dict[model_key]["name"]
            meta_info_path = self._preprocessing_single_model(
                model_name=model_name, emb_model=emb_model, save_path=save_path
            )
            model_info_dict[model_key] = {
                "name": model_name,
                "meta_info_vector_path": meta_info_path,
                "noise": (
                    self.model_dict[model_key]["noise"]
                    if "noise" in self.model_dict[model_key]
                    else 0.0
                ),
            }

        # save the model_info_dict for future reference
        with open(os.path.join(save_path, "meta_info_dict.json"), "w") as f:
            json.dump(model_info_dict, f, indent=4)

        self.model_info_dict = model_info_dict
        self.processed = True

    def _preprocessing_single_model(
        self, model_name: str, emb_model: SentenceTransformer, save_path: str
    ) -> str:

        self.console.log(f"Processing model {model_name} for GTA dataset")
        model_meta_dict = {}
        model_save_name = f"{model_name.split('/')[-1]}_meta_info.pt"
        model_save_dir = os.path.join(save_path, model_save_name)
        meta_info_vector = self._generate_meta_info_vector(
            model_name=model_name, emb_model=emb_model
        )
        model_meta_dict["model_meta_info"] = meta_info_vector

        # Process each layer's weights
        model = AutoModel.from_pretrained(model_name).to("cpu")
        """Traverse each weight by its depth in the model (e.g. layer 0, layer 1, etc.) and create a meta info including:
            - layer depth (normalized by total number of layers)
            - weight matrix size (normalized by max weight matrix size in the model)
            - weight type (e.g., attention k, v, q, mlp, mixture of experts, gated, etc.)
        """
        for name, _ in model.named_parameters(remove_duplicate=False):
            weight_name = name
            if "bias" in weight_name:
                weight_name = weight_name.replace(".bias", ".weight")
            weight_name = weight_name.replace(".weight", "")
            if weight_name in model_meta_dict:
                continue
            meta_info_vector = emb_model.encode(
                weight_name, convert_to_tensor=True
            ).cpu()
            weight_meta_info_vector_name = f"{weight_name.replace('.', '_')}"
            model_meta_dict[weight_meta_info_vector_name] = meta_info_vector

        torch.save(model_meta_dict, model_save_dir)
        self.console.log(
            f"Saved meta information for model {model_name} at {model_save_dir}"
        )
        return model_save_dir

    def _get_meta_information(
        self, source_model_name: str, target_model_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:

        source_meta_info_vector_name = f"{source_model_name.split("/")[-1]}_{self.meta_embedding_model.split("/")[-1]}_meta_info.pt"
        target_meta_info_vector_name = f"{target_model_name.split('/')[-1]}_{self.meta_embedding_model.split('/')[-1]}_meta_info.pt"

        if os.path.exists(os.path.join(self.save_dir, source_meta_info_vector_name)):
            source_meta_info_vector = torch.load(
                os.path.join(self.save_dir, source_meta_info_vector_name)
            )
        else:
            source_meta_info_vector = self._generate_meta_info_vector(source_model_name)
            torch.save(
                source_meta_info_vector,
                os.path.join(self.save_dir, source_meta_info_vector_name),
            )

        if os.path.exists(os.path.join(self.save_dir, target_meta_info_vector_name)):
            target_meta_info_vector = torch.load(
                os.path.join(self.save_dir, target_meta_info_vector_name)
            )
        else:
            target_meta_info_vector = self._generate_meta_info_vector(target_model_name)
            torch.save(
                target_meta_info_vector,
                os.path.join(self.save_dir, target_meta_info_vector_name),
            )

        return source_meta_info_vector, target_meta_info_vector

    def _generate_meta_info_vector(
        self, model_name: str, emb_model: SentenceTransformer
    ) -> torch.Tensor:
        config = AutoConfig.from_pretrained(model_name).to_json_string()
        meta_info_vector = emb_model.encode(config, convert_to_tensor=True).cpu()
        return meta_info_vector

    def _get_source_target_pair(self, idx) -> tuple[str, str]:
        source_model_keys = list(self.model_dict.keys())
        if self.do_shuffle:
            random.shuffle(source_model_keys)
        source_model_key = source_model_keys[idx % len(source_model_keys)]
        source_model_size = self.model_dict[source_model_key]["size"]

        target_model_keys = [
            k
            for k in self.model_dict.keys()
            if (k != source_model_key)
            and self.model_dict[k]["size"] > source_model_size
        ]
        target_model_key = random.choice(target_model_keys)

        source_model_name = self.model_dict[source_model_key]["name"]
        target_model_name = self.model_dict[target_model_key]["name"]
        return source_model_name, target_model_name


def GTA_collate_fn(batch, padding_size: int = 2048):
    """
    Collate function to combine a list of samples into a batch
    Need padding for the number of weights in the source and target models, but the weight matrices are already padded to the same size

    tensor_dict = {
        "meta_info": meta_info_tensor,  # (num_weights + 1, meta_info_dim)
        "weight": weight_tensor,  # (num_weights, padding_size, padding_size)
        "mask": mask_tensor,  # (num_weights, padding_size, padding_size)
        "has_bias": has_bias_tensor,  # (num_weights,)
    }

    create attention masks
    """

    souce_model_dict_final = {
        "meta_info": [],
        "meta_info_attn_mask": [],
        "weight": [],
        "mask": [],
        "has_bias": [],
        "weight_attn_mask": [],
    }
    target_model_dict_final = {
        "meta_info": [],
        "meta_info_attn_mask": [],
        "weight": [],
        "mask": [],
        "has_bias": [],
        "weight_attn_mask": [],
    }

    # take the max number of weights in the source and target models for padding
    source_max_num_weights = 0
    target_max_num_weights = 0

    for sample in batch:
        source_model_dict, target_model_dict = sample
        source_max_num_weights = max(
            source_max_num_weights, source_model_dict["weight"].size(0)
        )
        target_max_num_weights = max(
            target_max_num_weights, target_model_dict["weight"].size(0)
        )

    for sample in batch:
        source_model_dict, target_model_dict = sample
        source_return_dict = _process_one_dict(
            source_model_dict, source_max_num_weights, padding_size
        )
        for key in source_return_dict.keys():
            souce_model_dict_final[key].append(
                source_return_dict[key]
            )  # (batch_size, max_num_weights + 1, meta_info_dim)

        target_return_dict = _process_one_dict(
            target_model_dict, target_max_num_weights, padding_size
        )
        for key in target_return_dict.keys():
            target_model_dict_final[key].append(
                target_return_dict[key]
            )  # (batch_size, max_num_weights + 1, meta_info_dim)

    for key in souce_model_dict_final.keys():
        souce_model_dict_final[key] = torch.cat(souce_model_dict_final[key], dim=0)
        target_model_dict_final[key] = torch.cat(target_model_dict_final[key], dim=0)

    return souce_model_dict_final, target_model_dict_final


def _process_one_dict(
    model_dict: dict, max_num_weights: int, padding_size: int
) -> dict:

    return_dict = {
        "meta_info": [],
        "meta_info_attn_mask": [],
        "weight": [],
        "mask": [],
        "has_bias": [],
        "weight_attn_mask": [],
    }

    for key in model_dict.keys():
        if key == "meta_info":
            if model_dict[key].size(0) < max_num_weights + 1:
                pad_size = max_num_weights + 1 - model_dict[key].size(0)
                pad_tensor = torch.zeros(
                    (pad_size, model_dict[key].size(1)), dtype=model_dict[key].dtype
                )
                return_dict[key].append(
                    torch.cat([pad_tensor, model_dict[key]], dim=0).unsqueeze(0)
                )
                return_dict["meta_info_attn_mask"].append(
                    torch.cat(
                        [
                            torch.zeros((pad_size,), dtype=torch.bool),
                            torch.ones((model_dict[key].size(0),), dtype=torch.bool),
                        ],
                        dim=0,
                    ).unsqueeze(0)
                )  # (1, max_num_weights + 1, meta_info_dim)
        elif key == "weight":
            if model_dict[key].size(0) < max_num_weights:
                pad_size = max_num_weights - model_dict[key].size(0)
                pad_tensor = torch.zeros(
                    (pad_size, padding_size, padding_size), dtype=model_dict[key].dtype
                )
                return_dict[key].append(
                    torch.cat([pad_tensor, model_dict[key]], dim=0).unsqueeze(0)
                )
                return_dict["weight_attn_mask"].append(
                    torch.cat(
                        [
                            torch.zeros((pad_size,), dtype=torch.bool),
                            torch.ones((model_dict[key].size(0),), dtype=torch.bool),
                        ],
                        dim=0,
                    ).unsqueeze(0)
                )  # (1, max_num_weights, padding_size, padding_size)
        elif key == "mask":
            if model_dict[key].size(0) < max_num_weights:
                pad_size = max_num_weights - model_dict[key].size(0)
                pad_tensor = torch.zeros(
                    (pad_size, padding_size, padding_size), dtype=model_dict[key].dtype
                )
                return_dict[key].append(
                    torch.cat([pad_tensor, model_dict[key]], dim=0).unsqueeze(0)
                )  # (1, max_num_weights, padding_size, padding_size)
        else:
            if model_dict[key].size(0) < max_num_weights:
                pad_size = max_num_weights - model_dict[key].size(0)
                pad_tensor = torch.zeros((pad_size,), dtype=model_dict[key].dtype)
                return_dict[key].append(
                    torch.cat([pad_tensor, model_dict[key]], dim=0).unsqueeze(0)
                )  # (1, max_num_weights)
    return return_dict
