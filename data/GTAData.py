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
    ):
        self.num_samples = num_samples
        self.do_shuffle = args.shuffle_data
        self.meta_embedding_model = args.meta_embedding_model
        self.save_dir = save_dir
        self.console = console
        self.processed = False

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

        if os.path.exists(os.path.join(save_dir, "meta_info_dict.json")):
            with open(os.path.join(save_dir, "meta_info_dict.json"), "r") as f:
                self.model_info_dict = json.load(f)
            self.processed = True

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # Get a source-target model pair for this sample
        source_model_name, target_model_name = self._get_source_target_pair(idx)

        # Get meta_information for the source and target models (e.g. config details like hidden size, num layers, etc. that can be used as input features for GTA)

    def preprocessing(
        self,
    ) -> None:

        # create a model to extract embeddings for the meta information (e.g. model config details) using the specified meta_embedding_model

        self.console.log(
            f"Preprocessing GTA dataset and saving meta information vectors to {self.save_dir}"
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
                model_name=model_name, emb_model=emb_model
            )
            model_info_dict[model_key] = {
                "name": model_name,
                "meta_info_vector_path": meta_info_path,
            }

        # save the model_info_dict for future reference
        with open(os.path.join(self.save_dir, "meta_info_dict.json"), "w") as f:
            json.dump(model_info_dict, f, indent=4)

        self.model_info_dict = model_info_dict
        self.processed = True

    def _preprocessing_single_model(
        self, model_name: str, emb_model: SentenceTransformer
    ) -> str:

        self.console.log(f"Processing model {model_name} for GTA dataset")
        model_meta_dict = {}
        model_save_name = f"{model_name.split('/')[-1]}_{self.meta_embedding_model.split('/')[-1]}_meta_info.pt"
        model_save_dir = os.path.join(self.save_dir, model_save_name)
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
            meta_info_vector = emb_model.encode(name, convert_to_tensor=True).cpu()
            weight_meta_info_vector_name = f"{name.replace('.', '_')}"
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
