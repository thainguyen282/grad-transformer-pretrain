import torch
from torchvision import models
from transformers import AutoModel
from decnn import DECNN


class GTAModel(torch.nn.Module):
    def __init__(
        self,
        cnn_module: str = "resnet18",
        lm_module: str = "Qwen/Qwen2.5-3B-Instruct",
        padding_size: int = 512,
        meta_tensor_dim: int = 128,
        device: str = "cuda",
    ):
        super(GTAModel, self).__init__()
        if cnn_module == "resnet18":
            self.cnn = models.resnet18(pretrained=True)
            self.cnn = torch.nn.Sequential(
                *list(self.cnn.children())[:-1]
            )  # remove the final fully connected layer
            self.cnn_output_dim = 512  # resnet18's final feature dimension

        self.lm = AutoModel.from_pretrained(lm_module)
        lm_config = self.lm.config
        self.hidden_dim = lm_config.hidden_size

        self.deCNN = DECNN(
            hidden_dim=lm_config.hidden_size,  # similar to the hidden size of the language model
            target_shape=(1, padding_size, padding_size),
            base_channels=1,
            use_linear_projection=False,
        )

        self.meta_projection = torch.nn.Linear(meta_tensor_dim, lm_config.hidden_size)
        self.weight_projection = torch.nn.Linear(
            self.cnn_output_dim, lm_config.hidden_size
        )

    def forward(self, source_model_dict: dict, target_model_dict: dict):
        """
        Example of model dict:
            model_dict = {
                "meta_info": tensor_of_meta_info,  # (num_weights + 1, meta_info_dim)
                "weight": padded_weight of the model,  # (num_weights, padding_size, padding_size)
                "mask": mask,  # (num_weights, padding_size, padding_size)
                "has_bias": source_has_bias_tensor,  # (num_weights,)
            }
        """

        # project the meta info and weights to the hidden dimension
        source_meta_info = source_model_dict[
            "meta_info"
        ]  # (num_weights + 1, meta_info_dim)
        source_weight = source_model_dict[
            "weight"
        ]  # (num_weights, padding_size, padding_size)

        source_meta_emb = self.meta_projection(
            source_meta_info
        )  # (num_weights + 1, hidden_dim)
        source_weight_emb = self.cnn(
            source_weight
        ).squeeze()  # (num_weights, cnn_output_dim)
        source_weight_emb = self.weight_projection(
            source_weight_emb
        )  # (num_weights, hidden_dim)

        model_meta_emb = source_meta_emb[0, :].unsqueeze(0)  # (1, hidden_dim)
        weight_concat = [model_meta_emb]
        for i in range(0, source_weight_emb.size(0)):
            weight_concat.append(
                source_meta_emb[i + 1, :].unsqueeze(0)
            )  # (1, hidden_dim)
            weight_concat.append(
                source_weight_emb[i, :].unsqueeze(0)
            )  # (1, hidden_dim)
        source_combined_emb = torch.cat(
            weight_concat, dim=0
        )  # (num_weights * 2+1, hidden_dim)

        # Do for the target model as well.
        target_meta_info = target_model_dict["meta_info"]
        target_weight = target_model_dict["weight"]
        target_meta_emb = self.meta_projection(target_meta_info)
        target_weight_emb = self.cnn(target_weight).squeeze()
        target_weight_emb = self.weight_projection(target_weight_emb)
        target_model_meta_emb = target_meta_emb[0, :].unsqueeze(0)
        target_weight_concat = [target_model_meta_emb]
        for i in range(0, target_weight_emb.size(0)):
            target_weight_concat.append(
                target_meta_emb[i + 1, :].unsqueeze(0)
            )  # (1, hidden_dim)
            target_weight_concat.append(
                target_weight_emb[i, :].unsqueeze(0)
            )  # (1, hidden_dim)
        target_combined_emb = torch.cat(
            target_weight_concat, dim=0
        )  # (num_weights * 2+1, hidden_dim)
