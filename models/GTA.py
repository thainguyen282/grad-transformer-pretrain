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

        source_emb = self._extract_embedding_source(source_model_dict)
        target_emb, target_attn_mask = self._extract_embedding_target(target_model_dict)

        attention_mask = torch.cat(
            [torch.zeros(source_emb.size(1)), target_attn_mask], dim=0
        ).unsqueeze(0)

        # concatenate the source and target embeddings along the sequence dimension
        combined_emb = torch.cat([source_emb, target_emb], dim=1)  # (B, Ls * 4+2, h)

        output = self.deCNN(combined_emb)  # (B, 1, padding_size, padding_size)
        return output

    def _extract_embedding_source(self, model_dict: dict):

        meta_info = model_dict["meta_info"]  # (B, Ls, emb_dim)
        weight = model_dict["weight"]  # (B, Ls, P, P)
        meta_attn_mask = model_dict["meta_info_attn_mask"]  # (B, Ls)
        weight_attn_mask = model_dict["weight_attn_mask"]  # (B, Ls)

        meta_emb = self.meta_projection(meta_info)  # (B, Ls, h)
        weight_emb = []
        for i in range(weight.size(1)):
            weight_emb.append(
                self.cnn(weight[:, i, :, :].unsqueeze(1)).squeeze()
            )  # (B, cnn_output_dim)
        weight_emb = torch.stack(weight_emb, dim=1)  # (B, Ls, cnn_output_dim)
        weight_emb = self.weight_projection(weight_emb)  # (B, Ls, h)

        combined_emb = []
        for i in range(meta_emb.size(0)):
            comb_emb = []
            start = False
            for j in range(meta_emb.size(1)):
                if meta_attn_mask[i, j] == 0:
                    comb_emb.append(torch.zeros_like(meta_emb[i, j, :]).unsqueeze(0))
                if start == False:
                    start = True
                    comb_emb.append(meta_emb[i, j, :].unsqueeze(0))  # meta model
                    continue
                comb_emb.append(meta_emb[i, j, :].unsqueeze(0))  # (1, h)
                comb_emb.append(weight_emb[i, j, :].unsqueeze(0))  # (1, h)
            combined_emb.append(
                torch.cat(comb_emb, dim=0).unsqueeze(0)
            )  # (1, Ls * 2+1, h)
        combined_emb = torch.cat(combined_emb, dim=0)  # (B, Ls * 2+1, h)
        return combined_emb

    def _extract_embedding_target(self, model_dict: dict):

        meta_info = model_dict["meta_info"]  # (B, Ls, emb_dim)
        weight = model_dict["weight"]  # (B, Ls, P, P)
        meta_attn_mask = model_dict["meta_info_attn_mask"]  # (B, Ls)
        weight_attn_mask = model_dict["weight_attn_mask"]  # (B, Ls)

        meta_emb = self.meta_projection(meta_info)  # (B, Ls, h)
        weight_emb = []
        for i in range(weight.size(1)):
            weight_emb.append(
                self.cnn(weight[:, i, :, :].unsqueeze(1)).squeeze()
            )  # (B, cnn_output_dim)
        weight_emb = torch.stack(weight_emb, dim=1)  # (B, Ls, cnn_output_dim)
        weight_emb = self.weight_projection(weight_emb)  # (B, Ls, h)

        combined_emb = []
        attention_mask = []

        for i in range(meta_emb.size(0)):
            comb_emb = []
            pad_emb = []
            attn_mask = []
            start = False
            for j in range(meta_emb.size(1)):
                if meta_attn_mask[i, j] == 0:
                    pad_emb.append(torch.zeros_like(meta_emb[i, j, :]).unsqueeze(0))
                if start == False:
                    start = True
                    comb_emb.append(meta_emb[i, j, :].unsqueeze(0))  # meta model
                    attn_mask.append(0)  # meta model does not attend to any weight
                    continue
                comb_emb.append(meta_emb[i, j, :].unsqueeze(0))  # (1, h)
                attn_mask.append(0)  # meta info does not attend to any weight
                comb_emb.append(weight_emb[i, j, :].unsqueeze(0))  # (1, h)
                attn_mask.append(1)  # weight embedding attends to the meta info

            for pad in pad_emb:
                comb_emb.append(pad)
                attn_mask.append(0)  # padding does not attend to any weight
            combined_emb.append(
                torch.cat(comb_emb, dim=0).unsqueeze(0)
            )  # (1, Ls * 2+1, h)
            attention_mask.append(attn_mask)
        combined_emb = torch.cat(combined_emb, dim=0)  # (B, Ls * 2+1, h)
        attention_mask = torch.tensor(attention_mask).to(
            combined_emb.device
        )  # (B, Ls * 2+1)
        return combined_emb, attention_mask
