import torch
from transformers import AutoModel


class GTAModel(torch.nn.Module):
    def __init__(self, model_name: str, device: str = "cuda"):
        super(GTAModel, self).__init__()
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state
