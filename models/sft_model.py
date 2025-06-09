import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SFTModel(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask):
        # Standard output of a Causal LM
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        return outputs.loss, outputs.logits