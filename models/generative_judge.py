import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class GenerativeJudgeModel(nn.Module):
    """
    A generative 'judge' model that, given a prompt describing two candidate answers,
    is trained to generate the label (e.g. 'A' or 'B') indicating which is better.
    No extra classification head is used; we rely purely on the LM's generative capacity.
    """
    def __init__(self, base_model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Standard causal LM forward pass. If labels is not None, 
        the model will compute cross-entropy over those labels.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits
