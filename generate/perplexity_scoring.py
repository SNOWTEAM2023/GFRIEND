import torch
import math
import json

class PPLScorer:
    def __init__(self, model_path, device="cuda", max_length=512, temperature=1.0):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        self.max_length = max_length
        self.temperature = temperature

    def compute_perplexity(self, text):
        """
        Calculate the perplexity (ppl) of a given text.
        """
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
        input_ids = encodings['input_ids'].to(self.device)
        with torch.no_grad():
            # Calculate perplexity for autoregressive models with a one-token shift
            outputs = self.model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
        ppl = math.exp(neg_log_likelihood.item())
        return ppl

    def normalize_score(self, ppl, tau=10.0):
        """
        Map ppl to the range [0,1], where a lower ppl results in a higher score.
        In the paper: PPLscore = exp(-PPL / τ)
        This is an implementation with a tunable parameter.
        """
        score = math.exp(-ppl / tau)
        return score

    def parse_generation(self, gen_str):
        """
        Parse the model's output (assuming it is in JSON format),
        and extract: "CoT": "...", "Chosen answer": "A" or "B".
        """
        # Here, we use json.loads for simplicity.
        # If the actual string is not in standard JSON format, a robust parsing method is needed.
        try:
            data = json.loads(gen_str)
            return data.get("CoT", ""), data.get("Chosen answer", "A")
        except:
            # Fallback option
            return gen_str, "A"