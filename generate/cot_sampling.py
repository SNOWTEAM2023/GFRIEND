import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class CoTSampler:
    def __init__(self, model_path, max_new_tokens=512, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate_judgment(self, question, ansA, ansB, 
                          num_samples=3, temperature=1.0, top_p=0.9):
        """
        For the same (question, ansA, ansB), perform multiple samplings and return a list of [json_str, ...].
        Example of json_str:
        {
          "CoT": "...reasoning process...",
          "Chosen answer": "A" or "B"
        }
        """
        prompt = self.build_prompt(question, ansA, ansB)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        all_generations = []
        for _ in range(num_samples):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            gen_text = self.tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            # Here, we can perform a simple JSON formatting (or direct string parsing)
            # Assuming we prompt the model to output { "CoT": "...", "Chosen answer": "A" }
            # Simplified handling: treat the generated text directly as a JSON string
            all_generations.append(gen_text)
        return all_generations

    def build_prompt(self, question, ansA, ansB):
        """
        Construct a prompt to make the model output a judgment and CoT.
        This can be customized according to the specific requirements.
        """
        prompt = (
            "You are a helpful judge. "
            "Now let's reason step by step (chain-of-thought) to decide which answer is better.\n\n"
            f"Question: {question}\n\n"
            f"Answer A: {ansA}\n"
            f"Answer B: {ansB}\n\n"
            "Please output a JSON with your chain-of-thought as 'CoT' and the final chosen answer as 'Chosen answer', either 'A' or 'B'."
        )
        return prompt