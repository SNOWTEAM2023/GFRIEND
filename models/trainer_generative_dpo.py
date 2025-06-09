import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

class TrainerGenerativeDPO:
    """
    Trainer for a generative preference model that outputs a 'which candidate is better' token.
    Incorporates the multi-level weighting from the paper GFRIEND.
    """
    def __init__(self, model, tokenizer, alpha=1.0, lr=1e-5):
        """
        Args:
            model: GenerativeJudgeModel or similar
            tokenizer: the same tokenizer used for model
            alpha: the scaling parameter for w(g+, g-) = log(1 + exp(alpha * gap))
            lr: learning rate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # We force the label to be a short token, e.g. "A" or "B".
        # Make sure these tokens exist in the tokenizer vocabulary.
        # If not, consider adding special tokens or using "1"/"2".
        # (Below, we'll assume "A" and "B" are in-vocab.)
        self.label_a_id = self.tokenizer.convert_tokens_to_ids("A")
        self.label_b_id = self.tokenizer.convert_tokens_to_ids("B")
        # If these tokens don’t exist in your model’s vocab, 
        # pick a suitable token that does exist.

    def collate_fn(self, batch, max_length=512):
        """
        Expects each item in batch to have:
          - question
          - pos_cot
          - neg_cot
          - pos_ppl_score
          - neg_ppl_score
          ...
        We'll build a prompt like:

        Prompt: 
        ```
        Question: ...
        Candidate 1: [pos_cot]
        Candidate 2: [neg_cot]
        Which candidate is better? 
        ```

        Then we want the model to output either "Candidate 1" or "Candidate 2" 
        (or simply "A"/"B"), depending on which is correct (the 'pos' side).
        """
        input_texts = []
        labels = []
        weights = []

        for item in batch:
            q = item['question']
            pos_cot = item['pos_cot']
            neg_cot = item['neg_cot']
            pos_score = item['pos_ppl_score']
            neg_score = item['neg_ppl_score']

            # We know "pos_cot" is the better solution. 
            # Let's define the model's correct label to be "A", meaning candidate 1 is better.
            # We'll treat candidate 1 as the positive side. 
            # If you prefer the opposite arrangement, just swap them.

            prompt = (f"Question: {q}\n"
                      f"Candidate A: {pos_cot}\n"
                      f"Candidate B: {neg_cot}\n"
                      "Which candidate is better?\n")

            # We'll want the model to generate "A". 
            # So the 'labels' sequence would be that same prompt plus "A".
            # We'll do teacher forcing for that final token.

            # In standard language-model training, we can put the ground-truth token after 
            # the prompt, then mask out the prompt so that only the final token is trained.
            # E.g. input: [prompt + "A"] => label: [-100,...,-100, token_id_of_"A"].

            # We'll do a simple approach: we add "A" at the end, 
            # and ensure we only compute cross-entropy on that "A" token.

            full_text = prompt + "A"
            input_texts.append(full_text)

            # Weighted scheme:
            gap = abs(pos_score - neg_score)
            w = math.log(1 + math.exp(self.alpha * gap))
            weights.append(w)

            # We store that the correct label is literally the last token "A".
            # We'll handle labeling in compute_loss.

        # Now tokenize
        enc = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "weights": torch.tensor(weights, dtype=torch.float),
            # We'll store the text so we know how to find the final token if needed:
            "full_texts": input_texts,
        }

    def compute_loss(self, batch):
        """
        We'll do teacher-forcing over the final token. 
        The final token is known to be "A" in our setup.

        We only want to compute cross-entropy on that last "A" token,
        then multiply by the sample's weight. 
        Finally, average across the batch.
        """
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        weights = batch["weights"].cuda()

        # We want the final token to be "A". So let's find the token index for it:
        # Method 1: forcibly label the last token in the sequence. 
        # We'll find the actual length, set that position's label to "A" (label_a_id),
        # and set everything else to -100.

        # Because we appended "A" to the text, the last non-padding token should be "A".
        # We'll do a quick pass to identify that position for each sample.
        labels = input_ids.clone()
        # default to -100 for ignoring
        labels[:] = -100

        # find the last non-padding token for each sample
        seq_lens = attention_mask.sum(dim=1)  # number of real tokens
        for i in range(input_ids.size(0)):
            # The last token index
            last_idx = seq_lens[i] - 1
            if last_idx >= 0:
                labels[i, last_idx] = self.label_a_id  # "A"

        loss, logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        # 'loss' is already an average cross-entropy over all tokens with label != -100,
        # so it effectively only includes that final token in practice.

        # Multiply by each sample's weight. However, the HF model returns 
        # the average across the entire batch. 
        # If we want to incorporate the sample weights, we can't just multiply the final CE 
        # because it's an average. We can do a trick: 
        #   - manually compute token-level log probs for each sample, 
        #   - multiply by weight, and average. 
        #
        # For simplicity, let's do a manual approach:
        batch_size = input_ids.size(0)
        # Recompute log-probs manually, or do a small hack: 
        # We'll do the forward pass with 'labels' but get all token-level losses, 
        # then pick out the final token's loss. 
        # The HF API doesn't directly provide token-level losses, 
        # but we can do next-token log softmax ourselves. 
        # For a short example, let's do a simpler approach: 
        # We'll just multiply the final cross-entropy by the average weight ratio:
        #   WeightedLoss = (1/batch_size) * sum_i ( w_i * CE_i ) 
        # But 'loss' is the average (CE_i) across i. 
        # So WeightedLoss = ( sum_i w_i * CE_i ) / sum_i(1) 
        # => WeightedLoss = ( sum_i w_i / batch_size ) * average(CE_i) 
        # That doesn't properly reflect sample-by-sample weighting. 
        #
        # Proper approach: we need the token-level log-likelihood for each sample's last token. 
        # We'll do a second pass with "reduction='none'", then gather the last token's loss. 

        with torch.no_grad():
            outputs_no_reduce = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # We'll compute the log-prob of each token vs. its label ourselves:
            shift_logits = outputs_no_reduce.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # We'll see which positions we want to keep. 
            # Actually, easier might be to re-encode just the final token. 
            # For brevity, let's do a quick manual approach:
            # We know "the final token we want is at seq_len-1 of the input, 
            # the label is label_a_id." So let's get the logit there:
            final_token_logits = []
            for i in range(batch_size):
                final_idx = seq_lens[i] - 1
                # gather the logit for that position
                logit = outputs_no_reduce.logits[i, final_idx, :]  # shape [vocab_size]
                final_token_logits.append(logit.unsqueeze(0))
            final_token_logits = torch.cat(final_token_logits, dim=0)  # [B, vocab_size]

            # The ground-truth we want is "A" => label_a_id 
            # We'll compute -log p(A):
            log_probs_A = F.log_softmax(final_token_logits, dim=-1)[
                :, self.label_a_id
            ]  # [B]
            # cross-entropy for that single token is -log_probs_A
            per_sample_ce = -log_probs_A  # [B]

        # Weighted sum
        weighted_ce = (per_sample_ce * weights).mean()

        return weighted_ce

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            self.optimizer.zero_grad()
            loss_val = self.compute_loss(batch)
            loss_val.backward()
            self.optimizer.step()
            total_loss += loss_val.item()
        return total_loss / len(dataloader)
