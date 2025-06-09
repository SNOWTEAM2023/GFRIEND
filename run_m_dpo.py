import os
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from models.generative_judge import GenerativeJudgeModel
from models.trainer_generative_dpo import TrainerGenerativeDPO

class MultiPrefDataset(Dataset):
    """
    A simple dataset wrapper for the multi-level preference data (multi_pref_data.json).
    Each line in that JSON should contain fields:
      "question", "pos_cot", "neg_cot",
      "pos_ppl_score", "neg_ppl_score",
      ...
    """
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def run_m_dpo(args):
    dataset = MultiPrefDataset(args.multi_pref_data)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GenerativeJudgeModel(args.base_model_path).cuda()

    trainer = TrainerGenerativeDPO(model, tokenizer, alpha=args.alpha, lr=args.lr)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainer.collate_fn
    )

    for epoch in range(args.num_epochs):
        loss_val = trainer.train_one_epoch(dataloader)
        print(f"[Epoch {epoch+1}] Generative M-DPO loss: {loss_val:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Generative judge M-DPO training done. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_pref_data", type=str, required=True, 
                        help="Path to multi-level preference data (e.g., multi_pref_data.json).")
    parser.add_argument("--base_model_path", type=str, required=True, 
                        help="Base LM path (should be from your SFT checkpoint or similar).")
    parser.add_argument("--output_dir", type=str, default="./gen_judge_output")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scaling factor for multi-level weighting.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    run_m_dpo(args)
