import os
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from models.sft_model import SFTModel
from data.dataset import SFTCoTDataset

def train_sft(args):
    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    train_dataset = SFTCoTDataset(args.train_data, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Initialize model
    model = SFTModel(args.model_name_or_path)
    model.train()
    model.cuda()

    # 4. Optimizer & Learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(total_steps*args.warmup_ratio),
                                                num_training_steps=total_steps)

    # 5. Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader):
            input_ids = torch.tensor(batch['input_ids']).cuda()
            attention_mask = torch.tensor(batch['attention_mask']).cuda()

            loss, _ = model(input_ids, attention_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss = {epoch_loss/len(train_loader):.4f}")

    # 6. Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("SFT training complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_sft")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    train_sft(args)