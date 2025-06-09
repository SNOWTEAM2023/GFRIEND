import os
from datasets import load_dataset
from torch.utils.data import Dataset

class SFTCoTDataset(Dataset):
    """
    Dataset for the Supervised Fine-Tuning (SFT) stage:
    Each sample format: {
      'question': str,
      'chain_of_thought': str
    }
    """
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.dataset = load_dataset('json', data_files=data_path)['train']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['question']
        cot = item['chain_of_thought']

        # Concatenate: [question] + separator + [cot]
        # Additional prompt formats can be added as needed
        text = f"Question: {question}\nChain-of-thought: {cot}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

class PreferenceDataset(Dataset):
    """
    Dataset for preference judgment:
    Each sample format: {
      'question': str,
      'answer_neg': str,
      'answer_pos': str
    }
    """
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.dataset = load_dataset('json', data_files=data_path)['train']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        q = item['question']
        a_neg = item['answer_neg']
        a_pos = item['answer_pos']

        return q, a_neg, a_pos