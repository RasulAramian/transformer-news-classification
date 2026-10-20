import torch
from torch.utils.data import Dataset

from src.preprocessing import normalize_text


class NewsDataset(Dataset):
    """PyTorch Dataset for tokenized news classification data."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        if len(self.texts) != len(self.labels):
            raise ValueError("Texts and labels must have the same length.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = normalize_text(self.texts[index])
        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
