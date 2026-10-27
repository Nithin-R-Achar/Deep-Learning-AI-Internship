"""
Data loading and character tokenization utilities for LSTM Sequence Modeling.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    def __init__(self, text=None):
        if text is not None:
            self.chars = sorted(list(set(text)))
            self.vocab_size = len(self.chars)
            self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
            self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        else:
            self.chars = []
            self.vocab_size = 0
            self.char2idx = {}
            self.idx2char = {}

    def encode(self, text):
        """Encodes text to a list of integer token ids, ignoring unseen chars."""
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, indices):
        """Decodes integer token ids to text."""
        return "".join([self.idx2char.get(idx, "") for idx in indices])

    def save(self, filepath):
        """Saves vocabulary dictionary to JSON."""
        data = {
            "chars": self.chars,
            "vocab_size": self.vocab_size,
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath):
        """Loads vocabulary dictionary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.chars = data["chars"]
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.char2idx = data["char2idx"]
        tokenizer.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        return tokenizer


class CharDataset(Dataset):
    """
    Subsequences of length `seq_len` as inputs and inputs shifted by 1 as targets.
    """
    def __init__(self, encoded_data, seq_len=64, step=3):
        self.seq_len = seq_len
        self.inputs = []
        self.targets = []
        
        # Stride through text to collect training pairs
        data_len = len(encoded_data)
        for i in range(0, data_len - seq_len, step):
            self.inputs.append(encoded_data[i : i + seq_len])
            self.targets.append(encoded_data[i + 1 : i + seq_len + 1])

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_data_loader(encoded_data, seq_len=64, batch_size=64, step=3, shuffle=True):
    dataset = CharDataset(encoded_data, seq_len=seq_len, step=step)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader
