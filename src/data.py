"""Data loading, preparation, and dataset utilities for HealthBot."""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import BertTokenizer

from .preprocessing import clean_text


# ---------------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------------

def load_pickle(path: str | Path) -> Any:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save an object to a pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_jsonl_directory(directory: str | Path) -> Dict[str, list]:
    """Read all .jsonl files in *directory* into a dict with keys 'inputs' and 'labels'."""
    med_data: Dict[str, list] = defaultdict(list)
    directory = Path(directory)
    for jsonl_file in sorted(directory.glob("*.jsonl")):
        if jsonl_file.stem == "wants_antibiotics_for_a_cold":
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                med_data["inputs"].append(record.get("input", record.get("query", "")))
                med_data["labels"].append(jsonl_file.stem.replace("_", " "))
    return dict(med_data)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(labels: List[str]) -> Tuple[List[int], LabelEncoder, Dict[str, int]]:
    """Fit a LabelEncoder and return encoded labels + mapping.

    Returns:
        (encoded_labels, label_encoder, label_mapping)
    """
    le = LabelEncoder()
    encoded = le.fit_transform(labels).tolist()
    mapping = {name: idx for idx, name in enumerate(le.classes_)}
    return encoded, le, mapping


# ---------------------------------------------------------------------------
# Intent-classification tokenization
# ---------------------------------------------------------------------------

def tokenize_for_intent_classifier(
    texts: List[str],
    tokenizer: BertTokenizer,
    labels: Optional[List[int]] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, ...]:
    """Tokenize and pad texts for the LSTM intent classifier.

    Returns:
        (input_ids, attention_masks) or (input_ids, attention_masks, labels_tensor)
    """
    input_ids_list: List[torch.Tensor] = []
    attention_masks_list: List[torch.Tensor] = []
    max_length = 0

    for text in texts:
        encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        input_ids_list.append(encoded["input_ids"])
        attention_masks_list.append(encoded["attention_mask"])
        max_length = max(max_length, encoded["input_ids"].shape[1])

    # Pad to max length
    for i in range(len(input_ids_list)):
        pad_len = max_length - input_ids_list[i].shape[1]
        if pad_len > 0:
            input_ids_list[i] = torch.cat(
                [input_ids_list[i], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
            )
            attention_masks_list[i] = torch.cat(
                [attention_masks_list[i], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
            )

    input_ids = torch.cat(input_ids_list, dim=0).to(device)
    attention_masks = torch.cat(attention_masks_list, dim=0).to(device)

    if labels is not None:
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        return input_ids, attention_masks, labels_tensor

    return input_ids, attention_masks


# ---------------------------------------------------------------------------
# GPT-2 / FLAN-T5 prompt-completion datasets
# ---------------------------------------------------------------------------

class PromptCompletionDataset(Dataset):
    """Simple dataset of prompt-completion pairs for LLM fine-tuning."""

    def __init__(self, data: List[Dict[str, str]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


class TokenizedTextDataset(Dataset):
    """Pre-tokenized dataset for encoder-decoder models (FLAN-T5)."""

    def __init__(self, tokenizer, texts: List[str], max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Convenience: split + dataloader creation
# ---------------------------------------------------------------------------

def create_dataloaders(
    dataset: List[Dict[str, str]],
    batch_size: int = 8,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split a list of prompt-completion dicts into train/val/test DataLoaders."""
    train_val, test_data = train_test_split(dataset, test_size=test_size, random_state=seed)
    relative_val = val_size / (1 - test_size)
    train_data, val_data = train_test_split(train_val, test_size=relative_val, random_state=seed)

    train_loader = DataLoader(
        PromptCompletionDataset(train_data), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        PromptCompletionDataset(val_data), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        PromptCompletionDataset(test_data), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader
