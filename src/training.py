"""Training loops for all HealthBot models.

Includes:
    - Intent classifier (LSTM / BiRNN / Transformer)
    - BERT NER
    - GPT-2 fine-tuning
    - FLAN-T5 fine-tuning (with optional LoRA)
    - Reinforcement learning reward loop
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BertForTokenClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from .models import LSTMWithAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> str:
    """Return a human-readable summary of trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return (
        f"Trainable: {trainable:,}  |  Total: {total:,}  |  "
        f"Trainable %: {100 * trainable / total:.2f}%"
    )


# ---------------------------------------------------------------------------
# Intent Classifier Training
# ---------------------------------------------------------------------------

def train_intent_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    save_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Train an intent-classification model (LSTM/BiRNN/Transformer).

    Returns:
        Dict with keys 'train_acc', 'test_acc', 'loss'.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: Dict[str, List[float]] = {"train_acc": [], "test_acc": [], "loss": []}
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, masks, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)

        # --- Evaluate ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, masks, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100.0 * correct / total
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["loss"].append(avg_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]  "
            f"Loss: {avg_loss:.4f}  Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%"
        )

        # Save best model
        if save_dir and test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model, save_path / "best_model.pt")
            torch.save(model.state_dict(), save_path / "weights" / "best_weights.pt")

    return history


# ---------------------------------------------------------------------------
# NER Training
# ---------------------------------------------------------------------------

def train_ner(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_labels: int = 7,
    pretrained_model: str = "bert-base-uncased",
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    device: torch.device = torch.device("cpu"),
    save_dir: Optional[str] = None,
) -> Tuple[BertForTokenClassification, Dict[str, List[float]]]:
    """Fine-tune BERT for token-level NER.

    Returns:
        (model, history) where history has keys 'train_loss', 'val_loss',
        'train_acc', 'val_acc'.
    """
    model = BertForTokenClassification.from_pretrained(
        pretrained_model, num_labels=num_labels
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=2)
            mask = labels != 0  # ignore padding
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / max(total, 1)

        # --- Validate ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=2)
                mask = labels != 0
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / max(total, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]  "
            f"Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}  "
            f"Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%"
        )

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)

    return model, history


# ---------------------------------------------------------------------------
# GPT-2 Fine-tuning
# ---------------------------------------------------------------------------

def train_gpt2(
    model: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    eps: float = 1e-8,
    device: torch.device = torch.device("cpu"),
    save_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Fine-tune GPT-2 on prompt-completion pairs.

    DataLoaders should yield batches of (input_ids, attention_masks).

    Returns:
        Dict with 'train_loss' and 'val_loss'.
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=eps)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                val_loss += outputs.loss.item()

        avg_val = val_loss / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(f"Epoch [{epoch + 1}/{num_epochs}]  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)

    return history


# ---------------------------------------------------------------------------
# FLAN-T5 Fine-tuning
# ---------------------------------------------------------------------------

def train_flan_t5(
    model: T5ForConditionalGeneration,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    save_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Fine-tune FLAN-T5 (with or without LoRA adapters already applied).

    Returns:
        Dict with 'train_loss' and 'val_loss'.
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                val_loss += outputs.loss.item()

        avg_val = val_loss / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(f"Epoch [{epoch + 1}/{num_epochs}]  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)

    return history


# ---------------------------------------------------------------------------
# RL Reward Loop (Cosine Similarity)
# ---------------------------------------------------------------------------

def get_gpt2_embedding(
    text: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Get mean-pooled embedding from GPT-2's last hidden layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.hidden_states[-1]
    return last_hidden.mean(dim=1)


def compute_reward(
    generated: str,
    reference: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
) -> float:
    """Compute cosine-similarity reward between generated and reference responses."""
    emb_gen = get_gpt2_embedding(generated, model, tokenizer, device)
    emb_ref = get_gpt2_embedding(reference, model, tokenizer, device)
    similarity = F.cosine_similarity(emb_gen, emb_ref, dim=1)
    return similarity.item()


def train_rl_loop(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
    generate_fn,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    similarity_threshold: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    """Run the RL fine-tuning loop using cosine-similarity rewards.

    Args:
        model: GPT-2 model (must have output_hidden_states=True).
        tokenizer: Corresponding tokenizer.
        train_data: List of {'prompt': ..., 'response': ...}.
        val_data: Validation set in the same format.
        generate_fn: Callable(model, tokenizer, prompt) -> str.
        num_epochs: Number of RL epochs.
        batch_size: Batch size for updates.
        learning_rate: Optimizer learning rate.
        similarity_threshold: Reward threshold for valid responses.
        device: Torch device.

    Returns:
        Dict with 'train_reward' and 'val_reward'.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    history: Dict[str, List[float]] = {"train_reward": [], "val_reward": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_rewards: List[float] = []

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            batch_rewards: List[float] = []

            for item in batch:
                generated = generate_fn(model, tokenizer, item["prompt"])
                reward = compute_reward(
                    generated, item["response"], model, tokenizer, device
                )
                batch_rewards.append(reward)

            mean_reward = sum(batch_rewards) / len(batch_rewards)
            epoch_rewards.extend(batch_rewards)

            # Simple policy-gradient-style update: re-run forward and scale loss by -reward
            for item in batch:
                encoded = tokenizer.encode_plus(
                    item["prompt"] + item["response"],
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                ).to(device)
                optimizer.zero_grad()
                outputs = model(
                    encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    labels=encoded["input_ids"],
                )
                scaled_loss = outputs.loss * (1.0 - mean_reward)
                scaled_loss.backward()
                optimizer.step()

        avg_train_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)

        # --- Validate ---
        model.eval()
        val_rewards: List[float] = []
        for item in val_data:
            generated = generate_fn(model, tokenizer, item["prompt"])
            reward = compute_reward(
                generated, item["response"], model, tokenizer, device
            )
            val_rewards.append(reward)

        avg_val_reward = sum(val_rewards) / max(len(val_rewards), 1)
        history["train_reward"].append(avg_train_reward)
        history["val_reward"].append(avg_val_reward)

        print(
            f"RL Epoch [{epoch + 1}/{num_epochs}]  "
            f"Train Reward: {avg_train_reward:.4f}  Val Reward: {avg_val_reward:.4f}"
        )

    return history
