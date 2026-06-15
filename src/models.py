"""Neural network architectures for HealthBot.

Includes:
    - LSTMWithAttention  — Bidirectional LSTM with attention for intent classification
    - BiRNN              — Bidirectional vanilla RNN baseline
    - TransformerClassifier — Transformer-encoder baseline
    - LoRALayer          — Low-Rank Adaptation layer for FLAN-T5
    - PolicyNetwork      — Simple policy net used in the RL reward loop
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Intent Classification Models
# ---------------------------------------------------------------------------

class LSTMWithAttention(nn.Module):
    """Bidirectional LSTM with additive attention for sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def _attention(
        self, lstm_output: torch.Tensor, final_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention-weighted context vector.

        Args:
            lstm_output: (batch, seq_len, hidden*2)
            final_state: last hidden states from both directions (2, batch, hidden)
        """
        # Concatenate forward/backward final hidden states -> (batch, hidden*2, 1)
        hidden = final_state.view(-1, self.hidden_size * 2, 1)
        # Attention scores -> (batch, seq_len)
        scores = torch.bmm(lstm_output, hidden).squeeze(2)
        weights = F.softmax(scores, dim=1).unsqueeze(2)
        # Weighted sum -> (batch, hidden*2)
        context = torch.bmm(lstm_output.permute(0, 2, 1), weights).squeeze(2)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, (hn, _) = self.lstm(embedded)
        context = self._attention(lstm_out, hn[-2:])
        return self.fc(context)


class BiRNN(nn.Module):
    """Bidirectional vanilla RNN for intent classification (baseline)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = rnn_out[:, -1, :]
        return self.fc(out)


class TransformerClassifier(nn.Module):
    """Transformer-encoder classifier for intent classification (baseline)."""

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_classes: int,
        nhead: int = 4,
        d_model: int = 128,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        encoded = self.transformer_encoder(embedded)
        pooled = encoded.mean(dim=1)
        return self.fc(pooled)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# LoRA Adapter for FLAN-T5
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Low-Rank Adaptation wrapper around a nn.Linear layer."""

    def __init__(self, original_layer: nn.Linear, rank: int = 8) -> None:
        super().__init__()
        self.rank = rank
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.weight = nn.Parameter(original_layer.weight.clone())
        self.bias = original_layer.bias

        self.lora_B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_effective = self.weight + self.lora_B @ self.lora_A
        return F.linear(x, W_effective, self.bias)


def apply_lora(model: nn.Module, rank: int = 8) -> None:
    """Replace all nn.Linear layers inside T5 encoder/decoder blocks with LoRA layers.

    Modifies *model* in place.
    """
    for block in list(model.encoder.block) + list(model.decoder.block):
        for layer in block.children():
            for name, sub_layer in layer.named_children():
                if isinstance(sub_layer, nn.Linear):
                    setattr(layer, name, LoRALayer(sub_layer, rank=rank))


# ---------------------------------------------------------------------------
# RL Policy Network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Lightweight policy network used during the RL reward-update loop."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        hidden = F.relu(self.fc1(pooled))
        return self.fc2(hidden)
