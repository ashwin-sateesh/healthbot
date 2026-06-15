"""Centralized configuration for the HealthBot project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    """All path configurations. Override via environment or CLI."""

    base_dir: Path = Path("./artifacts")
    data_dir: Path = Path("./data")

    # Pickle data files
    med_data_file: str = "med_data_all.pkl"
    knowledge_graph_file: str = "knowledge_graph.pkl"
    input_queries_file: str = "input_queries.pkl"
    ques_resp_file: str = "ques_resp.pkl"
    ques_resp2_file: str = "ques_resp2.pkl"
    ner_queries_label_file: str = "ner_queries_label.pkl"

    # Model directories
    lstm_model_dir: str = "LSTMwA"
    lstm_model_file: str = "lstm_w_attn_full_final.pt"
    gpt2_model_dir: str = "gpt2-new"
    flan_t5_model_dir: str = "flan-t5-base-new"

    @property
    def lstm_model_path(self) -> Path:
        return self.base_dir / self.lstm_model_dir / self.lstm_model_file

    @property
    def lstm_tokenizer_path(self) -> Path:
        return self.base_dir / self.lstm_model_dir

    @property
    def gpt2_model_path(self) -> Path:
        return self.base_dir / self.gpt2_model_dir

    @property
    def flan_t5_model_path(self) -> Path:
        return self.base_dir / self.flan_t5_model_dir


@dataclass
class IntentClassifierConfig:
    """Hyperparameters for the LSTM-with-Attention intent classifier."""

    hidden_size: int = 128
    num_layers: int = 2
    embedding_dim: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 16
    num_epochs: int = 10
    train_split: float = 0.8


@dataclass
class NERConfig:
    """Hyperparameters for the BERT-based NER model."""

    pretrained_model: str = "bert-base-uncased"
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 5
    label2id: dict = field(default_factory=lambda: {
        "MISC": 0,
        "medicine": 1,
        "symptom": 2,
        "disease": 3,
        "severity": 4,
        "sensation": 5,
        "body": 6,
    })

    @property
    def id2label(self) -> dict:
        return {v: k for k, v in self.label2id.items()}

    @property
    def num_labels(self) -> int:
        return len(self.label2id)


@dataclass
class GPT2Config:
    """Hyperparameters for GPT-2 fine-tuning and generation."""

    pretrained_model: str = "gpt2"
    learning_rate: float = 2e-5
    eps: float = 1e-8
    batch_size: int = 8
    num_epochs: int = 10
    max_length: int = 512
    # Generation parameters
    gen_max_length: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    no_repeat_ngram_size: int = 2
    num_beams: int = 2


@dataclass
class FlanT5Config:
    """Hyperparameters for FLAN-T5 fine-tuning and generation."""

    pretrained_model: str = "google/flan-t5-base"
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 10
    max_length: int = 512
    gen_max_length: int = 100


@dataclass
class RLConfig:
    """Hyperparameters for the reinforcement learning loop."""

    learning_rate: float = 1e-3
    batch_size: int = 8
    num_epochs: int = 3
    similarity_threshold: float = 0.7
    reward_gen_max_length: int = 100


@dataclass
class LoRAConfig:
    """LoRA adapter configuration for FLAN-T5."""

    rank: int = 8
    lora_learning_rate: float = 1e-3
    base_learning_rate: float = 1e-5
    num_epochs: int = 5
