#!/usr/bin/env python3
"""Run the HealthBot interactive chatbot.

Usage:
    python scripts/chat.py --artifacts-dir ./artifacts --data-dir ./data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_pickle
from src.inference import HealthBot, interactive_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HealthBot interactive chat")
    parser.add_argument(
        "--artifacts-dir", type=str, default="./artifacts",
        help="Directory containing trained model artifacts",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory containing data files (knowledge graph, etc.)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-length", type=int, default=300,
        help="Max generation length (default: 300)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = Path(args.artifacts_dir)
    data_dir = Path(args.data_dir)

    # Load knowledge graph and label mapping
    knowledge_graph = load_pickle(data_dir / "knowledge_graph.pkl")
    label_mapping = load_pickle(artifacts / "LSTMwA" / "label_mapping.pkl")

    # Initialize the bot
    bot = HealthBot.from_pretrained(
        lstm_model_path=str(artifacts / "LSTMwA" / "best_model.pt"),
        lstm_tokenizer_path=str(artifacts / "LSTMwA"),
        gpt2_model_path=str(artifacts / "gpt2-rl"),
        knowledge_graph=knowledge_graph,
        label_mapping=label_mapping,
        temperature=args.temperature,
        max_length=args.max_length,
    )

    interactive_chat(bot)


if __name__ == "__main__":
    main()
