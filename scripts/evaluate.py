#!/usr/bin/env python3
"""Evaluate trained HealthBot models.

Usage:
    python scripts/evaluate.py --artifacts-dir ./artifacts --data-dir ./data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    encode_labels,
    load_pickle,
    tokenize_for_intent_classifier,
)
from src.evaluation import classification_metrics, evaluate_generation
from src.models import LSTMWithAttention
from src.preprocessing import clean_text
from src.prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HealthBot models")
    parser.add_argument("--artifacts-dir", type=str, default="./artifacts")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    artifacts = Path(args.artifacts_dir)
    data_dir = Path(args.data_dir)

    # ------------------------------------------------------------------
    # 1. Evaluate Intent Classifier
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Intent Classifier Evaluation")
    print("=" * 60)

    med_data = load_pickle(data_dir / "med_data_all.pkl")
    knowledge_graph = load_pickle(data_dir / "knowledge_graph.pkl")

    cleaned = clean_text(med_data["inputs"])
    encoded_labels, le, label_mapping = encode_labels(med_data["labels"])

    tokenizer = BertTokenizer.from_pretrained(str(artifacts / "LSTMwA"))
    input_ids, attention_masks, labels_tensor = tokenize_for_intent_classifier(
        cleaned, tokenizer, labels=encoded_labels, device=device,
    )

    # Load model
    model = torch.load(artifacts / "LSTMwA" / "best_model.pt", map_location=device)
    model.eval()

    # Run predictions
    all_preds, all_true = [], []
    dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for inputs, masks, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    metrics = classification_metrics(all_true, all_preds, class_names=list(le.classes_))
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nClassification Report:\n{metrics['report']}")
    print("\nPer-class Accuracy:")
    for cls, acc in metrics["per_class_accuracy"].items():
        print(f"  {cls}: {acc:.2%}")

    # ------------------------------------------------------------------
    # 2. Evaluate Response Generation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Response Generation Evaluation")
    print("=" * 60)

    gpt2_path = artifacts / "gpt2-rl"
    if not gpt2_path.exists():
        gpt2_path = artifacts / "gpt2"

    if gpt2_path.exists():
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(str(gpt2_path))
        gpt2_model = GPT2LMHeadModel.from_pretrained(
            str(gpt2_path), output_hidden_states=True
        ).to(device).eval()

        # Load test queries
        try:
            ques_resp = load_pickle(data_dir / "ques_resp.pkl")
            test_queries = ques_resp["queries"][:20]
            test_responses = ques_resp["responses"][:20]

            generated = []
            for q in test_queries:
                # Classify intent
                cleaned_q = clean_text([q])
                ids, _ = tokenize_for_intent_classifier(cleaned_q, tokenizer, device=device)
                with torch.no_grad():
                    logits = model(ids)
                    pred_idx = torch.argmax(logits, dim=1).item()
                disease = {v: k for k, v in label_mapping.items()}[pred_idx]

                # Build prompt and generate
                prompt = build_prompt(q, disease, knowledge_graph)
                enc_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
                output = gpt2_model.generate(
                    enc_ids, max_length=300,
                    pad_token_id=gpt2_tokenizer.eos_token_id,
                    temperature=0.7, top_p=0.9,
                    no_repeat_ngram_size=2, do_sample=True, num_beams=2,
                )
                gen_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
                if gen_text.startswith(prompt):
                    gen_text = gen_text[len(prompt):].strip()
                generated.append(gen_text)

            gen_metrics = evaluate_generation(generated, test_responses, device=device)
            print(f"\nGeneration Metrics (on {len(test_queries)} samples):")
            for key, val in gen_metrics.items():
                print(f"  {key}: {val:.4f}")

        except FileNotFoundError:
            print("  Test data not found, skipping generation evaluation.")
    else:
        print("  GPT-2 model not found, skipping generation evaluation.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
