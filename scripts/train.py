#!/usr/bin/env python3
"""Train all HealthBot models end-to-end.

Usage:
    python scripts/train.py --data-dir ./data --output-dir ./artifacts

This script runs the full training pipeline:
    1. Data loading and preprocessing
    2. Intent classifier training (LSTM with Attention)
    3. NER model training (BERT)
    4. GPT-2 fine-tuning for response generation
    5. RL reward loop for response refinement
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import (
    GPT2Config,
    IntentClassifierConfig,
    NERConfig,
    PathConfig,
    RLConfig,
)
from src.data import (
    encode_labels,
    load_pickle,
    save_pickle,
    tokenize_for_intent_classifier,
)
from src.evaluation import classification_metrics
from src.models import LSTMWithAttention
from src.preprocessing import clean_text
from src.prompts import build_prompt
from src.training import (
    count_parameters,
    train_gpt2,
    train_intent_classifier,
    train_ner,
    train_rl_loop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HealthBot models")
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory containing pickle data files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./artifacts",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--skip-ner", action="store_true",
        help="Skip NER training",
    )
    parser.add_argument(
        "--skip-gpt2", action="store_true",
        help="Skip GPT-2 training",
    )
    parser.add_argument(
        "--skip-rl", action="store_true",
        help="Skip RL fine-tuning",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    med_data = load_pickle(data_dir / "med_data_all.pkl")
    knowledge_graph = load_pickle(data_dir / "knowledge_graph.pkl")
    print(f"  Loaded {len(med_data['inputs'])} queries across {len(set(med_data['labels']))} diseases")

    # ------------------------------------------------------------------
    # 2. Train Intent Classifier
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Training Intent Classifier (LSTM with Attention)")
    print("=" * 60)

    ic_cfg = IntentClassifierConfig()
    cleaned_texts = clean_text(med_data["inputs"])
    encoded_labels, label_encoder, label_mapping = encode_labels(med_data["labels"])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    input_ids, attention_masks, labels_tensor = tokenize_for_intent_classifier(
        cleaned_texts, tokenizer, labels=encoded_labels, device=device,
    )

    dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
    train_size = int(ic_cfg.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=ic_cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=ic_cfg.batch_size, shuffle=False)

    num_classes = len(set(med_data["labels"]))
    model = LSTMWithAttention(
        vocab_size=len(tokenizer.vocab),
        hidden_size=ic_cfg.hidden_size,
        num_layers=ic_cfg.num_layers,
        num_classes=num_classes,
    )
    print(f"  {count_parameters(model)}")

    ic_save_dir = str(output_dir / "LSTMwA")
    history = train_intent_classifier(
        model, train_loader, test_loader,
        num_epochs=ic_cfg.num_epochs,
        learning_rate=ic_cfg.learning_rate,
        device=device,
        save_dir=ic_save_dir,
    )
    tokenizer.save_pretrained(ic_save_dir)

    # Save label mapping for inference
    save_pickle(label_mapping, output_dir / "LSTMwA" / "label_mapping.pkl")
    print(f"  Intent classifier saved to {ic_save_dir}")

    # ------------------------------------------------------------------
    # 3. Train NER (optional)
    # ------------------------------------------------------------------
    if not args.skip_ner:
        print("\n" + "=" * 60)
        print("STEP 3: Training NER (BERT)")
        print("=" * 60)

        ner_cfg = NERConfig()
        ner_data_path = data_dir / "ner_queries_label.pkl"

        if ner_data_path.exists():
            import re
            ner_raw = load_pickle(ner_data_path)

            # Parse NER label format
            pattern = r"\[.*?\] = \[.*?\]"
            matches = re.findall(pattern, ner_raw)
            text_lists, number_arrays = [], []
            for match in matches:
                parts = match.split(" = ")
                text_lists.append(eval(parts[0]))
                number_arrays.append(eval(parts[1]))

            texts = [" ".join(tl) for tl in text_lists]
            labels = number_arrays

            ner_tokenizer = BertTokenizer.from_pretrained(ner_cfg.pretrained_model)
            ner_input_ids = ner_tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]
            max_seq = ner_input_ids.shape[1]

            label_ids = torch.tensor([
                lbl + [0] * (max_seq - len(lbl)) for lbl in labels
            ], dtype=torch.long)

            ner_train_ids, ner_val_ids, ner_train_lbl, ner_val_lbl = train_test_split(
                ner_input_ids, label_ids, test_size=0.2, random_state=42,
            )
            ner_train_loader = DataLoader(
                TensorDataset(ner_train_ids, ner_train_lbl),
                batch_size=ner_cfg.batch_size, shuffle=True,
            )
            ner_val_loader = DataLoader(
                TensorDataset(ner_val_ids, ner_val_lbl),
                batch_size=ner_cfg.batch_size, shuffle=False,
            )

            ner_model, ner_history = train_ner(
                ner_train_loader, ner_val_loader,
                num_labels=ner_cfg.num_labels,
                num_epochs=ner_cfg.num_epochs,
                learning_rate=ner_cfg.learning_rate,
                device=device,
                save_dir=str(output_dir / "bert-ner"),
            )
            print(f"  NER model saved to {output_dir / 'bert-ner'}")
        else:
            print(f"  NER data not found at {ner_data_path}, skipping.")

    # ------------------------------------------------------------------
    # 4. Fine-tune GPT-2
    # ------------------------------------------------------------------
    if not args.skip_gpt2:
        print("\n" + "=" * 60)
        print("STEP 4: Fine-tuning GPT-2")
        print("=" * 60)

        gpt2_cfg = GPT2Config()

        # Load query-response data
        try:
            ques_resp = load_pickle(data_dir / "ques_resp.pkl")
            ques_resp2 = load_pickle(data_dir / "ques_resp2.pkl")
            input_queries_cu = load_pickle(data_dir / "input_queries_cu.pkl")
            input_queries_responses_cu = load_pickle(data_dir / "input_queries_responses_cu.pkl")

            # Combine all queries and responses
            queries, responses = [], []
            for disease, qs in input_queries_cu.items():
                queries.extend(qs)
            for disease, rs in input_queries_responses_cu.items():
                responses.extend(rs)
            queries.extend(ques_resp["queries"])
            queries.extend(ques_resp2["queries"])
            responses.extend(ques_resp["responses"])
            responses.extend(ques_resp2["responses"])

            # Run intent classification on queries for prompt construction
            print(f"  Building prompts for {len(queries)} query-response pairs...")
            model.eval()
            ic_diseases = []
            for q in queries:
                cleaned = clean_text([q])
                ids, _ = tokenize_for_intent_classifier(cleaned, tokenizer, device=device)
                with torch.no_grad():
                    logits = model(ids)
                    pred = torch.argmax(logits, dim=1).item()
                ic_diseases.append({v: k for k, v in label_mapping.items()}[pred])

            # Build prompt-completion pairs
            dataset_pairs = []
            for i, (q, r) in enumerate(zip(queries, responses)):
                prompt = build_prompt(q, ic_diseases[i], knowledge_graph)
                dataset_pairs.append({"prompt": prompt, "response": r})

            # Tokenize for GPT-2
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_cfg.pretrained_model)
            gpt2_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_cfg.pretrained_model)
            gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

            print(f"  {count_parameters(gpt2_model)}")

            def process_pairs(data, tok):
                input_ids_list, masks_list = [], []
                for item in data:
                    enc = tok.encode_plus(
                        item["prompt"] + item["response"],
                        add_special_tokens=True,
                        max_length=gpt2_cfg.max_length,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    input_ids_list.append(enc["input_ids"])
                    masks_list.append(enc["attention_mask"])
                return torch.cat(input_ids_list), torch.cat(masks_list)

            train_pairs, test_pairs = train_test_split(dataset_pairs, test_size=0.1, random_state=42)
            train_pairs, val_pairs = train_test_split(train_pairs, test_size=1 / 9, random_state=42)

            train_ids, train_masks = process_pairs(train_pairs, gpt2_tokenizer)
            val_ids, val_masks = process_pairs(val_pairs, gpt2_tokenizer)

            from torch.utils.data import RandomSampler, SequentialSampler

            gpt2_train_loader = DataLoader(
                TensorDataset(train_ids, train_masks),
                sampler=RandomSampler(TensorDataset(train_ids, train_masks)),
                batch_size=gpt2_cfg.batch_size,
            )
            gpt2_val_loader = DataLoader(
                TensorDataset(val_ids, val_masks),
                sampler=SequentialSampler(TensorDataset(val_ids, val_masks)),
                batch_size=gpt2_cfg.batch_size,
            )

            gpt2_save_dir = str(output_dir / "gpt2")
            gpt2_history = train_gpt2(
                gpt2_model, gpt2_train_loader, gpt2_val_loader,
                num_epochs=gpt2_cfg.num_epochs,
                learning_rate=gpt2_cfg.learning_rate,
                eps=gpt2_cfg.eps,
                device=device,
                save_dir=gpt2_save_dir,
            )
            gpt2_tokenizer.save_pretrained(gpt2_save_dir)
            print(f"  GPT-2 saved to {gpt2_save_dir}")

        except FileNotFoundError as e:
            print(f"  Could not load GPT-2 training data: {e}")
            print("  Skipping GPT-2 fine-tuning.")

    # ------------------------------------------------------------------
    # 5. RL Fine-tuning (optional)
    # ------------------------------------------------------------------
    if not args.skip_rl and not args.skip_gpt2:
        print("\n" + "=" * 60)
        print("STEP 5: RL Fine-tuning with Cosine Similarity Rewards")
        print("=" * 60)

        rl_cfg = RLConfig()

        def generate_fn(mdl, tok, prompt):
            ids = tok.encode(prompt, return_tensors="pt").to(device)
            out = mdl.generate(
                ids, max_length=rl_cfg.reward_gen_max_length,
                pad_token_id=tok.eos_token_id,
            )
            return tok.decode(out[0], skip_special_tokens=True)

        try:
            rl_history = train_rl_loop(
                gpt2_model, gpt2_tokenizer,
                train_data=train_pairs,
                val_data=val_pairs,
                generate_fn=generate_fn,
                num_epochs=rl_cfg.num_epochs,
                batch_size=rl_cfg.batch_size,
                learning_rate=rl_cfg.learning_rate,
                similarity_threshold=rl_cfg.similarity_threshold,
                device=device,
            )
            rl_save_dir = str(output_dir / "gpt2-rl")
            gpt2_model.save_pretrained(rl_save_dir)
            gpt2_tokenizer.save_pretrained(rl_save_dir)
            print(f"  RL-tuned GPT-2 saved to {rl_save_dir}")
        except Exception as e:
            print(f"  RL training failed: {e}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
