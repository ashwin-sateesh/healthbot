# HealthBot: Intelligent Healthcare Assistant using Large Language Models

An end-to-end NLP pipeline that answers medical queries by combining disease intent classification, named entity recognition, knowledge-graph-grounded prompt engineering, and fine-tuned language models with reinforcement learning from cosine-similarity rewards.

![Architecture](/assets/healthbot_architecture.png)

## Architecture Overview

The system operates in two phases:

**Phase 1 — Understanding:** A user query is processed in parallel by an **LSTM-with-Attention intent classifier** (to identify the target disease) and a **BERT-based NER model** (to extract medical entities such as symptoms, medications, and body parts). The classified disease is matched against a structured **knowledge graph** containing the top symptoms and medicines per condition.

**Phase 2 — Generation:** The extracted context is assembled into a knowledge-grounded prompt using a start/end prompt template. This prompt is fed to a **fine-tuned GPT-2** (or FLAN-T5) model for response generation. Generated responses are then refined via a **reinforcement learning loop** that uses cosine similarity between generated and reference response embeddings as the reward signal, iteratively updating model weights until responses exceed a similarity threshold.

## Project Structure

```
healthbot/
├── configs/
│   ├── __init__.py
│   └── config.py              # Centralized hyperparameters and path configs
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Text cleaning utilities
│   ├── data.py                # Data loading, tokenization, dataset classes
│   ├── models.py              # LSTMWithAttention, BiRNN, Transformer, LoRA, PolicyNetwork
│   ├── prompts.py             # Knowledge-graph prompt engineering
│   ├── training.py            # Training loops (intent, NER, GPT-2, FLAN-T5, RL)
│   ├── evaluation.py          # BLEU, ROUGE, semantic similarity, classification metrics
│   └── inference.py           # HealthBot class and interactive chat loop
├── scripts/
│   ├── train.py               # Full training pipeline entry point
│   ├── evaluate.py            # Model evaluation script
│   └── chat.py                # Interactive chatbot entry point
├── data/                      # Pickle/JSON data files (not tracked in git)
├── artifacts/                 # Trained model checkpoints (not tracked in git)
├── assets/                    # Architecture diagrams and images
├── requirements.txt
├── .gitignore
└── README.md
```

## Models

| Component | Model | Parameters | Metric |
|---|---|---|---|
| Intent Classification | Bidirectional LSTM + Attention | 4.6M | 96% validation accuracy |
| Named Entity Recognition | BERT Base Uncased | 109M | 0.84 F1 score |
| Response Generation | GPT-2 Base | 124M | 0.70 semantic similarity |
| Response Generation (alt) | FLAN-T5 Base | 248M | Semantic similarity |
| RL Refinement | Cosine similarity reward | — | Reward-weighted loss updates |

## Setup

```bash
# Clone the repository
git clone https://github.com/ashwin-sateesh/healthbot.git
cd healthbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Place the following pickle files in the `data/` directory:

- `med_data_all.pkl` — Combined query-label pairs (~5000 queries across 16 disease categories)
- `knowledge_graph.pkl` — Disease → symptoms/medicines hierarchical mapping
- `ques_resp.pkl`, `ques_resp2.pkl` — Query-response pairs for GPT-2 fine-tuning
- `input_queries_cu.pkl`, `input_queries_responses_cu.pkl` — Additional contextual query-response data
- `ner_queries_label.pkl` — NER-labeled token sequences

## Training

Train all models end-to-end:

```bash
python scripts/train.py --data-dir ./data --output-dir ./artifacts
```

Options:
- `--skip-ner` — Skip NER training
- `--skip-gpt2` — Skip GPT-2 fine-tuning
- `--skip-rl` — Skip reinforcement learning loop
- `--device cuda` — Force GPU (default: auto-detect)

## Evaluation

```bash
python scripts/evaluate.py --artifacts-dir ./artifacts --data-dir ./data
```

Outputs per-class intent classification metrics, NER precision/recall/F1, and generation-quality scores (BLEU, ROUGE, semantic similarity).

## Interactive Chat

```bash
python scripts/chat.py --artifacts-dir ./artifacts --data-dir ./data
```

Options:
- `--temperature 0.7` — Control response randomness
- `--max-length 300` — Maximum generation length

Example interaction:

```
HealthBot
----------------------------------------
Hello! I'm HealthBot. I can help you with general health questions.
Type 'quit' to leave the chat.

You: What are the symptoms of diabetes?
HealthBot: Here's what I found for you:
The symptoms of diabetes include increased thirst, frequent urination,
fatigue, blurred vision, and slow-healing wounds. Medicines for diabetes
include oral antidiabetic drugs, insulin, and lifestyle modifications.

You: quit
HealthBot: Goodbye!
```

## Programmatic Usage

```python
from src.data import load_pickle
from src.inference import HealthBot

knowledge_graph = load_pickle("data/knowledge_graph.pkl")
label_mapping = load_pickle("artifacts/LSTMwA/label_mapping.pkl")

bot = HealthBot.from_pretrained(
    lstm_model_path="artifacts/LSTMwA/best_model.pt",
    lstm_tokenizer_path="artifacts/LSTMwA",
    gpt2_model_path="artifacts/gpt2-rl",
    knowledge_graph=knowledge_graph,
    label_mapping=label_mapping,
)

response = bot.answer("What medicines are used for chest pain?")
print(response)
```

## Key Design Decisions

- **LSTM + Attention over Transformer for intent classification**: At 4.6M parameters versus the Transformer baseline, the LSTM achieved the highest validation accuracy (96%) while being significantly more lightweight — critical for low-latency inference.
- **Knowledge graph grounding**: Injecting structured symptom/medicine data into prompts reduces hallucination and keeps responses anchored in verified medical information.
- **Cosine similarity RL**: Using embedding-space similarity as the reward signal (rather than BLEU/ROUGE) captures semantic alignment — a generated response can use different phrasing while still receiving high reward if the meaning is preserved.
- **LoRA adapters for FLAN-T5**: Enables parameter-efficient fine-tuning with separate learning rates for adapter and base parameters.

## References

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805))
- Chung et al., "Scaling Instruction-Finetuned Language Models" ([arXiv:2210.11416](https://arxiv.org/abs/2210.11416))
- Liu et al., "NER-BERT: A Pre-trained Model for Low-Resource Entity Tagging" ([arXiv:2112.00405](https://arxiv.org/abs/2112.00405))

## License

This project was developed as part of coursework at Northeastern University.
