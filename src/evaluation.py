"""Evaluation metrics for HealthBot models."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Classification metrics (intent classifier / NER)
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix.

    Returns:
        Dict with 'accuracy', 'report' (str), 'confusion_matrix' (np.ndarray),
        and 'per_class_accuracy' (dict).
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = {}
    if class_names is not None:
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        per_class_acc = {
            name: float(class_accuracies[i])
            for i, name in enumerate(class_names)
        }

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
    }


def ner_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """Compute macro-averaged precision, recall, and F1 for NER.

    Returns:
        Dict with 'precision', 'recall', 'f1'.
    """
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def bleu_score(reference: str, hypothesis: str) -> float:
    """Compute smoothed sentence-level BLEU."""
    ref_tokens = [nltk.word_tokenize(reference)]
    hyp_tokens = nltk.word_tokenize(hypothesis)
    smoothing = SmoothingFunction()
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothing.method1)


def rouge_scores(reference: str, hypothesis: str, n_grams: List[int] = None) -> Dict[str, float]:
    """Compute ROUGE-N F-measure scores.

    Args:
        reference: Ground-truth text.
        hypothesis: Generated text.
        n_grams: List of n-gram sizes (default: [1, 2]).

    Returns:
        Dict mapping 'rouge-N' to F-measure.
    """
    if n_grams is None:
        n_grams = [1, 2]
    scorer = rouge_scorer.RougeScorer(
        [f"rouge{n}" for n in n_grams], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {f"rouge-{n}": scores[f"rouge{n}"].fmeasure for n in n_grams}


def semantic_similarity(
    generated_texts: List[str],
    reference_texts: List[str],
    sentence_model=None,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute average semantic similarity using a SentenceTransformer model.

    Args:
        generated_texts: Model outputs.
        reference_texts: Ground-truth answers.
        sentence_model: A SentenceTransformer instance. If None, falls back to
            bag-of-words cosine similarity.
        device: Torch device.

    Returns:
        Average cosine similarity score (0–1).
    """
    if sentence_model is not None:
        similarities = []
        for gen, ref in zip(generated_texts, reference_texts):
            emb_gen = sentence_model.encode(gen, convert_to_tensor=True).to(device)
            emb_ref = sentence_model.encode(ref, convert_to_tensor=True).to(device)
            sim = F.cosine_similarity(emb_gen.unsqueeze(0), emb_ref.unsqueeze(0), dim=1)
            similarities.append(sim.item())
        return float(np.mean(similarities))

    # Fallback: bag-of-words cosine similarity
    vectorizer = CountVectorizer()
    all_texts = generated_texts + reference_texts
    vectors = vectorizer.fit_transform(all_texts).toarray()
    n = len(generated_texts)
    gen_vecs = vectors[:n]
    ref_vecs = vectors[n:]
    sims = [
        float(sklearn_cosine(gen_vecs[i : i + 1], ref_vecs[i : i + 1])[0][0])
        for i in range(n)
    ]
    return float(np.mean(sims))


def evaluate_generation(
    generated: List[str],
    references: List[str],
    sentence_model=None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Run all generation metrics and return a summary dict."""
    bleu_scores_list = [
        bleu_score(ref, gen) for ref, gen in zip(references, generated)
    ]
    rouge_scores_list = [
        rouge_scores(ref, gen) for ref, gen in zip(references, generated)
    ]
    avg_bleu = float(np.mean(bleu_scores_list))
    avg_rouge = {
        key: float(np.mean([r[key] for r in rouge_scores_list]))
        for key in rouge_scores_list[0]
    }
    avg_semantic = semantic_similarity(generated, references, sentence_model, device)

    return {
        "bleu": avg_bleu,
        **avg_rouge,
        "semantic_similarity": avg_semantic,
    }
