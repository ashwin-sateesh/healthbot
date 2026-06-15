"""Text preprocessing utilities for HealthBot."""

import re
from typing import List

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))


def clean_text(texts: List[str], remove_stopwords: bool = False) -> List[str]:
    """Lowercase, strip special characters/digits, and optionally remove stopwords.

    Args:
        texts: Raw input strings.
        remove_stopwords: If True, filter English stopwords.

    Returns:
        List of cleaned strings.
    """
    cleaned: List[str] = []
    for text in texts:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        if remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in _STOP_WORDS]
            text = " ".join(words)
        cleaned.append(text)
    return cleaned
