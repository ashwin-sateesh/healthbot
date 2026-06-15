"""Inference pipeline for the HealthBot chatbot.

This module ties together intent classification, knowledge-graph lookup,
prompt engineering, and response generation into a single inference class
that can be used interactively or programmatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from .data import tokenize_for_intent_classifier
from .models import LSTMWithAttention
from .preprocessing import clean_text
from .prompts import build_prompt


class HealthBot:
    """End-to-end inference pipeline for the HealthBot assistant.

    Usage::

        bot = HealthBot.from_pretrained(
            lstm_model_path="artifacts/LSTMwA/best_model.pt",
            lstm_tokenizer_path="artifacts/LSTMwA",
            gpt2_model_path="artifacts/gpt2-new",
            knowledge_graph=knowledge_graph,
            label_mapping=label_mapping,
        )
        response = bot.answer("What are the symptoms of diabetes?")
    """

    def __init__(
        self,
        intent_model: LSTMWithAttention,
        intent_tokenizer: BertTokenizer,
        generator_model: GPT2LMHeadModel,
        generator_tokenizer: GPT2Tokenizer,
        knowledge_graph: Dict[str, Dict[str, List[str]]],
        label_mapping: Dict[str, int],
        device: torch.device = torch.device("cpu"),
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_length: int = 300,
    ) -> None:
        self.intent_model = intent_model.to(device).eval()
        self.intent_tokenizer = intent_tokenizer
        self.generator_model = generator_model.to(device).eval()
        self.generator_tokenizer = generator_tokenizer
        self.knowledge_graph = knowledge_graph
        self.label_mapping = label_mapping
        self.inv_label_mapping = {v: k for k, v in label_mapping.items()}
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        lstm_model_path: str,
        lstm_tokenizer_path: str,
        gpt2_model_path: str,
        knowledge_graph: Dict,
        label_mapping: Dict[str, int],
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "HealthBot":
        """Load all models from disk and return a ready-to-use HealthBot."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Intent classifier
        intent_model = torch.load(lstm_model_path, map_location=device)
        intent_tokenizer = BertTokenizer.from_pretrained(lstm_tokenizer_path)

        # Response generator
        generator_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
        generator_model = GPT2LMHeadModel.from_pretrained(
            gpt2_model_path, output_hidden_states=True
        )

        return cls(
            intent_model=intent_model,
            intent_tokenizer=intent_tokenizer,
            generator_model=generator_model,
            generator_tokenizer=generator_tokenizer,
            knowledge_graph=knowledge_graph,
            label_mapping=label_mapping,
            device=device,
            **kwargs,
        )

    def classify_intent(self, text: str) -> str:
        """Predict the disease intent for a single query.

        Returns:
            Predicted disease name (str).
        """
        cleaned = clean_text([text])
        input_ids, _ = tokenize_for_intent_classifier(
            cleaned, self.intent_tokenizer, device=self.device
        )
        with torch.no_grad():
            logits = self.intent_model(input_ids)
            pred_idx = torch.argmax(logits, dim=1).item()
        return self.inv_label_mapping[pred_idx]

    def generate_response(self, prompt: str) -> str:
        """Generate a text response given a fully constructed prompt.

        Returns:
            Generated answer string.
        """
        input_ids = self.generator_tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        output = self.generator_model.generate(
            input_ids,
            max_length=self.max_length,
            pad_token_id=self.generator_tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,
            num_beams=2,
        )
        generated = self.generator_tokenizer.decode(output[0], skip_special_tokens=True)

        # Strip the prompt prefix from the response
        if generated.startswith(prompt):
            generated = generated[len(prompt) :].strip()

        return generated

    def answer(self, user_query: str) -> str:
        """Full pipeline: classify intent → build prompt → generate response.

        Args:
            user_query: Natural-language health question.

        Returns:
            HealthBot's generated answer.
        """
        disease = self.classify_intent(user_query)
        prompt = build_prompt(user_query, disease, self.knowledge_graph)
        response = self.generate_response(prompt)
        return f"Here's what I found for you:\n{response}"


def interactive_chat(bot: HealthBot) -> None:
    """Run an interactive terminal chat loop.

    Type ``quit`` to exit.
    """
    print("HealthBot")
    print("-" * 40)
    print("Hello! I'm HealthBot. I can help you with general health questions.")
    print("Type 'quit' to leave the chat.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("HealthBot: Goodbye!")
            break
        response = bot.answer(user_input)
        print(f"HealthBot: {response}\n")
