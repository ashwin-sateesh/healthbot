"""Knowledge graph and prompt-engineering utilities for HealthBot."""

from __future__ import annotations

from typing import Any, Dict, List


def build_prompt(
    user_query: str,
    disease: str,
    knowledge_graph: Dict[str, Dict[str, List[str]]],
) -> str:
    """Construct a knowledge-grounded prompt for the response-generation LLM.

    The prompt injects symptom and medicine context from the knowledge graph so
    the model can produce medically relevant answers.

    Args:
        user_query: The user's original question.
        disease: Predicted disease label from the intent classifier.
        knowledge_graph: ``{disease: {"symptoms": [...], "medicines": [...]}}``

    Returns:
        Fully assembled prompt string.
    """
    disease_info = knowledge_graph.get(disease, {"symptoms": [], "medicines": []})
    symptom_info = "\n".join(f"- {s}" for s in disease_info["symptoms"])
    medicine_info = "\n".join(f"- {m}" for m in disease_info["medicines"])

    prompt = (
        f"For the context of {disease}, please consider the symptom and medicine "
        f"information below:\n\n"
        f"Symptoms for {disease}:\n{symptom_info}\n\n"
        f"Medicines for {disease}:\n{medicine_info}\n\n"
        f"{user_query}\n\n"
        f"Please provide an answer to the following question:"
    )
    return prompt


def build_flan_t5_prompt(
    user_query: str,
    disease: str,
    knowledge_graph: Dict[str, Dict[str, List[str]]],
    max_kg_items: int = 5,
) -> str:
    """Build a prompt formatted for FLAN-T5 instruction tuning.

    Args:
        user_query: The user's question.
        disease: Predicted disease.
        knowledge_graph: Disease → symptoms/medicines mapping.
        max_kg_items: Max number of symptoms/medicines to include.

    Returns:
        Instruction-style prompt.
    """
    disease_info = knowledge_graph.get(disease, {"symptoms": [], "medicines": []})
    symptoms = disease_info["symptoms"][:max_kg_items]
    medicines = disease_info["medicines"][:max_kg_items]

    prompt = (
        f"Answer the following question and give a valid response.\n"
        f"Also consider this information while answering.\n"
        f"Information: Symptoms and Medicines for {disease} are\n"
        f"  Symptoms - {symptoms}\n"
        f"  Medicines - {medicines}\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )
    return prompt
