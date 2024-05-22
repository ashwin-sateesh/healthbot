# HealthBot: Intelligent Healthcare Assistant using Large Language Models

Our approach revolves around the development of an intelligent bot,
empowered by advanced Large Language Models (LLMs), that excels in offering users comprehensive
and reliable advice. This HealthBot will emerge as a valuable resource, providing insights into
associated symptoms, recommended treatments, and relevant medical conditions. We intend to
achieve this by harnessing a diverse and meticulously curated dataset sourced from reputable medical
references. To further enhance its capabilities, we will fine-tune the LLM using state-of-the-art
transfer learning techniques.

The first step of this project involves identifying the disease intent through an
intent classifier. Simultaneously, a named entity recognizer detects relevant entities in the user input.
These two processes run in parallel, whose outputs forming the foundation for
prompt engineering. Through prompt engineering, prompt-completion pairs are generated, creating
a structured input-output framework. This framework is then employed for fine-tuning LLMs (GPT2 and FLAN-T5), which learns to generate appropriate responses based on the input. The
resulting responses undergo validation and feedback from a reward model (Cosine similarity as a reward function) through reinforcement
learning, ensuring that the model evolves to provide valid and meaningful responses. This intricate
interplay between intent classification, entity recognition, prompt engineering, fine-tuning, and reinforcement learning contributes to the robustness and effectiveness of the conversational system.

### Simplified Architecture:

![Healthbot-Architecture](https://github.com/ashwin-sateesh/healthbot/blob/main/assets/healthbot%20architecture.png)
