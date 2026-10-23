import os
from services.config.workout_config import PROMPT

# Groq retired llama-3.3-70b-versatile on 2026-08-16.
# Override with GROQ_MODEL in .env if Groq changes models again.
DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMCoach:
    def __init__(self, groq_client):
        self.client = groq_client
        self.history = []
        self.system_prompt = PROMPT
        self.model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    def give_feedback(self, event, issue):
        prompt = f"Event: {event}"

        if issue:
            prompt += f" Form Issue: {issue}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history[-10:],
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=200
        )

        text = (response.choices[0].message.content or "").strip()

        self.history.append({"role": "assistant", "content": text})

        return text