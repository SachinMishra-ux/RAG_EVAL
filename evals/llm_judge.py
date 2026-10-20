"""
evals/llm_judge.py
==================
DeepEval-compatible custom LLM Judge adapter.
Ensures students and practitioners can run the entire evaluation suite seamlessly
regardless of whether they are using OpenAI, Groq, or Google Gemini.

DeepEval natively expects an LLM instance inheriting from DeepEvalBaseLLM
or a model name string supported by its backends.
"""

import os
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

load_dotenv()


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge adapter wrapping Groq's high-speed inference."""
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name
        from groq import Groq
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.0,
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Groq/{self.model_name}"


class GeminiJudge(DeepEvalBaseLLM):
    """DeepEval judge adapter wrapping Google Gemini."""
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        from google import genai
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Gemini/{self.model_name}"


def get_judge_model():
    """
    Returns an appropriate LLM Judge for DeepEval metrics.
    Priority:
      1. OPENAI_API_KEY -> "gpt-4o-mini" (standard DeepEval default)
      2. GROQ_API_KEY   -> GroqJudge("openai/gpt-oss-20b")
      3. GEMINI_API_KEY -> GeminiJudge("gemini-2.5-flash")
    """
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    elif os.environ.get("GROQ_API_KEY"):
        model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        return GroqJudge(model_name=model_name)
    elif os.environ.get("GEMINI_API_KEY"):
        return GeminiJudge()
    else:
        # Fallback to standard string
        return "gpt-4o-mini"
