"""
evals/llm_judge.py
==================
DeepEval-compatible custom LLM Judge adapters with full asynchronous concurrency.
Ensures students and practitioners can run evaluation suites at maximum parallel speed
without blocking Python's asyncio event loop.

Supported Providers:
  1. GeminiJudge    : Google Gemini 3.1 Flash-lite / Flash via native async SDK
  2. GroqJudge      : High-speed Groq with async rate-limiting (~1.5s per eval)
  3. SenseNovaJudge : SenseNova via AsyncOpenAI with parallel JSON-mode enforcement
"""

import os
import asyncio
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

load_dotenv(override=True)


class GeminiJudge(DeepEvalBaseLLM):
    """DeepEval judge adapter wrapping Google Gemini with native async support."""
    def __init__(self, model_name: str = "gemini-3.1-flash-lite"):
        self.model_name = os.environ.get("GEMINI_MODEL", model_name)
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, *args, schema=None, **kwargs) -> str:
        from google.genai import types
        config = None
        if schema:
            prompt = f"{prompt}\n\nPlease respond strictly in valid JSON format matching schema:\n{schema}"
            config = types.GenerateContentConfig(response_mime_type="application/json")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text

    async def a_generate(self, prompt: str, *args, schema=None, **kwargs) -> str:
        from google.genai import types
        config = None
        if schema:
            prompt = f"{prompt}\n\nPlease respond strictly in valid JSON format matching schema:\n{schema}"
            config = types.GenerateContentConfig(response_mime_type="application/json")
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text

    def get_model_name(self) -> str:
        return f"Gemini/{self.model_name}"


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge adapter wrapping Groq with AsyncGroq and automatic rate-limit backoff."""
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = os.environ.get("GROQ_MODEL", model_name)
        from groq import Groq, AsyncGroq
        api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.async_client = AsyncGroq(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, *args, schema=None, **kwargs) -> str:
        extra = {}
        if schema:
            prompt = f"{prompt}\n\nPlease respond strictly in valid JSON format matching schema:\n{schema}"
            extra["response_format"] = {"type": "json_object"}
        for attempt in range(6):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=0.0,
                    **extra
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    import time
                    time.sleep(1.5 * (attempt + 1))
                else:
                    raise e

    async def a_generate(self, prompt: str, *args, schema=None, **kwargs) -> str:
        extra = {}
        if schema:
            prompt = f"{prompt}\n\nPlease respond strictly in valid JSON format matching schema:\n{schema}"
            extra["response_format"] = {"type": "json_object"}
        for attempt in range(6):
            try:
                chat_completion = await self.async_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=0.0,
                    **extra
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    await asyncio.sleep(1.5 * (attempt + 1))
                else:
                    raise e

    def get_model_name(self) -> str:
        return f"Groq/{self.model_name}"


class SenseNovaJudge(DeepEvalBaseLLM):
    """DeepEval judge adapter wrapping SenseNova with AsyncOpenAI concurrency and JSON mode."""
    def __init__(self, model_name: str = "sensenova-6.8-flash-lite"):
        self.model_name = os.environ.get("SENSENOVA_MODEL", model_name)
        from openai import OpenAI, AsyncOpenAI
        api_key = os.environ.get("SENSENOVA_API_KEY")
        base_url = "https://token.sensenova.ai/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.fallback_judge = None
        if os.environ.get("GROQ_API_KEY"):
            try:
                self.fallback_judge = GroqJudge()
            except Exception:
                pass

    def load_model(self):
        return self.client

    def generate(self, prompt: str, *args, schema=None, **kwargs) -> str:
        extra = {}
        if schema:
            prompt = f"{prompt}\n\nPlease respond strictly in valid JSON format matching schema:\n{schema}"
            extra["response_format"] = {"type": "json_object"}
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0,
                **extra
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if self.fallback_judge:
                print(f"[SenseNovaJudge] SenseNova error ({e}). Falling back to Groq...")
                return self.fallback_judge.generate(prompt, *args, schema=schema, **kwargs)
            raise e

    async def a_generate(self, prompt: str, *args, schema=None, **kwargs) -> str:
        extra = {}
        if schema:
            prompt = f"{prompt}\n\nPlease respond strictly in valid JSON format matching schema:\n{schema}"
            extra["response_format"] = {"type": "json_object"}
        try:
            chat_completion = await self.async_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0,
                **extra
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if self.fallback_judge:
                print(f"[SenseNovaJudge] SenseNova async error ({e}). Falling back to Groq...")
                return await self.fallback_judge.a_generate(prompt, *args, schema=schema, **kwargs)
            raise e

    def get_model_name(self) -> str:
        return f"SenseNova/{self.model_name}"


def get_judge_model(prefer_gemini: bool = True):
    """
    Returns an appropriate LLM Judge for DeepEval metrics.
    Directly uses Google Gemini (Gemini 3.1 Flash-Lite) for all evaluations.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is not set in .env. Please set GEMINI_API_KEY to run evaluations.")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return GeminiJudge(model_name=model_name)
