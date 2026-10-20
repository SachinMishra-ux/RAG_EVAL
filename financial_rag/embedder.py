"""
financial_rag/embedder.py
=========================
Embedding models for the Financial RAG pipeline.
Provides:
  1. GeminiEmbedder (multimodal: text + PDF page images via google-genai)
  2. LocalEmbedder (offline fast text embeddings via sentence-transformers)
"""

import os
import time
import random
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class GeminiEmbedder:
    """
    Multimodal embedder using Google GenAI API.
    Supports both text queries and high-resolution PDF page images.
    Implements robust exponential backoff for HTTP 429 rate limits.
    """
    def __init__(self, model_name: str = "text-embedding-004"):
        # If user specifies gemini-embedding-2 or text-embedding-004
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed_image(self, image: Image.Image, max_retries: int = 6, initial_backoff: float = 3.0) -> list[float]:
        """Embeds a single PIL Image with exponential backoff on rate limits."""
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=image
                )
                return result.embeddings[0].values
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(1.0, 2.5)
                    print(f"Rate limit hit on image embedding. Backing off {sleep_time:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    raise e

        # Final attempt
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=image
        )
        return result.embeddings[0].values

    def embed_text(self, text: str, max_retries: int = 5, initial_backoff: float = 2.0) -> list[float]:
        """Embeds a text query or document chunk with automatic retry."""
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                return result.embeddings[0].values
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(0.5, 1.5)
                    print(f"Rate limit hit on text embedding. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    raise e

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return result.embeddings[0].values


class LocalEmbedder:
    """
    Fast, offline, CPU-friendly text embedder using Sentence-Transformers.
    Ideal for local evaluation loops and low-latency testing without API keys.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


def get_default_embedder():
    """
    Factory helper: returns LocalEmbedder by default for rock-solid reliability and speed,
    or GeminiEmbedder if explicitly requested and key is verified.
    """
    use_gemini = os.environ.get("USE_GEMINI_EMBEDDINGS", "false").lower() == "true"
    if use_gemini and os.environ.get("GEMINI_API_KEY"):
        try:
            embedder = GeminiEmbedder()
            return embedder
        except Exception:
            pass
    return LocalEmbedder()
