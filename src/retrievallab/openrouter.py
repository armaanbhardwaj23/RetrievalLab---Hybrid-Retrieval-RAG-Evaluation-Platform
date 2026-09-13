"""Shared OpenRouter client configuration."""

import os

from dotenv import load_dotenv
from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_api_key() -> str:
    """Load the OpenRouter key without exposing it in application output."""

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required. Add it to .env or your environment."
        )
    return api_key


def openrouter_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client pointed at OpenRouter."""

    return OpenAI(api_key=api_key or openrouter_api_key(), base_url=OPENROUTER_BASE_URL)
