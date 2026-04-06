"""Environment variable loading and OpenAI key management."""

import os

def require_openai_api_key(reason: str = "OpenAI API calls") -> str:
    """Ensure OPENAI_API_KEY is set; raise error with context if not."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(f"OPENAI_API_KEY is not set. Required for: {reason}")
    return key
