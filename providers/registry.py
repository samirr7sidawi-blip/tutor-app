"""Model registry: dropdown display name -> provider instance.

Providers create their SDK client lazily, so instantiating all of them here is
cheap and safe even when some API keys are missing — only the selected model's
client is built, on its first chat() call.
"""
from .base import LLMProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

# Insertion order = dropdown order.
PROVIDERS: dict[str, LLMProvider] = {
    "Claude Sonnet 4.6": ClaudeProvider("claude-sonnet-4-6"),
    "Gemini 2.5 Pro": GeminiProvider("gemini-2.5-pro"),
    "Gemini 2.5 Flash": GeminiProvider("gemini-2.5-flash"),
    "GPT (current)": OpenAIProvider("gpt-5.5"),  # verified against the live OpenAI account at build
}

DEFAULT_MODEL = "Claude Sonnet 4.6"
