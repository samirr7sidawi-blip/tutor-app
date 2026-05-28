"""OpenAI provider — ``openai`` SDK, streaming + multi-turn.

OpenAI automatically caches long prompt prefixes (>1024 tokens) server-side,
so the system+PDF block we re-send each turn is cached with no extra config.
"""
import os
from typing import Iterator

from openai import OpenAI

from .base import LLMProvider, MissingAPIKey

TRUNCATION_NOTE = "\n\n_⚠️ Hit the length limit — say \"continue\" for the rest._"


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise MissingAPIKey("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key)
        return self._client

    def chat(self, system: str, history: list[dict], user: str, images: list = None) -> Iterator[str]:
        client = self._get_client()

        messages = [{"role": "system", "content": system}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        if images:
            content = [{"type": "text", "text": user}]
            for im in images:
                url = f"data:{im['media_type']};base64,{im['data']}"
                content.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user})

        finish_reason = None
        stream = client.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        for chunk in stream:
            if not chunk.choices:  # e.g. trailing usage-only chunk
                continue
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                yield choice.delta.content
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if finish_reason == "length":
            yield TRUNCATION_NOTE
