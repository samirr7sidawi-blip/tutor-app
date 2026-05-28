"""Claude provider — Anthropic SDK, streaming + multi-turn + prompt caching.

The system prompt + PDF knowledge base (~15k tokens) is re-sent every turn,
so it's passed as a ``cache_control: ephemeral`` block. Across a study
session's 20-30 turns that stable prefix is served from cache (~0.1x input
cost, lower latency) instead of being reprocessed each time. Sonnet 4.6's
minimum cacheable prefix is 2048 tokens, so the block is comfortably large
enough to cache.
"""
import os
from typing import Iterator

import anthropic

from .base import LLMProvider, MissingAPIKey

MAX_TOKENS = 8192  # headroom for long derivations/proofs; Sonnet 4.6 streams up to 64k


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self._client = None
        self.thinking = False      # set per-turn by the app (adaptive thinking toggle)
        self.last_usage = None     # populated after each chat() for the cost readout

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise MissingAPIKey("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def chat(self, system: str, history: list[dict], user: str, images: list = None) -> Iterator[str]:
        client = self._get_client()

        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        if images:
            content = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": im["media_type"], "data": im["data"]},
                }
                for im in images
            ]
            content.append({"type": "text", "text": user})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user})

        # System must be a block array (not a plain string) for cache_control to apply.
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

        kwargs = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            messages=messages,
        )
        # Adaptive thinking: Claude decides how much to reason. Better for proofs/
        # rigorous math, at the cost of a longer pause before the answer streams.
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        # SDK auto-retries 429/5xx with backoff, so no manual retry loop here.
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()

        self.last_usage = final.usage
        if final.stop_reason == "max_tokens":
            yield "\n\n_⚠️ Hit the length limit — say \"continue\" for the rest._"
