"""Gemini provider — Google ``google.genai`` SDK, native streaming + multi-turn.

Lifted from the original single-file ``tutor_app.py`` (string-concat) approach
and adapted to the :class:`LLMProvider` interface: a real multi-turn Content
array plus streaming, with the 429 retry/backoff preserved.
"""
import base64
import os
import time
from typing import Iterator

from google import genai
from google.genai import errors, types

from .base import LLMProvider, MissingAPIKey

MAX_RETRIES = 3
MAX_OUTPUT_TOKENS = 8192
TRUNCATION_NOTE = "\n\n_⚠️ Hit the length limit — say \"continue\" for the rest._"


def _hit_max_tokens(finish_reason) -> bool:
    """True if a Gemini finish_reason signals the output token cap (defensive)."""
    return finish_reason is not None and str(getattr(finish_reason, "name", finish_reason)) == "MAX_TOKENS"


class GeminiProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self._client = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise MissingAPIKey("GEMINI_API_KEY")
            self._client = genai.Client(api_key=key)
        return self._client

    def chat(self, system: str, history: list[dict], user: str, images: list = None) -> Iterator[str]:
        client = self._get_client()

        # Native multi-turn Content array. Gemini uses "model" for the assistant role.
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in history
        ]
        user_parts = [
            types.Part.from_bytes(data=base64.b64decode(im["data"]), mime_type=im["media_type"])
            for im in (images or [])
        ]
        user_parts.append(types.Part.from_text(text=user))
        contents.append(types.Content(role="user", parts=user_parts))

        config = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=MAX_OUTPUT_TOKENS
        )

        # Retry per-minute 429s with linear backoff, mirroring the original app.
        # Only retry before any text has been emitted, to avoid duplicating output.
        for attempt in range(MAX_RETRIES):
            yielded = False
            finish_reason = None
            try:
                stream = client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                )
                for chunk in stream:
                    if chunk.text:
                        yielded = True
                        yield chunk.text
                    try:
                        fr = chunk.candidates[0].finish_reason
                        if fr is not None:
                            finish_reason = fr
                    except (AttributeError, IndexError, TypeError):
                        pass
                if _hit_max_tokens(finish_reason):
                    yield TRUNCATION_NOTE
                return
            except errors.APIError as e:
                if getattr(e, "code", None) == 429 and not yielded and attempt < MAX_RETRIES - 1:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise
