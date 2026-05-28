"""Provider abstraction: one swappable interface over Claude, Gemini, and OpenAI.

Each provider streams text chunks so the Streamlit UI can render replies
progressively via ``st.write_stream()``.
"""
from abc import ABC, abstractmethod
from typing import Iterator


class MissingAPIKey(RuntimeError):
    """A provider was selected but its API key isn't set in the environment.

    Carries the missing env var name as its message (e.g. ``"GEMINI_API_KEY"``)
    so the UI can show a friendly hint instead of a raw traceback.
    """


class LLMProvider(ABC):
    """Common interface for all chat models.

    Implementations read their API key from the environment and create their
    SDK client lazily (on the first ``chat()`` call), so a missing key for a
    provider the user never selects can't break the app.
    """

    @abstractmethod
    def chat(self, system: str, history: list[dict], user: str, images: list = None) -> Iterator[str]:
        """Stream a reply, one text chunk at a time.

        Args:
            system: System instructions + PDF knowledge base, as one stable
                block (the orchestrator concatenates them).
            history: Prior turns as ``[{"role": "user"|"assistant",
                "content": str}, ...]``. An extra ``"model"`` key on assistant
                turns (recorded by the app) is ignored by providers.
            user: The latest user message.
            images: Optional list of images to attach to this user turn, each
                ``{"media_type": str, "data": <base64 str>}``. Images apply to
                the current turn only (not stored in history).

        Yields:
            Text chunks of the assistant's reply, in order.
        """
        ...
