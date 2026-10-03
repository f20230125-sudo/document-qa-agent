"""Shared test helpers. Gemini is always stubbed: no API key, no network."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class StubClient:
    """Stands in for genai.Client.

    `embed` maps a text to its embedding (a list of floats); `answer` is the
    text returned by the chat model. Every call is recorded.
    """

    def __init__(self, embed, answer=""):
        self.embed = embed
        self.answer = answer
        self.embed_calls = []
        self.chat_calls = []
        self.models = SimpleNamespace(
            embed_content=self._embed_content,
            generate_content=self._generate_content,
        )

    def _embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        values = self.embed(kwargs["contents"])
        return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])

    def _generate_content(self, **kwargs):
        self.chat_calls.append(kwargs)
        return SimpleNamespace(text=self.answer)
