"""Message with content blocks, matching Anthropic Messages API."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .content import ContentBlock, TextBlock

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock] = Field(default_factory=list)
    reasoning_content: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_str_content(cls, values: dict) -> dict:
        """Allow constructing with content='string' for convenience."""
        c = values.get("content")
        if isinstance(c, str):
            values["content"] = [TextBlock(text=c).model_dump()]
        return values

    @property
    def text(self) -> str:
        """Concatenate all TextBlock content, stripping thinking artifacts.

        Thinking-trained models (Qwen3, DeepSeek-R1, QwQ, etc.) emit
        internal reasoning that needs to be stripped before grading. Three
        shapes appear in practice:

        1. Server runs a reasoning-parser (e.g. vLLM --reasoning-parser
           qwen3) — content is already stripped, reasoning lives in
           reasoning_content. Parser may leak natural-language answers
           into reasoning_content when the model doesn't close </think>;
           fall back to reasoning_content if content is empty.
        2. Chat template primes assistant turns with `<think>\\n` and
           model continues thinking, then emits `</think>` and the
           answer. Response text starts mid-think with no opening tag —
           split on the last `</think>` and take what follows.
        3. Model emits both `<think>...</think>` inline — strip them.
        """
        text = "\n".join(b.text for b in self.content if hasattr(b, "text") and b.type == "text")
        # Case 2: thinking primer in prompt, model closes with </think>
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        # Case 3: full <think>...</think> blocks inline
        text = _THINK_RE.sub("", text).strip()
        if text:
            return text
        # Case 1: reasoning-parser leaked answer into reasoning_content
        if self.reasoning_content:
            return self.reasoning_content.strip()
        return text
