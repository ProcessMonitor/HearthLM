"""Chat session state: message history, prompt assembly, stop handling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data.chat_template import THINK_CLOSE, THINK_OPEN, render_text


@dataclass
class ChatSession:
    system: Optional[str] = None
    enable_thinking: bool = False
    tools: Optional[list[dict]] = None
    messages: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str, reasoning: Optional[str] = None) -> None:
        msg = {"role": "assistant", "content": text}
        if reasoning:
            msg["reasoning_content"] = reasoning
        self.messages.append(msg)

    def reset(self) -> None:
        self.messages.clear()

    def build_prompt(self) -> str:
        msgs = list(self.messages)
        if self.system:
            msgs = [{"role": "system", "content": self.system}] + msgs
        return render_text(
            msgs,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            tools=self.tools,
        )

    def trim(self, max_tokens: int, tokenizer, keep_last: int = 2) -> None:
        """Drop the oldest turns until the prompt fits, always keeping the tail."""
        while len(self.messages) > keep_last:
            if len(tokenizer.encode(self.build_prompt())) <= max_tokens:
                return
            del self.messages[0]
            if self.messages and self.messages[0]["role"] == "assistant":
                del self.messages[0]


def split_thinking(text: str) -> tuple[str, str]:
    """Return (reasoning, answer) from a raw generation."""
    if THINK_CLOSE in text:
        head, _, tail = text.partition(THINK_CLOSE)
        reasoning = head.replace(THINK_OPEN, "").strip()
        return reasoning, tail.lstrip("\n")
    return "", text