"""Chat template rendering and loss-mask derivation.

The template is a plain Python function rather than a Jinja string. Two reasons:
matching a Jinja template against the tokenizer is how assistant spans silently
break, and we need to know *where* the assistant span starts in token space in
order to build the loss mask. Rendering with explicit segments gives us both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

ROLE_TOKENS = ("<|system|>", "<|user|>", "<|assistant|>", "<|tool|>")


@dataclass
class Segment:
    """A piece of the rendered conversation and whether it contributes to loss."""

    text: str
    trainable: bool


def render_tools(tools: list[dict]) -> str:
    if not tools:
        return ""
    import json

    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    for tool in tools:
        lines.append(json.dumps(tool, ensure_ascii=False))
    lines += [
        "</tools>",
        "",
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:",
        "<tool_call>",
        '{"name": <function-name>, "arguments": <args-json-object>}',
        "</tool_call>",
    ]
    return "\n".join(lines)


def _tool_calls_text(tool_calls: Any) -> str:
    import json

    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except json.JSONDecodeError:
            return tool_calls
    chunks = []
    for call in tool_calls or []:
        fn = call.get("function", call) if isinstance(call, dict) else call
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args_text = args
        else:
            args_text = json.dumps(args, ensure_ascii=False)
        chunks.append(f'{TOOL_CALL_OPEN}\n{{"name": "{name}", "arguments": {args_text}}}\n{TOOL_CALL_CLOSE}')
    return "\n".join(chunks)


def render(
    messages: Iterable[dict],
    *,
    add_generation_prompt: bool = False,
    tools: Optional[list[dict]] = None,
    enable_thinking: bool = True,
    open_thinking: bool = False,
) -> list[Segment]:
    """Render messages into segments; assistant spans are marked trainable.

    `enable_thinking=False` writes an empty think block into the prompt so the
    model is asked for a direct answer, which is much faster at inference.
    """
    msgs = [dict(m) for m in messages]
    segments: list[Segment] = []

    sys_text = None
    if msgs and msgs[0].get("role") == "system":
        sys_text = msgs[0].get("content") or ""

    if tools:
        body = sys_text + "\n\n" + render_tools(tools) if sys_text else render_tools(tools)
        segments.append(Segment(f"{IM_START}system\n{body}{IM_END}\n", False))
        msgs = msgs[1:] if sys_text is not None else msgs
    elif sys_text is not None:
        segments.append(Segment(f"{IM_START}system\n{sys_text}{IM_END}\n", False))
        msgs = msgs[1:]

    for msg in msgs:
        role = msg.get("role")
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)

        if role == "assistant":
            reasoning = msg.get("reasoning_content")
            if reasoning is None and THINK_CLOSE in content:
                head, _, tail = content.partition(THINK_CLOSE)
                reasoning = head.replace(THINK_OPEN, "").strip("\n")
                content = tail.lstrip("\n")
            if reasoning is None:
                reasoning = ""
            think = f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n" if enable_thinking else ""
            body = think + content
            calls = msg.get("tool_calls")
            if calls:
                calls_text = _tool_calls_text(calls)
                body = body + ("\n" if body else "") + calls_text
            segments.append(Segment(f"{IM_START}assistant\n{body}{IM_END}\n", True))
        elif role == "tool":
            # Consecutive tool results are wrapped in one user turn.
            prefix = ""
            if not segments or not segments[-1].text.endswith(f"{TOOL_RESPONSE_CLOSE}"):
                prefix = f"{IM_START}user\n"
            segments.append(Segment(f"{prefix}{TOOL_RESPONSE_OPEN}\n{content}\n{TOOL_RESPONSE_CLOSE}", False))
        else:  # user / system appended mid-conversation
            segments.append(Segment(f"{IM_START}{role}\n{content}{IM_END}\n", False))

    # Close an unterminated tool-response turn.
    if segments and segments[-1].text.endswith(TOOL_RESPONSE_CLOSE):
        segments.append(Segment(f"{IM_END}\n", False))

    if add_generation_prompt:
        if enable_thinking and open_thinking:
            segments.append(Segment(f"{IM_START}assistant\n{THINK_OPEN}\n", False))
        elif enable_thinking:
            segments.append(Segment(f"{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n", False))
        else:
            segments.append(Segment(f"{IM_START}assistant\n", False))

    return segments


def render_text(messages: Iterable[dict], **kwargs) -> str:
    return "".join(s.text for s in render(messages, **kwargs))