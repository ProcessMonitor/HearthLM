"""Interactive chat in the terminal.

    python -m hearthlm.chat --checkpoint out/checkpoints/sft.pt
    python -m hearthlm.chat --checkpoint out/checkpoints/sft.pt --thinking

Commands inside the REPL: /reset, /system <text>, /thinking on|off, /tokens,
/quit. Streaming decode with a KV cache, so long replies do not re-read the
prompt at each step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..data import Tokenizer
from ..model import SamplingParams
from ..model.cache import GrowCache
from ..train.optim import load_model_from_file
from ..train.runtime import pick_device, resolve_dtype
from .session import ChatSession, split_thinking


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HearthLM chat")
    p.add_argument("--checkpoint", default="out/checkpoints/sft.pt")
    p.add_argument("--tokenizer", default="assets/tokenizer")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--system", default=None)
    p.add_argument("--thinking", action="store_true", help="ask the model to reason before answering")
    p.add_argument("--show-thinking", action="store_true", help="print the reasoning block")
    p.add_argument("--prompt", default=None, help="one-shot: answer this and exit")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    device = pick_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    from ..config import HearthConfig
    from ..model.hearth import HearthLM

    model, config, _ = load_model_from_file(args.checkpoint, HearthLM, HearthConfig)
    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    model.eval()
    tokenizer = Tokenizer(args.tokenizer)

    session = ChatSession(system=args.system, enable_thinking=args.thinking)
    params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        do_sample=args.temperature > 0,
    )

    def answer(text: str) -> None:
        session.add_user(text)
        prompt = session.build_prompt()
        ids = tokenizer.encode(prompt)
        max_ctx = config.max_position_embeddings
        if len(ids) > max_ctx - params.max_new_tokens:
            session.trim(max_ctx - params.max_new_tokens, tokenizer)
            ids = tokenizer.encode(session.build_prompt())

        inp = torch.tensor([ids], device=device)
        cache = GrowCache(config.num_hidden_layers, max_len=max_ctx)

        shown = 0
        buf: list[str] = []
        reasoning_shown = False
        with torch.inference_mode():
            from ..model.generate import apply_penalties, sample

            out = inp
            for _ in range(params.max_new_tokens):
                logits = model(out[:, cache.length:], cache=cache).logits[:, -1]
                logits = apply_penalties(
                    logits.clone(), out,
                    repetition_penalty=params.repetition_penalty,
                )
                nxt = sample(logits, params)
                if int(nxt[0, 0]) == config.eos_token_id:
                    break
                out = torch.cat([out, nxt], dim=-1)
                buf.append(int(nxt[0, 0]))
                text_so_far = tokenizer.decode(buf)
                if text_so_far.endswith("\ufffd"):
                    continue
                reasoning, body = split_thinking(text_so_far)
                if args.show_thinking and reasoning and not reasoning_shown:
                    print("\033[2m" + reasoning + "\033[0m", end="", flush=True)
                    reasoning_shown = bool(body)
                if body and len(body) > shown:
                    print(body[shown:], end="", flush=True)
                    shown = len(body)
        print()

        full = tokenizer.decode(buf)
        reasoning, body = split_thinking(full)
        session.add_assistant(body or full, reasoning or None)

    if args.prompt is not None:
        answer(args.prompt)
        return

    print(f"HearthLM — {Path(args.checkpoint).name} on {device}. /quit to exit.")
    while True:
        try:
            line = input("\033[1m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            if cmd in {"/quit", "/exit"}:
                break
            if cmd == "/reset":
                session.reset()
                print("history cleared")
            elif cmd == "/system":
                session.system = rest.strip() or None
                print(f"system prompt: {session.system!r}")
            elif cmd == "/thinking":
                session.enable_thinking = rest.strip().lower() in {"on", "1", "true"}
                print(f"thinking: {session.enable_thinking}")
            elif cmd == "/tokens":
                print(f"{len(tokenizer.encode(session.build_prompt()))} prompt tokens, "
                      f"{len(session.messages)} messages")
            else:
                print("commands: /reset /system <text> /thinking on|off /tokens /quit")
            continue
        answer(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()