"""Evaluation: perplexity, multiple-choice accuracy, and decoding throughput.

    python -m hearthlm.eval ppl   --checkpoint out/checkpoints/sft.pt --data data/demo/sft.jsonl
    python -m hearthlm.eval mcq   --checkpoint out/checkpoints/sft.pt --data data/demo/mcq.jsonl
    python -m hearthlm.eval bench --checkpoint out/checkpoints/sft.pt

Three small measurements that answer three separate questions. Perplexity says
whether the model fits the distribution. MCQ accuracy says whether it can choose
between options at all, which is a much weaker requirement than generation.
Throughput says whether the cache path is actually saving work.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ..config import HearthConfig
from ..data import Tokenizer, collate_sft, SFTDataset
from ..model import SamplingParams, generate
from ..model.cache import GrowCache
from ..model.hearth import HearthLM
from ..train.optim import load_model_from_file
from ..train.runtime import pick_device, resolve_dtype


def _load(path: str, device: torch.device, dtype: torch.dtype):
    model, config, _ = load_model_from_file(path, HearthLM, HearthConfig)
    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    return model.eval(), config


def cmd_ppl(args) -> None:
    device = pick_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, config = _load(args.checkpoint, device, dtype)
    tokenizer = Tokenizer(args.tokenizer)
    ds = SFTDataset(args.data, tokenizer, max_len=args.seq_len)

    total_loss, total_tokens = 0.0, 0
    per_row = []
    with torch.no_grad():
        for start in range(0, len(ds), args.batch_size):
            rows = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
            batch = collate_sft(rows, tokenizer.pad_token_id)
            if batch["input_ids"].numel() == 0:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"])
            n = int(batch["labels"][:, 1:].ne(-100).sum())
            total_loss += float(out.loss) * n
            total_tokens += n
            per_row.append(float(out.loss))

    ppl = float(torch.exp(torch.tensor(total_loss / max(total_tokens, 1))))
    print(f"tokens={total_tokens} loss={total_loss / max(total_tokens, 1):.4f} ppl={ppl:.3f}")
    if per_row:
        print(f"per-batch loss: min={min(per_row):.3f} median={statistics.median(per_row):.3f} max={max(per_row):.3f}")


def cmd_mcq(args) -> None:
    """Score each option by the log-prob of its letter and pick the best."""
    device = pick_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, config = _load(args.checkpoint, device, dtype)
    tokenizer = Tokenizer(args.tokenizer)

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    correct = 0
    for row in rows:
        question = row["question"]
        options = row["options"]
        answer = row["answer"]

        # All four options share one identical prefix so the comparison is fair.
        # Options are scored by the mean log-prob per token, which is the standard
        # length-normalised form. Summing raw log-probs would instead reward short
        # options, since every extra token subtracts from the total.
        prefix = (f"{question}\n"
                  + "".join(f"{letter}. {text}\n" for letter, text in zip("ABCD", options))
                  + "Answer:")
        ids = tokenizer.encode(prefix)

        scores = []
        for text in options:
            cont = tokenizer.encode(f" {text}")
            inp = torch.tensor([ids + cont], device=device)
            with torch.no_grad():
                logits = model(inp).logits[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            positions = range(len(ids) - 1, len(ids) - 1 + len(cont))
            total = sum(float(log_probs[p, tok]) for p, tok in zip(positions, cont))
            scores.append(total / max(len(cont), 1))

        pick = "ABCD"[max(range(4), key=lambda i: scores[i])]
        hit = pick == answer
        correct += hit
        if args.verbose:
            print(f"{'ok ' if hit else '   '} picked={pick} gold={answer} scores={[round(s, 2) for s in scores]}")

    acc = correct / max(len(rows), 1)
    print(f"mcq accuracy: {correct}/{len(rows)} = {acc:.1%} (chance = 25%)")


def cmd_bench(args) -> None:
    device = pick_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, config = _load(args.checkpoint, device, dtype)
    tokenizer = Tokenizer(args.tokenizer)

    prompt = "<|im_start|>user\n" + "Explain how the keep warm mode works. " * 8 + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt)

    def run(use_cache: bool, n: int) -> tuple[float, int]:
        inp = torch.tensor([ids], device=device)
        cache = GrowCache(config.num_hidden_layers) if use_cache else None
        t0 = time.perf_counter()
        with torch.no_grad():
            if cache is not None:
                model(inp, cache=cache)  # prefill
                produced = 0
                for _ in range(n):
                    step = torch.tensor([[1]], device=device)
                    model(step, cache=cache)
                    produced += 1
            else:
                cur = inp
                produced = 0
                for _ in range(n):
                    cur = torch.cat([cur, torch.tensor([[1]], device=device)], dim=1)
                    model(cur)
                    produced += 1
        return time.perf_counter() - t0, produced

    t_cached, n = run(True, args.tokens)
    t_full, _ = run(False, min(args.tokens, 32))  # quadratic; cap it

    print(f"device={device} dtype={dtype}")
    print(f"prompt tokens={len(ids)} decode={args.tokens}")
    print(f"with cache:    {n / t_cached:6.1f} tok/s  ({t_cached:.2f}s)")
    print(f"full recompute:{min(args.tokens, 32) / t_full:6.1f} tok/s  ({t_full:.2f}s, capped at 32)")

    gen_params = SamplingParams(max_new_tokens=args.tokens, temperature=0.7, seed=1)
    inp = torch.tensor([ids], device=device)
    t0 = time.perf_counter()
    out = generate(model, inp, gen_params, eos_token_id=config.eos_token_id)
    elapsed = time.perf_counter() - t0
    produced = out.shape[1] - len(ids)
    print(f"end-to-end generate(): {produced / elapsed:.1f} tok/s ({produced} tokens in {elapsed:.2f}s)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="HearthLM evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--checkpoint", default="out/checkpoints/sft.pt")
        sp.add_argument("--tokenizer", default="assets/tokenizer")
        sp.add_argument("--device", default="auto")
        sp.add_argument("--dtype", default="auto")

    p1 = sub.add_parser("ppl", help="perplexity on a conversation set")
    common(p1)
    p1.add_argument("--data", default="data/demo/sft.jsonl")
    p1.add_argument("--seq-len", type=int, default=512)
    p1.add_argument("--batch-size", type=int, default=8)
    p1.set_defaults(func=cmd_ppl)

    p2 = sub.add_parser("mcq", help="multiple-choice accuracy")
    common(p2)
    p2.add_argument("--data", default="data/demo/mcq.jsonl")
    p2.add_argument("--verbose", action="store_true")
    p2.set_defaults(func=cmd_mcq)

    p3 = sub.add_parser("bench", help="decoding throughput")
    common(p3)
    p3.add_argument("--tokens", type=int, default=64)
    p3.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()