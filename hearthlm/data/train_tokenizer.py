"""Train a byte-level BPE tokenizer.

    python -m hearthlm.data.train_tokenizer --data data/pretrain.jsonl --out assets/tokenizer --vocab-size 8192

Uses the pure-Python trainer in `tokenizer.py`. This exists so the vocab is
reproducible from repo contents; if you already have a `tokenizer.json`, skip it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tokenizer import SPECIAL_TOKENS, iter_corpus, train_bpe


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", default="assets/tokenizer")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--text-key", default="text")
    p.add_argument("--max-docs", type=int, default=None,
                   help="cap documents used for training (full corpus for timing)")
    args = p.parse_args(argv)

    docs = []
    for i, text in enumerate(iter_corpus(args.data, text_key=args.text_key)):
        docs.append(text)
        if args.max_docs and i + 1 >= args.max_docs:
            break
    print(f"training BPE on {len(docs)} documents, target vocab {args.vocab_size}")

    out = Path(args.out)
    result = train_bpe(
        docs,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        out_dir=out,
    )
    print(f"vocab={result['vocab_size']} merges={result['merges']} -> {out}")

    # Round-trip check on a few samples; a tokenizer that cannot reproduce its
    # input is worse than no tokenizer at all.
    from .tokenizer import Tokenizer

    tok = Tokenizer(out)
    for text in docs[:3]:
        ids = tok.encode(text)
        back = tok.decode(ids)
        assert back == text, f"round trip failed:\n{text!r}\n{back!r}"
    print("round trip ok")

    cfg = {"vocab_size": tok.vocab_size, "special_ids": tok.special_ids}
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()