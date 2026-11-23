"""Build a memory-mapped token corpus for pretraining.

    python -m hearthlm.data.build_corpus --data corpus/*.jsonl --out data/corpus

The output directory holds `tokens.bin` + `offsets.npy`, which `PretrainDataset`
memory-maps. Tokenizing once up front removes the dataloader bottleneck: the
usual alternative (tokenize inside `__getitem__`) makes a 4-worker loader the
slowest part of training on a single GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import TokenizedCorpus
from .tokenizer import Tokenizer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True, help=".jsonl or .txt files")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="assets/tokenizer")
    p.add_argument("--text-key", default="text")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--no-special-tokens", action="store_true",
                   help="do not wrap each document in bos/eos")
    args = p.parse_args(argv)

    tok = Tokenizer(args.tokenizer)
    TokenizedCorpus.build(
        args.data,
        tok,
        args.out,
        text_key=args.text_key,
        add_bos_eos=not args.no_special_tokens,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()