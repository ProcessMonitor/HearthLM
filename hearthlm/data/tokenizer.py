"""Tokenizer wrapper plus a from-scratch byte-level BPE trainer.

Inference/training uses `tokenizers` (fast, Rust) when a `tokenizer.json` exists.
`train_bpe` below is a dependency-free reference implementation used to *build*
that JSON from a text corpus, so the vocabulary is reproducible from the repo
alone without `sentencepiece` or a HuggingFace hub download.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

SPECIAL_TOKENS = [
    "<|endoftext|>",   # 0 bos
    "<|im_end|>",      # 1 eos
    "<|pad|>",         # 2 pad
    "<|unk|>",         # 3
    "<|im_start|>",    # 4
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|tool_call|>",
    "<|tool_response|>",
    "<think>",
    "</think>",
]


class Tokenizer:
    """Thin wrapper that fixes argument names and exposes the ids we care about."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as _T

        self.path = Path(path)
        json_path = self.path / "tokenizer.json" if self.path.is_dir() else self.path
        if not json_path.exists():
            raise FileNotFoundError(f"no tokenizer.json at {json_path}")
        self._tok = _T.from_file(str(json_path))
        meta_path = json_path.parent / "tokenizer_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.special_ids: dict[str, int] = meta.get("special_ids", {})
        self.vocab_size = self._tok.get_vocab_size()

    # ------------------------------------------------------------- properties
    @property
    def bos_token_id(self) -> int:
        return self.special_ids.get("<|endoftext|>", 0)

    @property
    def eos_token_id(self) -> int:
        return self.special_ids.get("<|im_end|>", 1)

    @property
    def pad_token_id(self) -> int:
        return self.special_ids.get("<|pad|>", 2)

    def id_of(self, token: str) -> int:
        return self.special_ids.get(token, self._tok.token_to_id(token))

    # ------------------------------------------------------------- conversion
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: Iterable[str], add_special_tokens: bool = False) -> list[list[int]]:
        encs = self._tok.encode_batch(list(texts), add_special_tokens=add_special_tokens)
        return [e.ids for e in encs]

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self._tok.encode(text, add_special_tokens=False)
        return enc.ids, list(enc.offsets)

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_span_for_chars(self, offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
        """Map a character range in the source text to a token index range.

        A token is included when its character span intersects [start, end).
        """
        first, last = None, None
        for i, (s, e) in enumerate(offsets):
            if e <= s:  # zero-width token (e.g. a lone special token marker)
                continue
            if e > start and s < end:
                if first is None:
                    first = i
                last = i + 1
        if first is None:
            return 0, 0
        return first, last

    def save(self, out_dir: str | Path, special_ids: Optional[dict[str, int]] = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(out / "tokenizer.json"))
        meta = {"special_ids": special_ids if special_ids is not None else self.special_ids}
        (out / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- BPE
def _byte_alphabet() -> dict[int, str]:
    """GPT-2 byte-to-unicode table: keeps every byte printable and reversible."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _pretokenize(text: str) -> list[str]:
    """Split on GPT-4-style boundaries: words, numbers, punctuation, whitespace runs."""
    import re

    pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\w]?\w+|\d{1,3}|[^\s\w]+|\s+(?!\S)|\s+""")
    return pattern.findall(text)


def train_bpe(
    corpus: Iterable[str],
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    special_tokens: Optional[list[str]] = None,
    out_dir: Optional[str | Path] = None,
    log_every: int = 500,
) -> dict:
    """Byte-level BPE, merging most-frequent adjacent pairs until `vocab_size`.

    Deliberately simple: full recount per merge. Fine for corpora up to a few
    hundred MB, which is the scale this project targets. For bigger data use the
    Rust implementation in `tokenizers`.
    """
    special_tokens = special_tokens or SPECIAL_TOKENS
    byte_to_uni = _byte_alphabet()

    from collections import defaultdict

    word_freqs: Counter[str] = Counter()
    for doc in corpus:
        for piece in _pretokenize(doc):
            word_freqs[piece] += 1

    def to_symbols(piece: str) -> tuple[str, ...]:
        return tuple(byte_to_uni[b] for b in piece.encode("utf-8"))

    vocab: dict[str, int] = {}
    for tok in special_tokens:
        vocab[tok] = len(vocab)
    # Seed *all* 256 byte symbols, not just the ones in this corpus. Otherwise
    # text containing an unseen byte (another script, an emoji, a stray control
    # char) has nowhere to go and gets silently dropped at encode time.
    for sym in sorted(set(byte_to_uni.values())):
        vocab[sym] = len(vocab)

    splits = {piece: list(to_symbols(piece)) for piece in word_freqs}
    merges: list[tuple[str, str]] = []

    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for piece, symbols in splits.items():
            if len(symbols) < 2:
                continue
            freq = word_freqs[piece]
            for a, b in zip(symbols, symbols[1:]):
                pair_counts[(a, b)] += freq
        if not pair_counts:
            break
        (best, count) = pair_counts.most_common(1)[0]
        if count < min_frequency:
            break

        merges.append(best)
        vocab[best[0] + best[1]] = len(vocab)

        new_splits = {}
        a, b = best
        merged = a + b
        for piece, symbols in splits.items():
            if len(symbols) < 2:
                new_splits[piece] = symbols
                continue
            out: list[str] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            new_splits[piece] = out
        splits = new_splits
        if log_every and len(merges) % log_every == 0:
            print(f"[bpe] merges={len(merges)} vocab={len(vocab)} last={best!r} count={count}")

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i, tok in enumerate(special_tokens)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [list(m) for m in merges],
        },
    }

    result = {"vocab_size": len(vocab), "merges": len(merges), "tokenizer": tokenizer_json,
              "special_ids": {tok: i for i, tok in enumerate(special_tokens)}}

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False), encoding="utf-8")
        (out / "tokenizer_meta.json").write_text(
            json.dumps({"special_ids": result["special_ids"]}, indent=2) + "\n", encoding="utf-8"
        )
    return result


def iter_corpus(paths: Iterable[str | Path], text_key: str = "text") -> Iterator[str]:
    """Yield documents from .jsonl (with `text_key`) or plain .txt files."""
    for p in paths:
        p = Path(p)
        if p.suffix in {".txt", ".md"}:
            yield p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get(text_key) if isinstance(obj, dict) else None
                    if text:
                        yield text
        else:
            raise ValueError(f"unsupported corpus file type: {p}")