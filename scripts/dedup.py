"""Deduplicate near-identical documents before pretraining.

    python scripts/dedup.py --data data/raw/*.jsonl --out data/clean.jsonl --threshold 0.9

Uses MinHash over character n-grams, which is the standard tool for this and
scales linearly. Exact duplicates are removed first with a hash set, then
near-duplicates by estimated Jaccard similarity.

Duplicates matter more at small scale than at large scale: a 30M-token corpus
with 20% repeats effectively trains on 24M tokens with a skewed distribution, and
a small model will happily memorise the repeats instead of learning structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def shingles(text: str, n: int = 5) -> set[int]:
    text = " ".join(text.split()).lower()
    if len(text) < n:
        return {hash(text)}
    return {hash(text[i:i + n]) for i in range(len(text) - n + 1)}


def minhash(sig: list[int], num_hashes: int, seeds: list[int]) -> list[int]:
    return [min((h * s) & 0xFFFFFFFF for h in sig) for s in seeds]


def jaccard_estimate(a: list[int], b: list[int]) -> float:
    if len(a) != len(b):
        raise ValueError("signatures differ in length")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text-key", default="text")
    p.add_argument("--threshold", type=float, default=0.9, help="Jaccard similarity above which a doc is dropped")
    p.add_argument("--hashes", type=int, default=64)
    p.add_argument("--shingle", type=int, default=5)
    p.add_argument("--min-chars", type=int, default=40)
    args = p.parse_args(argv)

    import random

    rng = random.Random(0)
    seeds = [rng.getrandbits(32) | 1 for _ in range(args.hashes)]

    rows = []
    for path in args.data:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(args.text_key) if isinstance(obj, dict) else None
                if text and len(text) >= args.min_chars:
                    rows.append(text)

    # Stage 1: exact duplicates.
    seen: set[str] = set()
    unique = []
    for text in rows:
        h = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(text)
    print(f"exact: {len(rows)} -> {len(unique)}")

    # Stage 2: near duplicates via MinHash signatures.
    signatures = []
    for text in unique:
        sig = sorted(shingles(text, args.shingle))
        if not sig:
            signatures.append(None)
            continue
        # LSH banding would be faster; at this corpus size a linear scan is fine
        # and exact. Swap in bands if you go past ~1M documents.
        signatures.append(minhash(sig, args.hashes, seeds))

    kept: list[str] = []
    kept_sigs: list[list[int]] = []
    dropped = 0
    for text, sig in zip(unique, signatures):
        if sig is None:
            kept.append(text)
            continue
        is_dup = False
        for other in kept_sigs:
            if jaccard_estimate(sig, other) >= args.threshold:
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        kept.append(text)
        kept_sigs.append(sig)

    print(f"near-dup: {len(unique)} -> {len(kept)} (dropped {dropped})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for text in kept:
            f.write(json.dumps({args.text_key: text}, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()