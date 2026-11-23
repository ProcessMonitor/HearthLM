"""Datasets.

Two decisions drive this file:

1. Pretraining packs several documents into one sequence instead of padding each
   to `seq_len`. Padding-to-fixed-length wastes 40-60% of the compute on short
   documents, which for a project whose selling point is training on one GPU is
   the single largest avoidable cost.
2. Supervision masks are computed from character offsets, not from searching for
   a sentinel string in the token stream. String search breaks the moment the
   tokenizer splits the sentinel, and it fails silently.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .chat_template import render


# --------------------------------------------------------------------- helpers
class TokenizedCorpus:
    """A flat token array on disk, memory-mapped for cheap random access.

    Layout: `tokens.bin` (uint32) + `offsets.npy` (int64, len = n_docs + 1).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tokens = np.memmap(self.root / "tokens.bin", dtype=np.uint32, mode="r")
        self.offsets = np.load(self.root / "offsets.npy")
        self.n_docs = len(self.offsets) - 1

    def __len__(self) -> int:
        return self.n_docs

    def doc(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i]: self.offsets[i + 1]]

    @classmethod
    def build(
        cls,
        sources: Iterable[str | Path],
        tokenizer,
        out_dir: str | Path,
        *,
        text_key: str = "text",
        add_bos_eos: bool = True,
        max_docs: Optional[int] = None,
        log_every: int = 20000,
    ) -> "TokenizedCorpus":
        from .tokenizer import iter_corpus

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bin_path = out / "tokens.bin"
        offsets: list[int] = [0]
        written = 0
        n = 0

        with bin_path.open("wb") as f:
            for text in iter_corpus(sources, text_key=text_key):
                ids = tokenizer.encode(text.strip())
                if not ids:
                    continue
                if add_bos_eos:
                    ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
                arr = np.asarray(ids, dtype=np.uint32)
                f.write(arr.tobytes())
                written += len(ids)
                offsets.append(written)
                n += 1
                if log_every and n % log_every == 0:
                    print(f"[corpus] docs={n} tokens={written}")
                if max_docs and n >= max_docs:
                    break

        np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        meta = {"n_docs": n, "n_tokens": written, "tokenizer_vocab_size": tokenizer.vocab_size}
        (out / "corpus_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[corpus] wrote {out}: {n} docs, {written} tokens")
        return cls(out)


# ------------------------------------------------------------------- pretrain
class PretrainDataset(Dataset):
    """Yields raw token arrays; packing happens in the collate function."""

    def __init__(self, corpus: TokenizedCorpus, seq_len: int, seed: int = 0):
        self.corpus = corpus
        self.seq_len = seq_len
        self.epoch = 0
        self.order: list[int] = []
        self._shuffle(seed)

    def _shuffle(self, seed: int) -> None:
        rng = random.Random(seed + self.epoch)
        self.order = list(range(len(self.corpus)))
        rng.shuffle(self.order)

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.epoch = epoch
            self._shuffle(0)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx: int) -> list[int]:
        doc = self.corpus.doc(self.order[idx])
        # Long documents are split into chunks rather than truncated so no text is
        # thrown away; chunks are offset deterministically by the doc index.
        if len(doc) <= self.seq_len:
            return doc.tolist()
        start = (idx * 7919) % max(1, len(doc) - self.seq_len)
        return doc[start:start + self.seq_len].tolist()


def pack_documents(
    docs: Sequence[Sequence[int]],
    seq_len: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Greedily concatenate documents into rows of at most `seq_len` tokens.

    Returns input_ids, labels, document_ids and attention_mask. The first token
    of every document gets label -100: its predecessor belongs to a different
    document, so that transition is not a real prediction target.
    """
    rows_ids: list[list[int]] = []
    rows_doc: list[list[int]] = []
    cur_ids: list[int] = []
    cur_doc: list[int] = []
    next_doc_id = 0

    for doc in docs:
        if not doc:
            continue
        start = 0
        while start < len(doc):
            room = seq_len - len(cur_ids)
            if room <= 0:
                rows_ids.append(cur_ids)
                rows_doc.append(cur_doc)
                cur_ids, cur_doc, next_doc_id = [], [], 0
                room = seq_len
            take = doc[start:start + room]
            cur_ids.extend(take)
            cur_doc.extend([next_doc_id] * len(take))
            next_doc_id += 1
            start += room
    if cur_ids:
        rows_ids.append(cur_ids)
        rows_doc.append(cur_doc)

    if not rows_ids:
        return {
            "input_ids": torch.zeros(0, seq_len, dtype=torch.long),
            "labels": torch.zeros(0, seq_len, dtype=torch.long),
            "document_ids": torch.zeros(0, seq_len, dtype=torch.long),
            "attention_mask": torch.zeros(0, seq_len, dtype=torch.long),
        }

    width = max(len(r) for r in rows_ids)
    bsz = len(rows_ids)
    input_ids = torch.full((bsz, width), pad_id, dtype=torch.long)
    document_ids = torch.full((bsz, width), -1, dtype=torch.long)
    attention_mask = torch.zeros((bsz, width), dtype=torch.long)

    for i, (ids, doc_ids) in enumerate(zip(rows_ids, rows_doc)):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        document_ids[i, :len(doc_ids)] = torch.tensor(doc_ids, dtype=torch.long)
        attention_mask[i, :len(ids)] = 1

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    # Mask the first token of each packed document (document_ids changes there).
    prev_doc = torch.cat([torch.full((bsz, 1), -1, dtype=torch.long), document_ids[:, :-1]], dim=1)
    first_of_doc = (document_ids != prev_doc) & (document_ids >= 0)
    labels[first_of_doc] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "document_ids": document_ids,
        "attention_mask": attention_mask,
    }


class PackCollator:
    def __init__(self, seq_len: int, pad_id: int):
        self.seq_len = seq_len
        self.pad_id = pad_id

    def __call__(self, docs: Sequence[list[int]]) -> dict[str, torch.Tensor]:
        return pack_documents(docs, self.seq_len, self.pad_id)


# ------------------------------------------------------------------------ SFT
def build_sft_example(
    messages: list[dict],
    tokenizer,
    *,
    max_len: int,
    add_eos: bool = True,
    enable_thinking: bool = False,
    tools: Optional[list[dict]] = None,
) -> Optional[dict[str, torch.Tensor]]:
    """Tokenize one conversation with a supervision mask over assistant spans."""
    segments = render(messages, tools=tools, enable_thinking=enable_thinking)
    text = "".join(s.text for s in segments)
    ids, offsets = tokenizer.encode_with_offsets(text)

    labels = [-100] * len(ids)
    cursor = 0
    last_trainable_end = 0
    for seg in segments:
        start = cursor
        end = cursor + len(seg.text)
        cursor = end
        if seg.trainable:
            a, b = tokenizer.token_span_for_chars(offsets, start, end)
            for j in range(a, b):
                labels[j] = ids[j]
            last_trainable_end = end

    # An EOS is only a training target when the conversation ends on an assistant
    # turn. Appending it after a user-only row teaches the model to stop talking,
    # which is exactly the wrong lesson.
    if add_eos:
        ends_on_assistant = bool(segments) and segments[-1].trainable and last_trainable_end == cursor
        ids = ids + [tokenizer.eos_token_id]
        labels = labels + ([tokenizer.eos_token_id] if ends_on_assistant else [-100])

    if max_len and len(ids) > max_len:
        ids = ids[:max_len]
        labels = labels[:max_len]

    if all(l == -100 for l in labels):
        return None  # nothing to learn from

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        *,
        max_len: int = 1024,
        enable_thinking: bool = False,
        val_ratio: float = 0.0,
        split: str = "train",
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.enable_thinking = enable_thinking
        rows = _read_jsonl(path)

        if val_ratio > 0:
            rng = random.Random(seed)
            rng.shuffle(rows)
            cut = int(len(rows) * (1 - val_ratio))
            rows = rows[:cut] if split == "train" else rows[cut:]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        msgs = row["conversations"] if "conversations" in row else row["messages"]
        tools = None
        if msgs and msgs[0].get("role") == "system" and msgs[0].get("tools"):
            tools = msgs[0]["tools"]
            if isinstance(tools, str):
                tools = json.loads(tools)
            msgs = [{"role": "system", "content": msgs[0].get("content") or ""}] + msgs[1:]
        ex = build_sft_example(
            msgs, self.tokenizer, max_len=self.max_len,
            enable_thinking=self.enable_thinking, tools=tools,
        )
        if ex is None:
            # Keep the dataset indexable; an empty example is filtered by the
            # collator's drop-empty behaviour.
            return {"input_ids": torch.zeros(0, dtype=torch.long), "labels": torch.zeros(0, dtype=torch.long)}
        return ex


def collate_sft(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    batch = [b for b in batch if b["input_ids"].numel() > 0]
    if not batch:
        return {"input_ids": torch.zeros(0, 0, dtype=torch.long), "labels": torch.zeros(0, 0, dtype=torch.long),
                "attention_mask": torch.zeros(0, 0, dtype=torch.long)}
    width = max(b["input_ids"].size(0) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


# ------------------------------------------------------------------------ DPO
class DPODataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, *, max_len: int = 1024,
                 val_ratio: float = 0.0, split: str = "train", seed: int = 0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        rows = _read_jsonl(path)
        if val_ratio > 0:
            rng = random.Random(seed)
            rng.shuffle(rows)
            cut = int(len(rows) * (1 - val_ratio))
            rows = rows[:cut] if split == "train" else rows[cut:]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, messages: list[dict]) -> dict[str, torch.Tensor]:
        ex = build_sft_example(messages, self.tokenizer, max_len=self.max_len)
        if ex is None:
            ex = {
                "input_ids": torch.zeros(1, dtype=torch.long),
                "labels": torch.full((1,), -100, dtype=torch.long),
            }
        return ex

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        chosen = self._encode(row["chosen"])
        rejected = self._encode(row["rejected"])
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
        }


def collate_dpo(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for prefix in ("chosen", "rejected"):
        ids_key, lab_key = f"{prefix}_input_ids", f"{prefix}_labels"
        width = max(b[ids_key].size(0) for b in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = b[ids_key].size(0)
            ids[i, :n] = b[ids_key]
            labels[i, :n] = b[lab_key]
            mask[i, :n] = 1
        out[ids_key] = ids
        out[lab_key] = labels
        out[f"{prefix}_mask"] = mask
    return out


# --------------------------------------------------------------------- readers
def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows