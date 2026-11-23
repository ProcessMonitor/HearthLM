"""Inference engine: one model, continuous-ish batching, per-request cache.

The design constraint is that a small model on one GPU is dominated by per-step
overhead, not FLOPs. So the engine keeps a single worker loop, admits requests
up to a token budget, and decodes all active requests in lockstep. Each request
owns a `StaticCache` slot, which is why the cache API has `reset()`.

This is not a production server. It is deliberately small enough to read, and it
handles the three things a naive `model.generate()` per request gets wrong:
concurrent requests, streaming, and per-request stop conditions.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import torch

from ..model.cache import StaticCache
from ..model.generate import SamplingParams, apply_penalties, sample


@dataclass
class Request:
    id: str
    prompt_ids: list[int]
    params: SamplingParams
    # callbacks
    on_token: Optional[Callable[[int], None]] = None
    on_done: Optional[Callable[[str], None]] = None
    tokenizer: object = None
    queue: Optional["queue.Queue[str]"] = field(default=None, repr=False)

    # runtime state
    output_ids: list[int] = field(default_factory=list)
    cache: Optional[StaticCache] = field(default=None, repr=False)
    finished: bool = False
    slot: int = -1
    _emitted: int = 0
    _raw_text: str = ""

    @property
    def n_prompt(self) -> int:
        return len(self.prompt_ids)

    def take_delta(self) -> Optional[str]:
        """Decode the tokens so far and return only the newly appended text.

        Byte-level BPE tokens can split a multi-byte character, which decodes to
        U+FFFD. Hold the text back until the character completes rather than
        streaming a replacement glyph the client cannot undo.
        """
        if self.tokenizer is None:
            return None
        text = self.tokenizer.decode(self.output_ids)
        if text.endswith("\ufffd"):
            return None
        delta = text[self._emitted:]
        self._emitted = len(text)
        return delta or None


class Engine:
    """Single-process batched decoder."""

    def __init__(
        self,
        model,
        *,
        device: torch.device,
        max_batch: int = 8,
        max_seq_len: int = 2048,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.dtype = dtype
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self._lock = threading.Lock()
        self._pending: list[Request] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        cfg = model.config
        self._caches: dict[int, StaticCache] = {}

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def submit(self, request: Request) -> None:
        with self._lock:
            self._pending.append(request)
        self.start()

    def stream(self, prompt_ids: list[int], params: SamplingParams, tokenizer) -> Iterator[str]:
        """Blocking helper: returns a generator of decoded deltas."""
        import queue

        q: "queue.Queue[str]" = queue.Queue()
        req = Request(
            id=str(time.time()),
            prompt_ids=prompt_ids,
            params=params,
            tokenizer=tokenizer,
            queue=q,
        )
        self.submit(req)
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    def generate(self, prompt_ids: list[int], params: SamplingParams, tokenizer=None) -> str:
        """Blocking full completion."""
        return "".join(self.stream(prompt_ids, params, tokenizer))

    # ------------------------------------------------------------------- loop
    def _slot_cache(self, slot: int) -> StaticCache:
        if slot not in self._caches:
            cfg = self.model.config
            self._caches[slot] = StaticCache(
                num_layers=cfg.num_hidden_layers,
                max_len=self.max_seq_len,
                batch_size=1,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                device=self.device,
                dtype=self.dtype if self.dtype != torch.float32 else torch.float32,
            )
        return self._caches[slot]

    def _admit(self, active: list[Request]) -> None:
        with self._lock:
            while self._pending and len(active) < self.max_batch:
                req = self._pending.pop(0)
                used = {r.slot for r in active}
                slot = next(i for i in range(self.max_batch) if i not in used)
                req.slot = slot
                cache = self._slot_cache(slot)
                cache.reset()
                active.append(req)

    @torch.inference_mode()
    def _run(self) -> None:
        active: list[Request] = []
        while not self._stop.is_set():
            self._admit(active)
            if not active:
                time.sleep(0.01)
                continue
            self._decode_step(active)
            active = [r for r in active if not r.finished]

    def _decode_step(self, active: list[Request]) -> None:
        """One forward per request. Batching these would need ragged-cache SDPA."""
        for req in list(active):
            cache = req.cache = self._slot_cache(req.slot)
            past = cache.length
            if past == 0:
                cur = torch.tensor([req.prompt_ids], device=self.device)
            else:
                cur = torch.tensor([[req.output_ids[-1]]], device=self.device)

            out = self.model(cur, cache=cache)
            logits = out.logits[:, -1]

            seen = torch.tensor([req.prompt_ids + req.output_ids], device=self.device)
            logits = apply_penalties(
                logits.clone(), seen,
                repetition_penalty=req.params.repetition_penalty,
                presence_penalty=req.params.presence_penalty,
                frequency_penalty=req.params.frequency_penalty,
            )
            next_id = int(sample(logits, req.params)[0, 0].item())
            req.output_ids.append(next_id)

            stop_ids = set(req.params.stop_token_ids)
            if self.model.config.eos_token_id is not None:
                stop_ids.add(self.model.config.eos_token_id)

            if next_id in stop_ids:
                self._finish(req)
                continue

            delta = req.take_delta()
            if delta and req.queue is not None:
                req.queue.put(delta)
            if req.on_token is not None:
                req.on_token(next_id)

            if len(req.output_ids) >= req.params.max_new_tokens or cache.length >= self.max_seq_len - 1:
                self._finish(req)

    def _finish(self, req: Request) -> None:
        req.finished = True
        # Flush whatever is still buffered, including a trailing partial character.
        if req.tokenizer is not None:
            text = req.tokenizer.decode(req.output_ids)
            tail = text[req._emitted:]
            if tail and req.queue is not None:
                req.queue.put(tail)
            req._emitted = len(text)
        if req.queue is not None:
            req.queue.put(None)
        if req.on_done is not None:
            text = req.tokenizer.decode(req.output_ids) if req.tokenizer else ""
            req.on_done(text)