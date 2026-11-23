"""Sampling and decoding.

Kept separate from the model so the serve layer and the CLI share one code path
and the sampling knobs are testable without a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    do_sample: bool = True
    seed: Optional[int] = None
    stop_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.do_sample and not (0 < self.temperature):
            raise ValueError("temperature must be > 0 when sampling")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")


def apply_penalties(
    logits: torch.Tensor,
    generated: torch.Tensor,
    *,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> torch.Tensor:
    """In-place-safe penalty application on (batch, vocab) logits.

    `generated` is (batch, n_tokens) of already emitted ids (prompt included).
    Penalties are applied once per distinct token, which is what the OpenAI
    semantics ask for.
    """
    if repetition_penalty == 1.0 and presence_penalty == 0.0 and frequency_penalty == 0.0:
        return logits

    batch, vocab = logits.shape
    rows = torch.arange(batch, device=logits.device).unsqueeze(1)
    counts = torch.zeros(batch, vocab, device=logits.device, dtype=logits.dtype)
    counts.index_put_((rows.expand_as(generated), generated), torch.ones_like(generated, dtype=logits.dtype),
                      accumulate=True)

    if repetition_penalty != 1.0:
        score = logits.gather(1, generated)
        score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
        logits.scatter_(1, generated, score)

    if presence_penalty != 0.0:
        logits -= presence_penalty * (counts > 0).to(logits.dtype)
    if frequency_penalty != 0.0:
        logits -= frequency_penalty * counts
    return logits


def filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> torch.Tensor:
    """Mask disallowed tokens to -inf. Order matches common practice: k, then p, then min-p."""
    if top_k and top_k > 0 and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Keep the smallest set whose cumulative mass reaches top_p. Shifting the
        # mask by one guarantees at least one token survives.
        remove = cumulative - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        threshold = min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < threshold, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return one token id per row from raw (batch, vocab) logits."""
    if not params.do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / params.temperature
    logits = filter_logits(logits, top_k=params.top_k, top_p=params.top_p, min_p=params.min_p)
    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or probs.sum(dim=-1).le(0).any():
        # Degenerate row (everything masked); fall back to greedy so we emit
        # something valid instead of corrupting the sequence.
        return logits.nan_to_num(nan=float("-inf")).argmax(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=generator)


TokenizerLike = Callable[[str], list[int]]


def generate(
    model,
    input_ids: torch.Tensor,
    params: SamplingParams,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    cache=None,
    on_token: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Autoregressive decode with a KV cache. Returns prompt+completion ids.

    `on_token(batch_index, token_id)` fires after each step, for streaming.

    Uses `no_grad`, not `inference_mode`: RL rollouts feed the same code path and
    tensors born under inference_mode cannot participate in a backward pass.
    """
    with torch.no_grad():
        return _generate(
            model, input_ids, params,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
            cache=cache,
            on_token=on_token,
        )


def _generate(
    model,
    input_ids: torch.Tensor,
    params: SamplingParams,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    cache=None,
    on_token: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    device = input_ids.device
    bsz = input_ids.shape[0]
    eos_ids = set(stop_token_ids or [])
    if eos_token_id is not None:
        eos_ids.add(eos_token_id)

    generator = None
    if params.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(params.seed)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.clone()

    out = input_ids
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    for step in range(params.max_new_tokens):
        past = cache.length if cache is not None else 0
        cur = out[:, past:] if cache is not None else out
        logits = model(cur, attention_mask=attention_mask, cache=cache).logits[:, -1]
        logits = apply_penalties(
            logits.clone(), out,
            repetition_penalty=params.repetition_penalty,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
        )
        next_id = sample(logits, params, generator)
        if eos_ids:
            eos_tensor = torch.tensor(sorted(eos_ids), device=device)
            hit = torch.isin(next_id.squeeze(-1), eos_tensor)
            # Rows that already finished keep emitting a stop token so the
            # padded tail stays uniform across the batch.
            filler = eos_tensor[0].expand(bsz, 1)
            next_id = torch.where(finished.unsqueeze(-1), filler, next_id)
            finished = finished | (hit & ~finished)
        out = torch.cat([out, next_id], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(bsz, 1)], dim=-1)
        if on_token is not None:
            for i in range(bsz):
                on_token(i, int(next_id[i, 0].item()))
        if bool(finished.all()):
            break

    return out