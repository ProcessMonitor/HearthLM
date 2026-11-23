"""KV cache implementations.

Two of them, because batch-1 chat and batched serving want different things:

* `GrowCache`  - append by concatenation. Simple, zero waste, right for a single
  stream with unknown final length.
* `StaticCache` - preallocated to max length with an explicit fill pointer. No
  allocator churn per step, fixed shapes for graph capture, and `reset()` lets a
  serving slot be reused for the next request.

Both expose the same five methods, so attention code never branches on type.
"""
from __future__ import annotations

from typing import Optional

import torch


class KVCache:
    """Interface. Subclasses store one (k, v) pair per layer."""

    length: int
    max_len: int

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append k,v (batch, heads, seq, dim) and return the full k,v so far."""
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def crop(self, length: int) -> None:
        """Drop everything past `length` tokens. Used to undo a rejected draft."""
        raise NotImplementedError


class GrowCache(KVCache):
    def __init__(self, num_layers: int, max_len: int = 0, batch_size: int = 1):
        self.num_layers = num_layers
        self.max_len = max_len
        self._k: list[Optional[torch.Tensor]] = [None] * num_layers
        self._v: list[Optional[torch.Tensor]] = [None] * num_layers
        self.length = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if self._k[layer_idx] is None:
            self._k[layer_idx], self._v[layer_idx] = k, v
        else:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], k], dim=2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], v], dim=2)
        if layer_idx == 0:
            self.length = self._k[0].shape[2]
            if self.max_len and self.length > self.max_len:
                raise RuntimeError(f"cache overflow: {self.length} > {self.max_len}")
        return self._k[layer_idx], self._v[layer_idx]

    def reset(self) -> None:
        self._k = [None] * self.num_layers
        self._v = [None] * self.num_layers
        self.length = 0

    def crop(self, length: int) -> None:
        for i in range(self.num_layers):
            if self._k[i] is not None:
                self._k[i] = self._k[i][:, :, :length]
                self._v[i] = self._v[i][:, :, :length]
        self.length = length

    # Convenience for callers that need the raw tensors (e.g. prefix sharing).
    def get(self, layer_idx: int):
        return self._k[layer_idx], self._v[layer_idx]

    def clone(self) -> "GrowCache":
        c = GrowCache(self.num_layers, self.max_len)
        c._k = [t.clone() if t is not None else None for t in self._k]
        c._v = [t.clone() if t is not None else None for t in self._v]
        c.length = self.length
        return c


class StaticCache(KVCache):
    """Preallocated cache. One slot per sequence in the batch."""

    def __init__(
        self,
        num_layers: int,
        max_len: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        shape = (batch_size, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(num_layers, *shape, device=device, dtype=dtype)
        self.v = torch.zeros(num_layers, *shape, device=device, dtype=dtype)
        self.num_layers = num_layers
        self.max_len = max_len
        self.length = 0
        # Write offset for the *current* forward call. It must be latched by the
        # first layer, not recomputed per layer: `length` advances once per step
        # and every layer in that step writes at the same offset.
        self._write_start = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if layer_idx == 0:
            self._write_start = self.length
        start = self._write_start
        end = start + k.shape[2]
        if end > self.max_len:
            raise RuntimeError(f"static cache overflow: {end} > {self.max_len}")
        # k: (batch, kv_heads, seq, dim); slots live on the seq axis
        self.k[layer_idx, :, :, start:end] = k
        self.v[layer_idx, :, :, start:end] = v
        if layer_idx == 0:
            self.length = end
        return self.k[layer_idx, :, :, :end], self.v[layer_idx, :, :, :end]

    def reset(self) -> None:
        self.length = 0
        self._write_start = 0

    def crop(self, length: int) -> None:
        self.length = length
        self._write_start = length

    def to_layer(self, layer_idx: int):
        return self.k[layer_idx], self.v[layer_idx]


def make_cache(
    kind: str,
    config,
    *,
    batch_size: int,
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KVCache:
    if kind == "grow":
        return GrowCache(config.num_hidden_layers, max_len=max_len, batch_size=batch_size)
    if kind == "static":
        return StaticCache(
            num_layers=config.num_hidden_layers,
            max_len=max_len,
            batch_size=batch_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"unknown cache kind {kind!r}")