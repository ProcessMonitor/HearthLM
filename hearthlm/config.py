"""Model configuration.

Plain dataclass instead of a transformers config so the training stack has no
dependency on transformers. `to_dict` / `from_dict` are the serialization
boundary: checkpoints and the serving layer both round-trip through them.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class HearthConfig:
    # vocabulary / embedding
    vocab_size: int = 4096
    hidden_size: int = 512
    num_hidden_layers: int = 8
    # attention
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None
    attention_bias: bool = False
    qk_norm: bool = True
    sliding_window: Optional[int] = None
    # mlp
    intermediate_size: Optional[int] = None
    hidden_act: str = "silu"
    # regularization
    dropout: float = 0.0
    # norm / rope
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 4096
    rope_scaling: Optional[dict] = None
    # embeddings
    tie_word_embeddings: bool = True
    # mixture of experts
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 1e-2
    router_z_loss_coef: float = 0.0
    # special tokens, kept here so tokenizer and model agree
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size={self.hidden_size} is not divisible by "
                    f"num_attention_heads={self.num_attention_heads}"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.intermediate_size is None:
            # SwiGLU uses three matrices; 8/3 keeps the parameter count on par
            # with a 4x two-matrix MLP. Round up to a multiple of 64 for kernels.
            raw = int(8 * self.hidden_size / 3)
            self.intermediate_size = int(math.ceil(raw / 64) * 64)
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    # ------------------------------------------------------------------ utils
    @property
    def num_parameters_per_layer(self) -> int:
        h = self.hidden_size
        d = self.head_dim
        attn = h * (self.num_attention_heads + 2 * self.num_key_value_heads) * d + h * self.num_attention_heads * d
        if self.use_moe:
            mlp = self.num_experts * 3 * h * self.moe_intermediate_size + h * self.num_experts
        else:
            mlp = 3 * h * self.intermediate_size
        return attn + mlp + 2 * h

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HearthConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HearthConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Named presets. Sizes are chosen so the whole pipeline is runnable on a laptop:
# `nano` trains on CPU in minutes, `base` matches the 60-70M class and still
# fits a single 8GB consumer GPU.
PRESETS: dict[str, dict[str, Any]] = {
    "nano": dict(
        hidden_size=256, num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=768, vocab_size=4096, max_position_embeddings=2048,
    ),
    "mini": dict(
        hidden_size=512, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
        vocab_size=8192, max_position_embeddings=4096,
    ),
    "base": dict(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4,
        vocab_size=8192, max_position_embeddings=4096,
    ),
    "moe": dict(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4,
        vocab_size=8192, max_position_embeddings=4096,
        use_moe=True, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=512,
    ),
}


def build_config(preset: str | None = None, **overrides: Any) -> HearthConfig:
    """Merge a preset with explicit overrides."""
    base: dict[str, Any] = dict(PRESETS[preset]) if preset else {}
    if preset is not None and preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; available: {sorted(PRESETS)}")
    # Changing hidden_size invalidates a preset intermediate_size.
    if "hidden_size" in overrides and "intermediate_size" not in overrides:
        base.pop("intermediate_size", None)
    base.update(overrides)
    return HearthConfig(**base)
