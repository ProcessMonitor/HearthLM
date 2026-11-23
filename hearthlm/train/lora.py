"""LoRA implemented directly, plus merge/unmerge.

`apply_lora` wraps matching `nn.Linear` modules with an additive low-rank branch
that is exactly zero at init, so the loss at step 0 equals the base model's.
Merging is done in fp32 and only then cast back, which avoids the visible drift
that fp16 accumulation causes when rank is high.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """base(x) + B(A(x)) * scaling, with B zero-initialised."""

    def __init__(self, base: nn.Linear, rank: int, alpha: Optional[float] = None, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("lora rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.scaling == 0:
            return out
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out + lora.to(out.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight, materialised in fp32. Only for merge/export paths."""
        delta = (self.lora_B.weight.float() @ self.lora_A.weight.float()) * self.scaling
        return self.base.weight.float() + delta

    def merged_weight(self) -> torch.Tensor:
        return self.weight.to(self.base.weight.dtype)


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> nn.Module:
    """Replace matching Linear submodules in place. Returns the same model."""
    targets = tuple(targets)
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(child_name.endswith(t) for t in targets):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    if replaced == 0:
        raise RuntimeError(f"no Linear layers matched targets={targets}")
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name
    return model


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors, with `base.` stripped so they load into any base."""
    out = {}
    for name, p in model.named_parameters():
        if "lora_" in name:
            out[name.replace(".base.", ".")] = p.detach().cpu()
    return out


def merge_lora(model: nn.Module) -> None:
    """Fold adapters into base weights in place; the module becomes a plain Linear."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                merged = child.merged_weight()
                new_linear = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)
                new_linear.weight.data.copy_(merged)
                if base.bias is not None:
                    new_linear.bias.data.copy_(base.bias.data)
                setattr(module, child_name, new_linear)


def unmerge_lora(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Reverse of merge: rebuild adapters from a saved state dict."""
    raise NotImplementedError("re-attach with apply_lora then load_state_dict instead")


def save_lora(model: nn.Module, path: str) -> None:
    torch.save(
        {
            "lora": {k: v.half() for k, v in lora_state_dict(model).items()},
            "meta": {
                "rank": next(m.rank for m in model.modules() if isinstance(m, LoRALinear)),
                "alpha": next(m.alpha for m in model.modules() if isinstance(m, LoRALinear)),
            },
        },
        path,
    )


def load_lora(model: nn.Module, path: str) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload["lora"] if "lora" in payload else payload
    own = lora_state_dict(model)
    # Keys are stored without `base.`; map them back onto the wrapped modules.
    mapping = {}
    for name in own:
        flat = name.replace(".base.", ".")
        if flat in state:
            mapping[name] = state[flat]
    missing = set(own) - set(mapping)
    if missing:
        raise RuntimeError(f"LoRA checkpoint is missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
    model.load_state_dict(mapping, strict=False)