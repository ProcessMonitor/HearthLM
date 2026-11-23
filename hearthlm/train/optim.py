"""Optimizer, schedule and checkpointing.

The checkpoint format is a directory with a model-only `.pt` and a separate
resume `.pt` (optimizer + step). Writing them separately means loading a model
for inference never touches the optimizer state, which is usually 3x larger.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch


def create_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    no_decay_patterns: tuple[str, ...] = ("norm", "bias", "embed_tokens", "lm_head"),
) -> torch.optim.Optimizer:
    """AdamW with decay skipped on norms, biases and embeddings.

    Decaying an embedding row only shrinks rarely-seen tokens toward zero, which
    measurably hurts small-vocabulary models where each token carries more load.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or any(p in name for p in no_decay_patterns):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


def cosine_lr(
    step: int,
    *,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay to `min_lr_ratio * base_lr`."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    model = getattr(model, "_orig_mod", model)  # torch.compile
    model = getattr(model, "module", model)     # DDP
    return model


# ------------------------------------------------------------------ checkpoint
class Checkpointer:
    def __init__(self, root: str | Path, tag: str, keep_last: int = 2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.keep_last = keep_last

    @property
    def model_path(self) -> Path:
        return self.root / f"{self.tag}.pt"

    @property
    def resume_path(self) -> Path:
        return self.root / f"{self.tag}_resume.pt"

    def save(
        self,
        model: torch.nn.Module,
        config,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        extra: Optional[dict] = None,
    ) -> None:
        raw = unwrap(model)
        state = {k: v.detach().to("cpu", torch.float32) for k, v in raw.state_dict().items()}
        payload = {"config": config.to_dict(), "model": state, "step": step, "epoch": epoch}
        _atomic_save(payload, self.model_path)

        if optimizer is not None:
            resume = {
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
                "config": config.to_dict(),
            }
            if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
                resume["scaler"] = scaler.state_dict()
            if extra:
                resume["extra"] = extra
            _atomic_save(resume, self.resume_path)
        self._rotate()

    @property
    def best_path(self) -> Path:
        return self.root / f"{self.tag}_best.pt"

    def save_best(self, model: torch.nn.Module, config, metric: float, step: int) -> None:
        """Write the best-so-far weights to `<tag>_best.pt`, separate from the last."""
        raw = unwrap(model)
        state = {k: v.detach().to("cpu", torch.float32) for k, v in raw.state_dict().items()}
        _atomic_save({"config": config.to_dict(), "model": state, "metric": metric, "step": step},
                     self.best_path)

    def _rotate(self) -> None:
        """Keep at most `keep_last` historical step files (tag_stepNNN.pt)."""
        if self.keep_last <= 0:
            return
        snaps = sorted(self.root.glob(f"{self.tag}_step*.pt"), key=lambda p: p.stat().st_mtime)
        for old in snaps[: max(0, len(snaps) - self.keep_last)]:
            old.unlink(missing_ok=True)

    def snapshot(self, step: int, src: Optional[Path] = None) -> Path:
        src = src or self.model_path
        dst = self.root / f"{self.tag}_step{step}.pt"
        shutil.copyfile(src, dst)
        return dst

    def load_model(self, model: torch.nn.Module) -> dict:
        return load_model_into(model, self.model_path)

    def load_resume(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler=None) -> Optional[dict]:
        if not self.resume_path.exists():
            return None
        payload = torch.load(self.resume_path, map_location="cpu", weights_only=False)
        model_state = torch.load(self.model_path, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(model_state, strict=False)
        optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and "scaler" in payload and getattr(scaler, "is_enabled", lambda: False)():
            scaler.load_state_dict(payload["scaler"])
        return payload

    def exists(self) -> bool:
        return self.model_path.exists()


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_model_into(model: torch.nn.Module, path: str | Path) -> dict:
    """Load weights from `path` into `model`; returns the raw payload."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    # rope tables are non-persistent; everything else missing is a real error
    missing = [k for k in missing if "rope_" not in k]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return payload


def load_model_from_file(path: str | Path, model_cls, config_cls):
    """Load a `.pt` written by Checkpointer and rebuild the model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = config_cls.from_dict(payload["config"])
    model = model_cls(config)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    missing = [k for k in missing if "rope_" not in k]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model, config, payload