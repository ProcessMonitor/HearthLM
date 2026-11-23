"""Device, precision, seeding and logging.

The point of this module is that every script gets the same device pick and the
same mixed-precision rules. Getting these subtly different between scripts is a
classic source of "the model trains but the loss curve looks wrong".
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


# --------------------------------------------------------------------- device
def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_bf16(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if device.type == "mps":
        # MPS bf16 support exists on macOS 14+ but fp16 is faster and equally
        # stable for inference-shaped workloads.
        return False
    if device.type == "cpu":
        # CPU bf16 matmul is emulated and slower than fp32 on most chips.
        return False
    return False


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Inference precision. Used by chat, serve and eval."""
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16 if supports_bf16(device) else torch.float32
    if requested == "float16":
        return torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    if requested == "auto":
        return torch.bfloat16 if supports_bf16(device) else (torch.float16 if device.type == "mps" else torch.float32)
    raise ValueError(f"unknown dtype {requested!r}")


def resolve_train_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Training precision.

    MPS is excluded from fp16 by default. Measured here, fp16 autocast on MPS is
    about 20% faster per step but overflows roughly a thousand steps into
    pretraining at a learning rate that is perfectly stable in fp32. MPS has no
    working GradScaler, so there is nothing to catch the overflow: the loss goes
    NaN and the run is dead. Twenty percent is not worth an overnight run that
    silently stops learning.

    CUDA keeps bf16 (or fp16 with a scaler), CPU stays fp32.
    """
    if requested == "auto" and device.type == "mps":
        return torch.float32
    if requested == "float16" and device.type == "mps":
        print("warning: fp16 training on MPS has no grad scaler and overflows on long "
              "runs; consider --dtype auto or float32", flush=True)
    return resolve_dtype(requested, device)



@contextlib.contextmanager
def autocast(device: torch.device, dtype: torch.dtype, enabled: bool = True):
    if not enabled or dtype == torch.float32 or device.type == "cpu":
        yield
        return
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=dtype):
            yield
    elif device.type == "mps":
        with torch.autocast("mps", dtype=dtype):
            yield
    else:
        yield


def make_grad_scaler(device: torch.device, dtype: torch.dtype) -> "torch.amp.GradScaler":
    """fp16 needs loss scaling; bf16 and fp32 do not. MPS has no scaler in torch 2.x."""
    enabled = device.type == "cuda" and dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


# --------------------------------------------------------------------- seeding
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------- logging
@dataclass
class History:
    """Append-only list of metric dicts, mirrored to a jsonl file."""

    path: Optional[Path] = None
    rows: list[dict] = field(default_factory=list)

    def log(self, **metrics: Any) -> None:
        row = {"time": round(time.time(), 3), **metrics}
        self.rows.append(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def maybe_init_distributed() -> bool:
    """Initialise torch.distributed when launched by torchrun."""
    if world_size() <= 1:
        return False
    import torch.distributed as dist

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return True


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def log(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def human_int(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{int(n)}"
        n /= 1000
    return f"{n:.1f}T"