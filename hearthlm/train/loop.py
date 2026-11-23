"""One training loop, many objectives.

Objective differences live in a small `Step` object that computes a loss from a
batch. Everything else - accumulation, clipping, scheduling, evaluation,
checkpointing, early stopping, metric logging - is shared. MiniMind's version of
this file is nine near-identical copies; that is how accumulation bugs and
inconsistent schedules creep in.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

import torch.distributed as dist

from .optim import Checkpointer, cosine_lr, create_optimizer, set_lr, unwrap
from .runtime import History, autocast, is_main_process, log, make_grad_scaler, world_size


# --------------------------------------------------------------------- sampler
class TokenBudgetSampler(Sampler[list[int]]):
    """Batch indices so each batch holds roughly the same number of tokens.

    Length-bucketed batches cut padding dramatically for SFT data, where samples
    range from 20 to 2000 tokens. `unit="samples"` groups n indices, which is what
    pretraining wants since packing already equalises length.
    """

    def __init__(self, lengths: Iterable[int], batch_size: int, *, unit: str = "tokens",
                 max_batch: Optional[int] = None, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = False):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.unit = unit
        # Cap on indices per batch so one huge sample cannot stall a step. With
        # token budgets this has to be generous: short documents should pack.
        self.max_batch = max_batch or (max(batch_size * 4, 8) if unit == "samples" else 4096)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterable[list[int]]:
        import random

        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        if self.unit == "tokens":
            # Shuffle inside a coarse bucket so batches are homogeneous but not
            # in a fixed order across epochs.
            order.sort(key=lambda i: self.lengths[i])
            block = max(self.max_batch * 16, 64)
            for start in range(0, len(order), block):
                chunk = order[start:start + block]
                rng.shuffle(chunk)
                order[start:start + block] = chunk
        elif self.shuffle:
            rng.shuffle(order)

        batch: list[int] = []
        cost = 0
        for idx in order:
            item = self.lengths[idx] if self.unit == "tokens" else 1
            if (cost + item > self.batch_size or len(batch) >= self.max_batch) and batch:
                yield batch
                batch, cost = [], 0
            batch.append(idx)
            cost += item
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.unit == "samples":
            n = len(self.lengths) // self.batch_size
            return n if self.drop_last else math.ceil(len(self.lengths) / self.batch_size)
        total = max(1, sum(self.lengths))
        return max(1, math.ceil(total / self.batch_size))


# ----------------------------------------------------------------------- steps
@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_metric: float = math.inf
    tokens_seen: int = 0


@dataclass
class TrainerConfig:
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    accumulation: int = 1
    max_grad_norm: Optional[float] = None
    log_every: int = 20
    eval_every: Optional[int] = None
    save_every: Optional[int] = None
    max_steps: Optional[int] = None
    early_stop_patience: Optional[int] = None
    compile: bool = False
    ema_decay: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_grad_norm is not None:
            self.grad_clip = self.max_grad_norm


class EMA:
    """Exponential moving average of weights, kept in fp32 on the CPU side of the device."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in unwrap(model).named_parameters() if v.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in unwrap(model).named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, param in unwrap(model).named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name].to(param.dtype))


def train(
    *,
    model: torch.nn.Module,
    step_fn: Callable[[torch.nn.Module, dict], tuple[torch.Tensor, dict]],
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    cfg: TrainerConfig,
    checkpointer: Optional[Checkpointer] = None,
    eval_fn: Optional[Callable[[torch.nn.Module], dict]] = None,
    state: Optional[TrainState] = None,
    history: Optional[History] = None,
    sampler: Optional[Sampler] = None,
    resume: bool = False,
    scheduler_steps: Optional[int] = None,
) -> TrainState:
    state = state or TrainState()
    history = history or History()
    model = model.to(device)
    if cfg.compile and hasattr(torch, "compile") and device.type != "mps":
        model = torch.compile(model)

    optimizer = create_optimizer(
        model, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )
    scaler = make_grad_scaler(device, dtype)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    if resume and checkpointer is not None:
        payload = checkpointer.load_resume(model, optimizer, scaler)
        if payload:
            state.step = payload.get("step", 0)
            state.epoch = payload.get("epoch", 0)
            log(f"resumed from step {state.step}")

    steps_per_epoch = len(dataloader)
    # `len(dataloader)` counts micro-batches; one optimizer step consumes
    # `accumulation` of them. Mixing the two units makes the schedule run long by
    # a factor of `accumulation` and makes --max-steps mean the wrong thing.
    accumulate = max(1, cfg.accumulation)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accumulate)
    total_steps = scheduler_steps or (optimizer_steps_per_epoch * cfg.epochs)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps + state.step)
    warmup = int(cfg.warmup_ratio * total_steps)
    # On a small dataset a fixed log interval can exceed the whole run, which
    # produces a silent training log. Never log less than five times per run.
    log_every = max(1, min(cfg.log_every, max(1, total_steps // 5)))

    raw_model = model
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    t0 = time.time()
    micro_since_log = 0
    running: dict[str, float] = {}
    patience = 0

    for epoch in range(state.epoch, cfg.epochs):
        state.epoch = epoch
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # `accumulate` is hoisted above the epoch loop; do not shadow it here.
        micro = 0
        for batch in dataloader:
            micro += 1
            batch = _to_device(batch, device)
            lr = cosine_lr(state.step, total_steps=total_steps, base_lr=cfg.lr,
                           warmup_steps=warmup, min_lr_ratio=cfg.min_lr_ratio)
            set_lr(optimizer, lr)

            with autocast(device, dtype):
                loss, info = step_fn(model, batch)
            if not torch.isfinite(loss):
                log(f"step {state.step}: non-finite loss, skipping update")
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            scaled = loss / accumulate
            scaler.scale(scaled).backward()
            state.tokens_seen += int(batch.get("n_tokens", 0))

            for k, v in info.items():
                running[k] = running.get(k, 0.0) + float(v)
            micro_since_log += 1

            if micro % accumulate == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                state.step += 1

                if state.step % log_every == 0:
                    avg = {k: v / max(micro_since_log, 1) for k, v in running.items()}
                    elapsed = time.time() - t0
                    tok_s = state.tokens_seen / max(elapsed, 1e-6)
                    msg = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    log(f"epoch {epoch + 1}/{cfg.epochs} step {state.step}/{total_steps} "
                        f"lr={lr:.2e} grad={float(grad_norm):.3f} {msg} tok/s={tok_s:.0f}")
                    history.log(step=state.step, epoch=epoch, lr=lr, grad_norm=float(grad_norm),
                                tokens_seen=state.tokens_seen, tok_per_s=tok_s, **avg)
                    running.clear()
                    micro_since_log = 0

                # Always evaluate on the final scheduled step, otherwise a short
                # run never produces a best checkpoint.
                due = eval_fn is not None and cfg.eval_every and (
                    state.step % cfg.eval_every == 0 or state.step >= total_steps
                )
                if due:
                    metrics = _evaluate(model, eval_fn, device, ema)
                    log(f"step {state.step} eval: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                    history.log(step=state.step, split="eval", **metrics)
                    monitor = metrics.get("eval_loss", metrics.get("loss", math.inf))
                    if monitor < state.best_metric - 1e-4:
                        state.best_metric = monitor
                        patience = 0
                        if checkpointer is not None and is_main_process():
                            # Best snapshot goes to a separate tag: sft_best.pt must
                            # not be clobbered by the final overfit weights.
                            checkpointer.save_best(model, unwrap(model).config, monitor, state.step)
                    else:
                        patience += 1
                        if cfg.early_stop_patience and patience >= cfg.early_stop_patience:
                            log(f"early stopping at step {state.step}")
                            _final_save(checkpointer, model, optimizer, scaler, state, epoch, cfg)
                            return state

                if (checkpointer is not None and cfg.save_every
                        and state.step % cfg.save_every == 0 and is_main_process()):
                    checkpointer.save(model, unwrap(model).config, optimizer=optimizer,
                                      scaler=scaler, step=state.step, epoch=epoch)
                    checkpointer.snapshot(state.step)

                if cfg.max_steps and state.step >= cfg.max_steps:
                    log(f"reached max_steps={cfg.max_steps}")
                    _final_save(checkpointer, model, optimizer, scaler, state, epoch, cfg)
                    return state

    _final_save(checkpointer, model, optimizer, scaler, state, cfg.epochs - 1, cfg)
    return state


def _final_save(checkpointer, model, optimizer, scaler, state, epoch, cfg) -> None:
    if checkpointer is None or not is_main_process():
        return
    checkpointer.save(model, unwrap(model).config, optimizer=optimizer, scaler=scaler,
                      step=state.step, epoch=epoch)


def _evaluate(model, eval_fn, device, ema) -> dict:
    was_training = model.training
    model.eval()
    backup = None
    if ema is not None:
        backup = {k: v.detach().clone() for k, v in unwrap(model).state_dict().items()
                  if k in ema.shadow}
        ema.copy_to(unwrap(model))
    try:
        with torch.no_grad():
            return eval_fn(model)
    finally:
        if backup is not None:
            unwrap(model).load_state_dict(backup, strict=False)
        if was_training:
            model.train()


def _to_device(batch: dict, device: torch.device) -> dict:
    # non_blocking only helps when the source is pinned CUDA host memory. On MPS
    # and CPU an async copy from pageable memory can read uninitialised pages,
    # which shows up as garbage labels and a NaN loss rather than an error.
    async_ok = device.type == "cuda"
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=async_ok)
        else:
            out[k] = v
    n_tokens = out.get("n_tokens")
    if not isinstance(n_tokens, int):
        ids = out.get("input_ids")
        mask = out.get("attention_mask")
        if ids is not None:
            out["n_tokens"] = int((mask if mask is not None else torch.ones_like(ids)).sum().item())
    return out


# --------------------------------------------------------- distributed helper
def build_dataloader(dataset, batch_size, *, collate_fn, sampler=None, num_workers=0,
                     pin_memory=True, drop_last=False):
    if dist.is_available() and dist.is_initialized() and sampler is None:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )