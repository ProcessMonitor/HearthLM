"""Training objectives. Each returns (loss, metrics) from a batch."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import HearthConfig


def _aux_total(model, output) -> torch.Tensor:
    aux = output.aux_loss
    if aux is None:
        return torch.zeros((), device=output.logits.device if output.logits is not None else "cpu")
    return aux


def pretrain_step(model, batch: dict) -> tuple[torch.Tensor, dict]:
    out = model(
        batch["input_ids"],
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
        document_ids=batch.get("document_ids"),
    )
    aux = _aux_total(model, out)
    loss = out.loss + aux
    metrics = {"loss": out.loss.detach()}
    if out.aux_loss is not None:
        metrics["aux"] = out.aux_loss.detach()
    return loss, metrics


def sft_step(model, batch: dict) -> tuple[torch.Tensor, dict]:
    out = model(
        batch["input_ids"],
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
    )
    aux = _aux_total(model, out)
    loss = out.loss + aux
    metrics = {"loss": out.loss.detach()}
    if out.aux_loss is not None:
        metrics["aux"] = out.aux_loss.detach()
    return loss, metrics


def _sequence_logprob(model, input_ids, labels, attention_mask) -> torch.Tensor:
    """Sum of log-probs over supervised tokens, per sequence."""
    out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1].float()
    tgt = labels[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(2, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = tgt.ne(-100)
    return (logp * mask).sum(dim=-1), logits, mask


def dpo_step(model, batch: dict, *, ref_model=None, beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """Reference-free when `ref_model` is None (RPO/SimPO-style implicit reference)."""
    c_logp, c_logits, c_mask = _sequence_logprob(
        model, batch["chosen_input_ids"], batch["chosen_labels"], batch.get("chosen_mask")
    )
    r_logp, r_logits, r_mask = _sequence_logprob(
        model, batch["rejected_input_ids"], batch["rejected_labels"], batch.get("rejected_mask")
    )

    if ref_model is not None:
        with torch.no_grad():
            c_ref, _, _ = _sequence_logprob(
                ref_model, batch["chosen_input_ids"], batch["chosen_labels"], batch.get("chosen_mask")
            )
            r_ref, _, _ = _sequence_logprob(
                ref_model, batch["rejected_input_ids"], batch["rejected_labels"], batch.get("rejected_mask")
            )
    else:
        c_ref = torch.zeros_like(c_logp)
        r_ref = torch.zeros_like(r_logp)

    # Length-normalise so long answers do not dominate the preference margin.
    c_len = c_mask.sum(-1).clamp_min(1)
    r_len = r_mask.sum(-1).clamp_min(1)
    c_margin = (c_logp - c_ref) / c_len
    r_margin = (r_logp - r_ref) / r_len
    diff = beta * (c_margin - r_margin)
    loss = -F.logsigmoid(diff).mean()

    with torch.no_grad():
        accuracy = (diff > 0).float().mean()
        margin = (c_margin - r_margin).mean()
    return loss, {"loss": loss.detach(), "acc": accuracy, "margin": margin}


def distillation_step(
    model,
    batch: dict,
    *,
    teacher=None,
    temperature: float = 1.0,
    kl_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Hard-label CE plus KL to the teacher on the same tokens."""
    out = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)

    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    ce = (ce * mask.reshape(-1)).sum() / mask.sum().clamp_min(1)

    if teacher is not None:
        with torch.no_grad():
            t_logits = teacher(batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float()
            t_logits = t_logits[..., : logits.size(-1)]
        t = temperature
        kl = F.kl_div(
            F.log_softmax(logits[mask] / t, dim=-1),
            F.log_softmax(t_logits[mask] / t, dim=-1),
            log_target=True,
            reduction="batchmean",
        ) * (t * t)
        loss = (1 - kl_weight) * ce + kl_weight * kl
        return loss, {"loss": loss.detach(), "ce": ce.detach(), "kl": kl.detach()}
    return ce, {"loss": ce.detach()}


def grpo_step(
    model,
    batch: dict,
    *,
    ref_model=None,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    pad_id: int = 2,
    eos_id: int = 1,
) -> tuple[torch.Tensor, dict]:
    """Group-relative policy optimisation on pre-rolled completions.

    Batch keys: prompt_ids (B*G, P), completion_ids (B*G, R), rewards (B*G),
    group_size. Groups are consecutive rows, which is how the rollout fills them.
    """
    prompt_ids = batch["prompt_ids"]
    completion_ids = batch["completion_ids"]
    rewards = batch["rewards"].float()
    group_size = int(batch["group_size"])

    full = torch.cat([prompt_ids, completion_ids], dim=1)
    attn = (full != pad_id).long()
    prompt_len = prompt_ids.size(1)

    # Advantages are computed inside each prompt's group of G samples.
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    advantages = ((grouped - mean) / std).reshape(-1)

    out = model(full, attention_mask=attn)
    logits = out.logits[:, :-1].float()
    tgt = full[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(2, tgt.unsqueeze(-1)).squeeze(-1)

    # Only the completion tokens carry the policy gradient.
    comp_mask = torch.zeros_like(tgt, dtype=torch.bool)
    comp_mask[:, prompt_len - 1:] = True
    comp_mask &= tgt.ne(pad_id)

    if ref_model is not None:
        with torch.no_grad():
            ref_logp = F.log_softmax(ref_model(full, attention_mask=attn).logits[:, :-1].float(), dim=-1)
            ref_logp = ref_logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
    else:
        kl = torch.zeros_like(logp)

    # Single-step PPO: the ratio is exp(logp - logp) = 1 at collection time, so
    # the clipped objective collapses to -A * logp with a KL penalty.
    per_token = -(advantages.unsqueeze(1) * logp - beta * kl)
    denom = comp_mask.sum(dim=1).clamp_min(1)
    loss = (per_token * comp_mask).sum(dim=1).div(denom).mean()

    with torch.no_grad():
        metrics = {
            "loss": loss.detach(),
            "reward": rewards.mean(),
            "reward_std": rewards.std() if rewards.numel() > 1 else rewards.new_zeros(()),
            "kl": (kl * comp_mask).sum().div(comp_mask.sum().clamp_min(1)),
        }
    return loss, metrics