"""Config-driven training entry point.

    python -m hearthlm.train --stage pretrain --data data/corpus.jsonl --preset nano
    python -m hearthlm.train --stage sft --data data/sft.jsonl --init out/pretrain.pt

Each stage reads a JSON config file if `--config` is given, with CLI flags
overriding. This is the only place that knows about all the stages, so the stage
list and its defaults live in one table instead of nine argparse blocks.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch

from ..config import HearthConfig, build_config
from ..data import (
    DPODataset,
    PackCollator,
    PretrainDataset,
    SFTDataset,
    TokenizedCorpus,
    Tokenizer,
    collate_dpo,
    collate_sft,
)
from .loop import EMA, TrainerConfig, TrainState, TokenBudgetSampler, train
from .optim import Checkpointer, create_optimizer, load_model_from_file
from . import objectives
from .runtime import (
    History,
    cleanup_distributed,
    human_int,
    is_main_process,
    log,
    log as log_msg,
    maybe_init_distributed,
    pick_device,
    resolve_dtype,
    resolve_train_dtype,
    seed_everything,
)
from ..model import HearthLM
from .lora import apply_lora, lora_parameters, lora_state_dict, save_lora

STAGES = {
    "pretrain": dict(lr=6e-4, epochs=1, batch_size=24, accumulation=4, seq_len=1024, warmup_ratio=0.02),
    "sft": dict(lr=8e-5, epochs=2, batch_size=12, accumulation=2, seq_len=1024, warmup_ratio=0.03),
    "lora": dict(lr=2e-4, epochs=3, batch_size=12, accumulation=2, seq_len=1024, warmup_ratio=0.03),
    "dpo": dict(lr=5e-7, epochs=1, batch_size=4, accumulation=4, seq_len=1024, warmup_ratio=0.0),
    "distill": dict(lr=8e-5, epochs=2, batch_size=8, accumulation=2, seq_len=1024, warmup_ratio=0.03),
    "grpo": dict(lr=1e-6, epochs=1, batch_size=2, accumulation=8, seq_len=768, warmup_ratio=0.0),
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HearthLM training")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--config", type=str, default=None, help="JSON file with any of these flags")
    p.add_argument("--data", type=str, required=False, help="dataset path (.jsonl or corpus dir)")
    p.add_argument("--tokenizer", type=str, default="assets/tokenizer")
    p.add_argument("--init", type=str, default=None, help="checkpoint .pt to start from")
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--preset", type=str, default="mini", choices=["nano", "mini", "base", "moe"])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any model config field, e.g. --set num_hidden_layers=10")

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accumulation", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--early-stop", type=int, default=None,
                   help="stop after this many evals without improvement")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--ema", type=float, default=None, help="EMA decay, e.g. 0.999")
    p.add_argument("--compile", action="store_true")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true")

    # dtype-specific
    p.add_argument("--thinking", action="store_true", help="train with think blocks enabled")
    p.add_argument("--beta", type=float, default=None, help="DPO/GRPO beta")
    p.add_argument("--ref", type=str, default=None, help="reference model .pt for DPO/GRPO/distill")
    p.add_argument("--teacher", type=str, default=None, help="teacher .pt for distillation")
    p.add_argument("--kl-weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--group-size", type=int, default=4, help="GRPO generations per prompt")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=None)
    return p


def _apply_config_file(args, path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in data.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            raise KeyError(f"config file has unknown key {key!r}")
        setattr(args, attr, value)


def _parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[key.strip()] = value
    return out


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.config:
        _apply_config_file(args, args.config)

    defaults = STAGES[args.stage]
    for key, value in defaults.items():
        attr = key.replace("-", "_")
        if getattr(args, attr) is None:
            setattr(args, attr, value)
    if args.device == "auto" and args.dtype == "auto" and args.stage == "pretrain":
        pass  # nothing special; keeps the branch explicit for future tuning

    distributed = maybe_init_distributed()
    seed_everything(args.seed)

    device = pick_device(args.device)
    dtype = resolve_train_dtype(args.dtype, device)
    log(f"device={device} dtype={dtype} stage={args.stage}")

    tokenizer = Tokenizer(args.tokenizer)
    model_cfg = _model_config(args, tokenizer, Path(args.init) if args.init else None)

    model = HearthLM(model_cfg)
    if args.init:
        model, loaded_cfg, _ = load_model_from_file(args.init, HearthLM, HearthConfig)
        model_cfg = loaded_cfg
        log(f"initialised from {args.init}")

    log(f"parameters: {human_int(model.num_parameters())} total, "
        f"{human_int(model.active_parameters())} active per token")

    if args.stage in {"lora", "dpo", "distill", "grpo"} and not args.init:
        raise SystemExit(f"--stage {args.stage} needs --init (a base checkpoint)")

    if args.stage == "lora":
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        trainable = sum(p.numel() for p in lora_parameters(model))
        log(f"lora rank={args.lora_rank} trainable={human_int(trainable)}")

    out_dir = Path(args.out)
    ckpt = Checkpointer(out_dir / "checkpoints", tag=args.stage)
    history = History(out_dir / f"{args.stage}_metrics.jsonl")

    tc = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay if args.weight_decay is not None else 0.1,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip if args.grad_clip is not None else 1.0,
        accumulation=args.accumulation,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_steps=args.max_steps,
        compile=args.compile,
        ema_decay=args.ema,
        early_stop_patience=args.early_stop,
    )

    ref_model = _maybe_load(args.ref, device) if args.ref else None
    teacher = _maybe_load(args.teacher, device) if args.teacher else None

    loader, eval_loader, sampler, step_fn, eval_fn = _build_stage(
        args, tokenizer, model, device, dtype, ref_model, teacher
    )

    if distributed:
        import torch.distributed as dist
        log_msg(f"distributed: world_size={dist.get_world_size()}")

    train(
        model=model,
        step_fn=step_fn,
        dataloader=loader,
        device=device,
        dtype=dtype,
        cfg=tc,
        checkpointer=ckpt,
        eval_fn=eval_fn,
        history=history,
        sampler=sampler,
        resume=args.resume,
    )

    if args.stage == "lora" and is_main_process():
        lora_path = out_dir / "checkpoints" / "lora.pt"
        save_lora(model, str(lora_path))
        log(f"lora adapter written to {lora_path}")

    cleanup_distributed()


def _model_config(args, tokenizer: Tokenizer, init_path: Optional[Path]) -> HearthConfig:
    overrides = _parse_set(args.set)
    overrides.setdefault("vocab_size", tokenizer.vocab_size)
    overrides.setdefault("bos_token_id", tokenizer.bos_token_id)
    overrides.setdefault("eos_token_id", tokenizer.eos_token_id)
    overrides.setdefault("pad_token_id", tokenizer.pad_token_id)
    if init_path is not None:
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        cfg = HearthConfig.from_dict(payload["config"])
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg
    return build_config(args.preset, **overrides)


def _maybe_load(path: str, device: torch.device):
    model, cfg, _ = load_model_from_file(path, HearthLM, HearthConfig)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    log(f"loaded auxiliary model from {path} ({human_int(model.num_parameters())} params)")
    return model


def _build_stage(args, tokenizer, model, device, dtype, ref_model, teacher):
    """Return (train_loader, eval_loader, sampler, step_fn, eval_fn)."""
    pad_id = tokenizer.pad_token_id

    if args.stage in {"pretrain", "sft", "lora", "distill"}:
        if args.stage == "pretrain":
            corpus_dir = Path(args.data)
            if not (corpus_dir / "tokens.bin").exists():
                raise SystemExit(
                    f"{corpus_dir} has no tokens.bin. Build it first:\n"
                    f"  python -m hearthlm.data.build_corpus --data <files> --out {corpus_dir}"
                )
            corpus = TokenizedCorpus(corpus_dir)
            dataset = PretrainDataset(corpus, args.seq_len, seed=args.seed)
            collate = PackCollator(args.seq_len, pad_id)
        else:
            dataset = SFTDataset(
                args.data, tokenizer, max_len=args.seq_len, enable_thinking=args.thinking,
                val_ratio=args.val_ratio, split="train", seed=args.seed,
            )
            if args.val_ratio > 0:
                eval_ds = SFTDataset(
                    args.data, tokenizer, max_len=args.seq_len, enable_thinking=args.thinking,
                    val_ratio=args.val_ratio, split="val", seed=args.seed,
                )
            collate = lambda b: collate_sft(b, pad_id)  # noqa: E731

        if args.stage == "pretrain":
            # Each row is already one packed sequence, so the batch unit is rows.
            sampler = TokenBudgetSampler([1] * len(dataset), args.batch_size,
                                         unit="samples", seed=args.seed)
        else:
            # SFT samples vary by an order of magnitude; group them by token count.
            # Note: len(dataset[i]) is the dict size, so read the tensor instead.
            lengths = [int(dataset[i]["input_ids"].numel()) for i in range(len(dataset))]
            sampler = TokenBudgetSampler(lengths, args.batch_size * args.seq_len,
                                         unit="tokens", seed=args.seed)

        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        eval_loader = None
        if args.stage != "pretrain" and args.val_ratio > 0:
            eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collate, num_workers=args.num_workers)

        base_step = objectives.sft_step if args.stage in {"sft", "lora"} else (
            objectives.pretrain_step if args.stage == "pretrain" else None)

        if args.stage == "distill":
            def step_fn(m, batch):
                return objectives.distillation_step(
                    m, batch, teacher=teacher, temperature=args.temperature, kl_weight=args.kl_weight
                )
        else:
            step_fn = base_step

        eval_fn = _make_eval_fn(base_step, eval_loader, dtype) if eval_loader is not None else None
        return loader, eval_loader, sampler, step_fn, eval_fn

    if args.stage == "dpo":
        from torch.utils.data import DataLoader

        ds = DPODataset(args.data, tokenizer, max_len=args.seq_len, val_ratio=args.val_ratio, split="train")
        val = DPODataset(args.data, tokenizer, max_len=args.seq_len, val_ratio=args.val_ratio, split="val")
        beta = args.beta if args.beta is not None else 0.1
        collate = lambda b: collate_dpo(b, pad_id)  # noqa: E731
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                            num_workers=args.num_workers)
        eval_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate) if args.val_ratio > 0 else None

        def step_fn(m, batch):
            return objectives.dpo_step(m, batch, ref_model=ref_model, beta=beta)

        def eval_fn(m):
            m.eval()
            totals, n = {}, 0
            with torch.no_grad():
                for b in eval_loader:
                    b = {k: v.to(device) for k, v in b.items()}
                    loss, info = objectives.dpo_step(m, b, ref_model=ref_model, beta=beta)
                    totals["eval_loss"] = totals.get("eval_loss", 0.0) + float(loss)
                    totals["eval_acc"] = totals.get("eval_acc", 0.0) + float(info["acc"])
                    n += 1
            m.train()
            return {k: v / max(n, 1) for k, v in totals.items()}

        return loader, eval_loader, None, step_fn, (eval_fn if eval_loader else None)

    if args.stage == "grpo":
        return _build_grpo(args, tokenizer, model, device, dtype, ref_model)

    raise ValueError(f"unhandled stage {args.stage}")


def _make_eval_fn(step_fn, eval_loader, dtype):
    # The model has no `.device` attribute; take it from its parameters.
    probe = next(iter(eval_loader), None)
    device = probe["input_ids"].device if probe is not None else torch.device("cpu")

    def eval_fn(m):
        target = next(m.parameters()).device
        totals: dict[str, float] = {}
        n = 0
        for batch in eval_loader:
            batch = {k: (v.to(target) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, info = step_fn(m, batch)
            totals["eval_loss"] = totals.get("eval_loss", 0.0) + float(loss)
            n += 1
        return {"eval_loss": totals.get("eval_loss", 0.0) / max(n, 1)}
    return eval_fn


def _build_grpo(args, tokenizer, model, device, dtype, ref_model):
    """Rollouts generated in-loop with the policy; rewards are pluggable."""
    from torch.utils.data import DataLoader, Dataset

    from ..model.generate import SamplingParams, generate

    class PromptDataset(Dataset):
        def __init__(self, path: str):
            self.rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            row = self.rows[i]
            from ..data.chat_template import render_text
            msgs = row.get("conversations", row.get("messages"))
            if row.get("prompt"):
                return {"prompt": row["prompt"], "answer": row.get("answer", "")}
            return {"prompt": render_text(msgs[:-1], add_generation_prompt=True), "answer": row.get("answer", "")}

    ds = PromptDataset(args.data)

    def collate(items):
        return {"prompts": [i["prompt"] for i in items], "answers": [i["answer"] for i in items]}

    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
    group_size = args.group_size
    beta = args.beta if args.beta is not None else 0.04
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id

    def make_step(m):
        def step_fn(model, batch):
            prompts = batch["prompts"]
            answers = batch["answers"]
            params = SamplingParams(max_new_tokens=args.seq_len, temperature=0.9, top_p=0.95, top_k=0,
                                    repetition_penalty=1.05)
            rows_ids, rows_comp, rewards = [], [], []
            # Rollout must not use inference_mode: tensors created there cannot be
            # used in the backward pass. no_grad is enough, and the ids are longs
            # that get re-embedded in the training forward anyway.
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    for prompt, answer in zip(prompts, answers):
                        ids = tokenizer.encode(prompt)[-args.seq_len:]
                        inp = torch.tensor([ids], device=device)
                        for _g in range(group_size):
                            out = generate(model, inp, params, eos_token_id=eos_id)
                            comp = out[0, len(ids):].tolist()
                            rows_ids.append(ids)
                            rows_comp.append(comp)
                            rewards.append(_rule_reward(tokenizer.decode(comp), answer))
            finally:
                if was_training:
                    model.train()

            width_p = max(len(r) for r in rows_ids)
            width_c = max(max((len(c) for c in rows_comp), default=1), 1)
            p_ids = torch.full((len(rows_ids), width_p), pad_id, dtype=torch.long, device=device)
            c_ids = torch.full((len(rows_comp), width_c), pad_id, dtype=torch.long, device=device)
            for i, (p, c) in enumerate(zip(rows_ids, rows_comp)):
                p_ids[i, :len(p)] = torch.tensor(p, device=device)
                if c:
                    c_ids[i, :len(c)] = torch.tensor(c, device=device)
            batch2 = {"prompt_ids": p_ids, "completion_ids": c_ids,
                      "rewards": torch.tensor(rewards, device=device), "group_size": group_size}
            return objectives.grpo_step(model, batch2, ref_model=ref_model, beta=beta,
                                        pad_id=pad_id, eos_id=eos_id)
        return step_fn

    return loader, None, None, make_step(model), None


def _rule_reward(text: str, answer: str) -> float:
    """Shaped reward: correctness when a reference answer exists, plus hygiene terms.

    Deliberately simple and deterministic. Swap in a model-based reward by
    passing a different function; nothing else in the GRPO step assumes this one.
    """
    import re

    text = text.strip()
    if not text:
        return -1.0
    score = 0.0
    if 4 <= len(text) <= 1200:
        score += 0.2
    if text.count("</think>") > 1:
        score -= 0.3
    # crude degenerate-repetition check: most common 4-gram share
    toks = re.findall(r"\w+", text.lower())
    if len(toks) >= 8:
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        share = max(grams.count(g) for g in set(grams)) / len(grams)
        score -= 2.0 * max(0.0, share - 0.2)
    if answer:
        target = answer.strip().lower()
        score += 1.0 if target and target in text.lower() else -0.5
    return score


if __name__ == "__main__":
    main()