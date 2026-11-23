"""Fold a LoRA adapter into a base checkpoint.

    python scripts/merge_lora.py --base out/checkpoints/sft.pt --lora out/checkpoints/lora.pt --out out/checkpoints/merged.pt

The merge happens in fp32 on the CPU, then the result is written in the same
format the trainers produce, so `--init out/checkpoints/merged.pt` works.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hearthlm.config import HearthConfig
from hearthlm.model.hearth import HearthLM
from hearthlm.train.lora import apply_lora, load_lora, lora_state_dict, merge_lora
from hearthlm.train.optim import load_model_from_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="base checkpoint .pt")
    p.add_argument("--lora", required=True, help="adapter .pt from a lora stage")
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=None, help="rank; read from the adapter if omitted")
    args = p.parse_args()

    model, config, _ = load_model_from_file(args.base, HearthLM, HearthConfig)
    payload = torch.load(args.lora, map_location="cpu", weights_only=False)
    rank = args.rank or int(payload.get("meta", {}).get("rank", 16))
    alpha = payload.get("meta", {}).get("alpha")

    apply_lora(model, rank=rank, alpha=alpha)
    load_lora(model, args.lora)
    model.eval()

    # Sanity check before writing: the merged forward must match the adapter path.
    ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        adapted = model(ids).logits
    merge_lora(model)
    with torch.no_grad():
        merged = model(ids).logits
    drift = float((adapted - merged).abs().max())
    if drift > 1e-2:
        raise SystemExit(f"merge drift {drift:.4f} is too large; aborting")
    print(f"merge drift {drift:.2e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": config.to_dict(), "model": {k: v.float() for k, v in model.state_dict().items()}},
        out,
    )
    print(f"merged checkpoint -> {out}")


if __name__ == "__main__":
    main()
