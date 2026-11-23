"""Export a checkpoint to a self-contained directory.

    python -m hearthlm.export --checkpoint out/checkpoints/sft.pt --out export/hearthlm-sft

Writes `config.json`, `model.safetensors` (or `model.pt` when safetensors is not
installed), and the tokenizer files. The exported layout matches what
`hearthlm.load()` reads, so nothing about reloading depends on the checkpoint
format the trainers use.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .config import HearthConfig
from .model.hearth import HearthLM


def export(checkpoint: str | Path, out_dir: str | Path, tokenizer_dir: str | Path,
           *, dtype: torch.dtype = torch.float16) -> Path:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = HearthConfig.from_dict(payload["config"])
    model = HearthLM(config)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    missing = [k for k in missing if "rope_" not in k]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model = model.to(dtype).eval()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "config.json")

    # RoPE tables are derived from config; keeping them out keeps the file small
    # and removes a way for the checkpoint to disagree with the config.
    state = {k: v for k, v in model.state_dict().items() if "rope_" not in k}

    try:
        from safetensors.torch import save_file

        weights = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}
        if config.tie_word_embeddings:
            # lm_head and embed_tokens share storage; safetensors refuses two names
            # on one buffer. The loader re-ties from config.
            weights.pop("lm_head.weight", None)
        save_file({k: v.contiguous() for k, v in weights.items()}, str(out / "model.safetensors"))
        weights_file = "model.safetensors"
    except (ImportError, RuntimeError):
        torch.save({"config": config.to_dict(), "model": state}, out / "model.pt")
        weights_file = "model.pt"

    for name in ("tokenizer.json", "tokenizer_meta.json", "tokenizer_config.json"):
        src = Path(tokenizer_dir) / name
        if src.exists():
            shutil.copyfile(src, out / name)

    meta = {
        "weights_file": weights_file,
        "dtype": str(dtype).replace("torch.", ""),
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_active": model.active_parameters(),
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"exported to {out} ({weights_file}, {meta['params_total'] / 1e6:.1f}M params)")
    return out


def load(export_dir: str | Path, device: str = "cpu", dtype: str = "auto"):
    """Inverse of `export`. Returns (model, config, tokenizer)."""
    from .data import Tokenizer
    from .train.runtime import pick_device, resolve_dtype

    root = Path(export_dir)
    config = HearthConfig.load(root / "config.json")
    model = HearthLM(config)

    meta_path = root / "export_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("weights_file", "").endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(str(root / meta["weights_file"]))
    else:
        state = torch.load(root / (meta.get("weights_file") or "model.pt"), map_location="cpu")["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    # RoPE tables are non-persistent, and a tied lm_head is omitted from
    # safetensors on purpose: it shares storage with the embedding.
    ignored = {"lm_head.weight"} if config.tie_word_embeddings else set()
    missing = [k for k in missing if "rope_" not in k and k not in ignored]
    if missing or unexpected:
        raise RuntimeError(f"export mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    dev = pick_device(device)
    dt = resolve_dtype(dtype, dev) if dtype != "auto" else None
    model = model.to(dev)
    if dt is not None and dt != torch.float32:
        model = model.to(dt)
    model.eval()
    return model, config, Tokenizer(root)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export a HearthLM checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="assets/tokenizer")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args(argv)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    export(args.checkpoint, args.out, args.tokenizer, dtype=dtype)


if __name__ == "__main__":
    main()