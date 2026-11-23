"""Convert a HearthLM checkpoint to a HuggingFace `Qwen3`-style directory.

    python scripts/to_huggingface.py --checkpoint out/checkpoints/sft_best.pt --out export/hf

HearthLM's architecture is deliberately a subset of Qwen3: RMSNorm, GQA, SwiGLU,
RoPE, optional QK-norm, optional MoE. That means the weights map one-to-one onto
`Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`, and you can then use vLLM, llama.cpp,
ollama or the `transformers` generate loop without depending on this repo.

Mapping notes:
  * `model.embed_tokens`      -> `model.embed_tokens`
  * `lm_head`                 -> `lm_head` (tied when tie_word_embeddings)
  * `q_norm` / `k_norm`       -> `self_attn.q_norm` / `k_norm` (Qwen3 has these too)
  * `mlp.experts[i]`          -> `mlp.experts[i]` for MoE checkpoints
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hearthlm.config import HearthConfig
from hearthlm.model.hearth import HearthLM


def to_hf_state(config: HearthConfig, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename HearthLM parameter paths onto the Qwen3 layout.

    The tensor shapes are already identical; only the module names differ, because
    HearthLM calls its attention submodule `attn` while Qwen3 calls it `self_attn`.
    """
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.startswith("model.rope_"):
            continue
        if ".attn." in name:
            name = name.replace(".attn.", ".self_attn.")
        out[name] = tensor

    expected_prefixes = ("model.embed_tokens.", "model.layers.", "model.norm.", "lm_head.")
    strays = [k for k in out if not k.startswith(expected_prefixes)]
    if strays:
        raise RuntimeError(f"unexpected parameter names, cannot map to Qwen3 layout: {strays[:5]}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="assets/tokenizer")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = HearthConfig.from_dict(payload["config"])
    state = to_hf_state(config, payload["model"])
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    state = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hf_config = {
        "architectures": ["Qwen3MoeForCausalLM" if config.use_moe else "Qwen3ForCausalLM"],
        "model_type": "qwen3_moe" if config.use_moe else "qwen3",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "vocab_size": config.vocab_size,
        "tie_word_embeddings": config.tie_word_embeddings,
        "hidden_act": config.hidden_act,
        "attention_bias": config.attention_bias,
        "use_sliding_window": False,
        "sliding_window": None,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "torch_dtype": args.dtype,
    }
    if config.use_moe:
        hf_config.update({
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "norm_topk_prob": config.norm_topk_prob,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
        })
    (out / "config.json").write_text(json.dumps(hf_config, indent=2) + "\n", encoding="utf-8")

    try:
        from safetensors.torch import save_file

        weights = {k: v.contiguous() for k, v in state.items()}
        if config.tie_word_embeddings:
            # Tied weights share storage; safetensors refuses to serialise two
            # names pointing at one buffer. The HF loader re-ties `lm_head` from
            # the config, so dropping it loses nothing.
            weights.pop("lm_head.weight", None)
        save_file(weights, str(out / "model.safetensors"))
    except ImportError:
        torch.save(state, out / "pytorch_model.bin")
        print("safetensors not installed; wrote pytorch_model.bin")

    tok_dir = Path(args.tokenizer)
    shutil.copyfile(tok_dir / "tokenizer.json", out / "tokenizer.json")
    # The template lives in code here, so materialise it for HF users.
    from hearthlm.data import render_text  # noqa: F401
    from hearthlm.data.chat_template import IM_END, IM_START

    template = (
        "{% for m in messages %}"
        "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "add_bos_token": False,
        "add_eos_token": False,
        "chat_template": template,
        "model_max_length": config.max_position_embeddings,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote HuggingFace layout to {out}")
    print("load with: AutoModelForCausalLM.from_pretrained(<path>, trust_remote_code=True)")


if __name__ == "__main__":
    main()