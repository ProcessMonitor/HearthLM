"""Building blocks: normalization, rotary embedding, attention, MLP, MoE, block.

Everything here is plain torch.nn. No transformers, no einops, no custom CUDA.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..config import HearthConfig

# torch >= 2.5 can broadcast grouped KV heads inside SDPA, which saves the
# repeat_kv copy. Fall back to explicit repeat on older builds.
_HAS_GQA = "enable_gqa" in F.scaled_dot_product_attention.__doc__ if F.scaled_dot_product_attention.__doc__ else False


class RMSNorm(nn.Module):
    """RMSNorm with weight applied in the input dtype (no fp32 round trip).

    The fp32 cast that many reference implementations do costs a full copy of
    the activation; it is only needed for fp16 training where the mean of
    squares underflows. We keep the statistics in fp32 but leave the scale
    multiply in the activation dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1_000_000.0,
    scaling: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) each of shape (max_seq_len, head_dim).

    With `scaling={"type": "yarn", "factor": f, "original_max_position_embeddings": n}`
    the inverse frequencies are stretched per the YaRN paper: a ramp interpolates
    between the untouched high frequencies and the fully divided low frequencies.
    """
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    attn_scale = 1.0  # YaRN's magnitude correction, kept at 1.0 like Qwen does
    if scaling:
        stype = scaling.get("type", "yarn")
        if stype != "yarn":
            raise ValueError(f"unsupported rope scaling type {stype!r}")
        factor = float(scaling["factor"])
        orig = float(scaling.get("original_max_position_embeddings", max_seq_len // factor))
        if max_seq_len > orig:
            beta_fast = float(scaling.get("beta_fast", 32.0))
            beta_slow = float(scaling.get("beta_slow", 1.0))
            attn_scale = float(scaling.get("attention_factor", 1.0))

            def inv_dim(beta: float) -> float:
                # frequency index where the period crosses beta tokens
                return head_dim * math.log(orig / (beta * 2 * math.pi)) / (2 * math.log(theta))

            lo = max(math.floor(inv_dim(beta_fast)), 0)
            hi = min(math.ceil(inv_dim(beta_slow)), head_dim // 2 - 1)
            idx = torch.arange(head_dim // 2, dtype=torch.float32, device=device)
            ramp = torch.clamp((idx - lo) / max(hi - lo, 0.001), 0.0, 1.0)
            inv = inv * (1 - ramp + ramp / factor)

    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv)  # (seq, head_dim/2)
    # Duplicate so the first and second half of the head rotate together; this
    # matches the "rotate_half" convention below.
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos() * attn_scale, emb.sin() * attn_scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """q,k: (batch, heads, seq, dim); cos,sin: (batch, 1, seq, dim)."""
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def _expand_batch(mask: torch.Tensor, bsz: int) -> torch.Tensor:
    """Broadcast a batch-1 mask to the real batch size.

    SDPA accepts broadcast bool masks in most versions but expanding explicitly
    avoids a per-backend inconsistency (MPS older builds reject size-1 batches on
    the mask while accepting them on q/k/v).
    """
    if mask.shape[0] != bsz:
        mask = mask.expand(bsz, *mask.shape[1:])
    return mask


def _ensure_self_attention(mask: torch.Tensor, past_len: int, device: torch.device) -> torch.Tensor:
    """Guarantee every query can see at least one key.

    A query row with no allowed key makes softmax sum to zero and produce NaN,
    which then poisons the whole batch's loss and every parameter. It happens in
    two ordinary situations: padding rows (all keys masked by the attention mask)
    and, in packed batches, a row whose document is entirely masked. Letting such
    a row attend to itself is well defined and costs one diagonal write.

    mask: (batch, 1, seq, full) boolean.
    """
    if mask.any(dim=-1).all():
        return mask
    seq_len = mask.shape[2]
    arange = torch.arange(seq_len, device=device)
    self_idx = (arange + past_len).clamp(max=mask.shape[-1] - 1)
    mask = mask.clone()
    mask[:, 0, arange, self_idx] = True
    return mask


def _causal_mask(seq_len: int, past_len: int, device: torch.device) -> torch.Tensor:
    """(1, 1, seq, past+seq) bool: query i sees keys 0..past+i."""
    q = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(1)
    k = torch.arange(past_len + seq_len, device=device).unsqueeze(0)
    return (k <= q).unsqueeze(0).unsqueeze(0)


def _band_mask(seq_len: int, past_len: int, window: int, device: torch.device) -> torch.Tensor:
    """Causal mask restricted to the last `window` keys per query."""
    q = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(1)
    k = torch.arange(past_len + seq_len, device=device).unsqueeze(0)
    return ((k <= q) & (k > q - window)).unsqueeze(0).unsqueeze(0)




class Attention(nn.Module):
    def __init__(self, config: HearthConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.dropout
        self.window = config.sliding_window

        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=config.attention_bias)

        # QK-norm stabilises training at small scale where per-head logits drift.
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional["KVCache"] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        # (b, h, t, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Rotate before caching: the cache must hold position-encoded keys,
        # otherwise every later step would re-rotate them.
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        # Derive the prefix length from the returned keys rather than reading
        # cache.length: the cache has already advanced by `seq_len` at this point.
        past_len = k.shape[2] - seq_len

        dropout = self.dropout if self.training else 0.0
        full_len = past_len + seq_len
        use_causal = False
        attn_mask = None

        if mask is not None:
            m = mask.to(x.device)
            if m.dtype != torch.bool:
                m = m.bool()
            if m.dim() == 4:
                raise ValueError(f"pass a 2-D key mask or 3-D full mask, got rank {m.dim()}")
            if m.dim() == 2:
                # Key-validity mask: (b or 1, full_len). Causality is still
                # required, so fold the (offset) causal triangle in.
                if m.shape[1:] != (full_len,) or m.shape[0] not in (1, bsz):
                    raise ValueError(f"2-D mask must be (batch, {full_len}), got {tuple(m.shape)}")
                attn_mask = m.view(m.shape[0], 1, 1, full_len) & _causal_mask(seq_len, past_len, x.device)
            else:
                # Full mask: (b or 1, seq, full_len), already carries causality.
                if m.shape[1:] != (seq_len, full_len) or m.shape[0] not in (1, bsz):
                    raise ValueError(f"3-D mask must be (batch, {seq_len}, {full_len}), got {tuple(m.shape)}")
                attn_mask = m.unsqueeze(1)
            attn_mask = _expand_batch(attn_mask, bsz)
        elif self.window is not None:
            attn_mask = _band_mask(seq_len, past_len, self.window, x.device)
        elif past_len == 0:
            use_causal = True
        elif seq_len == 1:
            # A single query with a full prefix may attend to everything.
            pass
        else:
            attn_mask = _causal_mask(seq_len, past_len, x.device)

        if attn_mask is not None and self.window is not None:
            attn_mask = attn_mask & _band_mask(seq_len, past_len, self.window, x.device)
        if attn_mask is not None:
            attn_mask = _ensure_self_attention(attn_mask, past_len, x.device)

        if _HAS_GQA and self.n_rep > 1:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout, is_causal=use_causal, enable_gqa=True
            )
        else:
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout, is_causal=use_causal
            )

        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(out))


ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu2": lambda x: F.relu(x) ** 2,
}


class MLP(nn.Module):
    def __init__(self, config: HearthConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        try:
            self.act = ACTIVATIONS[config.hidden_act]
        except KeyError as exc:
            raise ValueError(f"unknown activation {config.hidden_act!r}") from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """Token-choice top-k MoE with switch-style load balancing.

    Dispatch is a single sort followed by one gather/GEMM/scatter per expert.
    That is not as fast as a fused kernel but it is deterministic, portable and
    keeps gradient flow identical to a plain Python loop.
    """

    def __init__(self, config: HearthConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk = config.norm_topk_prob
        self.aux_coef = config.router_aux_loss_coef
        self.z_coef = config.router_z_loss_coef
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            MLP(config, config.moe_intermediate_size) for _ in range(self.num_experts)
        )
        self.aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        flat = x.reshape(-1, h)

        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)
        weights, idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk and self.top_k > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            # rows that picked expert e (possibly several times for top_k > 1)
            rows, slot = torch.where(idx == e)
            if rows.numel() == 0:
                if self.training:  # keep the expert in the graph so DDP does not stall
                    out[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
                continue
            y = expert(flat[rows])
            out.index_add_(0, rows, y * weights[rows, slot].unsqueeze(-1))

        if self.training and self.aux_coef > 0:
            # fraction of tokens routed to each expert vs mean routing probability
            onehot = F.one_hot(idx, self.num_experts).float().mean(0)
            load = onehot.sum(-1) / self.top_k
            self.aux_loss = (load * probs.mean(0)).sum() * self.num_experts * self.aux_coef
            if self.z_coef > 0:
                z = torch.logsumexp(logits, dim=-1)
                self.aux_loss = self.aux_loss + self.z_coef * (z ** 2).mean()
        elif self.training:
            self.aux_loss = logits.new_zeros(())
        else:
            self.aux_loss = None
        return out.view(b, t, h)


class Block(nn.Module):
    def __init__(self, config: HearthConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MoE(config) if config.use_moe else MLP(config)

    def forward(self, x, cos, sin, mask=None, cache=None):
        x = x + self.attn(self.input_layernorm(x), cos, sin, mask, cache, self.layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x