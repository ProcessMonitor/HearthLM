"""The model: embedding -> N blocks -> norm -> tied LM head.

`forward` accepts either plain ids or a batch of packed documents via
`document_ids`, so a single implementation serves pretraining and chat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..config import HearthConfig
from .cache import KVCache
from .layers import Block, MLP, MoE, RMSNorm, build_rope_cache


@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


class HearthModel(nn.Module):
    def __init__(self, config: HearthConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(Block(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        cos, sin = build_rope_cache(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling,
        )
        # persistent=False: the table is derived from config, so it must not be
        # written into checkpoints (it would triple the file size for nothing).
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._rope_len = config.max_position_embeddings

    # ------------------------------------------------------------------ rope
    def _rope_table(self, need: int, device: torch.device, dtype: torch.dtype):
        """Return cos/sin sliced to `need` positions, growing the table if asked.

        Growth only happens when a request exceeds max_position_embeddings; the
        new table is built with the same (possibly YaRN-scaled) frequencies.
        """
        if need > self._rope_len:
            cos, sin = build_rope_cache(
                self.config.head_dim, need, self.config.rope_theta, self.config.rope_scaling, device=device
            )
            self.rope_cos, self.rope_sin = cos, sin
            self._rope_len = need
        return self.rope_cos[:need].to(dtype), self.rope_sin[:need].to(dtype)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """Return hidden states (batch, seq, hidden).

        attention_mask : (batch, seq) or (batch, past+seq) bool/int, 1 = keep.
        document_ids   : (batch, seq) int, equal ids may attend to each other.
                         Used for sequence packing; combined with the padding
                         mask and causality to build a block-diagonal mask.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        x = self.dropout(self.embed_tokens(input_ids))

        past_len = cache.length if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0)
            if bsz > 1:
                position_ids = position_ids.expand(bsz, -1)
        else:
            if position_ids.shape[0] == 1 and bsz > 1:
                position_ids = position_ids.expand(bsz, -1)
            position_ids = position_ids + past_len  # cache offsets absolute positions
        max_pos = int(position_ids.max().item()) + 1

        cos_tab, sin_tab = self._rope_table(max_pos, device, x.dtype)
        # (batch, 1, seq, dim) so it broadcasts over heads
        cos = cos_tab[position_ids].unsqueeze(1)
        sin = sin_tab[position_ids].unsqueeze(1)

        mask = self._build_mask(attention_mask, document_ids, past_len, seq_len, device)

        for layer in self.layers:
            x = layer(x, cos, sin, mask, cache)
        return self.norm(x)

    def _build_mask(self, attention_mask, document_ids, past_len, seq_len, device):
        """Return a key-side mask (b, full_len) or a full (b, seq, full_len) mask.

        `Attention` normalises either layout into what SDPA wants.
        """
        # ---- padding / key-validity mask ---------------------------------
        if attention_mask is not None:
            am = attention_mask.to(device)
            if am.dtype != torch.bool:
                am = am.bool()
            if am.dim() == 1:
                am = am.unsqueeze(0)
            if am.shape[1] == seq_len and past_len:
                # Caller passed masks for the new tokens only; cached tokens are valid.
                am = F.pad(am, (past_len, 0), value=True)
            elif am.shape[1] != past_len + seq_len:
                raise ValueError(
                    f"attention_mask width {am.shape[1]} does not match {seq_len} new "
                    f"tokens plus {past_len} cached tokens"
                )
            if am.all():
                # An all-valid mask carries no information. Dropping it lets the
                # attention path use SDPA's causal flag instead of materialising a
                # (batch, seq, full) boolean each step, which dominates decode.
                am = None
        else:
            am = None

        if document_ids is None:
            return am

        # ---- packed documents: block diagonal ---------------------------
        docs = document_ids
        if docs.shape[1] != seq_len:
            raise ValueError("document_ids must cover the tokens in this forward call")
        if past_len:
            # With a cache a single document streams forward; its cached tokens
            # belong to the same document.
            docs = F.pad(docs, (past_len, 0), value=int(docs[0, 0]))
            docs[:, :past_len] = docs[:, past_len:past_len + 1]

        same_doc = docs.unsqueeze(2) == docs.unsqueeze(1)                     # (b, seq, full)
        causal = torch.ones(seq_len, past_len + seq_len, dtype=torch.bool, device=device).tril(
            diagonal=past_len
        )
        mask = same_doc & causal.unsqueeze(0)
        if am is not None:
            mask = mask & am.unsqueeze(1)
        return mask


class HearthLM(nn.Module):
    """Causal LM head on top of `HearthModel`."""

    def __init__(self, config: HearthConfig):
        super().__init__()
        self.config = config
        self.model = HearthModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 style); keeps deep stacks stable.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_hidden_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------- parameter
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.model.embed_tokens.weight.numel()
        return total

    def active_parameters(self) -> int:
        cfg = self.config
        if not cfg.use_moe:
            return self.num_parameters()
        embedding = cfg.vocab_size * cfg.hidden_size
        per_layer = sum(
            p.numel()
            for n, p in self.named_parameters()
            if n.startswith("model.layers.0.") and ".experts." not in n
        )
        one_expert = sum(
            p.numel()
            for n, p in self.named_parameters()
            if n.startswith("model.layers.0.mlp.experts.0.")
        )
        return embedding + cfg.num_hidden_layers * (per_layer + one_expert * cfg.num_experts_per_tok)

    # ---------------------------------------------------------------- checks
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        logits_to_keep: int = 0,
    ) -> ModelOutput:
        hidden = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
            cache=cache,
        )
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        logits = self.lm_head(hidden)

        aux = self._aux_loss()

        loss = None
        if labels is not None:
            # hidden was sliced only if logits_to_keep>0, in which case labels
            # must cover the same tail. Pretraining always passes full labels.
            if logits_to_keep:
                labels = labels[:, -logits_to_keep:]
            loss = self._loss(logits, labels, loss_mask)

        return ModelOutput(logits=logits, loss=loss, aux_loss=aux, hidden_states=hidden)

    def _aux_loss(self) -> Optional[torch.Tensor]:
        if not self.config.use_moe:
            return None
        losses = [l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MoE) and l.mlp.aux_loss is not None]
        if not losses:
            return None
        return torch.stack(losses).sum()

    @staticmethod
    def _loss(logits: torch.Tensor, labels: torch.Tensor, loss_mask: Optional[torch.Tensor]) -> torch.Tensor:
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
        shift_labels = labels[:, 1:].reshape(-1)
        per_token = F.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100, reduction="none"
        )
        valid = shift_labels.ne(-100)
        if loss_mask is not None:
            valid = valid & loss_mask[:, 1:].reshape(-1).bool()
        denom = valid.sum().clamp_min(1)
        return (per_token * valid).sum() / denom

    # -------------------------------------------------------------- helpers
    @torch.inference_mode()
    def hidden_states(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input_ids, **kwargs)


def _state_dict_no_rope(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if "rope_" not in k}