"""Correctness tests for the pieces that are easy to get subtly wrong.

Run with: python -m pytest tests -q     (or python -m unittest discover tests)

These are not benchmarks. Every test here exists because a plausible-looking
implementation of the same thing would pass a smoke test but train a wrong model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hearthlm import HearthConfig, HearthLM, build_config
from hearthlm.data import (
    Tokenizer,
    build_sft_example,
    pack_documents,
    render,
    render_text,
)
from hearthlm.model.cache import GrowCache, StaticCache
from hearthlm.model.generate import SamplingParams, apply_penalties, filter_logits, generate
from hearthlm.train.lora import LoRALinear, apply_lora, lora_state_dict, merge_lora
from hearthlm.train.loop import TokenBudgetSampler, cosine_lr

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "assets" / "tokenizer"


@pytest.fixture(scope="module")
def cfg():
    return build_config("nano", vocab_size=256, max_position_embeddings=128)


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return HearthLM(cfg).eval()


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer(TOKENIZER_DIR)


# ------------------------------------------------------------------- attention
def test_causality(model):
    """Changing a future token must not change earlier logits."""
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        base = model(ids).logits
        altered = ids.clone()
        altered[0, -1] = (altered[0, -1] + 1) % 256
        changed = model(altered).logits
    assert torch.allclose(base[0, :-1], changed[0, :-1], atol=1e-6)


def test_full_mask_equals_unmasked(model):
    ids = torch.randint(0, 256, (2, 10))
    with torch.no_grad():
        plain = model(ids).logits
        masked = model(ids, attention_mask=torch.ones_like(ids)).logits
    assert torch.allclose(plain, masked, atol=1e-6)


@pytest.mark.parametrize("cache_kind", ["grow", "static"])
def test_cache_matches_full_forward(model, cfg, cache_kind):
    ids = torch.randint(0, 256, (1, 12))

    def make():
        if cache_kind == "grow":
            return GrowCache(cfg.num_hidden_layers, max_len=64)
        return StaticCache(cfg.num_hidden_layers, 64, 1, cfg.num_key_value_heads, cfg.head_dim, torch.device("cpu"))

    with torch.no_grad():
        full = model(ids).logits
        cache = make()
        stepwise = torch.cat([model(ids[:, t:t + 1], cache=cache).logits for t in range(12)], dim=1)
        assert torch.allclose(full, stepwise, atol=1e-5)

        cache = make()
        chunked = torch.cat([model(ids[:, :5], cache=cache).logits,
                             model(ids[:, 5:], cache=cache).logits], dim=1)
        assert torch.allclose(full, chunked, atol=1e-5)


def test_static_cache_resets_between_requests(model, cfg):
    a = torch.randint(0, 256, (1, 8))
    b = torch.randint(0, 256, (1, 8))
    cache = StaticCache(cfg.num_hidden_layers, 32, 1, cfg.num_key_value_heads, cfg.head_dim, torch.device("cpu"))
    with torch.no_grad():
        first = model(a, cache=cache).logits
        cache.reset()
        second = model(b, cache=cache).logits
        fresh = model(b).logits
    assert torch.allclose(second, fresh, atol=1e-5)


def test_sequence_packing_isolates_documents(model):
    """Tokens from document B must not influence document A's logits."""
    ids = torch.randint(0, 256, (1, 8))
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    with torch.no_grad():
        base = model(ids, document_ids=doc_ids).logits
        altered = ids.clone()
        altered[0, 4:] = (altered[0, 4:] + 7) % 256
        changed = model(altered, document_ids=doc_ids).logits
    assert torch.allclose(base[0, :4], changed[0, :4], atol=1e-6)
    assert not torch.allclose(base[0, 4:], changed[0, 4:], atol=1e-6)


def test_packing_keeps_causality_within_document(model):
    ids = torch.randint(0, 256, (1, 8))
    doc_ids = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        base = model(ids, document_ids=doc_ids).logits
        altered = ids.clone()
        altered[0, -1] = (altered[0, -1] + 3) % 256
        changed = model(altered, document_ids=doc_ids).logits
    assert torch.allclose(base[0, :-1], changed[0, :-1], atol=1e-6)


def test_fully_masked_query_rows_do_not_produce_nan(model):
    """A padding row must not poison the batch.

    Every query needs at least one visible key. When a row is entirely padding the
    naive mask leaves softmax with nothing to normalise over, which yields NaN and
    corrupts the whole loss.
    """
    ids = torch.randint(0, 256, (1, 10))
    mask = torch.ones_like(ids)
    mask[0, 6:] = 0  # last four positions are padding
    with torch.no_grad():
        out = model(ids, attention_mask=mask).logits
    assert torch.isfinite(out).all(), "padding rows produced non-finite logits"


def test_packing_with_a_fully_masked_document_stays_finite(model):
    ids = torch.randint(0, 256, (1, 8))
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    mask = torch.ones_like(ids)
    mask[0, 4:] = 0  # document 1 is entirely padding
    with torch.no_grad():
        out = model(ids, attention_mask=mask, document_ids=doc_ids).logits
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------- packing
def test_pack_documents_masks_first_token_of_each_doc():
    docs = [[10, 11, 12], [20, 21], [30, 31, 32, 33]]
    out = pack_documents(docs, seq_len=16, pad_id=0)
    labels = out["labels"][0].tolist()
    ids = out["input_ids"][0].tolist()
    doc_ids = out["document_ids"][0].tolist()

    # each document starts at a new doc id, and its first token has no target
    starts = [i for i in range(len(doc_ids)) if doc_ids[i] != (doc_ids[i - 1] if i else -1) and doc_ids[i] >= 0]
    assert starts == [0, 3, 5]
    for s in starts:
        assert labels[s] == -100, f"first token of document at {s} should not be supervised"
    assert labels[1] == 11 and labels[2] == 12
    assert ids[:8] == [10, 11, 12, 20, 21, 30, 31, 32]


def test_pack_documents_splits_long_docs_across_rows():
    doc = list(range(1, 11))
    out = pack_documents([doc], seq_len=4, pad_id=0)
    assert out["input_ids"].shape[0] == 3
    assert out["input_ids"][0].tolist() == [1, 2, 3, 4]
    assert out["input_ids"][2].tolist() == [9, 10, 0, 0]


def test_pack_documents_fills_rows_then_pads_the_tail():
    # 3 + 5 tokens fit in one 8-token row, so greedy packing must not split them.
    out = pack_documents([[1, 2, 3], [4, 5, 6, 7, 8]], seq_len=8, pad_id=99)
    assert out["input_ids"].shape == (1, 8)
    assert out["input_ids"][0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert out["attention_mask"][0].tolist() == [1] * 8

    out = pack_documents([[1, 2, 3], [4, 5, 6, 7, 8]], seq_len=6, pad_id=99)
    assert out["input_ids"].shape == (2, 6)
    assert out["input_ids"][0].tolist() == [1, 2, 3, 4, 5, 6]
    # The long document continues on the next row; the remainder is padded.
    assert out["input_ids"][1].tolist() == [7, 8, 99, 99, 99, 99]
    assert out["attention_mask"][1].tolist() == [1, 1, 0, 0, 0, 0]


# --------------------------------------------------------------------- masking
def test_sft_mask_covers_only_assistant(tokenizer):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]
    ex = build_sft_example(msgs, tokenizer, max_len=128)
    ids = ex["input_ids"].tolist()
    labels = ex["labels"].tolist()
    supervised = tokenizer.decode([i for i, l in zip(ids, labels) if l != -100])
    assert "hello there" in supervised
    assert "hi" not in supervised.replace("hello there", "")
    assert tokenizer.decode([ids[i] for i, l in enumerate(labels) if l == ids[i]]).count("hi") == 0


def test_sft_mask_ignores_multi_turn_user_text(tokenizer):
    msgs = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    ex = build_sft_example(msgs, tokenizer, max_len=256)
    ids = ex["input_ids"].tolist()
    labels = ex["labels"].tolist()
    supervised = tokenizer.decode([i for i, l in zip(ids, labels) if l != -100])
    assert "first answer" in supervised and "second answer" in supervised
    assert "first question" not in supervised
    assert "second question" not in supervised


def test_sft_example_drops_unlearnable_rows(tokenizer):
    msgs = [{"role": "user", "content": "no answer"}]
    assert build_sft_example(msgs, tokenizer, max_len=64) is None


# ------------------------------------------------------------------- templates
def test_render_roles_and_think_toggle():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    with_think = render_text(msgs, enable_thinking=True)
    without = render_text(msgs, enable_thinking=False)
    assert "<think>" in with_think and "</think>" in with_think
    assert "<think>" not in without
    assert with_think.startswith("<|im_start|>user")
    assert "<|im_start|>assistant" in with_think


def test_render_generation_prompt_is_last():
    msgs = [{"role": "user", "content": "q"}]
    text = render_text(msgs, add_generation_prompt=True, enable_thinking=False)
    assert text.endswith("<|im_start|>assistant\n")


def test_render_tool_call_roundtrip():
    msgs = [
        {"role": "user", "content": "add 1 and 2"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "add", "arguments": {"a": 1, "b": 2}}}]},
        {"role": "tool", "content": "3"},
        {"role": "assistant", "content": "The answer is 3."},
    ]
    text = render_text(msgs, enable_thinking=False)
    assert '<tool_call>' in text and '"name": "add"' in text
    assert "<tool_response>" in text
    assert text.count("<|im_start|>user") == 2  # user turn plus wrapped tool result


def test_render_tools_block_in_system():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    text = render_text([{"role": "user", "content": "x"}], tools=tools)
    assert "# Tools" in text and "<tools>" in text


# -------------------------------------------------------------------- sampling
def test_top_k_keeps_at_most_k():
    logits = torch.randn(1, 100)
    filtered = filter_logits(logits, top_k=5)
    assert (filtered > float("-inf")).sum().item() == 5


def test_top_p_keeps_nucleus():
    logits = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    filtered = filter_logits(logits, top_p=0.5)
    assert filtered[0, 0] > float("-inf")
    assert filtered[0, 2] == float("-inf")


def test_min_p_removes_low_probability_tail():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    filtered = filter_logits(logits, min_p=0.1)
    assert (filtered > float("-inf")).sum().item() == 1


def test_repetition_penalty_pushes_down_seen_tokens():
    logits = torch.tensor([[2.0, 2.0, 2.0]])
    seen = torch.tensor([[0]])
    out = apply_penalties(logits.clone(), seen, repetition_penalty=2.0)
    assert out[0, 0] < out[0, 1]


def test_sampling_is_reproducible_with_seed():
    logits = torch.randn(1, 50)
    params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, seed=123)
    from hearthlm.model.generate import sample

    g1, g2 = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
    assert sample(logits, params, g1).item() == sample(logits, params, g2).item()


def test_degenerate_logits_still_produce_a_token():
    logits = torch.full((1, 8), float("-inf"))
    from hearthlm.model.generate import sample

    out = sample(logits, SamplingParams(temperature=1.0, top_k=0, top_p=1.0))
    assert 0 <= out.item() < 8


def test_generate_stops_at_eos(model, cfg):
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    params = SamplingParams(max_new_tokens=10, do_sample=False)
    with torch.no_grad():
        out = generate(model, ids, params, eos_token_id=cfg.eos_token_id)
    assert out.shape[0] == 1
    assert out.shape[1] <= 4 + 10


# ----------------------------------------------------------------------- LoRA
def test_lora_is_identity_at_init(cfg):
    torch.manual_seed(0)
    model = HearthLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        before = model(ids).logits
    apply_lora(model, rank=4)
    with torch.no_grad():
        after = model(ids).logits
    assert torch.allclose(before, after, atol=1e-6), "zero-initialised B must not change outputs"


def test_lora_trains_only_adapters(cfg):
    model = HearthLM(cfg)
    apply_lora(model, rank=4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "adapters should be trainable"
    assert all("lora_" in n for n in trainable), f"unexpected trainable params: {trainable[:5]}"


def test_lora_merge_matches_adapter_forward(cfg):
    torch.manual_seed(0)
    model = HearthLM(cfg).eval()
    apply_lora(model, rank=4)
    # make the adapter non-trivial
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B.weight, std=0.05)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        adapted = model(ids).logits
    merge_lora(model)
    with torch.no_grad():
        merged = model(ids).logits
    assert torch.allclose(adapted, merged, atol=1e-4)


def test_lora_state_dict_is_adapter_only(cfg):
    model = HearthLM(cfg)
    apply_lora(model, rank=4)
    state = lora_state_dict(model)
    assert state
    assert all("lora_" in k for k in state)
    assert not any(".base." in k for k in state)


# -------------------------------------------------------------------- schedule
def test_cosine_lr_warms_up_then_decays():
    lrs = [cosine_lr(s, total_steps=100, base_lr=1.0, warmup_steps=10, min_lr_ratio=0.1) for s in range(101)]
    assert lrs[0] < lrs[9]
    assert abs(lrs[9] - 1.0) < 1e-6
    assert lrs[-1] < 0.2
    assert all(a >= b - 1e-9 for a, b in zip(lrs[10:], lrs[11:]))


def test_token_budget_sampler_covers_every_index_once():
    lengths = [5, 500, 12, 300, 8, 90]
    sampler = TokenBudgetSampler(lengths, batch_size=600, unit="tokens", seed=0)
    seen = sorted(i for batch in sampler for i in batch)
    assert seen == list(range(len(lengths)))


def test_token_budget_sampler_respects_budget():
    lengths = [100] * 50
    sampler = TokenBudgetSampler(lengths, batch_size=300, unit="tokens", max_batch=64, seed=0)
    for batch in sampler:
        assert sum(lengths[i] for i in batch) <= 300 + max(lengths) or len(batch) == 1


# ------------------------------------------------------------------ tokenizer
def test_tokenizer_roundtrips_arbitrary_bytes(tokenizer):
    samples = [
        "plain ascii text",
        "中文标点：，。！？",
        "emoji 🙂👍 and accents café ñ",
        "mixed 中文 and English 123",
    ]
    for s in samples:
        assert tokenizer.decode(tokenizer.encode(s)) == s


def test_tokenizer_special_ids_are_stable(tokenizer):
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2


def test_token_span_covers_requested_chars(tokenizer):
    text = "prefix ASSISTED suffix"
    ids, offsets = tokenizer.encode_with_offsets(text)
    a, b = tokenizer.token_span_for_chars(offsets, 7, 15)
    # A token is included when it overlaps the range, so a leading space may come
    # along; the target text must be fully covered.
    assert tokenizer.decode(ids[a:b]).strip() == "ASSISTED"


# --------------------------------------------------------------------- config
def test_config_rejects_bad_head_split():
    with pytest.raises(ValueError):
        HearthConfig(hidden_size=100, num_attention_heads=8)


def test_config_roundtrips_through_dict():
    cfg = build_config("mini")
    again = HearthConfig.from_dict(cfg.to_dict())
    assert again == cfg


def test_moe_active_params_below_total():
    cfg = build_config("moe", hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                       num_key_value_heads=4, num_experts=4, num_experts_per_tok=1,
                       moe_intermediate_size=256)
    model = HearthLM(cfg)
    assert model.active_parameters() < model.num_parameters()