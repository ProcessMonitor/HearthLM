# Design notes

Decisions that took more than one attempt, and what the alternative cost.

## Attention mask shape

SDPA accepts either `is_causal=True` or an explicit `attn_mask` tensor, and the
two are not interchangeable once there is a prefix.

`is_causal=True` builds the triangle aligned to the **top-left** of the
`(query, key)` block. That is correct only when the key block starts at position
0. With a KV cache, or with a second prefill chunk, the correct triangle is
offset: query at absolute position `p` may attend to keys `0..p`, which is a
shifted triangle. Passing `is_causal=True` with `past_len > 0` silently drops the
first `past_len` keys from every query's view.

So the rule in `layers.py` is:

- no cache, no window, no mask  -> `is_causal=True` (fastest path)
- cache or chunk, no mask       -> explicit offset triangle
- any caller mask               -> fold the triangle in explicitly

The last point matters. A caller-supplied mask is a *key-validity* mask
(`(batch, seq)`, 1 = real token, 0 = padding). Applying it without causality lets
every query see every valid key including future ones. The first version of this
code did exactly that, and the test `test_full_mask_equals_unmasked` caught it
because an all-ones mask was producing different logits from no mask at all.

One extra wrinkle: `attention_mask` arrives either as `(batch, seq_len)` for the
new tokens or `(batch, past+seq_len)` covering the cache. The `hearth.py` layer
normalises to the second form, and drops it entirely when all values are 1, so
decode does not pay to build a `(batch, seq, full)` boolean each step. That
single change moved end-to-end generate from 4 tok/s to 25 tok/s on MPS.

## KV cache: two implementations

`GrowCache` concatenates. `StaticCache` preallocates.

The reason both exist is that they fail in opposite directions. `GrowCache` in
an engine that handles many concurrent requests causes allocator churn: every
step allocates a slightly larger tensor for every layer, the old one is freed,
and on CUDA that fragments the caching allocator into something unusable within
a thousand steps. `StaticCache` fixes that by allocating `(layers, batch, heads,
max_len, dim)` once and writing through a cursor, but it caps the sequence length
up front and wastes memory on short requests.

A single implementation would have been simpler and would have made one of the
two paths bad. The interface is five methods, so the duplication is small.

`StaticCache` has a bug worth documenting because it is easy to reintroduce: the
write cursor must be latched by the first layer of a forward call, not recomputed
per layer. Attention layers run in sequence, so `self.length` advances once per
step — if each layer reads it independently, layer 1 writes at offset 0+`seq_len`
in addition to layer 0's write, and the cache is corrupted. Caching
`_write_start` on `layer_idx == 0` fixes it. `test_cache_matches_full_forward`
runs both cache kinds against a full forward and would catch the regression.

## Position of the RoPE rotation

Rotate keys **before** inserting them into the cache.

The natural reading of the code is "project, reshape, cache, then rotate" —
which rotates the new keys but leaves previously-cached keys already rotated, so
they appear in the attention input rotated once. That is correct. The failure
mode is the other order: cache first, then rotate the full key tensor, which
rotates old keys a second time on every step. It does not crash. It produces
logits that drift with sequence position, which looks like a modelling problem
and trains a model that only works at short context.

The parity test caught this on the first run: max diff 0.34 against a full
forward, versus 8.9e-07 after the fix.

## Sequence packing

Padding to a fixed length wastes compute proportional to the padding. On mixed-
length text this is 40-60%. Packing fixes it and introduces the risk of
cross-document attention, which is much worse than wasting compute: the model
learns that tokens from unrelated documents predict each other, and quality
degrades in a way no metric flags.

The mask is built from a `document_ids` tensor of the same shape as the input.
Two tokens attend to each other iff they have the same document id AND the
causal condition holds AND neither is padding. That is a `(batch, seq, full)`
boolean, which is O(seq^2) memory — acceptable at these sequence lengths
(1024/2048) and unnecessary for unpacked data, so it is only built when
`document_ids` is passed.

The label for the first token of each document is set to -100. Its previous
token belongs to a different document, so predicting it is not a meaningful
target; leaving it in teaches the model to predict the edge between unrelated
texts.

Tests: `test_sequence_packing_isolates_documents` mutates the second document and
asserts the first document's logits are bit-identical.
`test_packing_keeps_causality_within_document` mutates the last token of a
single-document packed row and asserts earlier logits do not change. Both
directions are checked because a broken mask can fail either way — too permissive
(leaks across documents) or too restrictive (breaks within one).

## Supervision masks from offsets

The reference approach tokenizes the conversation, then searches the token
stream for the token sequence that `"<|im_start|>assistant\n"` produces, and
marks everything between those markers as trainable. Three things go wrong with
that:

1. The tokenizer may not produce that exact sequence. Byte-level BPE splits
   differently depending on what precedes it.
2. If the search fails, it fails silently — labels stay -100 and the row
   contributes nothing, or worse, the span shifts by one token and the model
   learns to predict the role marker instead of the answer.
3. Multi-turn conversations need to find every occurrence, and the loop is easy
   to get wrong at the boundary.

HearthLM renders into `Segment(text, trainable)` pieces and concatenates them.
Encoding gives `(ids, offsets)` where offsets are character positions in the
concatenated text. Each segment knows its character range, so the trainable flag
maps to a token range through the offset table. No sentinel search, no
dependence on tokenizer internals beyond "offsets are truthful", which is a
property `test_token_span_covers_requested_chars` checks.

A detail that came out of testing: EOS should only be a training target when the
conversation ends on an assistant turn. Appending it unconditionally means a
user-only row teaches the model to emit EOS immediately after a user message,
which is the exact behaviour that makes a tiny model useless.

## Optimizer

AdamW with weight decay 0.0 on norms, biases, embeddings and the LM head; 0.1
elsewhere. The reason to skip embeddings is that decaying an embedding row
shrinks rarely-seen tokens toward zero, and at a 1-8k vocabulary every token is
seen often enough that shrinkage is a real loss of capacity rather than
regularisation.

Under autocast, master weights stay fp32 and only the matmuls run in low
precision. This is not optional. Casting the model weights themselves to fp16
and then using AdamW breaks immediately: AdamW's `eps=1e-8` is below the fp16
minimum normal (6e-5), underflows to zero, and the update becomes
`lr * m / sqrt(v)` with no damping — the first step sends 90 of 91 parameter
tensors to NaN. The symptom is `non-finite loss` on batch 2, after a clean batch
1, which points at the data rather than the optimizer if you are not expecting it.

The rule enforced in `loop.py`: `model.to(device)` only. `dtype` goes to
`autocast` and to `GradScaler`, never to `.to(dtype)`.

## Data transfer

`non_blocking=True` on `.to(device)` is only useful when the source is pinned
host memory. On MPS and CPU the async path can complete before the copy
finishes, and on MPS in particular it produced garbage labels — `chosen_labels`
containing values around 7.5e18 for a 949-token vocabulary. That surfaced as a
NaN loss with finite logits, which is a miserable thing to debug.

`_to_device` now passes `non_blocking` only on CUDA, where the dataloader pins
memory.

## Checkpoints

Two files, not one.

`<tag>.pt` holds only weights plus the config. Loading a model for inference
never reads the optimizer state, which is 3-4x larger (two fp32 moment
estimates). `<tag>_resume.pt` holds optimizer, scaler, step and epoch for
resuming.

`<tag>_best.pt` holds the weights from the best evaluation, kept separately from
`<tag>.pt`. Before this split, the final save overwrote `sft.pt` with the
last-step weights, and on an overfitting dataset the last step is the worst one.
The README walkthrough uses `sft_best.pt` for that reason.

RoPE tables are registered with `persistent=False`. They are derived from
config; storing them inflates every checkpoint by the table size and creates a
class of bug where a checkpoint's table silently disagrees with its config.

## Serving

`http.server`, one worker thread, admission control up to `max_batch`, one
`StaticCache` slot per in-flight request.

Not FastAPI, because adding FastAPI plus pydantic plus uvicorn to serve a 13M
model is more unread code than the model itself. Not batched across requests,
because that needs paged attention to handle ragged caches; doing it without one
means padding caches to the longest active request, which wastes more than it
saves at these sizes.

Streaming a byte-level BPE token stream needs care. A single token can decode to
half a UTF-8 character, which renders as U+FFFD. The request holds back text
ending in U+FFFD until the next token completes the character, rather than
sending a replacement glyph the client cannot retract.

SSE has no content length, so the connection delimits the body. Answering with
`Connection: keep-alive` and not closing means every client hangs until its own
timeout. `Connection: close` plus `close_connection = True` after `[DONE]`.

## MoE dispatch

Token-choice top-k, one loop over experts, gather/GEMM/scatter per expert.

A per-expert Python loop is not fast. It is deterministic, portable, and
gradient-identical to the fused versions, and at 8 experts/top-2 the overhead is
visible but tolerable. The alternative — Triton kernels, `DeepSpeed-MoE` — is a
real project and would put the dispatch outside the "read every line" property
the rest of the repo is built on.

Two details that matter for correctness:

- When no token picks an expert, its parameters must still be in the graph in
  training, or DDP stalls waiting on a gradient that never arrives. The
  `out[0,0] += 0 * sum(p.sum())` line is for that, not stealth.
- The router aux loss is the switch-transformer form: `(fraction of tokens
  routed) * (mean routing probability)` summed over experts, scaled by
  `num_experts * coef`. Skipping the `num_experts` factor makes the loss vanish
  as expert count grows.

## Fully masked rows produce NaN, not zeros

If every key in a query row is masked, softmax has nothing to normalise over and
SDPA returns NaN for that row. One NaN row poisons the whole batch loss, the
backward pass writes NaN into every parameter with nonzero gradient, and training
is dead from that step on. It is not an exotic input: any padded position is
exactly this case, because the padding mask masks the key for every query at that
position.

The fix is `_ensure_self_attention`: after the mask is assembled, un-mask the
diagonal for any row left with no visible key. Those positions carry
`label = -100`, so the extra attention cannot affect the loss; it only keeps the
tensor finite.

This also explains an fp16 mystery. At `lr=1.5e-3` on MPS roughly 2500 of 3500
steps were skipped for non-finite loss; at `lr=5e-4`, 8 of 3448. The lower rate
happened to keep logits small enough that the degenerate softmax stayed finite.
The NaN was always there in the mask, and the learning rate only decided how
often it surfaced. Fixing the mask removes it at any rate.

## Tests that earned their place

Every test below failed at least once during development, or would have caught a
bug that did.

- `test_cache_matches_full_forward` — caught the rotate-after-cache bug (0.34 diff)
- `test_static_cache_resets_between_requests` — caught the write-cursor bug
- `test_full_mask_equals_unmasked` — caught the missing causal fold (1.3 diff)
- `test_sequence_packing_isolates_documents` — would catch a pack-mask leak
- `test_sft_example_drops_unlearnable_rows` — caught unconditional EOS labelling
- `test_lora_is_identity_at_init` — catches a non-zero B initialisation
- `test_lora_merge_matches_adapter_forward` — catches merge scaling errors
- `test_degenerate_logits_still_produce_a_token` — catches a NaN softmax emitting
  token 0 forever
- `test_tokenizer_roundtrips_arbitrary_bytes` — caught a vocab that only seeded
  byte symbols from the training corpus, so Chinese vanished on encode
- `test_fully_masked_query_rows_do_not_produce_nan` — catches the padded-row NaN
  that silently killed long fp16 runs

The last one is worth spelling out because the symptom was bizarre: the
tokenizer trained, round-tripped its own corpus perfectly, and silently dropped
every byte that never appeared in that corpus. Byte-level BPE must always seed
all 256 byte symbols. There is no such thing as an unseen byte.
