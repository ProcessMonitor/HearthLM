# HearthLM

A small decoder-only language model, plus everything needed to train, evaluate
and serve it, in about 5,700 lines of PyTorch you can read in an afternoon.

It is not a wrapper around `transformers`, `trl` or `peft`. There is no
`Trainer`, no `accelerate`, no config inheritance chain. The forward pass, the
loss functions, the KV cache, the LoRA implementation and the sampling loop are
all in this repository, and you can put a breakpoint in any of them.

The project is a from-scratch rebuild in the spirit of
[MiniMind](https://github.com/jingyaogong/minimind), which set out to make a
full LLM training pipeline reproducible on one consumer GPU. HearthLM keeps that
goal and changes the engineering underneath: the pieces that decide whether a
run converges, whether a mask is right, and whether you can tell the two apart
from the loss curve.

## What is different

The differences are not cosmetic; each one comes from a failure mode the
reference implementation hits at some point.

| | MiniMind | HearthLM |
|---|---|---|
| KV cache | list of tuples, `torch.cat` each step | `GrowCache` and `StaticCache`; slot-based, resettable, `crop()` for speculative decode |
| Attention mask | manual `repeat_kv`, one hand-rolled causal branch | SDPA with native GQA, offset-aware causal masks, block-diagonal packing masks |
| Pretraining batching | one document per sample, padded to length | token cache on disk, **sequence packing** with per-document attention isolation |
| SFT supervision | string-searches the token stream for a sentinel | computed from character offsets in the renderer, so it cannot silently drift |
| Training loop | nine near-identical scripts | one loop; accumulation, scheduler, EMA, eval, early stop, resumable state |
| Optimizer | plain AdamW on everything | decay split by parameter role, fp32 master weights under autocast |
| Serving | synchronous, one request | worker thread, admission control, per-request cache slots, SSE streaming |
| Device support | CUDA assumed | CUDA, MPS and CPU detected; dtype rules differ per backend |
| Tests | none | 39 tests covering the parts that fail quietly |

Two of those deserve emphasis because they are the ones that matter most for a
model this size.

**Sequence packing.** Padding every document to `seq_len` throws away 40-60% of
the compute on a corpus of mixed-length text. Packing concatenates documents
into full rows and sets a block-diagonal attention mask so token `i` in document
A cannot see document B. The test suite checks this directly: mutate document B,
assert document A's logits do not move.

**Offset-based supervision masks.** Tokenising `<|im_start|>assistant\n` and
searching for it is fragile — the tokenizer may split it, and then the mask
covers the wrong span with no error. HearthLM renders the conversation into
segments that carry a `trainable` flag, records character offsets, and maps each
segment to a token range through the tokenizer's own offset table.

## Results

Measured on one machine, an Apple-silicon laptop with 24 GB of unified memory.
The demo model is the `mini` preset at 24.1M parameters: pretrained on 139k
tokens for 1296 steps, then SFT'd for 40 steps on 2.7k conversations with a 5%
held-out split.

```
pretrain loss, final step            6.9  -> 0.28
sft eval loss, best checkpoint      0.127
ppl on held-out conversations       1.11
decode, 136-token prompt            110 tok/s cached / 15 tok/s full recompute
product-fact accuracy (15 probes)   10-15 / 15
FAQ accuracy (10 probes)            10/10
arithmetic (25 probes)              0/25
unit conversion (7 probes)          7/7
```

The three groups below differ in kind, not degree: one is what the pipeline
demonstrably teaches, one is a capacity limit at this size, and one is out of
reach entirely.

**What works.** The model copies procedural answers verbatim, 10 out of 10, and
gets every unit conversion asked of it. Format, turn structure, EOS placement
and system-prompt handling are all correct.

**What is marginal.** Binding an attribute to one of three near-identical product
names lands between 10/15 and 15/15 depending on the seed and step count. When it
fails it substitutes a sibling product, keeping the number attached to the wrong
name. The training data is balanced (215/201/200 rows per product, 72/72/72
answers), so this is a capacity limit, not a data bug: 24M parameters struggle to
keep three similar entities apart. More parameters or a longer schedule fixes it;
this project reports the number instead of picking the good seed.

**What does not work.** Arithmetic, 0 out of 25. The 569 arithmetic rows in the
training set are all correct (checked), so this is the model, not the data.
Single-digit addition has to be learned as an algorithm from examples, the
tokenizer emits digits one at a time, and 40 optimizer steps is nowhere near
enough. It needs a longer schedule, a tokenizer that keeps numbers whole, or a
bigger model. All three are real options; none is a code fix.

## Install

```
python -m pip install -r requirements.txt
```

Three dependencies: `torch`, `numpy`, `tokenizers`. Everything else is optional
and used by exactly one script (`safetensors` for export, `pytest` for tests,
`transformers` only if you want to verify the HF conversion).

Python 3.9 or newer. No GPU required; CPU training works, MPS is about 2x faster
for this size, CUDA is fastest.

## End to end from zero

Everything below runs offline. The demo dataset is generated rather than
downloaded, so the whole chain works without a network. On this laptop the
pretrain step takes about 20 minutes and SFT about 5; on a CUDA card both are
under a minute.

```bash
# 1. Build the demo data: a fictional countertop cooker manual with Q&A,
#    arithmetic, unit conversion, preference pairs and an MCQ set.
python scripts/generate_demo.py --out data/demo

# 2. Train the tokenizer on that text. Byte-level BPE, ~950 tokens.
python -m hearthlm.data.train_tokenizer \
    --data data/demo/pretrain.jsonl --out assets/tokenizer \
    --vocab-size 2048 --min-frequency 1

# 3. Tokenize the corpus once and memory-map it. Training then reads a flat
#    uint32 array instead of re-tokenizing in the dataloader.
#    pretrain_mix.jsonl is the manual plus the conversations in rendered form,
#    so the model sees the chat markers before SFT starts.
python -m hearthlm.data.build_corpus \
    --data data/demo/pretrain_mix.jsonl --out data/demo/corpus

# 4. Pretrain. `mini` is 24M parameters; about 20 minutes on a laptop CPU/GPU.
python -m hearthlm.train --stage pretrain \
    --data data/demo/corpus --preset mini --seq-len 256 \
    --batch-size 8 --accumulation 2 --lr 1e-3 --epochs 6 \
    --device auto --out out
```

Steps 5 and 6 differ by device: full finetuning on a small dataset overfits
within a few dozen steps, so the default is early stopping on a held-out slice.

```bash
# 5. Supervised fine-tuning.
python -m hearthlm.train --stage sft \
    --data data/demo/sft.jsonl --init out/checkpoints/pretrain.pt \
    --seq-len 256 --batch-size 24 --lr 1.2e-3 --epochs 4 \
    --val-ratio 0.05 --eval-every 20 --early-stop 5 --out out

# 6. Talk to it.
python -m hearthlm.chat --checkpoint out/checkpoints/sft_best.pt
```

Step 5 writes two checkpoints: `sft.pt` is the last step, `sft_best.pt` is the
weights from the best evaluation. On a small dataset the last step is usually the
worse one, so use `sft_best.pt` unless you have a reason not to.

## Training stages

Each stage is a flag on the same entry point. The shared loop handles
accumulation, learning-rate schedule, gradient clipping, evaluation, checkpoint
rotation and resumption; a stage only supplies a loss function and a dataset.

```bash
# Pretrain. Requires a corpus built by build_corpus.
python -m hearthlm.train --stage pretrain --data data/demo/corpus --preset mini

# Full-parameter SFT on conversations.
python -m hearthlm.train --stage sft --data data/sft.jsonl --init out/checkpoints/pretrain.pt

# LoRA. Trains adapters only; the base model is frozen.
python -m hearthlm.train --stage lora --data data/sft.jsonl \
    --init out/checkpoints/sft_best.pt --lora-rank 16 --lora-alpha 32

# Direct preference optimization against (chosen, rejected) pairs.
python -m hearthlm.train --stage dpo --data data/dpo.jsonl \
    --init out/checkpoints/sft_best.pt --beta 0.1 --lr 5e-7

# Knowledge distillation from a larger checkpoint in this same format.
python -m hearthlm.train --stage distill --data data/sft.jsonl \
    --init out/checkpoints/pretrain.pt --teacher out/checkpoints/sft_best.pt \
    --temperature 2.0 --kl-weight 0.5

# Group relative policy optimization. Rollouts are generated inline.
python -m hearthlm.train --stage grpo --data data/rl.jsonl \
    --init out/checkpoints/sft_best.pt --group-size 4 --beta 0.04
```

`--help` on any stage lists every knob. Model shape comes from `--preset`
(`nano`, `mini`, `base`, `moe`) plus `--set key=value` for anything else:

```bash
python -m hearthlm.train --stage pretrain --data corpus --preset mini \
    --set hidden_size=640 --set num_hidden_layers=14
```

Multi-GPU uses `torchrun`; the loop detects it and wraps in DDP:

```bash
torchrun --nproc_per_node 4 -m hearthlm.train --stage pretrain --data corpus --preset base
```

### What each stage needs

| Stage | Data | Needs `--init` | Notes |
|---|---|---|---|
| pretrain | token corpus directory | no | packs documents, block-diagonal mask |
| sft | `{"conversations": [...]}` | usually | masks loss to assistant turns |
| lora | same as sft | yes | rank/alpha, merges later with a script |
| dpo | `{"chosen": [...], "rejected": [...]}` | yes | length-normalised margins |
| distill | same as sft | yes | CE plus KL to `--teacher` |
| grpo | prompts, optional `answer` | yes | rule reward by default; see below |

## Presets

Parameter counts include an 8k vocabulary embedding.

| Preset | hidden | layers | heads / kv | experts | params | active |
|---|---|---|---|---|---|---|
| `nano` | 256 | 6 | 8 / 4 | – | 5.8M | 5.8M |
| `mini` | 512 | 8 | 8 / 4 | – | 27.8M | 27.8M |
| `base` | 768 | 12 | 12 / 4 | – | 81.8M | 81.8M |
| `moe` | 768 | 12 | 12 / 4 | 8, top-2 | 138.5M | 53.6M |

The architecture is Qwen3-shaped: pre-norm RMSNorm, grouped-query attention,
RoPE, SwiGLU, optional QK-norm, optional MoE with a top-k router. The `moe`
preset has no shared expert, matching Qwen3-MoE. That shape is not an accident —
it means checkpoints convert to the `transformers` Qwen3 classes with no
architectural translation, verified in `scripts/to_huggingface.py`.

## Inference

Terminal chat with history, streaming and a KV cache:

```bash
python -m hearthlm.chat --checkpoint out/checkpoints/sft_best.pt
```

Inside the prompt: `/reset`, `/system <text>`, `/thinking on|off`, `/tokens`,
`/quit`. `--thinking` asks the model to reason inside a think block first;
`--show-thinking` prints that block dimmed.

OpenAI-compatible HTTP server with a small built-in web page:

```bash
python -m hearthlm.serve --checkpoint out/checkpoints/sft_best.pt --port 8000
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 132 minus 13?"}],"max_tokens":32}'
```

Implements `POST /v1/chat/completions` (streaming and not), `GET /v1/models`,
`GET /health`, and a chat page at `/`. It is built on `http.server`, on purpose:
serving a 13M model does not justify FastAPI plus pydantic plus uvicorn. If you
want high concurrency, convert to HuggingFace format and run vLLM.

The server handles requests in a worker thread with admission control and one
preallocated cache slot per in-flight request. It does not batch forward passes
across requests; with ragged caches that needs a paged-attention kernel, which
is a different project.

## Evaluation

```bash
python -m hearthlm.eval ppl   --checkpoint out/checkpoints/sft_best.pt --data data/demo/sft.jsonl
python -m hearthlm.eval mcq   --checkpoint out/checkpoints/sft_best.pt --data data/demo/mcq.jsonl --verbose
python -m hearthlm.eval bench --checkpoint out/checkpoints/sft_best.pt --tokens 64
```

Perplexity says whether the model fits the distribution. MCQ accuracy says
whether it can discriminate between options, which is a much weaker requirement
than generating text, and reports chance rate next to the result. The bench
compares cached decode against full recompute, so you can see whether the cache
is earning its complexity. Sample MCQ output on a 2.6M model trained for 20
steps, which is honest about being undertrained:

```
mcq accuracy: 2/8 = 25.0% (chance = 25%)
```

## Export

Two formats, for two different reasons.

```bash
# Self-contained directory this repo can load back. Config, weights, tokenizer.
python -m hearthlm.export --checkpoint out/checkpoints/sft_best.pt --out export/hearthlm

# Qwen3-layout directory for transformers, vLLM, llama.cpp, ollama.
python scripts/to_huggingface.py --checkpoint out/checkpoints/sft_best.pt --out export/hf

# Fold a LoRA adapter into the base checkpoint.
python scripts/merge_lora.py --base out/checkpoints/sft_best.pt \
    --lora out/checkpoints/lora.pt --out out/checkpoints/merged.pt
```

The HuggingFace export is verified numerically, not just structurally: load the
result with `AutoModelForCausalLM`, run the same tokens through both
implementations, assert the logits match. Dense diff is `0.0`; MoE diff is
`2.4e-07`, which is fp32 rounding on the expert gather.

## Data formats

Pretraining text, one document per line:

```json
{"text": "The Hearth One draws 1200 W and holds 4.2 L."}
```

Conversations. `system` is optional; `tools` on a system message adds tool
definitions; `tool_calls` on an assistant message and `role: "tool"` results are
rendered into the tool-calling format the model trains on.

```json
{"conversations": [
  {"role": "system", "content": "Answer from the manual."},
  {"role": "user", "content": "How long is the warranty?"},
  {"role": "assistant", "content": "Two years from the date of purchase."}
]}
```

Preference pairs for DPO:

```json
{"chosen":   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "good"}],
 "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "bad"}]}
```

Cleaning a raw corpus before pretraining:

```bash
python scripts/dedup.py --data data/raw/*.jsonl --out data/clean.jsonl --threshold 0.9
```

Exact duplicates go first, then near-duplicates by MinHash Jaccard estimate.
Duplication hurts small models more than large ones: a 30M-token corpus with 20%
repeats effectively trains on 24M tokens with a skewed distribution, and the
model memorises the repeats instead of the structure.

## The code

```
hearthlm/
  config.py            HearthConfig, presets, validation
  model/
    layers.py          RMSNorm, RoPE, GQA attention, SwiGLU, MoE, Block
    cache.py           GrowCache, StaticCache
    hearth.py          HearthLM: embedding -> blocks -> norm -> head
    generate.py        sampling: top-k, top-p, min-p, penalties, decode loop
  data/
    tokenizer.py       tokenizer wrapper + from-scratch byte-level BPE trainer
    chat_template.py   segment renderer, produces supervision spans
    datasets.py        packing, SFT/DPO datasets, collators
    build_corpus.py    memory-mapped token cache
    train_tokenizer.py BPE training entry point
  train/
    runtime.py         device/dtype selection, autocast, logging, DDP
    optim.py           AdamW parameter groups, cosine schedule, checkpoints
    loop.py            the one training loop, budget sampler, EMA
    objectives.py      pretrain, sft, dpo, distill, grpo losses
    lora.py            LoRALinear, apply, merge, save, load
    cli.py             stage table, config file, stage builders
  chat/                terminal REPL with streaming
  serve/               OpenAI-compatible HTTP server, worker engine
  eval/                perplexity, MCQ accuracy, throughput
scripts/               demo data, dedup, merge, HF conversion
tests/                 37 correctness tests
docs/                  design notes
```

Run the tests:

```bash
python -m pytest tests -q
```

They are not a formality. Each one exists because a plausible implementation of
the same thing passes a smoke test and trains a subtly wrong model:

- changing a future token must not change earlier logits
- an all-ones attention mask must equal no mask
- cached decode must equal a full forward, token by token and in chunks
- a static cache must produce the same output after `reset()`
- documents packed into one sequence must not see each other
- the SFT mask must cover assistant turns and nothing else, across multi-turn data
- LoRA at init must be exactly the identity, and merge must match the adapter path
- the sampler must survive degenerate logits instead of emitting garbage
- the BPE tokenizer must round-trip Chinese, emoji and accented text byte for byte

## Limitations

Stated plainly, because the alternative is finding out the hard way.

- **Not competitive with a real model.** 13M parameters trained on 535
  conversations memorises answer shapes. It is a pipeline demonstration.
- **No production batching.** The server serialises forward passes. Real
  concurrency needs paged attention.
- **Reward shaping is a placeholder.** `--stage grpo` uses a rule reward based on
  length, repetition and optional substring match. Swap in a reward model by
  passing a different function to the step builder.
- **Token counts, not FLOPs.** The bench measures decode throughput. It does not
  model MFU, and the numbers will move a lot between backends.
- **`--stage distill` needs a teacher in this repo's format**, not an arbitrary
  HuggingFace checkpoint.
- **fp16 on MPS is off by default.** It is about 20% faster per step but
  overflows about a thousand steps into pretraining at a learning rate that is
  stable in fp32, and MPS has no working `GradScaler` to recover. `--dtype auto`
  picks fp32 there. Forcing `--dtype float16` still works and prints a warning.
- **A full-fp16 weight cast breaks AdamW.** `eps=1e-8` underflows to zero in
  fp16, so the first update sends most parameter tensors to NaN. Training keeps
  fp32 master weights and lets autocast handle the matmuls; only inference casts
  weights. Keep that separation if you modify the loop.

## Credits

The architecture is standard-issue modern LLM: pre-norm RMSNorm from GPT-3,
SwiGLU from PaLM, RoPE from RoFormer, grouped-query attention from Ainslie et
al., QK-norm and the Qwen3 configuration from the Qwen team, token-choice MoE
with a switch-style auxiliary loss from Switch Transformer. Chase's
[nanoGPT](https://github.com/karpathy/nanoGPT) for the argument that this is
worth writing by hand. MiniMind for the original small-model-from-scratch
pipeline this rebuilds.

License: Apache-2.0.
