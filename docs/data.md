# Data

Three formats, one per stage family. Every reader is in `hearthlm/data/datasets.py`;
this file is the contract.

## pretrain.jsonl

One document per line, `{"text": "..."}`. Also accepts `.txt` and `.md`, or a
`--text-key` if your field is named something else.

```json
{"text": "The Hearth One draws 1200 W and holds 4.2 L."}
{"text": "Brown the aromatics before adding liquid; it takes four extra minutes."}
```

`build_corpus` wraps each document in BOS/EOS, encodes it, and appends to a flat
`uint32` file with an offsets array. Documents are **not** truncated: a long one
is split into `seq_len` chunks at pack time, with the chunk offset derived from
the document index so it is stable across epochs.

Rules that matter:

- Strip boilerplate (license headers, nav text) before this stage. A small model
  will learn it as content.
- Don't pad. Padding happens at pack time and is cheap; padding in the corpus
  file wastes disk and confuses dedup.
- Deduplicate first (`scripts/dedup.py`). Repeats are memorised, not learned.

## sft.jsonl

```json
{"conversations": [
  {"role": "user", "content": "What is the capacity of the Hearth Pro?"},
  {"role": "assistant", "content": "6.5 L."}
]}
```

### Optional fields

**System message.** First in the list, rendered into its own turn:

```json
{"conversations": [
  {"role": "system", "content": "Answer from the manual. Be brief."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

**Reasoning.** Either a separate field or inline in `<think>` tags inside
`content`. The renderer accepts both.

```json
{"conversations": [
  {"role": "user", "content": "What is 17 times 23?"},
  {"role": "assistant",
   "reasoning_content": "17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391",
   "content": "391"}
]}
```

With `--thinking` the stage trains on the reasoning too. Without it, the think
block is written empty, which teaches direct answers and makes inference faster.

**Tools.** Attach definitions to the system message, and use `tool_calls` plus a
`role: "tool"` result:

```json
{"conversations": [
  {"role": "system", "content": "You can convert units.", "tools": [
    {"type": "function", "function": {"name": "convert", "description": "Convert cups to millilitres",
     "parameters": {"type": "object", "properties": {"cups": {"type": "number"}}}}}
  ]},
  {"role": "user", "content": "How many millilitres in 3 cups?"},
  {"role": "assistant", "content": "", "tool_calls": [
    {"function": {"name": "convert", "arguments": {"cups": 3}}}
  ]},
  {"role": "tool", "content": "{\"ml\": 709.8}"},
  {"role": "assistant", "content": "About 710 millilitres."}
]}
```

`tools` may be a list or a JSON string. Multiple general `role: "tool"` messages
in a row are wrapped into one user turn by the renderer, matching the Qwen
convention.

### What gets trained

Only assistant spans contribute to the loss. User turns, system prompts, tool
definitions and tool results are context. This is enforced by character offsets,
not by string matching, so it holds no matter how the tokenizer splits things.

A row whose final turn is not from the assistant contributes nothing and is
dropped. A row with any assistant content is kept, even if only the thinking
block is supervised.

## dpo.jsonl

```json
{"chosen":   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "preferred"}],
 "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "less preferred"}]}
```

Both sides are rendered and masked the same way as SFT. The loss is
`-logsigmoid(beta * (mean_logp_chosen - mean_logp_rejected))` with optional
reference model. Length normalisation is on by default; without it, longer
rejections dominate the margin for reasons unrelated to quality.

Pairs should differ in one dimension. A pair where the rejected answer is both
wrong and rude teaches the model to conflate the two.

## rl.jsonl

```json
{"conversations": [{"role": "user", "content": "What is 132 minus 13?"},
                   {"role": "assistant", "content": ""}],
 "answer": "119"}
```

The last assistant turn is the generation slot. `answer` is optional and used by
the default rule reward for a substring check. If your prompts are raw strings
rather than conversations, a `{"prompt": "...", "answer": "..."}` row works too.

The default reward, in `hearthlm/train/cli.py::_rule_reward`, scores length
sanity, penalises degenerate repetition, penalises multiple think blocks, and
checks the reference answer. It is a placeholder. Pass a different function to
`objectives.grpo_step` for a real reward model.

## mcq.jsonl

Only used by `hearthlm.eval mcq`.

```json
{"question": "What is the power rating of the Hearth One?",
 "options": ["1800 W", "2400 W", "700 W", "1200 W"],
 "answer": "D"}
```

Options are scored by mean log-prob of the option text after an identical
four-option prefix; the highest wins. Mean, not sum, because summing makes short
options win for free.

## Splits

`--val-ratio 0.05` holds out five percent, seeded and stable. Held-out rows feed
`--eval-every`, which drives `sft_best.pt` and `--early-stop`. Pretraining has no
held-out split by design; use the loss curve and a downstream eval instead.

## Sizes that work

Rough guidance from running this pipeline, not a rule:

| Stage | Rows needed to see it learn | Rows for a usable result |
|---|---|---|
| pretrain | 1k documents | 10M+ tokens |
| sft | 200 | 5k |
| dpo | 100 pairs | 2k |
| grpo | 50 prompts | 1k |

Below these, loss drops fast and the model memorises answer shapes without
learning the mapping. That is what the README reports honestly for the 535-row
demo set.
