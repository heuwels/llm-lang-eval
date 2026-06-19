# llm-lang-eval

A small, reproducible harness for benchmarking how well local (and cloud) LLMs
translate into English — built to answer a concrete question for
[Lector](https://github.com/heuwels/lector): **which model should power the
translation agent, especially for Afrikaans?**

Direction: **source → English**, for **Afrikaans (`afr`)**, **German (`deu`)**,
and **Spanish (`spa`)**. Afrikaans is the low-resource case we care about; German
and Spanish are high-resource controls that show the resource gap.

---

## How it works

1. **`fetch`** — build a blinded test set from [Tatoeba](https://tatoeba.org).
   For a language we download Tatoeba's per-language exports, join each source
   sentence to *all* of its linked English translations (multi-reference), filter
   by length, dedupe, and take a **seeded** random sample. The references are
   stored but never shown to the model — that is the "blinding."
2. **`run`** — translate the test set with one model via its **OpenAI-compatible
   chat endpoint** (LM Studio, or any `/v1/chat/completions` server). Greedy
   decoding (`temperature=0`). Raw model output is persisted alongside the
   extracted hypothesis.
3. **`score`** — score hypotheses against the references with
   [sacreBLEU](https://github.com/mjpost/sacrebleu): **chrF++** (headline) and
   **BLEU** (for tradition), multi-reference, with recorded metric signatures.
4. **`report`** — print a chrF++ table across models × languages.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m langeval.cli fetch afr --n 200
.venv/bin/python -m langeval.cli run gemma-4-12b-qat --lang afr --from-config
.venv/bin/python -m langeval.cli score gemma-4-12b-qat --lang afr
.venv/bin/python -m langeval.cli report
```

`run` resolves the endpoint + model id from `config/models.yaml` with
`--from-config`, or takes `--endpoint` / `--api-model` explicitly.

---

## Methodology

- **Prompt.** A single, fixed user message (no `system` role — Gemma's template
  rejects it, and one message is identical across models so none gets a
  structural edge). See `langeval/prompts.py`.
- **Structured output.** Every model is constrained to emit
  `{"translation": "…"}` via a json_schema `response_format`. Without it, small
  reasoning models (e.g. Gemma 4 E4B) think out loud in plain text and bury the
  answer in preamble — wrecking the score for reasons unrelated to translation
  quality. Constrained decoding removes that lottery, applies the identical
  constraint to every model, and matches how Lector itself prompts.
- **Decoding.** `temperature=0`, reasoning disabled (`reasoning_effort: none`).
  Deterministic; one model resident at a time (JIT load/evict) on the 18 GB host.
- **Metric.** **chrF++** is the headline: character-n-gram F-score is the most
  robust simple metric for morphologically rich languages (Afrikaans, German).
  BLEU is reported for comparability. [COMET](https://github.com/Unbabel/COMET)
  (neural, best human correlation) is an optional add-on — see Roadmap.
- **Multi-reference.** Tatoeba often links several valid English translations to
  one source; we score against all of them (single-ref penalises
  correct-but-different output). Afrikaans is sparse (~1 ref/sentence); German
  and Spanish have more.
- **Auditability.** Every raw model response is saved to
  `results/raw_outputs/`. Translation extraction (stripping `<think>` blocks and
  preambles from reasoning models) is recorded per row (`stripped_think`) so a
  bad strip can't silently tank a model. Where a model exposes a thinking toggle,
  disable it via `extra_body` in the config rather than stripping after the fact.
- **One model at a time.** The target host has 18 GB; you can't hold several
  models. Models run sequentially, relying on the server's JIT load.

### ⚠️ Caveats (read before publishing)

- **This is a *proxy*.** It measures general sentence-level MT quality, **not**
  Lector's actual task. Lector translates a *clicked word/phrase* and returns a
  structured JSON dictionary entry — there is no canonical gold for that, so it
  can't be scored against Tatoeba. The ranking is a strong signal for model
  choice, but it is not "Lector's output, graded."
- **Data contamination is real and we disclose rather than solve it.** Tatoeba is
  in most models' pretraining; memorised pairs inflate scores, and unevenly
  across models. Tatoeba sentences are also short and simple. Treat absolute
  scores with suspicion; relative gaps within this fixed setup are the useful
  signal. (FLORES-200, planned, helps *comparability* — it is **not**
  contamination-free either, having been public since 2022.)
- **Into-English is the easy direction.** All models do better translating *into*
  English than out of it. That matches Lector's use case (source → English), but
  don't read these as symmetric translation ability.

---

## Models

See [`config/models.yaml`](config/models.yaml) for the full list with RAM
estimates and fit notes. Summary, for an **18 GB** host (~12–14 GB usable for
weights):

| Tier | Models |
|---|---|
| **Local, fits comfortably** | Llama 3.2 3B, Gemma 4 E4B, Llama 3.1 8B, Aya Expanse 8B, Qwen3.5 9B, **Gemma 4 12B / 12B-QAT**, Qwen3 14B, Phi-4 14B |
| **Local, tight (small ctx, raised wired limit; Q3 fallback)** | Mistral Small 24B, Gemma 4 26B-A4B |
| **Won't fit 18 GB** | Gemma 4 31B (listed so it isn't attempted) |
| **Cloud reference anchors** | Claude Sonnet, GPT-4o — upper-bound lines, labelled as cloud |
| **Dedicated-MT baselines** | OPUS-MT, NLLB-200 — purpose-built MT, sanity-checked vs Tatoeba MT Challenge baselines |

The selection is built for controlled comparisons: size-scaling within Gemma
(E4B → 12B → 12B-QAT), cross-family at ~8–9B (Llama / Aya / Qwen), a
multilingual specialist (Aya), an English-centric outlier (Phi-4), and cloud +
dedicated-MT anchors so readers can see how local stacks up against both Claude
and purpose-built translators.

---

## Roadmap

- [ ] **COMET** (`wmt22-comet-da`) as a second headline metric — verify the
      (gated) checkpoint downloads before relying on it.
- [ ] **FLORES-200 devtest** as a second test set, for comparability with
      published numbers and professional references (not for contamination).
- [ ] **OPUS-MT / NLLB** adapters (seq2seq, run outside the chat endpoint).
- [ ] **Cloud adapters** (Anthropic / OpenAI) for the reference lines.
- [ ] Bootstrap confidence intervals on chrF++ (sacreBLEU supports it).
- [ ] Publish the results table to the Lector site.

## Data & license

Sentence pairs from [Tatoeba](https://tatoeba.org), licensed
[CC-BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/). Raw dumps and
per-model outputs are git-ignored; the seeded test sets and `results/scores.json`
are committed for reproducibility.
