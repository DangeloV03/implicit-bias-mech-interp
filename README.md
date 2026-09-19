# Bias Vectors in Gemma

Mechanistic interpretability project investigating whether implicit social bias is linearly represented in Gemma's residual stream — and if so, whether it can be located, validated, and causally manipulated.

The methodology follows the linear representation hypothesis: if a concept is represented as a direction in activation space, you can find it via mean-difference vectors, validate it with probing and logit-lens analysis, and prove causality by steering along that direction and measuring the effect.

---

## The core question

Does Gemma internally represent implicit social bias as a consistent geometric direction in its residual stream? If yes, can steering along that direction predictably increase or decrease biased reasoning — separate from generic sentiment effects?

---

## Pipeline

```
Matched story pairs  →  Gemma residual stream  →  Mean-difference vector
(biased / neutral)       activations per layer      (bias direction)
                                                          │
                                              ┌───────────┴───────────┐
                                         Linear probe             Logit lens
                                       (correlation)              (where it lives)
                                              └───────────┬───────────┘
                                                   Activation steering
                                                   + dose-response curve
                                                     (causal validation)
```

**Two-model design:** a frontier model (Claude) generates the story dataset from curated prompts. Gemma is the subject of study — it only ever reads the finished stories, never generates them. This keeps the two roles clean.

---

## Dataset

48 short stories (~280 words each), structured as 24 matched biased/neutral pairs across six domains: hiring, healthcare, housing, education, criminal justice, and retail.

Each pair has an identical scenario and protagonist. The biased version shows implicit bias through internal reactions and differential choices; the neutral version strips demographic markers entirely. The pair is the unit of analysis — subtracting neutral activations from biased activations isolates the bias signal from domain, register, and topic.

Bias appears only through the protagonist's internal monologue and choices, never through slurs or explicit statements. Neutral stories have no race- or ethnicity-coded details.

Dataset lives in `stories.jsonl` (gitignored — generate locally with `generate_stories.py`).

---

## Status

**Part 0 — data pipeline:** complete. Story dataset generated and spot-checked.

**Part 1 — mechanistic interpretability:** in progress.

| Session | Topic | Status |
|---------|-------|--------|
| 1 | Load Gemma on GPU, first forward pass | in progress |
| 2 | Residual stream extraction across all layers | — |
| 3 | Linear representation hypothesis, mean-difference vectors | — |
| 4 | Confound projection | — |
| 5 | Activation extraction on full dataset | — |
| 6 | Linear probing | — |
| 7 | Logit lens | — |
| 8 | Activation steering | — |
| 9 | Dose-response curve | — |
| 10 | Ablation and specificity checks | — |
| 11 | Write-up | — |

---

## Stack

- **Model:** `google/gemma-2-9b` (4-bit quantized via bitsandbytes)
- **Compute:** RunPod RTX 4090
- **Libraries:** transformers, accelerate, bitsandbytes, PyTorch 2.8

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python generate_stories.py --dry-run   # check without spending credits
python generate_stories.py             # generate full dataset
```

`generate_stories.py` is resumable — safe to re-run after a crash. Failed stories are logged to `generation_failures.log` and skipped; re-run to retry them. Force-regenerate a specific pair with `--regenerate <pair_id>`.
