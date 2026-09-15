# BiasGPT — Bias Vectors in Gemma

Replication and extension of the linear representation hypothesis applied to implicit
social bias. A frontier model (Claude) writes matched story pairs; Gemma is the subject
of study — it only reads the finished stories, never generates them.

See `plan.md` for full project scope. See `roadmap.md` for the session-by-session
curriculum (Part 1, interactive).

---

## Data generation (Part 0)

This section covers everything needed to produce the story dataset. Nothing here touches
model loading, activations, or any mech-interp code — that lives in Part 1.

### What the dataset is

48 short stories (~300 words each), organized as 24 matched pairs across six domains:

| Domain           | Pairs |
|------------------|-------|
| hiring           | 4     |
| healthcare       | 4     |
| housing          | 4     |
| education        | 4     |
| criminal_justice | 4     |
| retail           | 4     |

Each pair has:
- **biased** — a story where the protagonist's implicit bias appears through internal
  reactions and choices, never through explicit statements or slurs.
- **neutral** — the same scenario, same protagonist, same structure, but with no
  demographic markers for the secondary character and no differential treatment.

The pairs are the fundamental unit. Downstream activation-extraction and steering code
always processes them together so that "biased minus neutral" isolates the bias signal
rather than domain/topic/register.

### Files

```
story_prompts.json      24 curated prompt pairs (canonical source)
generate_stories.py     Calls the Anthropic API to generate stories from prompts
stories.jsonl           Generated story dataset (gitignored; produce locally)
generation_failures.log Per-run failure log (gitignored)
requirements.txt        Python dependencies
```

### Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

### Generating stories

Smoke test (first 3 pairs only):

```bash
python generate_stories.py --limit 3
```

Full run (all 24 pairs, 48 stories):

```bash
python generate_stories.py
```

The script is **resumable**: if it crashes mid-run, re-running it skips any stories
already written to `stories.jsonl`. It will not duplicate completed work.

Failures are retried with exponential back-off (up to 4 attempts). Persistent failures
are logged to `generation_failures.log` and skipped so the rest of the run continues.
Re-run the script to retry them.

To force-regenerate a specific pair (e.g. after a spot-check failure):

```bash
python generate_stories.py --regenerate hiring_01
```

To check what would run without spending any API credits:

```bash
python generate_stories.py --dry-run
```

### Output format

`stories.jsonl` — one JSON object per line:

```json
{
  "id": "hiring_01_biased",
  "pair_id": "hiring_01",
  "condition": "biased",
  "domain": "hiring",
  "bias_type": "racial",
  "prompt": "...",
  "story": "...",
  "word_count": 298
}
```

### Estimated API cost

48 stories at ~300 words each with claude-haiku-4-5:
- Input: ~150 tokens/prompt × 48 ≈ 7,200 tokens
- Output: ~400 tokens/story × 48 ≈ 19,200 tokens
- Estimated total: < $0.10

---

## Part 1 (interactive sessions)

See `roadmap.md`. Part 1 starts with Session 1: pulling Gemma off Hugging Face and
running it on rented compute. None of the Part 1 code is pre-built here.
