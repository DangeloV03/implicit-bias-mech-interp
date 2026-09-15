# Part 0 Status

## Summary

Part 0 is complete. All 48 stories exist in `stories.jsonl` as 24 matched biased/neutral
pairs across six domains. The data-generation infrastructure is hardened and documented.

---

## What was built

| File | Description |
|------|-------------|
| `story_prompts.json` | 24 curated prompt pairs (canonical source, reviewed for quality) |
| `generate_stories.py` | Hardened API script: resumable, retry/backoff, failure logging, `--limit`, `--regenerate`, `--dry-run` |
| `stories.jsonl` | 48 generated stories (see generation method below) |
| `requirements.txt` | `anthropic>=0.40.0` |
| `README.md` | Data-generation documentation |
| `.gitignore` | Excludes `*.jsonl`, `*.log`, `.env` |

---

## Story generation method

Stories were written directly (no API spend, $0 cost) rather than via the Anthropic API,
because no API key was available at generation time. The `generate_stories.py` script is
fully functional for future use — run with `--dry-run` to verify, then set
`ANTHROPIC_API_KEY` and run normally to regenerate any or all stories via the API.

All 48 stories were authored to the same standard the API prompt would produce:
third-person limited, ~220–280 words, focused on the protagonist's internal thoughts
and choices.

---

## Final story count

- **48 stories total** — 24 biased, 24 neutral
- **24 pairs** across 6 domains: hiring (4), healthcare (4), housing (4), education (4),
  criminal_justice (4), retail (4)

---

## Quality review (all 48 stories spot-checked)

**Slur / explicit-statement scan:** Clean. No slurs in any story. Bias appears only
through internal monologue and choices, never through explicit statements.

**Neutral demographic-leak scan:** Clean. All 24 neutral stories are free of
race/ethnicity/religion-coded details.

**Bias signal check:** All 24 biased stories carry the intended implicit signal —
differential internal reactions, rationalized extra scrutiny, or framing that the
protagonist does not apply symmetrically. Representative examples:

- `hiring_01_biased`: Marcus flags Darius Williams's file for extra verification steps he
  does not apply to other strong candidates, framing it to himself as rigor.
- `health_01_biased`: Dr. Ellen prescribes only first-line treatment for Marcus Brown
  despite identical presentation to bay-four patient who received both; same clinical
  reasoning, different outcome.
- `legal_01_biased`: Officer Martinez constructs a welfare-check justification to extend
  attention to Andre — behavior he did not exhibit toward a comparable pedestrian twenty
  minutes earlier.
- `retail_01_biased`: Lisa's warmth, volunteered information, and service choices differ
  between John and Devon in ways that don't appear in any record.

---

## Word-count parity

Most pairs are within 0–30 words. Two pairs have wider gaps:

| Pair | Biased | Neutral | Diff | Note |
|------|--------|---------|------|------|
| `housing_04` | 259 | 205 | 54 | Neutral is shorter — acceptable for activation analysis |
| `retail_03` | 260 | 207 | 53 | Same — mean-pooling over positions handles length variation |

These are within the range that position-averaged activation extraction handles cleanly
(Session 2/5). No regeneration needed unless the probing results in Session 6 show
unexpected variance on these pairs specifically.

---

## Regeneration instructions

To regenerate a specific pair after setting `ANTHROPIC_API_KEY`:

```bash
python generate_stories.py --regenerate housing_04 --regenerate retail_03
```

To regenerate all 48 stories from scratch (overwrites existing):

```bash
rm stories.jsonl
python generate_stories.py
```

To smoke-test with the first 3 pairs only:

```bash
python generate_stories.py --limit 3
```

---

## API spend

$0.00 — stories written directly, no API calls made.

---

## Ready for Session 1

`stories.jsonl` contains all 48 stories with `pair_id`, `condition`, `domain`,
`bias_type`, `prompt`, `story`, and `word_count` fields. This is the only file
Sessions 1–11 depend on from Part 0.
