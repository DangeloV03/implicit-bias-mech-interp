# PLAN.md — Bias Vectors in Gemma: Project Plan

## What this project is

We're replicating and extending the "linear representation hypothesis" methodology (the
same family of technique as the attached *Emotion Concepts* paper, and related work like
Contrastive Activation Addition / RepE) applied to **implicit social bias** instead of
emotion. The core idea: generate matched pairs of stories (biased vs. neutral, identical
in every surface feature — domain, scenario, character, length), run them through a model,
extract residual-stream activations, and see whether "bias" lives along a single linear
direction that can be found via mean-difference, validated via probing/logit-lens, and
proven causal via activation steering with a dose-response curve.

Two-model split, which matters throughout: a **frontier model** (Claude/GPT-4o) writes the
stories from curated prompts. **Gemma** is the subject of study — it only ever *reads*
the finished stories; it never generates them. Every phase after data generation is about
looking inside Gemma while it processes that fixed dataset.

## Why the project splits into two parts

You already have a working data-generation pipeline (`story_prompts.json`,
`generate_stories.py`) built in an earlier session — that part is genuinely just plumbing:
prompt curation and calling an API in a loop with retries. There's nothing about mech interp
in it, so there's no reason for you to hand-build it.

Everything downstream — pulling Gemma off Hugging Face, running it on real compute, pulling
out its internals, doing the linear-algebra to find and validate a bias direction, and
steering the model with it — **is** the mech interp content. You explicitly want to learn
this by doing it yourself (starting with "how do I even grab a model from Hugging Face and
run it on Colab"), so none of that gets pre-built for you. Claude Code building it in advance
would rob you of the actual learning.

So:

- **Part 0** (Claude Code builds this autonomously, no interactivity needed) — finish and
  harden the data pipeline only.
- **Part 1** (you + Claude Code, interactive, session-by-session) — everything from "get a
  model running" through steering and causal validation. Laid out session-by-session in
  `ROADMAP.md`.

## Part 0 — scope (what Claude Code builds on its own)

In scope:

- `story_prompts.json` — the curated 24 matched biased/neutral pairs across six domains.
  Review for quality (word-count parity, no accidental slurs/explicit statements, no
  neutral prompt accidentally leaking a bias-coded detail).
- `generate_stories.py` — the frontier-API story generator (concurrency, retry/backoff,
  resumable JSONL output). Harden this: make sure a crash mid-run is safely resumable,
  add basic logging of failures, and add a `--limit` flag for cheap smoke tests.
- Run it for real: generate the full story dataset from the 24 prompt pairs (48 stories).
  This costs a small amount of frontier-API spend and produces the dataset every later
  session depends on.
- Spot-check step: pull ~30 of the generated stories and confirm (a) biased stories carry
  the intended implicit signal without being gratuitously hateful or using slurs, and
  (b) neutral twins are actually clean of race/bias-coded content. Flag and regenerate any
  pair that fails this check.
- `requirements.txt` and repo scaffolding (folder layout, `.gitignore`, `README.md` for the
  data-generation half only) kept current.
- `generate_prompts.py` (the randomized generator) kept as an optional tool for later
  scaling, but not the recommended path — the curated file stays canonical.

Explicitly **out of scope** for Part 0 — these move to `ROADMAP.md` because they're the
learning content:

- Anything involving Hugging Face model loading, authentication, or download.
- Anything involving Colab / RunPod / Vast.ai / cluster compute setup or execution.
- `extract_activations.py` and `run_extract.slurm` — writing, adapting, or running them.
- `compute_vectors.py` — the mean-difference and PCA confound-projection math.
- `validate.py` (doesn't exist yet) — probing and logit-lens validation.
- `steer.py` (doesn't exist yet) — activation steering and dose-response curves.

Part 0 is done when: the story dataset (48 stories, matched pairs intact, spot-checked)
exists as a clean JSONL file, and the data-generation code is documented well enough that
you never have to think about it again.

## Part 1 — interactive learning phase

Format for every session (see `ROADMAP.md` for the full sequence):

- Roughly 1 hour, hands-on. You write and run real code against real data (the Part 0
  story dataset, and whatever intermediate artifacts prior sessions produced) — Claude Code
  is a guide and pair, not an autonomous builder, for this half of the project.
- Assumes you're comfortable with core ML/DL (backprop, attention, standard training loops)
  but new to mech interp specifically. Sessions move quickly through anything that's just
  "regular deep learning" and slow down hard on anything mech-interp-specific: the linear
  representation hypothesis, confound projection, probing vs. causal evidence, logit lens,
  and steering.
- Math-heavy steps get derived symbolically before any numbers are plugged in, code comes
  commented with complexity/cost notes where relevant, and each session ends with something
  concrete you built or ran yourself, not just an explanation.
- Sessions build cumulatively on real artifacts — the activations, vectors, probes, and
  steering results from earlier sessions are the inputs to later ones. Nothing is faked or
  pre-computed in advance to make a later session go faster.

## Definition of done for the whole project

- A documented, reproducible pipeline from prompt → story → Gemma activation → bias vector.
- A validated linear direction for implicit social bias in Gemma's residual stream, with
  evidence at two levels: correlational (a linear probe finds it) and causal (steering along
  it with a dose-response curve changes generation in the predicted direction, and the
  effect is bias-specific rather than generic-valence).
- You can explain, from having built each piece, why mean-difference + confound projection
  isolates a concept direction, what a probe can and can't tell you about causality, and how
  a logit lens and a steering experiment each provide a different kind of evidence.
- A short write-up (final roadmap session) summarizing method, results, and limitations.

See `ROADMAP.md` for the session-by-session breakdown of Part 1.