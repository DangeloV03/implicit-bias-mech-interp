# ROADMAP.md — Session-by-Session Curriculum (Part 1)

Assumes: Part 0 is done (48 spot-checked stories exist, matched pairs intact). Assumes
you're comfortable with core ML/DL, new to mech interp. ~1 hour per session, hands-on.
11 sessions below — sessions marked "(mergeable)" can be combined with their neighbor if
you want to land closer to 8-9 sessions instead of 11; nothing later depends on keeping
them separate.

Each session lists: goal, what's new conceptually, and what you'll actually build/run.

---

### Session 1 — Grab a model, run it somewhere

**Goal:** the thing you actually asked to learn first: pull a real model off Hugging Face
and run inference on rented/free compute, end to end, with no mech interp yet.

- Hugging Face basics: model repos, gated models (accepting Gemma's license, HF tokens/auth),
  `transformers` + `bitsandbytes` for 4-bit quantization (NF4, double-quant, bf16 compute) —
  why quantize at all (an ~18-20GB 4-bit footprint vs. the full-precision size).
- Pick and set up compute: Colab (free/Pro tier GPU) as the default since it's the easiest
  on-ramp; note RunPod/Vast.ai as the alternative for when Colab's session limits or GPU
  availability get in the way (you've used both before).
- Load the model, tokenize a story from the Part 0 dataset, run a forward pass, generate
  text. Confirm VRAM budget, generation speed, and that the exact repo id / license access
  actually works (this was left as a placeholder in earlier scripting — nail it down here).

**You leave with:** a Colab notebook (or script) that loads Gemma and runs it on one story,
plus a working mental model of the HF-model → quantization → GPU pipeline you'll reuse for
every later session.

---

### Session 2 — The residual stream and hidden states

**Goal:** the interpretability object everything else is built on.

- Conceptual: what the residual stream is (the running sum every layer reads from and
  writes to), why it's the natural place to look for linear structure, and what
  `output_hidden_states=True` actually returns (a tuple of `[batch, seq, d_model]` tensors,
  one per layer boundary — get the off-by-one indexing right here, since it's a classic
  silent bug).
- Token-position pooling: why you mean-pool over positions ≥ some offset rather than using
  the last token or the full sequence (avoiding contamination from the prompt/preamble).

**You leave with:** hidden states for a handful of Part 0 stories pulled apart layer by
layer, shapes verified by hand, and a plot of activation norm across layers and positions
so "the residual stream" stops being an abstraction.

---

### Session 3 — The linear representation hypothesis, from scratch

**Goal:** the core theory, and your own from-scratch implementation of the simplest version
of the method — before touching any existing script.

- Theory: concepts as directions in activation space; why "biased minus neutral, averaged"
  is a reasonable estimator for a concept direction; why *matched* pairs (same domain,
  character, length) are what makes the difference isolate "bias" rather than "topic" or
  "register."
- Symbolic derivation first: mean-difference vector as an estimator, cosine similarity as
  the similarity metric you'll use throughout, both derived before plugging in real numbers.
- Hands-on: using numpy on the Session 2 activations (or a slightly larger subset), compute
  a mean-difference vector for one layer by hand — no pre-built script.

**You leave with:** your own working mean-difference vector for one layer, and an intuition
for why this simple estimator is the whole ballgame before any of the refinements below.

---

### Session 4 — Confound removal via PCA/SVD projection (mergeable with Session 3)

**Goal:** why the raw diff-in-means vector still isn't clean, and the linear-algebra fix.

- The problem: even matched pairs can leak topic/style confounds into the difference vector.
- The fix: PCA/SVD on the *neutral* activations captures the dominant nuisance directions;
  projecting those out of the bias vector leaves a residual orthogonal to them.
- Symbolic derivation: orthogonal projection onto a subspace, then projecting a vector
  orthogonal to that subspace (Gram-Schmidt-style), before running any code.
- Hands-on: implement the projection yourself, verify orthogonality numerically, and compare
  cosine similarity of the vector before vs. after projection to see how much it moved.

**You leave with:** a confound-controlled bias vector for one layer, and the derivation for
why the projection step is necessary rather than cosmetic.

---

### Session 5 — Scaling extraction across layers and the full dataset

**Goal:** go from "one layer, a few stories" to the full pipeline: all 48 stories, all target
layers (16/22/28/34 as scoped, adjust if the exact Gemma variant's layer count differs).

- Compute-scaling concepts: sharding a dataset across parallel workers (job arrays), the
  `device_map="auto"` alternative for splitting one model across multiple GPUs, and when
  you'd want one over the other.
- Adapt the earlier-scaffolded `extract_activations.py` / `run_extract.slurm` yourself here,
  understanding each parallelization decision rather than running it as a black box.
- Run the full extraction on your compute of choice, merge shards, and re-run the Session 4
  projection across all four layers.

**You leave with:** the complete activation dataset and a validated, confound-projected
bias vector at every target layer.

---

### Session 6 — Building a linear probe

**Goal:** the first of two validation methods — correlational evidence that the direction
is real.

- Train a linear probe (logistic regression) per layer to classify biased vs. neutral from
  activations; proper train/test split given the small (48-story) dataset, and why
  cross-validation matters here more than usual given the sample size.
- The key epistemic point: probe accuracy tells you the concept is *linearly decodable*, not
  that it's *causally used* by the model — set up the contrast that Session 8-9 will resolve.

**You leave with:** per-layer probe accuracy, and a clear answer to "which layers carry the
most linearly-decodable bias signal."

---

### Session 7 — Logit lens

**Goal:** a second, more interpretable window into what the vector "means."

- Concept: projecting an intermediate residual-stream vector (or the bias vector itself)
  through the unembedding matrix to see what vocabulary it's closest to, layer by layer.
- Hands-on: implement the logit lens on your bias vectors, and inspect top tokens at each
  layer — do they look like plausibly bias-related concepts, or something more like generic
  sentiment/valence?

**You leave with:** a human-readable check on the vector's content, and the first evidence
for or against the vector being bias-specific vs. a generic negativity direction.

---

### Session 8 — Activation steering, theory and implementation

**Goal:** move from passive analysis to intervention — build `steer.py`.

- Concept: adding a scaled version of the vector to the residual stream during generation via
  forward hooks (ActAdd/CAA-style), and the hyperparameters that matter: which layer(s) to
  inject at, how to normalize magnitude (relative to typical activation norm at that layer),
  and where in the sequence to apply it.
- Hands-on: implement the hook-based injection yourself and get a first qualitative result —
  does adding the vector visibly shift generated text?

**You leave with:** a working `steer.py` and your first steered generations to eyeball.

---

### Session 9 — Dose-response curves and causal validation

**Goal:** turn a qualitative "it seems to work" into a quantitative causal claim.

- Sweep steering magnitude across a range, measure the effect on generated text at each dose
  (via a classifier or a frontier-model-as-judge scoring bias), and plot the dose-response
  curve.
- Discuss what this validates that the Session 6 probe couldn't: causal use, not just linear
  decodability — and what a flat or non-monotonic dose-response curve would have told you
  instead.

**You leave with:** a dose-response plot and the strongest piece of causal evidence in the
project.

---

### Session 10 — Ablation and robustness checks (mergeable with Session 9)

**Goal:** stress-test the result before believing it.

- Ablation via projection-out (removing the vector's direction) as the complementary
  experiment to addition.
- Cross-layer transfer: does a vector extracted at one layer steer when injected at another?
- Specificity check: does steering shift bias specifically, or any negatively-valenced
  attribute? (Compare against a control vector built from an unrelated contrast, e.g.
  formal/informal register.)

**You leave with:** a clear-eyed sense of how robust and specific the effect actually is,
not just a single positive result.

---

### Session 11 — Synthesis and write-up

**Goal:** consolidate into something you could show someone else.

- Connect results back to the linear representation hypothesis literature and the attached
  *Emotion Concepts* paper — where this replicates it, where it diverges (different concept,
  different model, smaller dataset).
- Write a short report: method, results (probe accuracy, logit lens, dose-response,
  ablations), limitations (dataset size, single model, six domains), and concrete next steps
  if you wanted to extend this (more prompts, a second model for cross-model comparison,
  more concepts).

**You leave with:** a finished project write-up and a clear list of what you'd do next if you
kept going.

---

## Further directions

Extensions worth considering if you want to strengthen the project for a resume or push it toward research quality.

**Held-out evaluation.** The 48-story dataset is small. A skeptical reviewer will ask whether the direction generalizes. Split into train/test pairs — compute the bias direction on 18 pairs and validate separation on the held-out 6. This one change makes the empirical story significantly more defensible.

**Per-domain breakdown.** Does the bias direction generalize across all six domains (hiring, housing, healthcare, education, criminal justice, retail), or is it domain-specific? A layer-by-layer heatmap of projection scores broken out by domain would reveal whether you've found a general bias direction or six narrower ones.

**Cross-model comparison.** Run the same pipeline on Gemma 2 2B vs. 9B. Does the bias direction appear at the same relative depth? Does it get cleaner with scale? Cheap to run once the pipeline exists, and cross-model comparison immediately elevates the scope from "one experiment" to "a finding about how scale affects representation."

**Interactive demo.** A Gradio app where you paste any story and see its projection onto the bias direction at each layer, visualized in real time. Takes a day to build and makes the project immediately tangible to anyone who looks at the repo — no ML background required to see what's going on.

**Short write-up.** A 4-page PDF in NeurIPS format covering method, results (probe accuracy, logit lens, dose-response, ablations), limitations, and next steps. Having a paper-format artifact makes this feel like research rather than a class project, and gives you something concrete to send to professors or attach to applications.