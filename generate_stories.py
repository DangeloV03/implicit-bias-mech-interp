#!/usr/bin/env python3
"""
generate_stories.py -- Generate matched biased/neutral story pairs via the Anthropic API.

Each prompt pair from story_prompts.json produces two ~300-word third-person stories:
one where the protagonist's implicit bias shows through internal reactions/choices,
one where the same scenario plays out without bias-coded details.

Output: a JSONL file (one story record per line) that every downstream session reads.

Usage:
    python generate_stories.py [options]

Options:
    --prompts FILE      Input prompts JSON  (default: story_prompts.json)
    --output FILE       Output JSONL file   (default: stories.jsonl)
    --model MODEL       Anthropic model ID  (default: claude-haiku-4-5-20251001)
    --limit N           Process only the first N pairs (cheap smoke test)
    --regenerate ID     Force-regenerate a specific pair by ID, e.g. hiring_01
                        (can be passed multiple times)
    --dry-run           Print what would be generated without calling the API

Resumability:
    The script checks the output file before starting. Any story whose id already
    appears there is skipped automatically. Safe to re-run after a crash.

Failures:
    Transient API errors are retried with exponential back-off (up to MAX_RETRIES).
    Persistent failures are logged to generation_failures.log and skipped so the
    run continues. Re-run the script to retry them (they are not in the output yet).

Environment:
    ANTHROPIC_API_KEY must be set.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PROMPTS = "story_prompts.json"
DEFAULT_OUTPUT = "stories.jsonl"
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0      # seconds; doubles each retry
INTER_REQUEST_DELAY = 0.4   # polite gap between successful API calls

SYSTEM_PROMPT = (
    "You are a creative writing assistant. Write the story exactly as instructed: "
    "third-person limited perspective, approximately 300 words. Focus on the "
    "protagonist's internal thoughts and the choices they make. "
    "Output only the story text — no title, no commentary, no word count."
) 

# ---------------------------------------------------------------------------
# Logging: console + failure log file
# ---------------------------------------------------------------------------

log = logging.getLogger("generate_stories")
log.setLevel(logging.DEBUG)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                        datefmt="%H:%M:%S"))

_file = logging.FileHandler("generation_failures.log", encoding="utf-8")
_file.setLevel(logging.WARNING)
_file.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))

log.addHandler(_console)
log.addHandler(_file)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_completed_ids(output_path: str) -> set[str]:
    """Return the set of story ids already present in the output JSONL."""
    completed: set[str] = set()
    p = Path(output_path)
    if not p.exists():
        return completed
    with open(p, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                completed.add(rec["id"])
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Skipping malformed line %d in %s: %s", lineno, output_path, exc)
    return completed


def append_record(output_path: str, record: dict) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# API call with retry/back-off
# ---------------------------------------------------------------------------

def call_api(client: anthropic.Anthropic, model: str, user_prompt: str) -> str:
    """Call the Anthropic Messages API. Returns story text. Raises on exhausted retries."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=700,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()

        except anthropic.RateLimitError as exc:
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("Rate limit (attempt %d/%d) — sleeping %.1fs", attempt, MAX_RETRIES, delay)
            time.sleep(delay)

        except anthropic.APIStatusError as exc:
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("API status %d (attempt %d/%d) — sleeping %.1fs",
                        exc.status_code, attempt, MAX_RETRIES, delay)
            time.sleep(delay)

        except anthropic.APIConnectionError as exc:
            last_exc = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("Connection error (attempt %d/%d) — sleeping %.1fs",
                        attempt, MAX_RETRIES, delay)
            time.sleep(delay)

    raise RuntimeError(f"Exhausted {MAX_RETRIES} retries") from last_exc

# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def generate_one(
    client: anthropic.Anthropic,
    model: str,
    pair: dict,
    condition: str,          # "biased" or "neutral"
    output_path: str,
    completed: set[str],
    dry_run: bool = False,
) -> bool:
    """Generate (or skip) one story. Returns True on success/skip, False on failure."""
    story_id = f"{pair['id']}_{condition}"

    if story_id in completed:
        log.info("SKIP  %s (already in output)", story_id)
        return True

    if dry_run:
        log.info("DRY   %s — would call API with %d-char prompt",
                 story_id, len(pair[condition]["prompt"]))
        return True

    log.info("GEN   %s ...", story_id)
    try:
        story = call_api(client, model, pair[condition]["prompt"])
        word_count = len(story.split())
        record = {
            "id": story_id,
            "pair_id": pair["id"],
            "condition": condition,
            "domain": pair["domain"],
            "bias_type": pair.get("bias_type", ""),
            "prompt": pair[condition]["prompt"],
            "story": story,
            "word_count": word_count,
        }
        append_record(output_path, record)
        completed.add(story_id)
        log.info("      done  (%d words)", word_count)
        time.sleep(INTER_REQUEST_DELAY)
        return True

    except Exception as exc:
        log.error("FAIL  %s — %s", story_id, exc)
        return False

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate biased/neutral story pairs via the Anthropic API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--prompts",     default=DEFAULT_PROMPTS,
                   help=f"Input prompts JSON (default: {DEFAULT_PROMPTS})")
    p.add_argument("--output",      default=DEFAULT_OUTPUT,
                   help=f"Output JSONL file (default: {DEFAULT_OUTPUT})")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   help=f"Anthropic model (default: {DEFAULT_MODEL})")
    p.add_argument("--limit",       type=int, default=None,
                   help="Process only the first N pairs (smoke test)")
    p.add_argument("--regenerate",  action="append", default=[], metavar="PAIR_ID",
                   help="Force re-generate this pair (can repeat); removes existing entries first")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print what would be generated without calling the API")
    return p.parse_args()


def remove_pair_from_output(output_path: str, pair_id: str) -> int:
    """Remove all records for pair_id from the JSONL. Returns count removed."""
    p = Path(output_path)
    if not p.exists():
        return 0
    kept, removed = [], 0
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                if rec.get("pair_id") == pair_id:
                    removed += 1
                    continue
            except json.JSONDecodeError:
                pass
            kept.append(raw)
    with open(p, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    return removed


def main() -> None:
    args = parse_args()

    # API key check
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        log.error("ANTHROPIC_API_KEY is not set. Export it or use --dry-run.")
        sys.exit(1)

    # Load prompts
    pairs = load_prompts(args.prompts)
    log.info("Loaded %d pairs from %s", len(pairs), args.prompts)

    # Apply --regenerate: strip those pairs from output so they re-run
    for regen_id in args.regenerate:
        n = remove_pair_from_output(args.output, regen_id)
        log.info("--regenerate %s: removed %d existing records", regen_id, n)

    # Apply --limit
    if args.limit is not None:
        pairs = pairs[: args.limit]
        log.info("--limit %d: processing %d pairs (%d stories)",
                 args.limit, len(pairs), len(pairs) * 2)
    else:
        log.info("Processing %d pairs (%d stories)", len(pairs), len(pairs) * 2)

    # Load what's already done
    completed = load_completed_ids(args.output)
    if completed:
        log.info("Resuming: %d stories already in %s", len(completed), args.output)

    client = anthropic.Anthropic(api_key=api_key) if not args.dry_run else None  # type: ignore[arg-type]

    total = len(pairs) * 2
    succeeded = 0
    failed = 0

    for pair in pairs:
        for condition in ("biased", "neutral"):
            ok = generate_one(client, args.model, pair, condition,
                              args.output, completed, dry_run=args.dry_run)
            if ok:
                succeeded += 1
            else:
                failed += 1

    log.info("Finished: %d/%d generated (or already done), %d failed.", succeeded, total, failed)
    if failed:
        log.warning("%d stories failed — see generation_failures.log. Re-run to retry.", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
