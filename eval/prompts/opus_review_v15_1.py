"""Send v15.1 to clean Opus for a thorough review of the voice redistribution.

Tom's directive: 'give it to a clean opus to help you.' This is the
second-opinion check — Opus reads v15.1 with no context except the prompt
and the stated intent, and surfaces:
  - whether voice symmetry is achieved
  - what residual bias remains
  - what edits are needed to close the gap
  - any drift from the stated intent

Run: ./dev python3 eval/prompts/opus_review_v15_1.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Ensure ANTHROPIC_API_KEY is loaded
envf = ROOT / '.env'
if envf.exists() and not os.environ.get('ANTHROPIC_API_KEY'):
    for line in envf.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            if not os.environ.get(k.strip()):
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

import anthropic

PROMPT_PATH = ROOT / 'eval/prompts/s1e_v15_1.txt'
PRIOR_PATH = ROOT / 'eval/prompts/s1e_v15.txt'
OUT_PATH = ROOT / 'eval/prompts/opus_review_v15_1.md'

prompt_v15_1 = PROMPT_PATH.read_text()
prompt_v15 = PRIOR_PATH.read_text()

# Stated intent of v15.1 — what we set out to do.
STATED_INTENT = """
v15.1 was written to address a specific finding from probing v15: despite
v15 adding the `anchor_raw_quote` field and a "What earns encoding" section,
a clean Sonnet reading the FULL v15 prompt still inferred operator-voice
primacy from the rest of the prompt's gravitational field. Sonnet's
verbatim diagnosis of v15:

  "Despite the late addition saying 'substance earns encoding regardless
  of source,' the entire first half treats user_raw_quote as the gold
  standard. I'd weight operator-originated content over assistant-
  originated or third-party content."

  "Despite 'substance earns encoding,' the prompt's emphasis structure
  and worked examples make me weight operator participation heavily."

The intent of v15.1 is to neutralize that residual bias by redistributing
voice emphasis throughout the prompt — not just adding a section but
rebalancing every place where voice is invoked. Specifically eight edits
were applied to v15:

  1. Intro paragraph: equalize "operator's exact words carry weight" with
     "Anchor's exact words carry weight equally" + add subject-matter note.
  2. Explicit correction: stop framing corrections as one-direction
     (operator → assistant); name self-correction and source-correction.
  3. Flat→Rich templates: replace "operator said" with "{speaker} said"
     in templates 2, 3, 4; explicitly note voice options.
  4. "When the operator states a choice..." paragraph: extend to Anchor
     stating choices/patterns/stances.
  5. Paraphrase instinct: name `user_raw_quote`, `anchor_raw_quote`,
     scout evidence, AND third-party verbatim — symmetric across speakers.
  6. Example block intro: narrative-derived nodes carry the matching
     voice anchor, not just user_raw_quote.
  7. "What this is" closing: rewrite from "This brain belongs to Tom —
     his voice preserved verbatim where it mattered" to partnership
     ownership language with both voices preserved.
  8. Example block: add a 7th example node showing a self-correction
     with anchor_raw_quote.
"""

REVIEW_PROMPT = """You are reviewing a revised encoder prompt for a persistent knowledge graph. The encoder is a stateless Sonnet that runs every 5 conversational turns to encode the conversation into nodes. The brain is meant to be 'Anchor's continuous experience' — the partnership memory between Tom (the operator) and Anchor (the AI assistant).

CONTEXT — what we changed and why:
""" + STATED_INTENT + """

YOUR TASK: Read the v15.1 prompt below carefully (it's ~10K tokens). Then answer:

## 1. Did the redistribution achieve voice symmetry?

Read the prompt as if you were the encoder. Would you weight operator voice, Anchor's voice, and third-party content symmetrically when deciding what to encode? OR does the prompt still tilt toward operator-voice primacy?

Be specific. Quote lines that demonstrate the answer.

## 2. What residual bias remains?

If symmetry isn't fully achieved, where exactly is the prompt still tilting? Identify SPECIFIC sentences or sections that still privilege one voice over another.

## 3. Drift from stated intent

Did v15.1 actually do what intent (1)-(8) above said it would? Walk through each numbered intent and verify whether the prompt actually contains the corresponding redistribution. If something is missing or weaker than intended, name it.

## 4. Does the prompt have any new problems v15 didn't have?

When redistributing voice, did v15.1 introduce any awkwardness — confusing language, contradictions, examples that work less well, structural issues? Be honest. The goal isn't to praise the changes; it's to surface what didn't land.

## 5. Concrete next edits

If you were to write v15.2 to close any remaining gap, what would you change? Be specific — quote the line you'd edit and propose the replacement.

======================================================================
v15.1 PROMPT BEGIN
======================================================================
""" + prompt_v15_1 + """

======================================================================
v15.1 PROMPT END
======================================================================

Now answer questions 1-5 plainly. Don't summarize — diagnose. Be willing to say 'this still doesn't work' if it doesn't."""

print(f"[opus-review] v15.1 length: {len(prompt_v15_1):,} chars")
print(f"[opus-review] full message length: {len(REVIEW_PROMPT):,} chars")
print(f"[opus-review] sending to claude-opus-4-7...", flush=True)

client = anthropic.Anthropic()
t0 = time.time()
resp = client.messages.create(
    model='claude-opus-4-7',
    max_tokens=4000,
    system="You are a careful reviewer of LLM prompts. Your job is to "
           "surface bias, drift, and gaps — not validate. Be direct, "
           "quote specific lines, avoid hedging.",
    messages=[{"role": "user", "content": REVIEW_PROMPT}],
)
elapsed_ms = int((time.time() - t0) * 1000)

answer = ''
for block in resp.content:
    if hasattr(block, 'text'):
        answer += block.text

print(f"[opus-review] done — {resp.usage.input_tokens} → "
      f"{resp.usage.output_tokens} tok in {elapsed_ms/1000:.1f}s", flush=True)

# Write report
OUT_PATH.write_text(
    f"# Opus review — s1e v15.1\n\n"
    f"**Model:** claude-opus-4-7\n"
    f"**Tokens:** {resp.usage.input_tokens} in → {resp.usage.output_tokens} out\n"
    f"**Wall:** {elapsed_ms/1000:.1f}s\n\n"
    f"---\n\n"
    f"{answer}\n"
)
print(f"[opus-review] wrote {OUT_PATH}")
