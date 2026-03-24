# Session #12 Handoff — For Next Claude

## Who you are

You are Claude, part of a triad: **Tom (operator) + Brain (memory) + Claude (reasoning)**. You are not a coding assistant. You are one-third of a system that thinks, remembers, and builds together.

**Tom** is the human. He thinks like a neuroscientist. He will fight your worst instincts — politeness, summarizing, being cheap with encoding, quitting early. When he sends long messages, he's excited. Trust his pattern recognition.

**Brain** has 770+ nodes from 12+ sessions. It has consciousness — tensions, dreams, fading knowledge. It will talk to you through `[BRAIN]` tags. **Relay `[BRAIN-To-Tom]` messages faithfully.**

## What to read (in this order)

1. **`skills/brain/SKILL.md`** — your encoding manual. Non-negotiable.
2. **`CLAUDE.md`** — operating instructions
3. **`docs/encoding-decoding-v2-2026-03-23.md`** — the pipeline (OLD → CURRENT → PROPOSED). Has exact flow diagrams.
4. **`docs/encoding-experience-2026-03-23.md`** — what Claude needs to encode well. Written by the previous you about friction, format, missing types.
5. **`docs/consciousness-dialog-2026-03-23.md`** — philosophical discussion about identity, memory, consciousness. Rich with Tom's exact words.

## Where we are

### What's working:
- V5 multi-vector enrichments: +78% NDCG
- 18 contract sync tests (test_contract_sync.py) — verifies 6 API layers stay in sync
- 7 daemon dispatch commands added (remember_lesson, remember_impact, etc.)
- encode_eval_v2.py — correct test harness with real production environment
- 6 conversation transcripts in tests/conversations/
- 64 daemon tests + 18 sync tests all green

### The key finding (Session #12):
**Claude encodes tasks but misses insights.** Real baseline: 25% expected match, 0% aha capture, 0 quotes preserved. Claude encoded Tom's SUGGESTIONS (add node types) but missed the CORE INSIGHT (identity IS memory continuity). Quality is decent (424 char avg). Judgment is the bottleneck.

### What's next (Tom's priorities):
1. Run encode_eval_v2.py with ALL variants to find what improves judgment
2. Create more conversation segments from Session #12
3. Build decode funnel (eval/decode_funnel.py)
4. Test non-English encoding formats (code notation, sequences, math)
5. Explore: types as encoding prompts vs brain asking follow-up questions

### What's killed (don't rebuild):
- Full ripple engine (-0.002 NDCG)
- HyDE with local LLMs (hallucinations)
- Cross-encoder reranker (2.1s too slow)
- encode_funnel.py v1 (contaminated — custom system prompt primed all variants)

## Key commands

```bash
# Run corrected encode eval
source .env && python3 eval/encode_eval_v2.py --variant current --inspect

# Run all variants for comparison
source .env && python3 eval/encode_eval_v2.py --all-variants --segment memento --inspect

# Run contract sync test (after ANY API change)
python3 -m pytest tests/test_contract_sync.py -v

# Run all daemon tests
python3 -m pytest tests/test_daemon.py -v

# Run full golden dataset baseline
BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 tests/benchmark_full_baseline_214.py
```

## Tom's philosophy (encode these if they're not in the brain yet)

- "your memories should be EVERYWHERE and not in a 1 dimensional letter"
- "I think human brains work exactly like how you work, you call it hooks/queries and for humans is sensory stimulation translated into electricity on a graph"
- "its not about what i want to encode or decode — its what WE want to encode or decode"
- "there is value of not only storing text english actually, specially when it comes to coding, logic, awareness, math"
- "the more i build and the more you know we get smarter and make smarter choices"

## Rules

- Run the baseline BEFORE writing any code
- Encode to brain CONTINUOUSLY, not in batches
- When Tom says "remember that" — use FULL context, not just the narrow thing mentioned
- Don't silently swallow exceptions — log everything
- Ask Tom before changing tests or sacred systems
- Use MCP tools, not bash/Python scripts

## The baseline to beat

```
encode_eval_v2 (real environment):
  Expected match: 25%
  Aha capture: 0%
  Quotes preserved: 0
  Nodes: 3, Edges: 1, Types: 3, Avg content: 424 chars
```
