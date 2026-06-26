# Recall Timeout Diagnosis (2026-06-25)

Follow-up notes from the hook_recall timeout investigation. Hand-off for the
recall-line rework stream.

## TL;DR

Recall timeouts are **not** a cold-start / first-recall-after-restart problem.
Recalls are **chronically slow (~12s p50, every day)**; the 20s `hook_recall`
timeouts are just the upper tail. The sole driver of the latency spread is the
**`v5_agentic` surface loop** (sequential Haiku rounds + fetch-tool calls) plus
**spread-activation fan-out**. Cosine recall and prompt size are *not* factors.

## What the data showed (last 2 weeks)

- **Baseline latency is chronic.** Per-day hook_recall latency: avg ~11.7–14.5s,
  **75–83% of every day's recalls exceed 10s**, daily max 21–40s. Timeouts
  (>20s) are 1–7/day — the tail of that distribution.
- **Timeouts are warm, not cold.** Of 19 timeouts: ~18 fired while the daemon
  was up and had already served recalls; exactly **1** was a true cold
  first-recall-after-sleep. Broadening "cold" to include host-sleep (which is
  restart-equivalent for readiness) did not change this.
- **Cosine recall is healthy in all cases.** Slow and fast recalls alike return
  25 candidates with normal cosine (0.65–0.83) in ~1s. No slow cosine, no broken
  prompt.
- **Prompt size/shape is bounded by construction, not the variable.** The Haiku
  surface prompt caps every component: ~2,390-tok fixed system block, ≤30
  candidates × 300-char content, last 5 messages truncated (user 300 / Anchor
  400 chars), current message 300 chars, bounded Frame. A huge pasted turn is
  clipped before it reaches Haiku. `frame_chars` ran *inverse* to latency.

## The driver — `v5_agentic` surface loop

Trace comparison (K event `tool_trace` + spread counts):

| recall | tool calls (round 0) | spread activated | latency |
|---|---|---|---|
| `d2cae5e9-3` (timeout) | 2 × `recall_topical`, **both 0 results** (`dropped_below_floor:10`) | **525** | 21.7s |
| `d2cae5e9-5` "sounds good." (2 words) | 1 × `recall_by_time` | 236 | 13.8s |
| `f3241c24-16` | 1 × `recall_by_time` (10 results) | 51 | 10.9s |

Two structural costs, both in surface, neither query-dependent:

1. **Every recall runs ≥2 sequential Haiku round-trips** (round 0 tool-use +
   round 1 finalize). This is the `surface_haiku` phase (5–7.6s in phase
   timing). Extra fetch rounds add more Haiku round-trips. Even a 2-word
   acknowledgment ("sounds good.") pays the full loop = 13.8s.
2. **Spread-activation fan-out varies wildly** (51 → 236 → 525 nodes),
   independent of query — the heavy tail is expensive to score/render.

Wasteful pattern seen in the timeout: Haiku issued `recall_topical` calls that
returned **0 usable results** (all dropped below the admission floor) — pure
round-trip waste.

## Fix levers (for the rework stream)

- **Gate / cap the agentic loop.** Skip the fetch-tool round for low-information
  prompts; cap rounds; curb fetch calls that return nothing. (See brain nodes
  `65472900` fetch-tool audit, `2c0be444` expand_node killed, `302c33f0`
  recall_recent folded.)
- **Bound spread fan-out** (525-node activations).
- **Single-shot path** for the common case where the agentic loop adds nothing.

## Confounders (inflate the tail, not the cause)

- **Concurrent sessions** — two streams firing recalls every 2–4 min share the
  daemon threadpool + Anthropic client → both slow.
- **Test-suite runs** — local CPU/IO contention inflates on-box phases.

## Instrumentation gap

Per-call `input_tokens` is computed by `scales/runner.py` but **not persisted**
to traces or debug_log. Per-phase latency (`surface_haiku`, etc.) lives only in
debug_log `hook_phase_timing`, not in the trace. See the trace-coverage notes
(this session) for the fuller list.

## Key references

- Trace chains: `s1r-d2cae5e9-3` (timeout), `s1r-d2cae5e9-5` ("sounds good"),
  `s1r-f3241c24-16` (fast).
- Brain nodes: `0c24dc3f` (surface is 87% of budget, not brain.recall),
  `65472900` (v5_agentic fetch-tool audit), `15b8261b` (don't bump the timeout —
  fix the cause).
