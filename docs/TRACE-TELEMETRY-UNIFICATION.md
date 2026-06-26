# Trace Telemetry: Coverage Gaps + Unification (2026-06-25)

Hand-off for the next stream. Covers (1) what each agent currently persists to
its trace and the gaps, and (2) whether the mechanisms can be unified.

Grew out of the recall-timeout investigation
([RECALL-TIMEOUT-DIAGNOSIS.md](RECALL-TIMEOUT-DIAGNOSIS.md)) — the diagnosis was
hard precisely because Surface persists no cost telemetry.

## 1. Current state — who persists what

Two agent families, each missing what the other has:

| Telemetry | Surface (S1 decoder, `s1r` K event) | S1 Scribe + S2 units (`delta` event) |
|---|---|---|
| **Cost: input/output tokens, cache_*, elapsed_ms** | ❌ **missing** | ✅ present (`build_delta_metadata`) |
| **Tool/loop detail** (per round: tool, args, result_count, latency_ms, error) | ✅ **rich** (`tool_trace`) | ❌ counts only (`rounds`, `actions`, `read_calls` list — and `action_details`/`read_calls` often empty) |
| **Spread/kernel** (per step: new_nodes, edges_considered/transmitted, max_act, threshold) | ✅ `kernel_trace` | n/a |
| **Domain payload** | candidates+cosine, selected/dropped, frame, activations | created/revised/archived, journal_entry, outcomes |

**Q1 — are input_tokens missing from both Surface and S2?** No, asymmetric:
present for S2 + S1 Scribe, **missing for Surface**. S2's were added in the
2026-06-24 fleet-wide token fix; Surface was never on that path. Confirmed live:
healer delta carries `input_tokens:31327, output_tokens:1017, elapsed_ms:15257`;
Surface K carries none.

**Q2 — easy tool/loop telemetry in traces?** Yes for Surface (the rich
`tool_trace` is what made the agentic-loop diagnosis possible), **thin for S2**
(aggregate counts only — no per-tool latency, args, or result_count; the detail
fields are frequently empty).

Net: **no agent has both cost and tool detail.** Surface has detail, no cost;
encoders have cost, no detail.

## 2. Can it be unified?

**Yes — unify the core, specialize the shell.** These aren't fundamentally
different agents; they're the same agentic-loop substrate with two different
domain outputs.

### Already shared (everyone but Surface)

A unified telemetry stack exists and the encoders use it:

| Primitive | Definition | Reused by |
|---|---|---|
| `run_llm_loop` | `scales/runner.py` | S1 Scribe (`encode.py`), 3 S2 encoders, scouts |
| `USAGE_FIELDS` + `read_usage()` | `scales/runner.py:43,47` | runner, S2 base |
| `_sum_telemetry()` | `scales/s2/base.py:464` | healer / consolidation / community |
| `build_delta_metadata()` | `trace_contract.py:281` | S1 Scribe + all 4 S2 units |
| loud telemetry-missing detector | (brain `1e65ea2a`) | enforces token threading on deltas |

**Surface is the un-unified duplicate.** It calls `client.messages.create()`
directly (`surface.py:134` single-shot, `:229` agentic), hand-rolls its own loop
and its own trace metadata, and reuses none of the above — which is exactly why
it has no tokens.

### The two layers

- **Shared core (one mechanism):** every agent runs an LLM agentic loop whose
  loop+cost telemetry is identical in shape — `rounds`, per-tool trace (`tool`,
  `args`, `result_count`, `latency_ms`, `error`), `USAGE_FIELDS` tokens,
  `cache_*`, `elapsed_ms`, `truncated`. No reason for this to differ by agent.
- **Specialized shell (stays per-agent):** the domain payload reflects role and
  is genuinely different —
  - **decoder (Surface):** candidates+cosine, spread `kernel_trace`,
    selected/dropped, frame, activations → describes *selection*.
  - **encoder (Scribe/S2):** created/revised/archived, `journal_entry`,
    outcomes → describes *mutation*.

  Forcing these two payloads into one schema would be a leaky abstraction
  (a "candidates" list ≠ a "created/revised" set). Keep them separate.

## 3. Recommended path (next stream)

1. **Surface adopts the existing primitives.** Call `read_usage()` on each Haiku
   response + `_sum_telemetry()` across rounds; emit `input_tokens` /
   `output_tokens` / `cache_*` / `elapsed_ms` into the K trace. Zero new
   abstraction — a third reuse point. Closes the cost gap (the single
   highest-value, lowest-effort fix; would have turned the timeout investigation
   into one query).
2. **One "agent run telemetry" sub-object** in `trace_contract.py` — `rounds` +
   `tool_trace` shape + `USAGE_FIELDS` + `elapsed_ms` — embedded by *both* the
   Surface K trace and the encoder delta. Converges Surface's rich `tool_trace`
   and S2's thin counts onto one schema, so S2 gains the per-tool detail it lacks
   (closes the Q2 asymmetry). Guard it with the existing loud detector pattern.
3. **Do NOT force-merge the loop bodies.** Surface's loop has decode-specific
   control (spread interleaving, fetch-tool admission floor) and is
   latency-critical; merging it into `run_llm_loop` is high-risk for little gain.
   Share the *telemetry*, not necessarily the *loop*.
4. **Keep the domain payload specialized** — compose the shared sub-object +
   per-agent fields. Don't over-unify.

### Also worth folding in (from Q3 — wider gaps)

Beyond cost+tool parity, these would close the "understand functionality" gaps:
- **Per-phase latency, structured in the trace** (`surface_haiku`, `candidates`,
  `spread`, `render`) — today only in debug_log `hook_phase_timing` as an
  unqueryable string.
- **Hook-outcome flag** on the trace (`served` / `timeout` / `empty`) — today a
  timed-out recall writes a normal-looking trace; you must cross-reference
  `hook_errors` by timestamp to know it failed.
- **Recall usefulness** — no signal for whether surfaced nodes were used (no
  closed loop on recall quality).
- **Spread fan-out as a summary field** (51→525 varies wildly and drives cost).

## Precedent

This is the same "unify in `trace_contract.py`" move already underway for encoder
journals (brain `56871770` four-layer journal architecture; `4530ef2b` "five
reinventions, no shared contract"). Extend that pattern to loop/cost telemetry —
and this time include Surface (a decoder, no journal, but the same loop).

## References

- Code: `scales/runner.py` (USAGE_FIELDS/read_usage/run_llm_loop),
  `scales/s2/base.py` (_sum_telemetry), `trace_contract.py:281`
  (build_delta_metadata), `scales/s1/surface.py:134,229` (the un-unified loop).
- Brain: `6a7d605b` (these gaps as open thread), `5827910d` (S2 telemetry
  unification), `1e65ea2a` (loud detector), `0c24dc3f` (surface is 87% of recall
  budget).
