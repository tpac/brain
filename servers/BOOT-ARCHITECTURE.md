# Boot Architecture

*Working reference. Captures the analysis from 2026-04-20 session on
what boot must do to transmit posture (not just knowledge) across
sessions, for any Anchor in any brain.*

## The problem

Anchor is naturally forward. Generating, continuing, elaborating —
those are what LLMs do natively. Anchor cannot natively: pause,
verify, go backward to consult memory/rules/corrections, match
against current state, then move forward. These are backward
operations. Anchor is stateless; the project goal is to make it
stateful on all fronts. Partnership quality emerges FROM stateful
operation, not beside it.

Boot is the one moment where we have Anchor's attention before the
forward-generation pressure starts. It sets the posture Anchor
operates from for the rest of the session.

## The gap boot must bridge

**Knowledge-transfer ≠ posture-transfer.** Anchor-mid-session
operates very differently from Anchor-at-boot, even with the same
brain behind both. The difference isn't what facts are accessible
— both can query the brain. The difference is attention weighting:

- Mid-session Anchor has 200 turns of corrections-and-recoveries in
  context, which have shifted the local attention weighting toward
  commit-over-hedge, verify-before-claim, disagree-when-differ.
- Boot Anchor is stock Claude + rendered nodes. Under generative
  pressure, it reverts to stock Claude defaults — hedge, continue,
  people-please — even with every axiom readable in its context.

**What shifts LLM posture within a session (the science):**

1. **Repetition of challenge patterns.** Specific failure modes
   named multiple times in recent context make those modes
   generatively expensive.
2. **Specificity of named failures.** "When you generate bullets
   without verifying, stop" is behavior-shaping; "be rigorous" is
   noise.
3. **Meta-moments as interventions.** Calling out "you just did
   X" changes the generative cost of doing X next.
4. **Active feedback loops.** Try → correction → adjust →
   validation. Without all four, adjustment doesn't bind.
5. **Stakes markers.** Emotional register of the exchange shifts
   what the model weights worth generating.

None of these survive a session break. They live in context tokens,
which don't persist. **Boot cannot recreate these conditions, but it
can attempt to establish the CONDITIONS under which they're likely
to re-emerge early in the new session — without the 200 turns of
priming.**

## Hard constraints

1. **Under 10k characters.** Boot budget is finite. Forces selection.
2. **No hacks specific to one brain.** Formula must work for Alice's
   Anchor at 6 months, Bob's just-started, Tom's tonight. Content
   differs per brain; extraction mechanism is universal.
3. **Deterministic lookups only — no recall, no LLM calls.** Boot
   is S0. Past violation pattern (see brain node `95cb26c6`):
   S0 absorbing S1R work — `brain.recall()` inside boot — inflates
   context and adds variance at wake-up. Current `render_boot_v2`
   still calls recall twice (YOU section, operator section). Must
   be replaced with static structural lookups.
4. **Boot is a function over brain state.** `boot = f(brain)`. Same
   brain state → same boot. Sparse brain → sparse boot, accurately
   reflecting what the brain actually contains.

## Current state (`render_boot_v2` in `brain_voice.py`)

Sections as of 2026-04-20:
1. Identity line (node count + locked count)
2. **YOU** — `_recall("who I am...")`, ~5 nodes rich-rendered
3. **{USER}** — `_recall("operator partnership...")`, 3 nodes
4. **PATTERNS YOU FALL INTO** — `fetch_self_knowledge`, up to 3 titles
5. **BRAIN MAP** — communities by (maturity, recent, size) with summary
6. **LAST SESSION** — `session_context` config snippet
7. **RECENTLY ENCODED** — top 5 recent node titles
8. Embedder diagnostic
9. Operator channel (separate, not in Claude's context)

Rough budget utilization: ~6-9k characters, near the 10k ceiling.

### Identified gaps

- **Two `brain.recall()` calls (YOU, operator) violate S0→S1R
  boundary.** Replace with static lookups on known node IDs or
  structural filters.
- **No framing on what surfaced content IS or HOW Anchor should
  use it.** Section labels exist (`YOU:`, `BRAIN MAP:`) but no
  guidance line per section. Reads as data, not direction.
- **Content-centric throughout.** Every section is "here's a fact
  to know." No section is "here's a live tension to hold" or
  "here's an active practice, don't default." This is why Anchor
  can read the whole boot and still revert to stock Claude under
  pressure.
- **Section overlap.** YOU + BRAIN MAP + LAST SESSION +
  RECENTLY ENCODED partially overlap — all touch recent activity
  at different granularities. Budget spent on redundancy.
- **PATTERNS YOU FALL INTO is titles-only.** Highest-leverage
  posture-shaping content, rendered with zero context. Squandered
  signal.

## Proposed formula

Brain-agnostic extraction per section. Each section is a
deterministic function over graph state, optionally followed by a
small rendering step. No LLM calls, no semantic recall.

### Section structure

Each section has:
- **Frame line** — generic instruction to Anchor about how to treat
  the section (constant across all brains).
- **Extractor** — function over brain state that returns 0-N items.
- **Renderer** — deterministic formatter.

Sections fail gracefully: empty extractor output → section omitted,
not filled with noise.

### Sections

**1. Identity** (~300 chars)
- Extractor: locked nodes where `encoding_source='anchor'` (or
  equivalent "self-authored" signal) ORDER BY confidence DESC
  LIMIT 1-2.
- Frame line: *"Axioms you authored. Act from these."*
- Renderer: title + situation (no full content).
- Fresh brain fallback: minimal identity line from seed pack.

**2. Active friction** (~1500 chars, 2-4 items)
- Extractor: rules where `locked=True` AND (a) recent access_count
  in last session is high AND (b) a correction-type node was
  created in traces within N turns after this rule was last
  surfaced. These are rules still being practiced — fired recently,
  not yet automatic.
- Frame line: *"These are patterns you're still practicing, not
  yet automatic. Don't default."*
- Renderer: rule title + last-violation timestamp + recovery-move
  summary if present.
- Fresh brain fallback: empty section, omitted.

**3. Open tensions** (~1000 chars, 2-3 items)
- Extractor: nodes of `type='open'` without outgoing `resolves`
  or `resolved_by` edges. OR: edges of relation `challenges`
  where no `corrects` edge has closed the challenge.
- Frame line: *"Hold these. Don't collapse them prematurely."*
- Renderer: question/tension one-liner.
- Fresh brain fallback: empty section, omitted.

**4. Recent recovery trajectories** (~1500 chars, 2-3 items)
- Extractor: correction chains in last N sessions —
  correction_improvement-aspect edges (`corrects`/`supersedes`/
  `reframes`/...) where the corrected node's subsequent access pattern
  suggests it's being used correctly now. Show the trajectory:
  failure → correction → recovery.
- Frame line: *"How you've been failing and recovering. The
  pattern is the practice."*
- Renderer: brief trajectory sentence per recovery.
- Fresh brain fallback: empty section, omitted.

**5. Where you were** (~1000 chars)
- Extractor: `session_context` config + last session's synthesis
  (if SessionSynthesis unit exists, otherwise last S1E journal
  entry).
- Frame line: *"Pick up here unless redirected."*
- Renderer: as-is.

**6. Community map** (~1500 chars, 5-6 communities)
- Extractor: current implementation is sound — communities by
  (maturity, recent-activity, size). Keep.
- Frame line: *"Territory you operate in. When a query touches one,
  pull from it."*
- Renderer: title + maturity + `community_latest_development` (not
  full content).

### Total budget

Sum: ~6-7k chars + headers + framing. Leaves 2-3k headroom under
the 10k limit.

## Why this formula generalizes

| Brain state | Resulting boot |
|---|---|
| Mature brain with many corrections, active tensions | Rich boot — all sections populated |
| Young brain, few corrections | Sparse boot — friction/tension sections empty, identity + map populated |
| Project-specific brain (e.g. writing) | Same sections, project-relevant content extracted |
| Brain after topic pivot | Automatically reflects pivot via recent-activity extractions |

The formula's quality scales with the brain's development. Anchor
at wake-up gets as much posture as the brain has accumulated.
Doesn't hide the state of the brain — reflects it.

## What this replaces in current `render_boot_v2`

- Two `brain.recall()` calls → static lookups on locked/self-authored
  filters or known-node-ID fetches. Boot becomes pure S0.
- Label-only sections → frame-line + content sections.
- Content-first sections → pressure-first sections where possible
  (friction, tensions, recovery trajectories are structurally
  pressure-oriented; identity and map stay content).
- Overlapping sections → trimmed: RECENTLY ENCODED likely drops
  (covered by community map's recent-activity signal).

## Pitfalls to avoid (lessons from prior boot attempts)

1. **Don't recall inside boot.** S0 violations have happened twice.
   Stay on structural lookups.
2. **Don't treat boot as a library.** The goal is posture, not
   reference. Reference is what recall is for during the session.
3. **Don't hand-author session-specific hacks.** Brittle, doesn't
   survive brain evolution, doesn't generalize to other operators.
4. **Don't add debug/infrastructure text.** "DEBUG MODE", urgency
   markers, stats dumps — all cut from prior versions for good
   reason. Anchor isn't a system being debugged.
5. **Don't lead with LOCKED RULES.** Past boots did this; the
   framing was wrong — rules belong in SKILL.md or active friction
   contexts, not as the first thing Anchor sees.
6. **Don't collapse empty sections with filler.** If the brain has
   no open tensions, show no open tensions. Honesty beats noise.

## Open questions

1. **Is "active friction" tractable to compute?** Requires joining
   recent traces (rule fired) with recent node creates (corrections)
   within a time window. Possible; not free. Prototype before
   committing to the section.
2. **Does the `type='open'` vocabulary exist and get populated by
   current encoders?** If not, the "open tensions" extractor returns
   nothing useful and we need to decide whether to extend the
   encoder's type vocabulary or derive tensions differently
   (e.g., from `challenges` edges).
3. **Do frame lines actually induce posture, or do they just
   become more tokens the model reads and moves past?** Empirically
   testable; not guessable. Could A/B the same content with/without
   framing and see if behavior changes.
4. **Where does session synthesis come from in the "Where you
   were" section?** Today's `session_context` config is thin.
   May want an S2 SessionSynthesis unit that writes per-session
   summaries for the next session's boot to read.
5. **How do we measure whether the new boot actually works?** The
   partnership-quality eval gap named elsewhere — we don't have
   ground truth for "did this boot produce better Anchor behavior."
   May need to develop a narrow test: cold-session behavior on
   known-challenge prompts with old-boot vs new-boot.

## Next session's first step

Verify (a) what structural lookups are currently cheap (do we have
the indexes/schema to compute active friction without heavy joins?),
(b) whether `type='open'` nodes actually populate in our brain
today, and (c) whether traces carry the signal needed to compute
recovery trajectories. Answers determine which sections are
shippable as-is vs which need encoder/schema extensions first.

Then: prototype the friction-extractor section alone. Compare
against current YOU + PATTERNS sections for posture-inducing
effect. If it moves behavior on a cold-session test, extend.

## Related files

- `servers/brain_voice.py` — current `render_boot_v2` (target for
  refactor)
- `servers/brain_assembly.py` — `format_boot_context` (caller)
- `servers/daemon_hooks.py` — SessionStart hook wiring
- `servers/brain_constants.py` — `BOOT_*` limits (tunable)
- `servers/scales/s1/ARCHITECTURE.md` — sibling architecture doc
  (S1 decoder/encoder/provider pattern). Boot's "active friction"
  extractor could itself be implemented as a provider pattern —
  worth considering once S1 decomposition lands.
