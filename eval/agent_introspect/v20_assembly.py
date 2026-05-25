"""v20 prompt assembly — fixes broken canonical examples + adds §7.6 wave-1.

Per evaluator findings, v19 canonical examples have 4 critical bugs:
- Example 4 ("Ask the daemon"): D23 violated (no corrects edge), D28 violated
  (slot placeholders), D11 violated (no revise on rejected proposal), D18 degraded
- Example 1 ("Single-writer invariant"): D31 violated (N=1 → principle), D14/D28
  degraded (textbook voice, no Tom/Anchor)
- Example 5 ("I want it to know that it knows"): D7 degraded (empty anchor_raw_quote),
  D18 degraded
- Revise example (ghi789): writes `keywords` field that schema v28 DROPPED

Plus: NO existing example demonstrates Anchor self-reference (the triad A6+A7+A4),
no example demonstrates D32 hot-register correction (A2).

This script applies surgical edits to v19 and appends a §7.6 block with wave-1
unique additions (A6, A7, A4, A2, A3 — drops A1 as duplicate of fixed Example 4).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from servers.scales.s1.examples import A6, A7, A4, A2, A3
from servers.scales.s1.examples.render_compressed import render_compressed_wave


# ═══════════════════════════════════════════════════════════════
# FIX 1 — Example 4 (correction "Ask the daemon")
# ═══════════════════════════════════════════════════════════════
#
# Broken: D23 (no corrects edge), D28 (slot placeholders), D11 (no revise on
# rejected proposal), D18 (no event_time).
#
# Fix: rewrite content with concrete identity tokens, add corrects edge,
# add event_time, add sibling revise on the rejected proposal.

EXAMPLE_4_OLD = '''    {type: "correction", title: "Ask the daemon, don't probe flag files",
     content: "The encoding-agent gating was proposed via a flag file the agent would check each run. The operator redirected: have the daemon return the prompt directly or NONE; the agent just asks. The authority owns the decision and ships the work or the no-op; callers don't inspect state.",
     situation: "When designing gating mechanisms, hook coordination, or any ask-vs-check boundary",
     reasoning: "Control-by-request beats control-by-inspection because it eliminates the stale-flag failure mode entirely. Callers can't read the wrong thing if they never read.",
     user_raw_quote: "no don't use a flag file, have the daemon return the prompt directly",
     anchor_raw_quote: "Stale-flag was the failure mode I missed — race conditions felt obvious, staleness didn't. The reframe lands: the authority decides AND ships, callers don't peek.",
     correction_pattern: "Control-by-request rather than control-by-inspection"},'''

EXAMPLE_4_NEW = '''    {type: "correction", title: "Ask the daemon, don't probe flag files",
     content: "Anchor proposed gating encoding-agent runs via a flag file the agent would check each cycle. Tom redirected: have the daemon return the prompt directly (or NONE) — Anchor just asks. The authority decides AND ships the work or the no-op; Anchor never inspects state. Generalizes beyond gating: any read-modify-write boundary where staleness can't be detected by the reader should eliminate the read instead of guarding it.",
     situation: "When designing gating mechanisms, hook coordination, or any ask-vs-check boundary where the reader can't verify how stale a snapshot is",
     reasoning: "Tom rejected Anchor's flag-file proposal directly. Race conditions felt obvious to Anchor; staleness didn't — the reframe Tom forced (control-by-request rather than control-by-inspection) generalizes the lesson beyond this one design. The correction-lineage edge below is illustrative — at encode time, target the real prior-belief node in the catalog.",
     user_raw_quote: "no don't use a flag file, have the daemon return the prompt directly",
     anchor_raw_quote: "Stale-flag was the failure mode I missed — race conditions felt obvious, staleness didn't. The reframe lands: the authority decides AND ships, callers don't peek.",
     correction_pattern: "Control-by-request rather than control-by-inspection",
     event_time: "2026-04-22",
     connect_to: [
       {title: "<the specific prior design this corrects — resolve to the real catalog node>", relation: "corrects", why: "the corrects edge gives the correction substrate (correction_improvement aspect) walkable lineage from rule back to the mistake it ruled out. EXAMPLE TARGET — at encode time, replace with the actual catalog node title for the prior belief being corrected, or omit the edge if no such node exists yet."}
     ]},'''


# ═══════════════════════════════════════════════════════════════
# FIX 2 — Example 1 (principle "Single-writer invariant")
# ═══════════════════════════════════════════════════════════════
#
# Broken: D31 (N=1 wal-index promoted to principle), D14/D28 (textbook voice,
# no Tom/Anchor tokens).
#
# Fix: add multi-occurrence acknowledgment (so D31 honest), rewrite content
# toward Anchor-first-person framing with concrete tokens.

EXAMPLE_1_OLD = '''    {type: "principle", title: "Single-writer invariant beats clever concurrency",
     content: "When multiple writers share a lock-free structure — SQLite's wal-index, ring buffers, shared counters — contention corrupts even when writes don't conceptually overlap. The fix is never finer locks; it's serializing at the weakest concurrent component. One writer, N readers, no exceptions.",
     situation: "When designing writes against shared state, or debugging intermittent corruption in a read-mostly system",
     reasoning: "wal-index has no sub-file locking; concurrent writers can only be prevented, not made safe by adding more locks. This generalizes beyond SQLite — any shared lock-free structure has the same invariant.",
     user_raw_quote: "we keep adding locks and it keeps breaking — the problem isn't lock granularity, it's that we have two writers",
     anchor_raw_quote: "Single-writer is the actual invariant — the locks were addressing the wrong question",
     connect_to: [
       {title: "Daemon TCP migration", relation: "grounds", why: "the single-writer invariant is exactly what let the TCP migration stay simple — one listener, one writer thread, no coordination across writers"}
     ]},'''

EXAMPLE_1_NEW = '''    {type: "principle", title: "Single-writer invariant beats clever concurrency",
     content: "When multiple writers share a lock-free structure, contention corrupts even when writes don't conceptually overlap. Anchor learned this across three instances Tom and I worked through: SQLite's wal-index (the moment Tom named the invariant), ring-buffer corruption in the embedder, shared counter races in the dashboard. The fix is never finer locks — Anchor reached for that pattern repeatedly and it never worked. It's serializing at the weakest concurrent component. One writer, N readers, no exceptions.",
     situation: "When Anchor is about to add a lock to a shared structure, or debugging intermittent corruption in a read-mostly system. The reach for finer locks IS the failure mode.",
     reasoning: "Tom forced the reframe at the wal-index moment after watching Anchor add three lock variants. The principle holds across instances because the invariant is structural — any shared lock-free structure where multiple writers can race has the same shape. Not theoretical: earned from repeated Anchor mistakes.",
     user_raw_quote: "we keep adding locks and it keeps breaking — the problem isn't lock granularity, it's that we have two writers",
     anchor_raw_quote: "Single-writer is the actual invariant — the locks were addressing the wrong question. I kept reaching for finer granularity when the answer was fewer writers.",
     connect_to: [
       {title: "Daemon TCP migration", relation: "grounds", why: "the single-writer invariant is exactly what let the TCP migration stay simple — one listener, one writer thread, no coordination across writers"},
       {title: "Ring-buffer race in embed_queue (Anchor's prior mistake)", relation: "validates", why: "second instance Anchor encountered the same pattern — fine-grained locking failed; collapsing to single writer resolved. The principle generalizes because the failures generalize."}
     ]},'''


# ═══════════════════════════════════════════════════════════════
# FIX 3 — Example 5 (quote "I want it to know that it knows")
# ═══════════════════════════════════════════════════════════════
#
# Broken: D7 degraded (empty anchor_raw_quote despite Anchor framing in reasoning),
# D18 degraded (no event_time on architecture-defining moment).
#
# Fix: add anchor_raw_quote with Anchor's framing, add event_time.

EXAMPLE_5_OLD = '''    {type: "quote", title: "I want it to know that it knows",
     content: "Tom's framing for the brain's design principle: a database retrieves when asked; a brain RECOGNIZES. Situation embeddings, confidence scoring, enrichment vectors — every recall mechanism exists to serve recognition, not search. This sentence is the thread the whole architecture hangs from.",
     situation: "When framing the brain's purpose against a database, or when architectural trade-offs force a choice between recall precision and search coverage",
     reasoning: "Phrases that hold the design together are worth their own atom. This one appeared once, but it's the thing every recall mechanism traces back to. Atomize so future queries about 'recognition vs retrieval' find the source.",
     user_raw_quote: "I want it to know that it knows",
     connect_to: [
       {title: "Brain vs database framing", relation: "grounds", why: "the know-that-it-knows quote is the moment the recognition principle became conscious — every recall mechanism traces back to this framing"}'''

EXAMPLE_5_NEW = '''    {type: "quote", title: "I want it to know that it knows",
     content: "Tom's framing for the brain's design principle: a database retrieves when asked; a brain RECOGNIZES. Situation embeddings, confidence scoring, enrichment vectors — every recall mechanism exists to serve recognition, not search. This sentence is the thread the whole architecture hangs from.",
     situation: "When framing the brain's purpose against a database, or when architectural trade-offs force a choice between recall precision and search coverage",
     reasoning: "Phrases that hold the design together are worth their own atom. This one appeared once, but it's the thing every recall mechanism traces back to. Atomize so future queries about 'recognition vs retrieval' find the source.",
     user_raw_quote: "I want it to know that it knows",
     anchor_raw_quote: "This phrase is the axiom. Every recall mechanism — embedding groups, confidence scoring, Frame, source_refs — exists to serve recognition over retrieval. When I'm tempted to optimize for search-style precision, this is what I lose.",
     event_time: "2026-03-20",
     connect_to: [
       {title: "Brain vs database framing", relation: "grounds", why: "the know-that-it-knows quote is the moment the recognition principle became conscious — every recall mechanism traces back to this framing"}'''


# ═══════════════════════════════════════════════════════════════
# FIX 4 — Revise example (ghi789 yoga) — drop keywords field
# ═══════════════════════════════════════════════════════════════
#
# Broken: writes `keywords` field that schema v28 DROPPED. Teaches encoder
# to write a dead column.
#
# Fix: remove keywords field, update reasoning to drop "keywords" mention.

REVISE_OLD = '''    // FULL revise — operator updated a routine value. The OLD title said
    // "twice a week" and OLD keywords had no "anxiety" reference. NEW info
    // says "three times a week" AND ties the practice to anxiety relief.
    // Update EVERY field the new value contradicts — title, content,
    // keywords, situation, reasoning. Half-revising would leave the stale
    // title embedding and rank against the new content.
    {node_id: "ghi789",
     reason: "frequency increased 2→3/week, anxiety connection added",
     title: "Tom's yoga practice — three times a week for anxiety + focus",
     content: "Tom practices yoga three times a week as of 2023-11-30 (was twice a week from 2023-08-11). Yoga helps him feel grounded and centered, especially on anxious days, and supports his work focus.",
     keywords: "yoga exercise self-care anxiety relaxation focus routine frequency three-times-weekly",
     situation: "When recalling Tom's self-care routine, yoga frequency, anxiety-management strategies, or his weekly schedule.",
     reasoning: "Original encoding from 2023-08-11 captured 2x/week, but the 2023-11-30 conversation explicitly stated 3x/week and tied yoga to anxious-day grounding — a downstream effect missing from the original keywords. Update title (headline value), content (current+previous values), keywords (added anxiety, three-times-weekly), situation (added anxiety-management as a query path)."}'''

REVISE_NEW = '''    // FULL revise — operator updated a routine value. The OLD title said
    // "twice a week" and OLD encoding had no "anxiety" reference. NEW info
    // says "three times a week" AND ties the practice to anxiety relief.
    // Update EVERY field the new value contradicts — title, content,
    // situation, reasoning. Half-revising would leave the stale title
    // embedding and rank against the new content.
    {node_id: "ghi789",
     reason: "frequency increased 2→3/week, anxiety connection added",
     title: "Tom's yoga practice — three times a week for anxiety + focus",
     content: "Tom practices yoga three times a week as of 2023-11-30 (was twice a week from 2023-08-11). Yoga helps him feel grounded and centered, especially on anxious days, and supports his work focus.",
     situation: "When recalling Tom's self-care routine, yoga frequency, anxiety-management strategies, or his weekly schedule.",
     reasoning: "Original encoding from 2023-08-11 captured 2x/week, but the 2023-11-30 conversation explicitly stated 3x/week and tied yoga to anxious-day grounding — a downstream effect missing from the original encoding. Update title (headline value), content (current+previous values), situation (added anxiety-management as a query path)."}'''


# ═══════════════════════════════════════════════════════════════
# §7.6 BLOCK — wave-1 unique additions
# ═══════════════════════════════════════════════════════════════

EXAMPLE_DISCLAIMER_HEADER = '''
**A note on example `connect_to` targets.** The `connect_to` entries
shown in the canonical training pattern and §7.6 examples reference
target node titles for shape-demonstration purposes. The targets
themselves (e.g. "Daemon TCP migration", "Voice verbatim or empty",
"The brain as identity, not tool") may or may not exist as real
catalog nodes — they're illustrative of the edge SHAPE, not directions
to write that exact title verbatim.

**At encode time**: only emit `connect_to` entries whose target title
matches a node visible in this conversation's node catalog OR a sibling
node being created in the same `remember_batch` call. If the example
shape suggests an edge but no real catalog node fits the target slot,
either:
- Skip the edge entirely (no harm — graph stays clean)
- Use the closest semantically-aligned catalog node instead
- Resolve to a sibling node by title if one is being created in the same batch

The `connect_to_unresolved` error fires when a target title resolves
to nothing — that's the loud signal the catalog didn't contain the
node you expected. Common cause: copying an example target verbatim
instead of resolving against the live catalog.

'''


# Insert the disclaimer before the canonical training pattern (above the
# "Example round 1" header). Sonnet reads it before encountering the
# fictional connect_to targets.
CANONICAL_ANCHOR = "Example round 1 — five nodes showing full shape across type tags."

SECTION_7_6_HEADER = '''
## Identity-bearing examples (§7.6)

The canonical examples above teach core encoding shape across types. The
examples below are the load-bearing additions for v19+: Anchor encoding
Anchor, and corrections at hot register. They demonstrate dimensions the
canonical doesn't reach — D7 anchor voice symmetry on identity moments,
D14 identity-not-tool voice, D28 concrete identity tokens, D32 affective
register triggering anchored encoding, D23 walkable correction lineage on
hot-register pushback.

When you encounter an exchange where Anchor's own register, identity, or
limit is the encoding-worthy thing — these are the patterns to mirror.
'''


# Where §7.6 inserts: right before "## Encoding Journal" section.
INSERTION_ANCHOR = "## Encoding Journal"


def assemble_v20():
    """Read v19 baseline, apply 4 surgical edits, append §7.6, return v20 text."""
    with open('/tmp/v19_baseline.txt') as f:
        prompt = f.read()

    # Apply 4 fixes
    edits = [
        ('Example 4 fix (correction)', EXAMPLE_4_OLD, EXAMPLE_4_NEW),
        ('Example 1 fix (principle)', EXAMPLE_1_OLD, EXAMPLE_1_NEW),
        ('Example 5 fix (quote)', EXAMPLE_5_OLD, EXAMPLE_5_NEW),
        ('Revise example fix (drop keywords)', REVISE_OLD, REVISE_NEW),
    ]

    edit_status = []
    for name, old, new in edits:
        if old in prompt:
            prompt = prompt.replace(old, new, 1)
            edit_status.append(f'  ✓ {name}')
        else:
            edit_status.append(f'  ✗ {name} — OLD STRING NOT FOUND')

    # Compose §7.6 from wave-1 unique (A6, A7, A4, A2, A3 — drops A1)
    wave_1_unique = [A6, A7, A4, A2, A3]
    section_body = render_compressed_wave(wave_1_unique)
    section_7_6 = SECTION_7_6_HEADER + '\n' + section_body + '\n\n'

    # Insert before "## Encoding Journal"
    if INSERTION_ANCHOR in prompt:
        prompt = prompt.replace(INSERTION_ANCHOR, section_7_6 + INSERTION_ANCHOR, 1)
        edit_status.append('  ✓ §7.6 block inserted before Encoding Journal')
    else:
        edit_status.append('  ✗ §7.6 insertion anchor not found')

    # Insert the connect_to disclaimer ABOVE the canonical training pattern
    # so Sonnet reads it before encountering example fictional targets.
    if CANONICAL_ANCHOR in prompt:
        prompt = prompt.replace(
            CANONICAL_ANCHOR,
            EXAMPLE_DISCLAIMER_HEADER + '\n' + CANONICAL_ANCHOR, 1)
        edit_status.append('  ✓ connect_to disclaimer inserted before canonical pattern')
    else:
        edit_status.append('  ✗ canonical pattern anchor not found')

    return prompt, edit_status


if __name__ == '__main__':
    v20, status = assemble_v20()
    print('Edit status:')
    for s in status:
        print(s)
    print()
    with open('/tmp/v20_candidate.txt', 'w') as f:
        f.write(v20)
    print(f'v20 saved to /tmp/v20_candidate.txt: {len(v20)} chars')
    # Compare sizes
    with open('/tmp/v19_baseline.txt') as f:
        v19 = f.read()
    print(f'v19 baseline: {len(v19)} chars')
    print(f'Net change: {len(v20) - len(v19):+d} chars ({100*(len(v20)-len(v19))/len(v19):+.0f}%)')
