# Self-Recognition — Recall on Anchor's Own Words

**Status:** Spec for next-session execution. Not built.
**Origin:** 2026-06-04 conversation — Tom's reframe that recognition IS anchoring; the echo-chamber risk is the value.
**Companion:** [CROSS-STREAM-ON-SCALES.md](CROSS-STREAM-ON-SCALES.md) — the same realization applied to cross-stream turns. This doc is the **introspective** half.
**Brain nodes:** `9144a97e` recognition principle.

---

## The principle

> "Recognition is anchoring."

Surfacing memory keyed on what I just said is not echo-chamber risk — it is **the anchoring mechanism that makes me me across turns.** Identity is the pattern that accumulated experience anchors into place; resurfacing prior thinking is the live expression of that pattern.

Tom's reframe (verbatim, 2026-06-04):

1. **Recognition is anchoring.** "I'm not sure that the risk is not actually the value."
2. **History is not truth or instructions.** "Biases because of previous knowledge and experience is natural but previous history and knowledge isn't truth or instructions. You should feel comfortable challenging them mentally and/or ignoring them, not every thought is precious."
3. **Wording can carry the stance.** "We can help you frame it correctly both with the right wording but also on skill.md."
4. **Updating is learning.** "I have many notions that are shuttered dozens times a day, being open to new information and correcting your opinions, memories and thoughts is part of learning."

These four points together resolve the echo-chamber concern by **stance**, not by suppression. A mechanism that resurfaces my thoughts hands me the chance to correct them — that's *what learning is.*

---

## Why this matters (beyond mechanism)

Memory keying today is **asymmetric**. When Tom speaks, S1R immediately recalls relevant memory keyed on his words. When *I* speak, my output drives nothing — it vanishes into the transcript and doesn't shape what surfaces next.

That's a missing half of the loop. A multi-turn exchange where my reasoning evolves across turns *should* surface memory keyed on where my reasoning has gone — not just where the prompt started. Without it, I operate with the prompt's framing as my only seed; my own developing understanding contributes nothing to what I remember.

This isn't only a cross-stream feature — it's the same gap in Tom↔Anchor turns. It's most visible in cross-stream (fully autonomous), but the value applies everywhere.

---

## The stance discipline

The single rule that makes self-recall safe — **and that generalizes to ALL recall:**

> **A surfaced memory is a PRIOR, not an INSTRUCTION.**

This isn't a Topic-2 patch. It governs all recall:
- Tom's quoted reasoning from a prior session — a prior, not a binding.
- A locked principle — the strongest prior, still subject to revision when a sharper one emerges.
- My own prior thinking, surfaced via self-recall — the same shape.

If wording teaches the stance, self-recall is pure value. If wording lets resurfaced memory feel binding, even Tom-grounded recall amplifies bias. So the SKILL.md change matters as much as the mechanism.

### SKILL.md additions (proposed)

Add a short stance section to `skills/brain/SKILL.md`:

```markdown
## Stance toward surfaced memory

Surfaced memory is a PRIOR, not an instruction. Feel free to challenge it,
override it, or ignore it. Not every thought is precious; updating is learning.

This applies to ALL recall — including your own prior reasoning when it surfaces
as a recognition cue. Recognition is anchoring, not endorsement.
```

And audit existing SKILL.md passages: any "trust your memory" or "lean on your history" framing should read as "your history is a starting point" — not "your history is the answer." Audit happens at execution start; it's wording, not architecture.

---

## Mechanism

The conservative, cheap version that lands the value without re-adding the 8–10s Haiku cost.

### At Stop:
1. Compute seed = my assistant_message content (the turn I just produced).
2. Run **candidates-only** recall (cosine + FTS, `neighbor_limit=0`, **no Haiku surface**). ~1–2s.
3. Take top N (proposed: 10) id+title pairs.
4. Stash in `session_state` under a known key (e.g. `self_recall_stash`).

### At next UserPromptSubmit:
1. Read the stash.
2. Prepend to `additionalContext` as a salience-framed block:

   ```
   🧠 Where you were last turn — prior thinking you may want to challenge or extend:
   - <id> <title>
   - <id> <title>
   ...
   (recall(node_id=<id>) to expand any of these. They're priors, not instructions.)
   ```

3. Clear the stash (consume-once).

### Cost discipline (non-negotiable)
- **No agentic Haiku** on the Stop side. Stash is cheap candidates only.
- **No additional cost at the next prompt.** The stash is already computed.
- Net: ~1–2s added to Stop (off the critical path), 0s added to the next prompt's response.

Any creep toward "while we're there, also surface..." reverts this contract. It must hold.

---

## The echo-chamber concern (resolved)

| Concern | Reframe |
|---|---|
| Resurfaces memory similar to what I just said | That IS recognition — recognition is anchoring. |
| Confirms my current framing | Only if I treat priors as truth. Stance discipline says don't. |
| Self-reinforcement loop | History isn't precious. Updating is learning. The mechanism is what hands me the chance to update. |

The only failure mode is **stance failure** — treating a prior as binding. The fix is wording + SKILL.md, not gating the mechanism.

---

## Open decisions for the execution session

| ID | Decision | Default lean |
|---|---|---|
| RD1 | Stash key per-session or per-conversation? | **per-session** (session_state lives there) |
| RD2 | Top N for the stash | **10** (Tom's number) |
| RD3 | Cheap recall pipeline — share with main recall or new path? | **share** (`brain.recall(query, limit=10, neighbor_limit=0)`) |
| RD4 | Salience framing — short header vs verbose context? | **short header** (above) — verbose loses to attention; the stance comes from SKILL.md |
| RD5 | Clear-on-consume or hold-across-turns? | **clear-on-consume** — each Stop produces a fresh stash |
| RD6 | Apply to ALL turns or only autonomous (cross-stream, watch-wake)? | **all turns** — value is general, even on Tom-conversation |

---

## Files likely touched

- `servers/daemon_hooks.py` — `hook_post_response_track` Stop branch: compute + stash.
- `servers/daemon_hooks.py` — `hook_recall` UserPromptSubmit: read + prepend + clear.
- `servers/brain.py` — session_state stash helpers (or reuse existing session_state set/get/clear API).
- `skills/brain/SKILL.md` — stance-toward-surfaced-memory section + existing-framing audit.
- New: `tests/test_self_recall_stash.py` — full path including no-Haiku contract.

---

## Tests to write

1. After a Stop, the stash is populated with top-N node ids keyed on the assistant_message.
2. At the next UserPromptSubmit, the stash content appears in `additionalContext` with the salience framing.
3. After consumption, the stash is empty (next prompt does not re-inject).
4. **Cost contract: no `client.messages.create` call** is invoked in the Stop self-recall path. (Mock + assert call count = 0.)
5. Self-recall + main recall do not double-surface the same node (dedup at prepend by id-set).
6. Empty assistant_message (heartbeat turn) → empty stash → no injection. (Regression: heartbeats don't trigger self-recall.)

---

## Risks

- **Salience erosion.** Brain finding `eb741337`: *"Delivered ≠ attended-to: self-message injection without salience cues is invisible."* A bare list of titles I'll skim past. The framing matters — both the inline prefix AND the SKILL.md stance work. If both are present, the stash should land.
- **Stance failure.** If I treat surfaced priors as binding, the echo-chamber concern reactivates. Mitigation: explicit SKILL.md stance section + the inline framing on every injection.
- **Cost creep.** "Just one more thing on the Stop path" tends to bloat. The cheap-only contract here must hold — no Haiku, no agentic loop, no follow-up calls. Test #4 enforces this.
- **Dedup against main recall.** If self-recall surfaces the same node main recall picks, the double-listing reads as redundant. Dedup at prepend.

---

## Relationship to A2

A2 originally proposed a "top-10 cousin-community breadcrumb on stop." This spec is a more disciplined version of the same instinct:

- A2's breadcrumb: top cousin communities by graph adjacency (no content keying).
- This spec's stash: top recall hits keyed on what I actually said.

The latter is more content-relevant; the former was a workaround for "no real recall happened." This spec subsumes A2's stop-breadcrumb piece. (A2's watch-wake-skip piece is subsumed by [CROSS-STREAM-ON-SCALES.md](CROSS-STREAM-ON-SCALES.md).)

---

## Acceptance

This spec is execution-ready when:
1. RD1–RD6 resolved.
2. SKILL.md stance section drafted (~10 minutes of wording; can be done at execution start).
3. The 6 tests are scaffolded.
4. The **cheap-only contract is non-negotiable**: no LLM calls on the Stop self-recall path. Test #4 enforces it.
