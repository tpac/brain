# Community Metadata Denormalization

**Status:** Finding + proposed direction (not yet implemented). Surfaced **2026-06-24** by the new community journal's first production run — the encoder's own `surprise` note caught `community_size` drifting. Brain nodes: `49b34921` (the bug), `e978b69f` (Phase-5 landing that made the journal live).

> **One-line:** the community encoder is told to hand-write five *structural* fields the graph already knows; it maintains them blind, so they drift — and the only authoritative consumer (the decoder) recomputes the real values from edges every cycle and never reads the stored copies. Make structural fields **derived**, not authored.

---

## 1. The finding

The community encoder agent (Haiku, interaction `s2_community_enrichment`) is instructed, on every `new_community` / `add_to_existing` / `merge` / `revise`, to set a batch of metadata fields. Five of them are **structural facts derivable from the `community_member` edges**:

| Field the encoder sets | Graph-derivable? | How | Who reads the *stored* value |
|---|---|---|---|
| `community_size` | **yes** | `COUNT(community_member edges)` | display only |
| `community_internal_fraction` | **yes** | decoder computes it fresh — `community_decoder.py:650` | display only |
| `community_is_corridor` | **yes** | decoder computes — `community_decoder.py:652` | display only |
| `community_dominant_type` | **yes** | mode of member node-types | display only |
| `community_members` (the id:title list) | **yes** | it *is* the member edge-targets | display **+ reconcile** (see §3) |
| `community_key_decisions` | no — judgment | — | display |
| `community_latest_development` | mostly judgment | — | display |
| `community_maturity` | semi-structural | int_frac thresholds + health flow | decoder health-update |

Prompt sites that instruct this: `servers/scales/s2/community_enrichment_prompt.py:120-133,154` ("set `community_size` to new member count (integer as string)", "`community_members` — ALL member IDs as id:title pairs", etc.).

## 2. The smell: stored copies the authority ignores

The **decoder** is the only thing that makes structural decisions about communities (clustering, placement, drift, merge, health). It computes membership and the structural metrics **fresh from the live edges every cycle** and never trusts the stored metadata:

- membership set = `get_community_members(...)` which reads `community_member` **edges** (`servers/dal.py:2957`), not the `community_members` string;
- `internal_fraction`, `is_corridor`, `member_count` are recomputed each run and passed in the proposal (`community_decoder.py:650-652`, `1215`, `1219-1220`);
- all the decoder's size/affinity/merge/health math is `len(members)` on the edge-derived set (`community_decoder.py:371,424,461,622,625,647,846,913,950-952`).

So the encoder-stored `community_size` / `community_internal_fraction` / `community_is_corridor` / `community_dominant_type` are **write-only, stale-the-moment-the-graph-changes snapshots** used only for display.

## 3. Why they drift (the `community_size` mechanism)

On an `add_to_existing` / `merge`, the encoder is told to write the *new* count. But it **cannot see the true current count** — what it sees in the rendered community node is the **stale stored `community_size`**. It reads a wrong base, guesses the delta, and writes a new wrong value, compounding each cycle.

Observed in production (the journal note that started this, `49b34921`): community `fe73f0b8` went `13→9` on two member-adds (should be 15); `2e6986a2` `16→14` on one add (should be 17); another `35→16` after a revision. The note's own words: *"Revision ops don't know prior-to-call member count."*

## 4. The `community_members` nuance (not pure vanity)

`community_members` is the same denormalization — encoder-authored, derivable from edges — **but it has a real consumer.** `reconcile_community_membership` (`servers/dal.py:2815`) reads it as the community's **"declared intent"** to **back-fill `community_member` edges for ZERO-edge orphan communities** — the case where Haiku creates the community node + metadata but **omits the `connect_to` edges entirely** (a known failure mode; reconcile is scoped to the all-or-nothing zero-edge case only).

So the list is a **band-aid for unreliable edge-writing**, not just a display copy. The deeper smell: *we keep a hand-written membership list because the real edge-write is flaky — and the band-aid itself drifts.*

## 5. Blast radius — why this is low-risk to fix

- **Decoder decisions: ZERO impact.** Membership and structure come from edges, never from `community_size`/`community_internal_fraction`/`community_is_corridor`. The bug **never corrupted a clustering / drift / merge / health decision** — it was display-only the entire time. *(This is the key safety property.)*
- **Display (dashboard run-cards, recall render of a community node): corrected.** `community_size` appears in the render field list (`servers/contract.py:559`) — it's metadata, not a recall ranking input.
- **`reconcile_community_membership`: unaffected** by `community_size`; it uses `community_members` (intent) + edges.
- **Encoder prompt: simplified** — dropping the structural-field instructions is a prompt change (register → activate → `./dev sync-prompts`).

## 6. Proposed direction

1. **Structural fields → derived, not authored.** For `community_size` / `community_internal_fraction` / `community_is_corridor` / `community_dominant_type`:
   - **Read-time derivation** (simplest, can't-drift): compute from the live graph wherever displayed; stop storing them (or keep-but-recompute). One cheap indexed COUNT per render.
   - **OR per-cycle stamp**: broaden the existing integrity pass (`reconcile_community_membership`, which already runs every cycle) to stamp these from the graph on **every** live community, not just zero-edge orphans.
   - Remove the "set these" instructions from the prompt; optionally have the dispatch drop any encoder-supplied structural values as a belt.
2. **`community_members` → keep only while edge-writing is unreliable.** It's the orphan-recovery source. The real fix is making the encoder's edge-write reliable (Haiku reliably emits `connect_to` member edges), after which the list becomes derivable too. **Treat as a separate decision** from the structural-field cleanup.
3. **Encoder authors only judgment:** narrative/`content`, `community_key_decisions`, `community_latest_development`, `community_maturity` (semi-structural). Everything countable, the graph already knows.

## 7. Open decisions for the implementer

- Read-time derive **vs** per-cycle stamp for the structural fields?
- Drop the stored fields entirely **vs** keep-but-derive (cheap display reads)?
- **Verify first** (grep): confirm *nothing* on a decision path reads the stored `community_internal_fraction` / `community_is_corridor` / `community_dominant_type` before removing them — §2 confirms the decoder *computes* them fresh, but the implementer should prove no other reader exists.
- Migration of existing wrong values: self-correct on read (read-time flavor) vs a one-time backfill.
- Tackle edge-write reliability (the `community_members` root) now, or defer?
- Eval: an IsolatedBrain run confirming derived values match the true edge counts + the prompt change doesn't regress community-creation quality.

## 8. References

- Prompt: `servers/scales/s2/community_enrichment_prompt.py:120-133,154`
- Decoder (recomputes structure from edges): `servers/scales/s2/community_decoder.py:650-652,1215,1219-1220` and `:371,461,622,950`
- Edge-truth read: `servers/dal.py:2957` (`get_community_members`)
- Orphan backfill from the list: `servers/dal.py:2815` (`reconcile_community_membership`)
- Display field list: `servers/contract.py:559`
- Brain: `49b34921` (the journal-surfaced bug), `e978b69f` (Phase-5 landing), `fe73f0b8` (a drifted community)

*This finding is itself a validation of the encoder-journal redesign: the residue review caught a real architectural smell on its first live community run, unprompted.*
